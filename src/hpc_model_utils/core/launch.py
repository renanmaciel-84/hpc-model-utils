"""ADR-051/ADR-009: render a resubmittable model.sbatch from a LaunchSpec."""

from __future__ import annotations

import enum
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.workspace import Phase, Workspace

_QUEUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_ENV_KEY_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*")
_ROOT_FORBIDDEN_CHARS = frozenset("%\"'\\")
_DEFAULT_JOB_NAME = "hpcmu"
_FINALIZE_MODEL_PATTERN = re.compile(r"^[a-z0-9_]+$")
_FINALIZE_JOB_ID_PATTERN = re.compile(r"^[0-9]+$")
_FINALIZE_GUARDS = (
    "ulimit -u unlimited 2>/dev/null || ulimit -u 65535 2>/dev/null || true\n"
    "export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
)

_BODY_TEMPLATE = Template(
    "$prelude\n"
    "rm -f $exit_path\n"
    "$libpath$env$launch\n"
    "rc=$$?\n"
    "printf '%s\\n' \"$$rc\" > $exit_path\n"
    'exit "$$rc"\n'
)


class Launcher(enum.StrEnum):
    DIRECT = "DIRECT"
    MPIEXEC_HYDRA = "MPIEXEC_HYDRA"
    SRUN_PMIX = "SRUN_PMIX"


@dataclass(frozen=True, slots=True)
class Resources:
    queue: str
    cores: int
    max_cores_per_node: int | None = None
    time_limit_hours: int | None = None

    def __post_init__(self) -> None:
        if not _QUEUE_PATTERN.fullmatch(self.queue):
            raise UsageError(
                f"queue must match {_QUEUE_PATTERN.pattern}: {self.queue!r}"
            )
        if self.cores < 1:
            raise UsageError(f"cores must be at least 1: {self.cores}")


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    launcher: Launcher
    argv: tuple[str, ...]
    ntasks: int | None = None
    nodes: int | None = None
    ntasks_per_node: int | None = None
    cpus_per_task: int = 1
    prepend_mpich_lib: bool = False
    env: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must be non-empty")
        if not Path(self.argv[0]).is_absolute():
            raise ValueError(f"argv[0] must be absolute: {self.argv[0]!r}")
        for name, value in (
            ("ntasks", self.ntasks),
            ("nodes", self.nodes),
            ("ntasks_per_node", self.ntasks_per_node),
            ("cpus_per_task", self.cpus_per_task),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1: {value}")
        if self.launcher is Launcher.MPIEXEC_HYDRA and self.ntasks is None:
            raise ValueError("MPIEXEC_HYDRA requires ntasks")
        if self.launcher is Launcher.SRUN_PMIX and self.nodes is None:
            raise ValueError("SRUN_PMIX requires nodes")


@dataclass(frozen=True, slots=True)
class Toolchain:
    mpich_bin: Path
    slurm_bin: Path
    cli_bin: Path

    def __post_init__(self) -> None:
        for name, value in (
            ("mpich_bin", self.mpich_bin),
            ("slurm_bin", self.slurm_bin),
            ("cli_bin", self.cli_bin),
        ):
            if not value.is_absolute():
                raise UsageError(f"{name} must be absolute: {value}")

    @property
    def mpich_lib(self) -> Path:
        return self.mpich_bin.parent / "lib"

    @property
    def python_bin(self) -> Path:
        return self.cli_bin.parent / "python"


def format_time_limit(hours: int) -> str:
    return f"{hours // 24}-{hours % 24:02d}:00:00"


def render_prelude(tools: Toolchain) -> str:
    mpich_bin = shlex.quote(str(tools.mpich_bin))
    slurm_bin = shlex.quote(str(tools.slurm_bin))
    lines = [
        "set -u",
        f'export PATH={mpich_bin}:{slurm_bin}:"$PATH"',
        "export HPCMU_PLATFORM=off",
        "unset PYTHONPATH PYTHONHOME",
        "export PYTHONSAFEPATH=1",
        (
            'echo "HPCMU_START job=${SLURM_JOB_ID:-?} '
            "restart=${SLURM_RESTART_COUNT:-0} "
            'at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"'
        ),
        (
            'echo "HPCMU_SHAPE nodes=${SLURM_JOB_NUM_NODES:-?} '
            "cpus_on_node=${SLURM_CPUS_ON_NODE:-?} nproc=$(nproc) "
            "mem_per_node=${SLURM_MEM_PER_NODE:-?} "
            "mem_total_kb=$(awk '/^MemTotal:/ {print $2}' "
            '/proc/meminfo 2>/dev/null || echo ?)"'
        ),
    ]
    return "\n".join(lines)


def _check_root_characters(ws: Workspace) -> None:
    root_str = str(ws.root)
    if any(ch in root_str for ch in _ROOT_FORBIDDEN_CHARS):
        raise UsageError(f"workspace root contains %, \", ' or \\: {ws.root}")


def _job_name(ws: Workspace) -> str:
    name = ws.root.name
    return name if _JOB_NAME_PATTERN.fullmatch(name) else _DEFAULT_JOB_NAME


def _render_sbatch_lines(
    pairs: Sequence[tuple[str, str | None]],
) -> list[str]:
    lines = ["#!/bin/bash"]
    for flag, value in pairs:
        if value is None:
            continue
        if value == "":
            lines.append(f"#SBATCH {flag}")
        else:
            lines.append(f"#SBATCH {flag}={value}")
    return lines


def _render_libpath_seam(tools: Toolchain) -> str:
    quoted_lib = shlex.quote(str(tools.mpich_lib))
    return (
        f"export LD_LIBRARY_PATH={quoted_lib}"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n"
    )


def _render_env_exports(env: Mapping[str, str]) -> str:
    for key in env:
        if key == "LD_LIBRARY_PATH":
            raise UsageError(
                "LD_LIBRARY_PATH must not be set via spec.env; "
                "use prepend_mpich_lib instead"
            )
        if not _ENV_KEY_PATTERN.fullmatch(key):
            raise UsageError(
                f"env key {key!r} must match {_ENV_KEY_PATTERN.pattern}"
            )
    lines = [f"export {key}={shlex.quote(env[key])}" for key in sorted(env)]
    return "\n".join(lines) + ("\n" if lines else "")


def _render_launch_line(spec: LaunchSpec) -> str:
    quoted_argv = " ".join(shlex.quote(item) for item in spec.argv)
    if spec.launcher is Launcher.DIRECT:
        return quoted_argv
    if spec.launcher is Launcher.MPIEXEC_HYDRA:
        if spec.ntasks == 1:
            return quoted_argv
        return f'mpiexec -np "$SLURM_NTASKS" {quoted_argv}'
    return (
        "srun --mpi=pmix --ntasks-per-node=1 "
        '--cpus-per-task="$SLURM_CPUS_PER_TASK" '
        f"--kill-on-bad-exit=1 {quoted_argv}"
    )


def render_model_script(
    ws: Workspace, spec: LaunchSpec, res: Resources, tools: Toolchain
) -> str:
    _check_root_characters(ws)
    sbatch_pairs: list[tuple[str, str | None]] = [
        ("--job-name", _job_name(ws)),
        ("--partition", res.queue),
        ("--exclusive", ""),
        ("--ntasks", None if spec.ntasks is None else str(spec.ntasks)),
        ("--nodes", None if spec.nodes is None else str(spec.nodes)),
        (
            "--ntasks-per-node",
            None if spec.ntasks_per_node is None else str(spec.ntasks_per_node),
        ),
        ("--cpus-per-task", str(spec.cpus_per_task)),
        (
            "--time",
            None
            if res.time_limit_hours is None
            else format_time_limit(res.time_limit_hours),
        ),
        ("--chdir", str(ws.root)),
        ("--output", str(ws.log_pattern(Phase.MODEL))),
    ]
    sbatch_block = "\n".join(_render_sbatch_lines(sbatch_pairs))
    body = _BODY_TEMPLATE.substitute(
        prelude=render_prelude(tools),
        exit_path=shlex.quote(str(ws.model_exit_path)),
        libpath=_render_libpath_seam(tools) if spec.prepend_mpich_lib else "",
        env=_render_env_exports(spec.env),
        launch=_render_launch_line(spec),
    )
    return f"{sbatch_block}\n{body}"


def write_model_script(
    ws: Workspace, spec: LaunchSpec, res: Resources, tools: Toolchain
) -> Path:
    ws.ensure_layout()
    path = ws.job_script(Phase.MODEL)
    path.write_text(render_model_script(ws, spec, res, tools), encoding="utf-8")
    path.chmod(0o755)
    return path


@dataclass(frozen=True, slots=True)
class FinalizeInvocation:
    model: str
    model_job_id: str | None
    cores: int
    synthesis_bin: Path | None

    def __post_init__(self) -> None:
        if not _FINALIZE_MODEL_PATTERN.fullmatch(self.model):
            raise UsageError(
                f"model must match {_FINALIZE_MODEL_PATTERN.pattern}: "
                f"{self.model!r}"
            )
        if (
            self.model_job_id is not None
            and not _FINALIZE_JOB_ID_PATTERN.fullmatch(self.model_job_id)
        ):
            raise UsageError(
                "model_job_id must be None or match "
                f"{_FINALIZE_JOB_ID_PATTERN.pattern}: {self.model_job_id!r}"
            )
        if self.cores < 1:
            raise UsageError(f"cores must be at least 1: {self.cores}")
        if (
            self.synthesis_bin is not None
            and not self.synthesis_bin.is_absolute()
        ):
            raise UsageError(
                f"synthesis_bin must be absolute: {self.synthesis_bin}"
            )


def _render_finalize_exec_line(cli_bin: Path, inv: FinalizeInvocation) -> str:
    job_id = inv.model_job_id if inv.model_job_id is not None else "none"
    parts = [
        "exec",
        shlex.quote(str(cli_bin)),
        "finalize",
        shlex.quote(inv.model),
        "--model-job-id",
        shlex.quote(job_id),
        "--cores",
        shlex.quote(str(inv.cores)),
    ]
    if inv.synthesis_bin is not None:
        parts += ["--synthesis-bin", shlex.quote(str(inv.synthesis_bin))]
    return " ".join(parts)


def render_finalize_script(
    ws: Workspace, res: Resources, tools: Toolchain, inv: FinalizeInvocation
) -> str:
    _check_root_characters(ws)
    sbatch_pairs: list[tuple[str, str | None]] = [
        ("--job-name", _job_name(ws)),
        ("--partition", res.queue),
        ("--nodes", "1"),
        ("--ntasks", "1"),
        ("--exclusive", ""),
        ("--mem", "0"),
        (
            "--time",
            None
            if res.time_limit_hours is None
            else format_time_limit(res.time_limit_hours),
        ),
        ("--chdir", str(ws.root)),
        ("--output", str(ws.log_pattern(Phase.FINALIZE))),
    ]
    sbatch_block = "\n".join(_render_sbatch_lines(sbatch_pairs))
    body = (
        f"{render_prelude(tools)}\n"
        f"{_FINALIZE_GUARDS}\n"
        f"{_render_finalize_exec_line(tools.cli_bin, inv)}\n"
    )
    return f"{sbatch_block}\n{body}"


def write_finalize_script(
    ws: Workspace, res: Resources, tools: Toolchain, inv: FinalizeInvocation
) -> Path:
    ws.ensure_layout()
    path = ws.job_script(Phase.FINALIZE)
    path.write_text(
        render_finalize_script(ws, res, tools, inv), encoding="utf-8"
    )
    path.chmod(0o755)
    return path
