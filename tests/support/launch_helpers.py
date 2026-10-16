"""Shared ``Toolchain``/script-execution and sbatch-golden helpers for the
launch-related unit tests (tests/unit/core/test_launch_model.py,
tests/unit/core/test_launch_finalize.py and
tests/unit/models/decomp/test_decomp_plugin.py), which built
identical/near-identical copies of these independently."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from hpc_model_utils.core.launch import TelemetryConfig, Toolchain
from hpc_model_utils.core.workspace import Workspace

REGEN_ENV = "HPCMU_REGEN_GOLDENS"
SBATCH_GOLDEN_DIR = (
    Path(__file__).resolve().parents[1] / "goldens" / "v2" / "sbatch"
)
GOLDEN_TOOLS = Toolchain(
    mpich_bin=Path("/opt/mpich/bin"),
    slurm_bin=Path("/opt/slurm/bin"),
    cli_bin=Path("/opt/hpcmu/bin"),
)
GOLDEN_TELEMETRY = TelemetryConfig(
    enabled=True, interval=5, workspace_interval=60, tool_version="9.9.9"
)
RM_EXIT_GOLDEN_LINE = "rm -f <WS>/.hpcmu/jobs/model.exit"
_SIDECAR_LOG_TIMEOUT_S = 5
_SIDECAR_LOG_POLL_S = 0.05


def sidecar_golden_line(phase: str) -> str:
    return (
        'srun --overlap --nodes="${SLURM_JOB_NUM_NODES:-?}" '
        '--ntasks="${SLURM_JOB_NUM_NODES:-?}" --ntasks-per-node=1 '
        "--cpus-per-task=1 --cpu-bind=none --mpi=none --kill-on-bad-exit=0 "
        "--input=none --wait=0 --job-name=hpcmu-telemetry "
        f"--output=<WS>/.hpcmu/telemetry/{phase}-%j-r"
        '"${SLURM_RESTART_COUNT:-0}"-%t.log /opt/hpcmu/python -I -m '
        "hpc_model_utils.infra.telemetry.sampler --staging "
        f"<WS>/.hpcmu/telemetry --phase {phase} --workspace <WS> "
        "--interval 5 --workspace-interval 60 --tool-version 9.9.9 > "
        f'<WS>/.hpcmu/telemetry/{phase}-"${{SLURM_JOB_ID:-?}}"-r'
        '"${SLURM_RESTART_COUNT:-0}"-srun.log 2>&1 & disown'
    )


def toolchain(tmp_path: Path, *, cli_bin_name: str | None = None) -> Toolchain:
    cli_bin = tmp_path / "toolchain" / "cli" / "bin"
    if cli_bin_name is not None:
        cli_bin = cli_bin / cli_bin_name
    return Toolchain(
        mpich_bin=tmp_path / "toolchain" / "mpich" / "bin",
        slurm_bin=tmp_path / "toolchain" / "slurm" / "bin",
        cli_bin=cli_bin,
    )


def run_script(
    script_path: Path, tmp_path: Path, env_overrides: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def assert_matches_golden(
    rendered: str, ws: Workspace, golden_path: Path, tmp_path: Path
) -> None:
    normalized = rendered.replace(str(ws.root), "<WS>")
    if os.environ.get(REGEN_ENV) == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized, encoding="utf-8")
        pytest.fail(
            f"{REGEN_ENV}=1: regenerated {golden_path}; "
            "re-run without it to verify"
        )

    assert normalized == golden_path.read_text(encoding="utf-8")

    script_path = tmp_path / "golden-check.sbatch"
    script_path.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def assert_sidecar_after_rm_exit(on_golden: Path, off_golden: Path) -> None:
    off_lines = off_golden.read_text(encoding="utf-8").split("\n")
    on_lines = on_golden.read_text(encoding="utf-8").split("\n")
    rm_index = off_lines.index(RM_EXIT_GOLDEN_LINE)
    assert on_lines[rm_index + 1] == sidecar_golden_line("model")
    assert on_lines[: rm_index + 1] + on_lines[rm_index + 2 :] == off_lines


def await_text(path: Path, needle: str) -> str:
    deadline = time.monotonic() + _SIDECAR_LOG_TIMEOUT_S
    while True:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if needle in text:
            return text
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{needle!r} not in {path} within "
                f"{_SIDECAR_LOG_TIMEOUT_S} s: {text!r}"
            )
        time.sleep(_SIDECAR_LOG_POLL_S)
