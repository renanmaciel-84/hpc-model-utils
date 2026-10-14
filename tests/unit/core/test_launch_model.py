"""ADR-051/ADR-009: LaunchSpec rendering into a resubmittable model.sbatch."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.launch import (
    Launcher,
    LaunchSpec,
    Resources,
    Toolchain,
    format_time_limit,
    render_model_script,
    render_prelude,
    write_model_script,
)
from hpc_model_utils.core.workspace import Workspace
from tests.support.launch_helpers import run_script, toolchain

_REGEN_ENV = "HPCMU_REGEN_GOLDENS"
_GOLDEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "goldens"
    / "v2"
    / "sbatch"
    / "model_mpiexec.sbatch"
)
_SRUN_PMIX_GOLDEN_PATH = _GOLDEN_PATH.with_name("model_srun_pmix.sbatch")


def _write_stub(bin_dir: Path, name: str, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / name
    stub.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    stub.chmod(0o755)
    return stub


# -- format_time_limit -------------------------------------------------


def test_format_time_limit_multiple_days_returns_d_hh_mm_ss() -> None:
    assert format_time_limit(24) == "1-00:00:00"


def test_format_time_limit_under_one_day_returns_zero_days() -> None:
    assert format_time_limit(5) == "0-05:00:00"


# -- Launcher ------------------------------------------------------------


def test_launcher_members_are_direct_mpiexec_hydra_srun_pmix() -> None:
    assert {member.value for member in Launcher} == {
        "DIRECT",
        "MPIEXEC_HYDRA",
        "SRUN_PMIX",
    }


# -- Resources -------------------------------------------------------------


def test_resources_invalid_queue_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="queue"):
        Resources(queue="-bad", cores=1)


@pytest.mark.parametrize("queue", ["batch\n", "batch\r\n", "\nbatch"])
def test_resources_queue_trailing_newline_raises_usage_error(
    queue: str,
) -> None:
    with pytest.raises(UsageError, match="queue"):
        Resources(queue=queue, cores=1)


def test_resources_cores_less_than_one_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="cores"):
        Resources(queue="batch", cores=0)


def test_resources_valid_values_construct() -> None:
    res = Resources(queue="batch", cores=4, time_limit_hours=24)
    assert res.queue == "batch"
    assert res.cores == 4


# -- Toolchain -------------------------------------------------------------


def test_toolchain_relative_path_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="mpich_bin"):
        Toolchain(
            mpich_bin=Path("relative/bin"),
            slurm_bin=Path("/opt/slurm/bin"),
            cli_bin=Path("/opt/hpcmu/bin"),
        )


def test_toolchain_mpich_lib_trailing_slash_free_sibling_dir() -> None:
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path("/opt/hpcmu/bin"),
    )
    assert tools.mpich_lib == Path("/opt/mpich/lib")
    assert str(tools.mpich_lib) == "/opt/mpich/lib"


def test_toolchain_mpich_lib_nested_bin_path() -> None:
    tools = Toolchain(
        mpich_bin=Path("/shared-zfs/mpich-4.2.3-install/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path("/opt/hpcmu/bin"),
    )
    assert tools.mpich_lib == Path("/shared-zfs/mpich-4.2.3-install/lib")


def test_toolchain_python_bin_is_sibling_of_cli_bin() -> None:
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path(
            "/shared-zfs/tools/hpc-model-utils/abc/.venv/bin/hpc-model-utils"
        ),
    )
    assert tools.python_bin == Path(
        "/shared-zfs/tools/hpc-model-utils/abc/.venv/bin/python"
    )


def test_toolchain_python_bin_missing_interpreter_raises_nothing(
    tmp_path: Path,
) -> None:
    cli_bin = tmp_path / "bin" / "hpc-model-utils"
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=cli_bin,
    )
    assert tools.python_bin == tmp_path / "bin" / "python"
    assert not tmp_path.joinpath("bin").exists()


def test_toolchain_python_bin_symlinked_interpreter_is_not_resolved(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli_bin = bin_dir / "hpc-model-utils"
    cli_bin.write_text("", encoding="utf-8")
    (bin_dir / "python").symlink_to(sys.executable)
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=cli_bin,
    )
    assert tools.python_bin == bin_dir / "python"
    assert tools.python_bin != (bin_dir / "python").resolve()


# -- LaunchSpec.__post_init__ ---------------------------------------------


def test_launch_spec_empty_argv_raises_value_error() -> None:
    with pytest.raises(ValueError, match="argv"):
        LaunchSpec(launcher=Launcher.DIRECT, argv=())


def test_launch_spec_relative_argv0_raises_value_error() -> None:
    with pytest.raises(ValueError, match="absolute"):
        LaunchSpec(launcher=Launcher.DIRECT, argv=("relative/bin",))


def test_launch_spec_ntasks_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ntasks"):
        LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",), ntasks=0)


def test_launch_spec_nodes_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="nodes"):
        LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",), nodes=0)


def test_launch_spec_ntasks_per_node_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ntasks_per_node"):
        LaunchSpec(
            launcher=Launcher.DIRECT,
            argv=("/bin/true",),
            ntasks_per_node=0,
        )


def test_launch_spec_cpus_per_task_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="cpus_per_task"):
        LaunchSpec(
            launcher=Launcher.DIRECT, argv=("/bin/true",), cpus_per_task=0
        )


def test_launch_spec_mpiexec_hydra_without_ntasks_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ntasks"):
        LaunchSpec(launcher=Launcher.MPIEXEC_HYDRA, argv=("/bin/true",))


def test_launch_spec_srun_pmix_without_nodes_raises_value_error() -> None:
    with pytest.raises(ValueError, match="nodes"):
        LaunchSpec(launcher=Launcher.SRUN_PMIX, argv=("/bin/true",))


def test_launch_spec_mpiexec_hydra_with_ntasks_constructs() -> None:
    spec = LaunchSpec(
        launcher=Launcher.MPIEXEC_HYDRA, argv=("/bin/true",), ntasks=1
    )
    assert spec.ntasks == 1


def test_launch_spec_srun_pmix_with_nodes_constructs() -> None:
    spec = LaunchSpec(launcher=Launcher.SRUN_PMIX, argv=("/bin/true",), nodes=1)
    assert spec.nodes == 1


# -- render_prelude ---------------------------------------------------------


def test_render_prelude_line_order(tmp_path: Path) -> None:
    tools = toolchain(tmp_path)
    lines = render_prelude(tools).split("\n")
    assert lines[0] == "set -u"
    assert lines[1] == (
        f'export PATH={tools.mpich_bin}:{tools.slurm_bin}:"$PATH"'
    )
    assert lines[2] == "export HPCMU_PLATFORM=off"
    assert lines[3] == "unset PYTHONPATH PYTHONHOME"
    assert lines[4] == "export PYTHONSAFEPATH=1"
    assert lines[5].startswith('echo "HPCMU_START job=')
    assert lines[6].startswith('echo "HPCMU_SHAPE nodes=')
    assert len(lines) == 7


# -- render_model_script: root-character rejection (amendment) -------------


def test_render_model_script_root_with_percent_raises_usage_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "a%b"
    root.mkdir()
    ws = Workspace.at(root)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    with pytest.raises(UsageError, match="%"):
        render_model_script(ws, spec, res, toolchain(tmp_path))


def test_render_model_script_root_with_quote_raises_usage_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "a'b"
    root.mkdir()
    ws = Workspace.at(root)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    with pytest.raises(UsageError, match="'"):
        render_model_script(ws, spec, res, toolchain(tmp_path))


# -- render_model_script: env validation ------------------------------------


def test_render_model_script_env_ld_library_path_key_raises_usage_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.DIRECT,
        argv=("/bin/true",),
        env={"LD_LIBRARY_PATH": "/x"},
    )
    res = Resources(queue="batch", cores=1)
    with pytest.raises(UsageError, match="LD_LIBRARY_PATH"):
        render_model_script(ws, spec, res, toolchain(tmp_path))


def test_render_model_script_invalid_env_key_raises_usage_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.DIRECT, argv=("/bin/true",), env={"bad-key": "1"}
    )
    res = Resources(queue="batch", cores=1)
    with pytest.raises(UsageError, match="bad-key"):
        render_model_script(ws, spec, res, toolchain(tmp_path))


@pytest.mark.parametrize("key", ["FOO\n", "FOO\r\n", "\nFOO"])
def test_render_model_script_env_key_trailing_newline_raises_usage_error(
    tmp_path: Path, key: str
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.DIRECT, argv=("/bin/true",), env={key: "1"}
    )
    res = Resources(queue="batch", cores=1)
    with pytest.raises(UsageError, match=re.escape(repr(key))):
        render_model_script(ws, spec, res, toolchain(tmp_path))


def test_render_model_script_env_exports_sorted(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.DIRECT,
        argv=("/bin/true",),
        env={"ZEBRA": "1", "ALPHA": "a b", "MID": "x"},
    )
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    lines = rendered.split("\n")
    alpha_idx = lines.index("export ALPHA='a b'")
    mid_idx = lines.index("export MID=x")
    zebra_idx = lines.index("export ZEBRA=1")
    assert alpha_idx < mid_idx < zebra_idx


# -- render_model_script: job-name fallback ---------------------------------


def test_render_model_script_job_name_uses_valid_root_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study-01"
    root.mkdir()
    ws = Workspace.at(root)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "#SBATCH --job-name=study-01" in rendered


def test_render_model_script_job_name_fallback_for_invalid_characters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study with spaces"
    root.mkdir()
    ws = Workspace(root.resolve())
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "#SBATCH --job-name=hpcmu" in rendered


def test_render_model_script_job_name_fallback_for_trailing_newline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study\n"
    root.mkdir()
    ws = Workspace(root.resolve())
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "#SBATCH --job-name=hpcmu" in rendered


# -- render_model_script: optional flags / --exclusive ----------------------


def test_render_model_script_optional_sbatch_flags_omitted_when_none(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "--ntasks=" not in rendered
    assert "--nodes=" not in rendered
    assert "--ntasks-per-node=" not in rendered
    assert "--time=" not in rendered
    assert "#SBATCH --cpus-per-task=1" in rendered


@pytest.mark.parametrize(
    "spec",
    [
        LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",)),
        LaunchSpec(
            launcher=Launcher.MPIEXEC_HYDRA, argv=("/bin/true",), ntasks=1
        ),
        LaunchSpec(launcher=Launcher.SRUN_PMIX, argv=("/bin/true",), nodes=1),
    ],
)
def test_render_model_script_exclusive_present_for_every_launcher(
    tmp_path: Path, spec: LaunchSpec
) -> None:
    ws = Workspace.at(tmp_path)
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "#SBATCH --exclusive" in rendered


# -- render_model_script: per-launcher launch line (inline expected text) --


def test_render_model_script_direct_launcher_runs_argv_directly(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true", "-x"))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "\n/bin/true -x\nrc=$?\n" in rendered


def test_render_model_script_mpiexec_hydra_ntasks_one_runs_argv_directly(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.MPIEXEC_HYDRA, argv=("/bin/true",), ntasks=1
    )
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "\n/bin/true\nrc=$?\n" in rendered
    assert "mpiexec" not in rendered


def test_render_model_script_mpiexec_hydra_multi_task_uses_mpiexec(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.MPIEXEC_HYDRA, argv=("/bin/true",), ntasks=4
    )
    res = Resources(queue="batch", cores=4)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert '\nmpiexec -np "$SLURM_NTASKS" /bin/true\nrc=$?\n' in rendered


def test_render_model_script_srun_pmix_launch_line_expected_text(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(launcher=Launcher.SRUN_PMIX, argv=("/bin/true",), nodes=2)
    res = Resources(queue="batch", cores=2)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    expected = (
        'srun --mpi=pmix --ntasks-per-node=1 --cpus-per-task="'
        '$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 /bin/true'
    )
    assert expected in rendered
    assert rendered.split(expected)[1].startswith("\nrc=$?\n")


# -- render_model_script: library-path seam ----------------------------------


def test_render_model_script_libpath_seam_off_by_default(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, toolchain(tmp_path))
    assert "LD_LIBRARY_PATH" not in rendered


def test_render_model_script_libpath_seam_on_renders_expansion(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    spec = LaunchSpec(
        launcher=Launcher.DIRECT,
        argv=("/bin/true",),
        prepend_mpich_lib=True,
    )
    res = Resources(queue="batch", cores=1)
    rendered = render_model_script(ws, spec, res, tools)
    expected = (
        f"export LD_LIBRARY_PATH={tools.mpich_lib}"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    )
    assert expected in rendered
    assert not expected.endswith(":")


# -- write_model_script --------------------------------------------------


def test_write_model_script_creates_executable_file(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    path = write_model_script(ws, spec, res, toolchain(tmp_path))
    assert path == ws.jobs_dir / "model.sbatch"
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o755


# -- AC2: the regenerable NEWAVE-shaped golden -------------------------------


def test_render_model_script_mpiexec_hydra_matches_golden(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study-newave"
    root.mkdir()
    ws = Workspace.at(root)
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path("/opt/hpcmu/bin"),
    )
    spec = LaunchSpec(
        launcher=Launcher.MPIEXEC_HYDRA,
        argv=(str(ws.root / "assets" / "newave"),),
        ntasks=64,
        ntasks_per_node=32,
        cpus_per_task=2,
    )
    res = Resources(queue="batch", cores=64, time_limit_hours=24)
    rendered = render_model_script(ws, spec, res, tools)
    normalized = rendered.replace(str(ws.root), "<WS>")

    if os.environ.get(_REGEN_ENV) == "1":
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN_PATH.write_text(normalized, encoding="utf-8")
        pytest.fail(
            f"{_REGEN_ENV}=1: regenerated {_GOLDEN_PATH}; "
            "re-run without it to verify"
        )

    expected = _GOLDEN_PATH.read_text(encoding="utf-8")
    assert normalized == expected
    assert "#SBATCH --exclusive" in rendered
    assert "LD_LIBRARY_PATH" not in rendered

    script_path = tmp_path / "golden-check.sbatch"
    script_path.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_render_model_script_srun_pmix_matches_golden(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study-cobre"
    root.mkdir()
    ws = Workspace.at(root)
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path("/opt/hpcmu/bin"),
    )
    spec = LaunchSpec(
        launcher=Launcher.SRUN_PMIX,
        argv=(
            str(ws.root / "assets" / "cobre-mpi"),
            "run",
            str(ws.root),
            "--threads",
            "96",
            "--comm-backend",
            "mpi",
        ),
        nodes=4,
        ntasks_per_node=1,
        cpus_per_task=96,
        prepend_mpich_lib=True,
        env={"FI_PROVIDER": "efa"},
    )
    res = Resources(
        queue="batch", cores=384, max_cores_per_node=96, time_limit_hours=24
    )
    rendered = render_model_script(ws, spec, res, tools)
    normalized = rendered.replace(str(ws.root), "<WS>")

    if os.environ.get(_REGEN_ENV) == "1":
        _SRUN_PMIX_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SRUN_PMIX_GOLDEN_PATH.write_text(normalized, encoding="utf-8")
        pytest.fail(
            f"{_REGEN_ENV}=1: regenerated {_SRUN_PMIX_GOLDEN_PATH}; "
            "re-run without it to verify"
        )

    expected = _SRUN_PMIX_GOLDEN_PATH.read_text(encoding="utf-8")
    assert normalized == expected
    assert "srun --mpi=pmix --ntasks-per-node=1" in rendered
    assert "export FI_PROVIDER=efa" in rendered
    assert "export LD_LIBRARY_PATH=" in rendered

    script_path = tmp_path / "golden-check.sbatch"
    script_path.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script_path)], check=True)


# -- AC3: library-path seam executed under bash with a stub srun -----------


def test_write_model_script_srun_pmix_libpath_seam_with_existing_value(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    _write_stub(tools.slurm_bin, "srun", 'printf %s "${LD_LIBRARY_PATH:-}"')
    spec = LaunchSpec(
        launcher=Launcher.SRUN_PMIX,
        argv=("/bin/true",),
        nodes=4,
        prepend_mpich_lib=True,
    )
    res = Resources(queue="batch", cores=4)
    script = write_model_script(ws, spec, res, tools)
    rendered = script.read_text(encoding="utf-8")
    assert rendered.startswith("#!/bin/bash\n#SBATCH")
    assert "srun --mpi=pmix --ntasks-per-node=1" in rendered

    result = run_script(
        script,
        tmp_path,
        {"LD_LIBRARY_PATH": "/efa/lib", "SLURM_CPUS_PER_TASK": "4"},
    )
    assert result.returncode == 0, result.stderr
    stub_output = result.stdout.split("\n", 2)[-1]
    assert stub_output == f"{tools.mpich_lib}:/efa/lib"


def test_write_model_script_srun_pmix_libpath_seam_without_existing_value(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    _write_stub(tools.slurm_bin, "srun", 'printf %s "${LD_LIBRARY_PATH:-}"')
    spec = LaunchSpec(
        launcher=Launcher.SRUN_PMIX,
        argv=("/bin/true",),
        nodes=4,
        prepend_mpich_lib=True,
    )
    res = Resources(queue="batch", cores=4)
    script = write_model_script(ws, spec, res, tools)

    env = {**os.environ, "SLURM_CPUS_PER_TASK": "4"}
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    stub_output = result.stdout.split("\n", 2)[-1]
    assert stub_output == str(tools.mpich_lib)


# -- AC4: hostile argv arrives as one argument -------------------------------


def test_write_model_script_hostile_argv_arrives_as_one_argument(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    _write_stub(tools.mpich_bin, "mpiexec", 'printf %s "${@: -1}"')
    hostile = "/ws/assets/new ave;rm -rf x"
    spec = LaunchSpec(
        launcher=Launcher.MPIEXEC_HYDRA, argv=(hostile,), ntasks=2
    )
    res = Resources(queue="batch", cores=2)
    script = write_model_script(ws, spec, res, tools)

    result = run_script(script, tmp_path, {"SLURM_NTASKS": "2"})
    assert result.returncode == 0, result.stderr
    stub_output = result.stdout.split("\n", 2)[-1]
    assert stub_output == hostile
    assert not (tmp_path / "x").exists()


# -- AC5: stale model.exit removal and exit-code capture ---------------------


def test_write_model_script_stale_model_exit_replaced_by_launch_rc(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    stub = _write_stub(tmp_path / "bin", "model_stub", "exit 7")
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=(str(stub),))
    res = Resources(queue="batch", cores=1)
    script = write_model_script(ws, spec, res, tools)
    ws.model_exit_path.write_text("3", encoding="utf-8")

    result = run_script(script, tmp_path, {})
    assert result.returncode == 7
    assert ws.model_exit_path.read_text(encoding="utf-8") == "7\n"


def test_write_model_script_killed_before_capture_leaves_no_model_exit(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    stub = _write_stub(tmp_path / "bin", "model_kill", 'kill -9 "$PPID"')
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=(str(stub),))
    res = Resources(queue="batch", cores=1)
    script = write_model_script(ws, spec, res, tools)
    ws.model_exit_path.write_text("3", encoding="utf-8")

    result = run_script(script, tmp_path, {})
    assert result.returncode < 0
    assert not ws.model_exit_path.exists()


# -- AC6: the HPCMU_START / HPCMU_SHAPE lines --------------------------------


def test_write_model_script_shape_line_fields_with_mem_per_node_zero(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    spec = LaunchSpec(launcher=Launcher.DIRECT, argv=("/bin/true",))
    res = Resources(queue="batch", cores=1)
    script = write_model_script(ws, spec, res, tools)
    rendered = script.read_text(encoding="utf-8")
    assert "/sys/fs/cgroup" not in rendered

    result = run_script(script, tmp_path, {"SLURM_MEM_PER_NODE": "0"})
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert re.match(r"^HPCMU_START job=", lines[0])
    assert re.fullmatch(
        r"HPCMU_SHAPE .* mem_per_node=0 mem_total_kb=[0-9]+", lines[1]
    )


# -- bash -n over every rendered variant -------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        LaunchSpec(
            launcher=Launcher.DIRECT,
            argv=("/bin/true",),
            prepend_mpich_lib=True,
            env={"FOO": "bar"},
        ),
        LaunchSpec(
            launcher=Launcher.MPIEXEC_HYDRA,
            argv=("/bin/true",),
            ntasks=1,
            prepend_mpich_lib=True,
            env={"FOO": "bar"},
        ),
        LaunchSpec(
            launcher=Launcher.MPIEXEC_HYDRA,
            argv=("/bin/true",),
            ntasks=4,
            prepend_mpich_lib=True,
            env={"FOO": "bar"},
        ),
        LaunchSpec(
            launcher=Launcher.SRUN_PMIX,
            argv=("/bin/true",),
            nodes=2,
            prepend_mpich_lib=True,
            env={"FOO": "bar"},
        ),
    ],
)
def test_render_model_script_bash_n_on_every_variant(
    tmp_path: Path, spec: LaunchSpec
) -> None:
    ws = Workspace.at(tmp_path)
    tools = toolchain(tmp_path)
    res = Resources(queue="batch", cores=4, time_limit_hours=10)
    rendered = render_model_script(ws, spec, res, tools)
    script_path = tmp_path / f"variant-{spec.launcher}.sbatch"
    script_path.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script_path)], check=True)
