"""ADR-057/ADR-051/R119/R135/R138: the cobre launch shape (ticket-069)."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

import pytest

from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.launch import (
    Launcher,
    LaunchSpec,
    Resources,
    Toolchain,
    render_model_script,
)
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models.cobre import CobrePlugin
from tests.support.cobre_case import cobre_workspace, write_elf_stub

_FLAG_TEXT = (
    "cobre needs --max-cores-per-node: threads per node T = "
    "max_cores_per_node, nodes K = cores / T (400 cores with 100 per node "
    "gives 4 nodes x 100 threads)"
)
_MISSING_TEXT = "novomodelo-mpi missing from the fetched cobre executables"
_SRUN = (
    "srun --mpi=pmix --ntasks-per-node=1 "
    '--cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 '
)
_TOOLS = Toolchain(
    mpich_bin=Path("/opt/mpich/bin"),
    slurm_bin=Path("/opt/slurm/bin"),
    cli_bin=Path("/opt/cli/hpc-model-utils"),
)


def _workspace(tmp_path: Path, *, top: str | None = "caso_cobre") -> Workspace:
    ws = cobre_workspace(tmp_path, top=top).ws
    write_elf_stub(ws.assets)
    return ws


# -- max_cores ----------------------------------------------------------


def test_max_cores_flag_is_required(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    with pytest.raises(DataError) as excinfo:
        CobrePlugin().launch(ws, Resources("fila", 400))
    assert str(excinfo.value) == _FLAG_TEXT


@pytest.mark.parametrize(("cores", "threads"), [(250, 100), (50, 100)])
def test_max_cores_not_a_multiple_raises_data_error(
    tmp_path: Path, cores: int, threads: int
) -> None:
    ws = _workspace(tmp_path)
    res = Resources("fila", cores, max_cores_per_node=threads)
    with pytest.raises(DataError) as excinfo:
        CobrePlugin().launch(ws, res)
    assert str(excinfo.value) == (
        "cobre needs cores to be a multiple of --max-cores-per-node: "
        f"{cores} is not a multiple of {threads}"
    )
    assert str(tmp_path) not in str(excinfo.value)


def test_max_cores_equal_to_cores_is_one_node(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    spec = CobrePlugin().launch(
        ws, Resources("fila", 100, max_cores_per_node=100)
    )
    assert (spec.nodes, spec.cpus_per_task) == (1, 100)


# -- missing ------------------------------------------------------------


def _stray_cobre(assets: Path) -> None:
    write_elf_stub(assets, name="cobre")


def _directory(assets: Path) -> None:
    (assets / "novomodelo-mpi").mkdir(parents=True)


@pytest.mark.parametrize(
    "arrange",
    [lambda assets: None, _stray_cobre, _directory],
    ids=["empty-assets", "stray-cobre", "directory"],
)
def test_missing_cobre_mpi_raises_data_error(
    tmp_path: Path, arrange: Callable[[Path], None]
) -> None:
    ws = cobre_workspace(tmp_path).ws
    arrange(ws.assets)
    res = Resources("fila", 100, max_cores_per_node=100)
    with pytest.raises(DataError, match=f"^{_MISSING_TEXT}$") as excinfo:
        CobrePlugin().launch(ws, res)
    assert str(tmp_path) not in str(excinfo.value)


# -- shape --------------------------------------------------------------


@pytest.mark.parametrize("top", ["caso_cobre", None], ids=["top", "flat"])
@pytest.mark.parametrize(
    ("cores", "threads", "nodes"), [(100, 100, 1), (400, 100, 4), (4, 2, 2)]
)
def test_shape_is_one_srun_pmix_spec_for_every_node_count(
    tmp_path: Path, top: str | None, cores: int, threads: int, nodes: int
) -> None:
    ws = _workspace(tmp_path, top=top)
    res = Resources("fila", cores, max_cores_per_node=threads)
    spec = CobrePlugin().launch(ws, res)
    case_dir = ws.root if top is None else ws.root / top
    assert spec == LaunchSpec(
        Launcher.SRUN_PMIX,
        (
            str(ws.assets / "novomodelo-mpi"),
            "run",
            str(case_dir),
            "--threads",
            str(threads),
            "--comm-backend",
            "mpi",
        ),
        nodes=nodes,
        ntasks_per_node=1,
        cpus_per_task=threads,
        prepend_mpich_lib=True,
        env={"FI_PROVIDER": "efa"},
    )
    assert spec.argv[0].endswith("/assets/novomodelo-mpi")
    assert spec.argv[2] == str(case_dir)
    assert spec.argv[-2:] == ("--comm-backend", "mpi")


# -- render -------------------------------------------------------------


@pytest.mark.parametrize(("cores", "nodes"), [(100, 1), (400, 4)])
def test_render_script_holds_nodes_lib_efa_and_launch_line(
    tmp_path: Path, cores: int, nodes: int
) -> None:
    ws = _workspace(tmp_path)
    res = Resources("q", cores, max_cores_per_node=100)
    spec = CobrePlugin().launch(ws, res)
    lines = render_model_script(ws, spec, res, _TOOLS).splitlines()
    for line in (
        f"#SBATCH --nodes={nodes}",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=100",
    ):
        assert line in lines
    lib = (
        "export LD_LIBRARY_PATH=/opt/mpich/lib"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    )
    launch = _SRUN + " ".join(shlex.quote(arg) for arg in spec.argv)
    assert (
        lines.index(lib)
        < lines.index("export FI_PROVIDER=efa")
        < lines.index(launch)
    )
