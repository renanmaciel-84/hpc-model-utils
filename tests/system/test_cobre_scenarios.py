"""ADR-025/ADR-038/ADR-051/ADR-052/ADR-057/R73/R81/R105/R134/R138: the
cobre end-to-end scenarios (ticket-072).

The real ``CobrePlugin`` runs through ``core.lifecycle.run.run``, the
executing fake SLURM (whose ``srun`` shim runs the command once), a
local CLI shim over the real registry (so ``finalize cobre`` resolves
the registered plugin) and ``publish``. ``assets/cobre-mpi`` is a
Python stub with a baked-in payload: it records its call, prints the
``Execution`` lines cobre prints to stderr, writes the output tree and
exits with a chosen code. The case itself is the synthetic one of
``tests/support/cobre_case``.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from hpc_model_utils.core.diagnosis import RunStatus
from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.launch import Resources, Toolchain
from hpc_model_utils.core.lifecycle.publish import publish
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import ModelInfo, RunState, StateStore
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.infra.slurm import Slurm
from hpc_model_utils.models.cobre import CobrePlugin
from hpc_model_utils.platform.modelops import Reporter
from tests.support.cli_shim import write_python_sibling
from tests.support.cobre_case import (
    COBRE_EXECUTION_LINES,
    cobre_workspace,
    simulation_metadata,
    training_metadata,
    write_outputs,
)
from tests.support.fake_slurm import FakeSlurm
from tests.support.hooks import Hook, parse_hooks
from tests.support.object_store import RecordingObjectStore

_PLUGIN = CobrePlugin()
_ARTIFACTS_URI = "s3://bucket-a/artifacts/" + "a" * 64 + "/"
_STATUS_HOOK_METHODS = frozenset(
    {"SetSuccess", "SetModelError", "SetDataError", "SetRuntimeError"}
)
_LD_LIBRARY_PATH_SUFFIX = "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
_FLAG_TEXT = (
    "cobre needs --max-cores-per-node: threads per node T = "
    "max_cores_per_node, nodes K = cores / T (400 cores with 100 per node "
    "gives 4 nodes x 100 threads)"
)

_STUB_BODY = r"""
import json
import os
import sys
from pathlib import Path

nnodes = os.environ.get("SLURM_NNODES", "1")
record = {
    "argv": sys.argv,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    "FI_PROVIDER": os.environ.get("FI_PROVIDER"),
    "SLURM_NNODES": os.environ.get("SLURM_NNODES"),
}
with open(CALLS, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
if PAYLOAD.get("execution_lines", True):
    for line in EXECUTION_LINES:
        print(line, file=sys.stderr)
for line in PAYLOAD["stderr"]:
    print(line, file=sys.stderr)
world_size = PAYLOAD.get("world_size", int(nnodes))
output = Path(sys.argv[2]) / "output"
for name, hex_data in PAYLOAD["files"].items():
    data = bytes.fromhex(hex_data)
    if name.endswith("/metadata.json"):
        metadata = json.loads(data)
        metadata["distribution"]["world_size"] = world_size
        data = json.dumps(metadata).encode("utf-8")
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
sys.exit(PAYLOAD["exit_code"])
"""


_BRIDGE_BODY = r"""
import json
import os
import sys
from pathlib import Path

record = {"argv": sys.argv[1:], "cwd": os.getcwd()}
with open(CALLS, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
if EXIT_CODE:
    print("dashboard failed", file=sys.stderr)
    sys.exit(EXIT_CODE)
target = Path(sys.argv[sys.argv.index("--output") + 1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("<html></html>\n", encoding="utf-8")
"""


def _write_cli_shim(dest_dir: Path) -> Path:
    """Real-registry shim: tests/support/cli_shim.py installs the fake one."""
    shim = dest_dir / "hpcmu-cobre-cli"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import hpc_model_utils.cli\n"
        "\n"
        "raise SystemExit(hpc_model_utils.cli.main())\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    write_python_sibling(dest_dir)
    return shim


def _write_bridge_stub(dest_dir: Path, calls: Path, exit_code: int) -> Path:
    """A ``cobre-bridge`` stub: it records its call; like ``build_dashboard``
    it then creates the parent of the ``--output`` path and writes a
    minimal HTML file, or fails with ``exit_code`` and writes nothing."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stub = dest_dir / "cobre-bridge"
    stub.write_text(
        f"#!{sys.executable}\n"
        f"CALLS = {str(calls)!r}\n"
        f"EXIT_CODE = {exit_code!r}\n"
        f"{_BRIDGE_BODY}",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


@dataclass(frozen=True, slots=True)
class _Stub:
    exit_code: int = 0
    execution_lines: bool = True
    stderr: tuple[str, ...] = ()
    world_size: int | None = None
    outputs: bool = True
    training: Mapping[str, object] = field(default_factory=training_metadata)
    simulation: Mapping[str, object] = field(
        default_factory=simulation_metadata
    )
    dashboard_exit: int = 0


@dataclass(frozen=True, slots=True)
class _Outcome:
    ws: Workspace
    state: RunState
    hooks: list[Hook]
    store: RecordingObjectStore
    submitted: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    bridge_calls: list[dict[str, Any]]


def _payload(stub: _Stub, scratch: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    if stub.outputs:
        names = write_outputs(
            scratch, training=stub.training, simulation=stub.simulation
        )
        files = {
            name: (scratch / "output" / name).read_bytes().hex()
            for name in names
        }
    payload: dict[str, object] = {
        "exit_code": stub.exit_code,
        "execution_lines": stub.execution_lines,
        "stderr": list(stub.stderr),
        "files": files,
    }
    if stub.world_size is not None:
        payload["world_size"] = stub.world_size
    return payload


def _write_stub(ws: Workspace, calls: Path, payload: dict[str, object]) -> None:
    ws.assets.mkdir(exist_ok=True)
    binary = ws.assets / "cobre-mpi"
    binary.write_text(
        f"#!{sys.executable}\n"
        f"CALLS = {str(calls)!r}\n"
        f"EXECUTION_LINES = {list(COBRE_EXECUTION_LINES)!r}\n"
        f"PAYLOAD = {payload!r}\n"
        f"{_STUB_BODY}",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _run_cobre(
    tmp_path: Path,
    fake_slurm: FakeSlurm,
    res: Resources,
    stub: _Stub = _Stub(),
) -> _Outcome:
    ws = cobre_workspace(tmp_path).ws
    calls = tmp_path / "calls.jsonl"
    bridge_calls = tmp_path / "bridge-calls.jsonl"
    _write_stub(ws, calls, _payload(stub, tmp_path / "scratch"))

    state_store = StateStore(ws)
    state_store.save(
        replace(
            state_store.load_or_create(_PLUGIN.name),
            model=ModelInfo(_PLUGIN.model_name, None),
        )
    )
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    tools = Toolchain(
        cli_bin=_write_cli_shim(tmp_path),
        mpich_bin=tmp_path / "mpich" / "bin",
        slurm_bin=fake_slurm.bin_dir,
    )
    settings = EngineSettings(
        poll_interval=0.05,
        settle_window=0.2,
        outcome_attempts=3,
        outcome_backoff=0.1,
        cancel_timeout=5.0,
    )

    run(
        ws,
        _PLUGIN,
        Slurm(fake_slurm.bin_dir),
        reporter,
        state_store,
        SubmitRequest(
            resources=res,
            tools=tools,
            skip_model=False,
            synthesis_bin=_write_bridge_stub(
                tmp_path / "bridge", bridge_calls, stub.dashboard_exit
            ),
        ),
        JobLedger(),
        settings=settings,
        emit=lambda _line: None,
    )
    store = RecordingObjectStore()
    publish(ws, _PLUGIN, store, reporter, state_store, _ARTIFACTS_URI)

    persisted = state_store.load()
    assert persisted is not None
    return _Outcome(
        ws=ws,
        state=persisted,
        hooks=parse_hooks("\n".join(channel.lines)),
        store=store,
        submitted=fake_slurm.submitted(),
        calls=[json.loads(line) for line in calls.read_text().splitlines()],
        bridge_calls=(
            [json.loads(line) for line in bridge_calls.read_text().splitlines()]
            if bridge_calls.exists()
            else []
        ),
    )


def _uploaded_keys(store: RecordingObjectStore) -> set[str]:
    return {
        detail.removeprefix(_ARTIFACTS_URI)
        for kind, detail in store.ops
        if kind == "upload"
    }


def _status_hooks(outcome: _Outcome) -> list[Hook]:
    return [h for h in outcome.hooks if h.method in _STATUS_HOOK_METHODS]


def _annotations(outcome: _Outcome) -> list[str]:
    return [h.args[0] for h in outcome.hooks if h.method == "SetAnnotation"]


def _expected_exports(tmp_path: Path) -> list[str]:
    lib = shlex.quote(str(tmp_path / "mpich" / "lib"))
    return [
        f"export LD_LIBRARY_PATH={lib}{_LD_LIBRARY_PATH_SUFFIX}",
        "export FI_PROVIDER=efa",
    ]


def _expected_argv(ws: Workspace, threads: int) -> list[str]:
    return [
        str(ws.assets / "cobre-mpi"),
        "run",
        str(ws.root / "caso_cobre"),
        "--threads",
        str(threads),
        "--comm-backend",
        "mpi",
    ]


def _model_script_lines(ws: Workspace) -> list[str]:
    return ws.job_script(Phase.MODEL).read_text(encoding="utf-8").splitlines()


def test_cobre_shim_python_sibling_is_a_regular_file(tmp_path: Path) -> None:
    shim = _write_cli_shim(tmp_path)

    python_bin = Toolchain(Path("/m"), Path("/s"), shim).python_bin

    assert python_bin.is_file()
    assert not python_bin.is_symlink()


@pytest.mark.timeout(60)
def test_cobre_scenario_one_node_runs_cobre_mpi_through_srun_and_publishes(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path, fake_slurm, Resources("batch", 4, max_cores_per_node=4)
    )

    assert [Path(s["script"]).name for s in outcome.submitted] == [
        "model.sbatch",
        "finalize.sbatch",
    ]
    script = _model_script_lines(outcome.ws)
    assert "#SBATCH --nodes=1" in script
    for export in _expected_exports(tmp_path):
        assert export in script

    assert len(outcome.calls) == 1
    call = outcome.calls[0]
    assert call["argv"] == _expected_argv(outcome.ws, 4)
    assert call["LD_LIBRARY_PATH"].startswith(str(tmp_path / "mpich" / "lib"))
    assert call["SLURM_NNODES"] == "1"

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.SUCCESS
    assert diagnosis.rule_id == "cobre.completed"

    keys = _uploaded_keys(outcome.store)
    assert {
        "saidas/training.zip",
        "saidas/policy.zip",
        "saidas/simulation.zip",
        "saidas/training/metadata.json",
        "saidas/simulation/metadata.json",
        "entradas/eco_deck.zip",
        "entradas/deck_processado.zip",
        "saidas/metadata.modelops",
        "saidas/run.json",
    } <= keys
    assert not any(key.endswith("cortes.zip") for key in keys)
    metadata = json.loads(
        outcome.store.get_bytes(
            S3Uri.parse(_ARTIFACTS_URI).join("saidas/metadata.modelops")
        )
    )
    assert metadata["model_name"] == "COBRE"


@pytest.mark.timeout(60)
def test_cobre_scenario_k_nodes_runs_cobre_mpi_once_with_two_nodes(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path, fake_slurm, Resources("batch", 4, max_cores_per_node=2)
    )

    script = _model_script_lines(outcome.ws)
    assert "#SBATCH --nodes=2" in script
    for export in _expected_exports(tmp_path):
        assert export in script

    assert len(outcome.calls) == 1
    call = outcome.calls[0]
    assert call["argv"] == _expected_argv(outcome.ws, 2)
    assert call["FI_PROVIDER"] == "efa"
    assert call["SLURM_NNODES"] == "2"

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.SUCCESS


@pytest.mark.timeout(60)
def test_cobre_scenario_rank_mismatch_diagnoses_rank_count_mismatch(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=2),
        _Stub(world_size=1),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.RUNTIME_ERROR
    assert diagnosis.rule_id == "cobre.rank_count_mismatch"


@pytest.mark.timeout(60)
def test_cobre_scenario_missing_flag_raises_data_error_before_submission(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    with pytest.raises(DataError) as excinfo:
        _run_cobre(tmp_path, fake_slurm, Resources("batch", 4))

    assert str(excinfo.value) == _FLAG_TEXT
    assert fake_slurm.submitted() == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("exit_code", "status", "rule_id", "hook"),
    [
        (1, RunStatus.DATA_ERROR, "cobre.validation_error", "SetDataError"),
        (2, RunStatus.DATA_ERROR, "cobre.io_error", "SetDataError"),
        (3, RunStatus.RUNTIME_ERROR, "cobre.solver_error", "SetRuntimeError"),
        (
            4,
            RunStatus.RUNTIME_ERROR,
            "cobre.internal_error",
            "SetRuntimeError",
        ),
    ],
)
def test_cobre_scenario_exit_table_maps_exit_code_to_status_and_hook(
    fake_slurm: FakeSlurm,
    tmp_path: Path,
    exit_code: int,
    status: RunStatus,
    rule_id: str,
    hook: str,
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(exit_code=exit_code, outputs=False),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is status
    assert diagnosis.rule_id == rule_id
    assert _status_hooks(outcome) == [Hook(hook, ())]


@pytest.mark.timeout(60)
def test_cobre_scenario_incomplete_simulation_at_exit_zero_is_runtime_error(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(
            simulation=simulation_metadata(
                scenarios={"total": 10, "completed": 7, "failed": 0}
            )
        ),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.RUNTIME_ERROR
    assert diagnosis.rule_id == "cobre.simulation_incomplete"
    assert _status_hooks(outcome) == [Hook("SetRuntimeError", ())]
    assert "7 completed, 0 failed, 10 total" in _annotations(outcome)[0]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("release", ["0.17.0", "0.18.0"])
def test_cobre_scenario_both_metadata_releases_succeed(
    fake_slurm: FakeSlurm, tmp_path: Path, release: str
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(
            training=training_metadata(release=release),
            simulation=simulation_metadata(release=release),
        ),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.rule_id == "cobre.completed"
    assert _status_hooks(outcome) == [Hook("SetSuccess", ())]


@pytest.mark.timeout(60)
def test_cobre_scenario_stop_on_request_is_cancelled_and_archives_outputs(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(
            exit_code=5,
            training=training_metadata(
                status="partial",
                iterations={"completed": 37},
                convergence={"termination_reason": "graceful_shutdown"},
            ),
            simulation=simulation_metadata(
                status="partial",
                scenarios={"total": 100, "completed": 0, "failed": 0},
            ),
        ),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.CANCELLED
    assert diagnosis.rule_id == "cobre.shutdown_cancelled"
    assert _status_hooks(outcome) == [Hook("SetRuntimeError", ())]
    assert _annotations(outcome)[0].startswith(
        "CANCELLED: training stopped on request after 37 iterations"
    )
    assert outcome.bridge_calls == []
    assert {
        "saidas/training.zip",
        "saidas/policy.zip",
        "saidas/training/metadata.json",
    } <= _uploaded_keys(outcome.store)


@pytest.mark.timeout(60)
def test_cobre_scenario_not_started_backend_is_runtime_error_mpi_not_started(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(
            exit_code=4,
            execution_lines=False,
            stderr=(
                "error: communication backend error: backend not available",
            ),
            outputs=False,
        ),
    )

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.status is RunStatus.RUNTIME_ERROR
    assert diagnosis.rule_id == "cobre.mpi_not_started"
    assert _status_hooks(outcome) == [Hook("SetRuntimeError", ())]
    assert "no Backend line" in _annotations(outcome)[0]


@pytest.mark.timeout(60)
def test_cobre_scenario_success_publishes_the_dashboard(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path, fake_slurm, Resources("batch", 4, max_cores_per_node=4)
    )

    assert outcome.bridge_calls == [
        {
            "argv": [
                "dashboard",
                "caso_cobre",
                "--output",
                "sintese/dashboard.html",
            ],
            "cwd": str(outcome.ws.root),
        }
    ]
    assert {
        "sintese/dashboard.html",
        "saidas/logs/synthesis.out",
    } <= _uploaded_keys(outcome.store)
    assert (
        outcome.store.get_bytes(
            S3Uri.parse(_ARTIFACTS_URI).join("sintese/dashboard.html")
        )
        == b"<html></html>\n"
    )
    assert _status_hooks(outcome) == [Hook("SetSuccess", ())]
    assert ("SetMetadata", ("synthesis_status", "ok")) in {
        (h.method, h.args) for h in outcome.hooks
    }

    diagnosis = outcome.state.diagnosis
    assert diagnosis is not None
    assert diagnosis.rule_id == "cobre.completed"


@pytest.mark.timeout(60)
def test_cobre_scenario_dashboard_failure_keeps_success(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    outcome = _run_cobre(
        tmp_path,
        fake_slurm,
        Resources("batch", 4, max_cores_per_node=4),
        _Stub(dashboard_exit=1),
    )

    assert _status_hooks(outcome) == [Hook("SetSuccess", ())]
    annotations = _annotations(outcome)
    assert len(annotations) == 1
    assert annotations[0].startswith(
        "SUCCESS: synthesis failed: synthesis tool exited 1; "
    )
    assert ("SetMetadata", ("synthesis_status", "failed")) in {
        (h.method, h.args) for h in outcome.hooks
    }
    keys = _uploaded_keys(outcome.store)
    assert "sintese/dashboard.html" not in keys
    assert "saidas/logs/synthesis.out" in keys

    run_json = json.loads(
        outcome.store.get_bytes(
            S3Uri.parse(_ARTIFACTS_URI).join("saidas/run.json")
        )
    )
    assert run_json["diagnosis"]["rule_id"] == "cobre.completed"
    assert run_json["diagnosis"]["evidence"][0]["source"] == "synthesis"
