"""Job-telemetry ticket-052 (R65, R46, ADR-077, ADR-083, ADR-085): the
fake-Slurm lifecycle proof. Each test drives the real ``run`` (rendered job
scripts, sampler sidecar) and then the real ``publish`` (the export
subprocess, the uploads, the headline hooks) against an in-memory store.

"Every plugin" is read as every ``Launcher`` a plugin reaches, as in
``test_telemetry_job_logs.py``: ``DIRECT``, ``MPIEXEC_HYDRA`` and
``SRUN_PMIX``. The three telemetry artifacts must be published for each. A
sampler that crashes at start changes nothing in the status, the hooks, the
uploads or the job logs (K1). A model killed by ``TIMEOUT`` still yields
telemetry up to the kill (R36).

The helpers are local copies adapted from ``test_telemetry_job_logs.py`` and
``test_engine_end_to_end.py``, whose own helpers are module-private.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deploy.modelops.compare_job_logs import main as compare_job_logs
from hpc_model_utils.core.diagnosis import RunStatus
from hpc_model_utils.core.launch import (
    TELEMETRY_OFF,
    Launcher,
    LaunchSpec,
    Resources,
    TelemetryConfig,
    Toolchain,
)
from hpc_model_utils.core.lifecycle.finalize import legacy_synthesis_bin
from hpc_model_utils.core.lifecycle.publish import publish
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import RunState, StateStore, current_tool
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.infra.slurm import Slurm
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADLINE_KEY_PREFIX,
    SAMPLES_FILE,
    SUMMARY_FILE,
)
from hpc_model_utils.platform.modelops import Reporter
from tests.support.cli_shim import write_cli_shim
from tests.support.fake_plugin import FakePlugin, install_fake_model
from tests.support.fake_slurm import FakeSlurm
from tests.support.hooks import Hook, parse_hooks
from tests.support.object_store import RecordingObjectStore

pytestmark = pytest.mark.slow

_ARTIFACTS_URI = "s3://test-bucket/artifacts/" + "a" * 64 + "/"
_TELEMETRY_PREFIX = "saidas/telemetry/"
_TELEMETRY_KEYS = [f"{_TELEMETRY_PREFIX}{name}" for name in EXPORT_FILES]
_METADATA_KEY = "saidas/metadata.modelops"
_STATUS_KEY = "saidas/status.modelops"
_MEM_MIN_HOOKS = [
    f"{HEADLINE_KEY_PREFIX}{phase}.mem_available_bytes_min"
    for phase in ("model", "finalize")
]
_STATUS_HOOK_METHODS = frozenset(
    {"SetSuccess", "SetModelError", "SetDataError", "SetRuntimeError"}
)
_TELEMETRY_ON = replace(
    TELEMETRY_OFF, enabled=True, tool_version=current_tool().version
)
_CRASHING_SAMPLER = """\
#!/bin/sh
echo 'Traceback (most recent call last):' >&2
echo 'RuntimeError: injected sampler crash' >&2
exit 1
"""
_SETTINGS = EngineSettings(
    poll_interval=0.05,
    settle_window=0.2,
    outcome_attempts=3,
    outcome_backoff=0.1,
    cancel_timeout=5.0,
)


@pytest.fixture(autouse=True)
def _chdir_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class _SequencedChannel:
    """Feeds the hook lines it writes into a ``sequence`` shared with
    ``_SequencedStore``, so hooks and uploads share one real call-order
    timeline."""

    def __init__(self, sequence: list[tuple[str, str]]) -> None:
        self._sequence = sequence

    def write_line(self, line: str) -> None:
        self._sequence.append(("hook", line))


class _SequencedStore(RecordingObjectStore):
    def __init__(self, sequence: list[tuple[str, str]]) -> None:
        super().__init__()
        self._sequence = sequence

    def upload(self, src: Path, uri: S3Uri) -> None:
        super().upload(src, uri)
        self._sequence.append(("upload", str(uri)))


def _make_workspace(arm_dir: Path) -> Workspace:
    root = arm_dir / "study-fake"
    root.mkdir()
    return Workspace.at(root)


def _toolchain(arm_dir: Path, bin_dir: Path) -> Toolchain:
    return Toolchain(
        cli_bin=write_cli_shim(arm_dir),
        mpich_bin=arm_dir / "mpich" / "bin",
        slurm_bin=bin_dir,
    )


def _write_legacy_sintetizador(ws: Workspace) -> Path:
    # The leading sleep keeps the finalize job alive past the sampler's
    # start-up, so the job-end teardown reaches a live step.
    stub = legacy_synthesis_bin(ws, FakePlugin())
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        "#!/bin/bash\nsleep 1\nmkdir -p sintese\n: > sintese/x.parquet\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


# The plugin variants: only `launch` differs, the name stays "fake" so the
# finalize job resolves the base plugin through the CLI shim's registry.


class _HydraPlugin(FakePlugin):
    def launch(self, ws: Workspace, res: Resources) -> LaunchSpec:
        return LaunchSpec(
            Launcher.MPIEXEC_HYDRA,
            (str(ws.assets / self.executables.entrypoint),),
            ntasks=2,
        )


class _PmixPlugin(FakePlugin):
    def launch(self, ws: Workspace, res: Resources) -> LaunchSpec:
        return LaunchSpec(
            Launcher.SRUN_PMIX,
            (str(ws.assets / self.executables.entrypoint),),
            nodes=2,
        )


@dataclass(frozen=True)
class _Published:
    ws: Workspace
    state: RunState
    sequence: list[tuple[str, str]]
    store: RecordingObjectStore


def _run_and_publish(
    arm_dir: Path,
    controller: FakeSlurm,
    plugin: FakePlugin,
    telemetry: TelemetryConfig,
    *,
    sibling: str | None = None,
    force: tuple[str, float] | None = None,
    model_sleep: float = 1.0,
) -> _Published:
    """``sibling`` replaces the ``python`` the toolchain writes, after
    ``_toolchain`` has created it and before ``run`` submits. ``force`` is
    the ``(state, after)`` of the model job."""
    ws = _make_workspace(arm_dir)
    install_fake_model(ws, sleep=model_sleep)
    _write_legacy_sintetizador(ws)
    tools = _toolchain(arm_dir, controller.bin_dir)
    if sibling is not None:
        tools.python_bin.write_text(sibling, encoding="utf-8")
    if force is not None:
        controller.force("model", state=force[0], after=force[1])
    sequence: list[tuple[str, str]] = []
    reporter = Reporter(_SequencedChannel(sequence), enabled=True)
    state_store = StateStore(ws)
    run(
        ws,
        plugin,
        Slurm(controller.bin_dir),
        reporter,
        state_store,
        SubmitRequest(
            resources=Resources(queue="batch", cores=2),
            tools=tools,
            skip_model=False,
            synthesis_bin=None,
            telemetry=telemetry,
        ),
        JobLedger(),
        settings=_SETTINGS,
        emit=lambda _line: None,
    )
    controller.wait_idle()
    store = _SequencedStore(sequence)
    publish(ws, plugin, store, reporter, state_store, _ARTIFACTS_URI)
    state = state_store.load()
    assert state is not None
    return _Published(ws, state, sequence, store)


def _keys(published: _Published) -> list[str]:
    return [
        uri.removeprefix(_ARTIFACTS_URI)
        for kind, uri in published.sequence
        if kind == "upload"
    ]


def _hooks(published: _Published) -> list[tuple[int, Hook]]:
    return [
        (index, hook)
        for index, (kind, line) in enumerate(published.sequence)
        if kind == "hook"
        for hook in parse_hooks(line)
    ]


def _hook_args(published: _Published, method: str) -> list[tuple[str, ...]]:
    return [hook.args for _, hook in _hooks(published) if hook.method == method]


def _telemetry_hooks(published: _Published) -> list[tuple[int, str]]:
    return [
        (index, hook.args[0])
        for index, hook in _hooks(published)
        if hook.method == "SetMetadata"
        and hook.args[0].startswith(HEADLINE_KEY_PREFIX)
    ]


def _status_hooks(published: _Published) -> list[Hook]:
    return [
        hook
        for _, hook in _hooks(published)
        if hook.method in _STATUS_HOOK_METHODS
    ]


def _object(published: _Published, name: str) -> bytes:
    return published.store.get_bytes(
        S3Uri.parse(_ARTIFACTS_URI).join(f"{_TELEMETRY_PREFIX}{name}")
    )


def _samples(published: _Published) -> set[tuple[str, str, int]]:
    table = pq.read_table(pa.BufferReader(_object(published, SAMPLES_FILE)))
    phases: list[str] = table.column("phase").to_pylist()
    job_ids: list[str] = table.column("job_id").to_pylist()
    nodes: list[int] = table.column("node_index").to_pylist()
    return set(zip(phases, job_ids, nodes, strict=True))


def _summary_records(
    published: _Published, section: str
) -> list[dict[str, object]]:
    summary = json.loads(_object(published, SUMMARY_FILE))
    assert isinstance(summary, dict)
    records = summary[section]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return records


def _assert_telemetry_keys(published: _Published) -> None:
    keys = _keys(published)
    last_sintese = max(
        index for index, key in enumerate(keys) if key.startswith("sintese/")
    )
    first = last_sintese + 1
    assert keys[first : first + len(_TELEMETRY_KEYS)] == _TELEMETRY_KEYS
    assert keys[first + len(_TELEMETRY_KEYS)] == _STATUS_KEY


def _assert_hook_placement(published: _Published) -> None:
    telemetry = _telemetry_hooks(published)
    assert set(_MEM_MIN_HOOKS) <= {key for _, key in telemetry}
    metadata_upload = next(
        index
        for index, (kind, uri) in enumerate(published.sequence)
        if kind == "upload" and uri.endswith(_METADATA_KEY)
    )
    (artifacts_path,) = (
        index
        for index, hook in _hooks(published)
        if hook.method == "SetExecutionArtifactsPath"
    )
    for position, _ in telemetry:
        assert metadata_upload < position < artifacts_path


def _compare(off: _Published, on: _Published, phase: Phase, job_id: str) -> int:
    return compare_job_logs(
        [
            "--mode",
            "strict",
            "--off-root",
            str(off.ws.root),
            "--on-root",
            str(on.ws.root),
            str(off.ws.log_path(phase, job_id)),
            str(on.ws.log_path(phase, job_id)),
        ]
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("plugin", "model_nodes"),
    [
        pytest.param(FakePlugin(), (0,), id="direct"),
        pytest.param(_HydraPlugin(), (0,), id="hydra"),
        pytest.param(_PmixPlugin(), (0, 1), id="pmix"),
    ],
)
def test_lifecycle_published_telemetry_artifacts(
    fake_slurm: FakeSlurm,
    tmp_path: Path,
    plugin: FakePlugin,
    model_nodes: tuple[int, ...],
) -> None:
    published = _run_and_publish(tmp_path, fake_slurm, plugin, _TELEMETRY_ON)

    assert published.state.diagnosis is not None
    assert published.state.diagnosis.status is RunStatus.SUCCESS
    _assert_telemetry_keys(published)

    samples = _samples(published)
    for node in model_nodes:
        assert ("model", "1000", node) in samples
    assert {(phase, job) for phase, job, _ in samples} >= {
        ("model", "1000"),
        ("finalize", "1001"),
    }

    jobs = _summary_records(published, "jobs")
    assert [job["job_id"] for job in jobs] == ["1000", "1001"]
    warnings = _summary_records(published, "warnings")
    assert "bad_header" not in [warning["code"] for warning in warnings]

    _assert_hook_placement(published)


@pytest.mark.timeout(60)
def test_lifecycle_sampler_crash_changes_nothing(
    fake_slurm_pair: tuple[FakeSlurm, FakeSlurm], tmp_path: Path
) -> None:
    plugin = _PmixPlugin()
    off = _run_and_publish(
        tmp_path / "off", fake_slurm_pair[0], plugin, TELEMETRY_OFF
    )
    on = _run_and_publish(
        tmp_path / "on",
        fake_slurm_pair[1],
        plugin,
        _TELEMETRY_ON,
        sibling=_CRASHING_SAMPLER,
    )

    assert off.state.diagnosis is not None
    assert on.state.diagnosis is not None
    assert (off.state.diagnosis.status, off.state.diagnosis.rule_id) == (
        on.state.diagnosis.status,
        on.state.diagnosis.rule_id,
    )
    assert _status_hooks(off) == _status_hooks(on) != []
    assert _hook_args(off, "SetAnnotation") == _hook_args(on, "SetAnnotation")
    assert _keys(off) == _keys(on)
    for arm in (off, on):
        assert not any(key.startswith(_TELEMETRY_PREFIX) for key in _keys(arm))
        assert _telemetry_hooks(arm) == []
    assert _compare(off, on, Phase.MODEL, "1000") == 0
    assert _compare(off, on, Phase.FINALIZE, "1001") == 0


@pytest.mark.timeout(60)
def test_lifecycle_model_timeout_still_publishes_telemetry(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    published = _run_and_publish(
        tmp_path,
        fake_slurm,
        FakePlugin(),
        _TELEMETRY_ON,
        force=("TIMEOUT", 1.5),
        model_sleep=5.0,
    )

    assert published.state.diagnosis is not None
    assert published.state.diagnosis.status is RunStatus.TIMEOUT
    assert published.state.diagnosis.rule_id == "slurm.timeout"
    assert _status_hooks(published) == [Hook("SetRuntimeError", ())]
    assert set(_TELEMETRY_KEYS) <= set(_keys(published))

    samples = _samples(published)
    assert {(phase, job) for phase, job, _ in samples} >= {
        ("model", "1000"),
        ("finalize", "1001"),
    }
