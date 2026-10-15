"""ADR-084/R85/R50/R78 tests for the publish failure matrix (ticket-049).

Every export-failure mode leaves the other uploads, the projections, the hooks
and the saved diagnosis equal to a run with nothing to export (K1). A failed
telemetry upload is the one exception: a StorageError with the best-effort
RUNTIME_ERROR overwrite (ADR-084/D8).
"""

from __future__ import annotations

import functools
import json
import logging
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from hpc_model_utils.core.diagnosis import Diagnosis, RunStatus
from hpc_model_utils.core.errors import StorageError
from hpc_model_utils.core.lifecycle import publish as publish_module
from hpc_model_utils.core.lifecycle.publish import publish
from hpc_model_utils.core.state import (
    JobRecord,
    RunState,
    StateStore,
    ToolInfo,
)
from hpc_model_utils.core.telemetry import (
    Exported,
    ExportResult,
    NothingToExport,
    export_telemetry,
)
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.infra.telemetry.schema import EXPORT_FILES
from hpc_model_utils.platform.modelops import Reporter
from tests.support.fake_plugin import FakePlugin
from tests.support.hooks import Hook, parse_hooks
from tests.support.object_store import RecordingObjectStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUCKET = "test-bucket"
_HASH = "a" * 64
_URI = f"s3://{_BUCKET}/artifacts/{_HASH}/"
_AT = "2000-01-01T00:00:00.000Z"
_DIAGNOSIS = Diagnosis(RunStatus.SUCCESS, "core.ok", "model converged", at=_AT)
_HEADLINE = (("telemetry.model.cpu_util_mean", "0.5000"),)
_BASELINE_UPLOADS = [
    "saidas/logs/model-1000.out",
    "saidas/logs/finalize-1001.out",
    "sintese/x.parquet",
    "saidas/status.modelops",
    "saidas/run.json",
    "saidas/metadata.modelops",
]
_PUBLISH_BUDGET_S = 10.0

_SAMPLES, _WORKSPACE, _SUMMARY = EXPORT_FILES
_FIND_OUT = (
    'while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done; '
    'mkdir -p "$out"'
)
_WRITE_PARQUETS = f'{_FIND_OUT}; touch "$out/{_SAMPLES}" "$out/{_WORKSPACE}"'

type _Export = Callable[[Workspace, RunState], ExportResult]


@dataclass(frozen=True, slots=True)
class _Case:
    id: str
    body: str
    timeout_s: float
    reason: str | None
    prefix: bool = False


_CASES = (
    _Case(
        "exit1",
        'echo "hpcmu-telemetry-export: OSError: disk full" >&2; exit 1',
        30,
        "hpcmu-telemetry-export: OSError: disk full",
    ),
    _Case(
        "exit2",
        'echo "hpcmu-telemetry-export: error: bad argv" >&2; exit 2',
        30,
        "hpcmu-telemetry-export: error: bad argv",
    ),
    _Case("exit3", "exit 3", 30, None),
    _Case("sigkill", "kill -KILL $$", 30, "killed by signal 9"),
    _Case("sigsegv", "ulimit -c 0; kill -SEGV $$", 30, "killed by signal 11"),
    _Case("timeout", "exec sleep 30", 0.5, "timed out after 0.5 s"),
    _Case(
        "traceback",
        "printf 'Traceback (most recent call last):\\n"
        '  File "x", line 1\\nMemoryError\\n\' >&2; exit 1',
        30,
        "MemoryError",
    ),
    _Case(
        "no_summary",
        f"{_WRITE_PARQUETS}; exit 0",
        30,
        f"export exited 0 without {_SUMMARY}",
    ),
    _Case(
        "bad_summary",
        f"""{_WRITE_PARQUETS}; echo '{{"kind": "other"}}' > "$out/{_SUMMARY}"; exit 0""",
        30,
        f"malformed {_SUMMARY}",
        prefix=True,
    ),
)
_BY_ID = {case.id: case for case in _CASES}
_BOUNDED = (_BY_ID["timeout"], _BY_ID["sigsegv"])


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


@dataclass(frozen=True, slots=True)
class _Outcome:
    uploads: list[str]
    status: bytes
    metadata: bytes
    artifacts: list[tuple[str, int]]
    hooks: list[Hook]
    diagnosis: Diagnosis | None
    raised: bool


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "export-interpreter"
    path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _runner(tmp_path: Path, case: _Case) -> _Export:
    return functools.partial(
        export_telemetry,
        interpreter=_script(tmp_path, case.body),
        timeout_s=case.timeout_s,
    )


def _workspace(root: Path) -> Workspace:
    root.mkdir()
    ws = Workspace.at(root)
    ws.ensure_layout()
    jobs: list[JobRecord] = []
    for phase, job_id, second in (
        (Phase.MODEL, "1000", "00"),
        (Phase.FINALIZE, "1001", "01"),
    ):
        log = ws.log_path(phase, job_id)
        log.write_bytes(f"{phase.value} {job_id} log\n".encode())
        jobs.append(
            JobRecord(
                phase,
                job_id,
                f"2000-01-01T00:00:{second}.000Z",
                ws.relative(log),
            )
        )
    (ws.root / "sintese").mkdir()
    (ws.root / "sintese" / "x.parquet").write_bytes(b"parquet-bytes")
    StateStore(ws).save(
        RunState(
            run_id="run-1",
            plugin="fake",
            tool=ToolInfo(name="hpc-model-utils", version="2.0.0"),
            jobs=tuple(jobs),
            reported_job_id="1001",
            diagnosis=_DIAGNOSIS,
        )
    )
    return ws


def _exported(ws: Workspace) -> Exported:
    ws.telemetry_outputs_dir.mkdir(parents=True)
    samples, workspace, summary = (
        ws.telemetry_outputs_dir / name for name in EXPORT_FILES
    )
    for path in (samples, workspace, summary):
        path.write_bytes(b"fake")
    return Exported((samples, workspace, summary), _HEADLINE)


def _key(name: str) -> S3Uri:
    return S3Uri(_BUCKET, f"artifacts/{_HASH}/{name}")


def _uploads(store: RecordingObjectStore) -> list[str]:
    return [uri[len(_URI) :] for op, uri in store.ops if op == "upload"]


def _publish(
    ws: Workspace, store: RecordingObjectStore, channel: _ListChannel
) -> None:
    publish(
        ws,
        FakePlugin(),
        store,
        Reporter(channel, enabled=True),
        StateStore(ws),
        _URI,
    )


def _publish_case(
    root: Path, monkeypatch: pytest.MonkeyPatch, export: _Export
) -> _Outcome:
    ws = _workspace(root)
    monkeypatch.setattr(publish_module, "export_telemetry", export)
    store = RecordingObjectStore()
    channel = _ListChannel()
    raised = False
    try:
        _publish(ws, store, channel)
    except StorageError:
        raised = True
    state = StateStore(ws).load()
    assert state is not None
    root_bytes = str(ws.root).encode()
    artifacts: list[tuple[str, int]] = []
    if not raised:
        run = json.loads(store.get_bytes(_key("saidas/run.json")))
        artifacts = [(item["path"], item["bytes"]) for item in run["artifacts"]]
    return _Outcome(
        uploads=_uploads(store),
        status=store.get_bytes(_key("saidas/status.modelops")).replace(
            root_bytes, b"<WS>"
        ),
        metadata=store.get_bytes(_key("saidas/metadata.modelops")).replace(
            root_bytes, b"<WS>"
        ),
        artifacts=artifacts,
        hooks=parse_hooks("\n".join(channel.lines)),
        diagnosis=state.diagnosis,
        raised=raised,
    )


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _is_reason(message: str, case: _Case) -> bool:
    expected = f"telemetry export failed: {case.reason}"
    return message.startswith(expected) if case.prefix else message == expected


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_publish_export_failure_leaves_the_outcome_equal_to_the_baseline(
    case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _publish_case(
        tmp_path / "baseline",
        monkeypatch,
        lambda ws, state: NothingToExport(),
    )
    row = _publish_case(tmp_path / "row", monkeypatch, _runner(tmp_path, case))

    assert baseline.uploads == _BASELINE_UPLOADS
    assert not row.raised
    assert not [u for u in row.uploads if u.startswith("saidas/telemetry/")]
    assert not [
        hook
        for hook in row.hooks
        if any(arg.startswith("telemetry.") for arg in hook.args)
    ]
    assert row == baseline


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_publish_export_logging_warns_once_with_the_reason(
    case: _Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _publish_case(tmp_path / "row", monkeypatch, _runner(tmp_path, case))

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    if case.reason is None:
        assert warnings == []
    else:
        assert len(warnings) == 1
        assert _is_reason(warnings[0], case)


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_publish_s3_failure_raises_storage_error_with_the_overwrite(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _workspace(tmp_path / "ws")
    exported = _exported(ws)
    monkeypatch.setattr(
        publish_module, "export_telemetry", lambda _ws, _state: exported
    )
    store = RecordingObjectStore()
    store.fail_on(f"saidas/telemetry/{name}")
    channel = _ListChannel()

    with pytest.raises(
        StorageError,
        match=rf"upload of saidas/telemetry/{re.escape(name)} failed",
    ):
        _publish(ws, store, channel)

    assert _uploads(store)[-2:] == [
        "saidas/metadata.modelops",
        "saidas/status.modelops",
    ]
    metadata = json.loads(store.get_bytes(_key("saidas/metadata.modelops")))
    assert metadata["status"] == "RUNTIME_ERROR"
    assert parse_hooks("\n".join(channel.lines)) == [
        Hook("SetExecutionArtifactsPath", (_URI,))
    ]


@pytest.mark.parametrize("case", _BOUNDED, ids=lambda case: case.id)
def test_publish_export_bounded_and_clean_leaves_no_core_file(
    case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _git_status()

    started = time.monotonic()
    _publish_case(tmp_path / "row", monkeypatch, _runner(tmp_path, case))
    elapsed = time.monotonic() - started

    assert elapsed < _PUBLISH_BUDGET_S
    assert _git_status() == before
