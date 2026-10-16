"""ADR-042/ADR-012/ADR-028/R13/R15/R56/R115/R55/R5/R14/R136 tests for
the ``publish`` lifecycle step (ticket-040)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from hpc_model_utils.core.diagnosis import (
    ANNOTATION_MAX_LENGTH,
    Diagnosis,
    RunStatus,
    utc_now_iso,
)
from hpc_model_utils.core.errors import (
    StateFormatError,
    StorageError,
    UsageError,
)
from hpc_model_utils.core.lifecycle import publish as publish_module
from hpc_model_utils.core.lifecycle.publish import (
    publish,
    validate_artifacts_prefix,
)
from hpc_model_utils.core.outputs import RealizedOutputs
from hpc_model_utils.core.run_record import RUN_KIND, RUN_SCHEMA_VERSION
from hpc_model_utils.core.state import (
    ExecutionSource,
    FinalizeRecord,
    JobRecord,
    RunState,
    StateStore,
    StepOutcome,
    ToolInfo,
    write_finalize,
)
from hpc_model_utils.core.telemetry import (
    EXPORT_TIMEOUT_S,
    Exported,
    ExportFailed,
    ExportResult,
    NothingToExport,
)
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.infra.telemetry.schema import EXPORT_FILES
from hpc_model_utils.platform.encoding import ANNOTATION_CAP
from hpc_model_utils.platform.modelops import Reporter
from tests.support.fake_plugin import FakePlugin
from tests.support.hooks import Hook, parse_hooks
from tests.support.object_store import RecordingObjectStore
from tests.support.telemetry_staging import header, sample, write_staging

_BUCKET = "test-bucket"
_HASH = "a" * 64


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _hooks(channel: _ListChannel) -> list[Hook]:
    return parse_hooks("\n".join(channel.lines))


def _uri(*, hash_: str = _HASH) -> str:
    return f"s3://{_BUCKET}/artifacts/{hash_}/"


def _key(name: str) -> str:
    return f"artifacts/{_HASH}/{name}"


def _s3(name: str) -> S3Uri:
    return S3Uri(_BUCKET, _key(name))


def _base(name: str = "") -> str:
    return f"s3://{_BUCKET}/{_key(name)}"


def _upload_keys(store: RecordingObjectStore) -> list[str]:
    base = _base()
    return [uri[len(base) :] for op, uri in store.ops if op == "upload"]


class _CapturingObjectStore(RecordingObjectStore):
    """Like ``RecordingObjectStore``, but also keeps the exact bytes
    of each successful upload in order -- needed to tell a pre-marker
    PUT's RUNTIME_ERROR payload apart from a later PUT to the same
    key, since the backing dict only ever holds the latest value."""

    def __init__(self) -> None:
        super().__init__()
        self.uploaded: list[tuple[str, bytes]] = []

    def upload(self, src: Path, uri: S3Uri) -> None:
        data = src.read_bytes()
        super().upload(src, uri)
        self.uploaded.append((str(uri), data))


def _uploaded_keys(store: _CapturingObjectStore) -> list[str]:
    base = _base()
    return [key[len(base) :] for key, _ in store.uploaded]


def _ws(tmp_path: Path) -> Workspace:
    workspace = Workspace.at(tmp_path)
    workspace.ensure_layout()
    return workspace


def _state(
    ws: Workspace,
    *,
    diagnosis: Diagnosis | None,
    jobs: tuple[JobRecord, ...] = (),
    run_id: str = "run-1",
    reported_job_id: str | None = "222",
    execution_source: ExecutionSource = ExecutionSource.CLUSTER,
) -> RunState:
    state = RunState(
        run_id=run_id,
        plugin="fake",
        tool=ToolInfo(name="hpc-model-utils", version="2.0.0"),
        jobs=jobs,
        reported_job_id=reported_job_id,
        diagnosis=diagnosis,
        execution_source=execution_source,
    )
    StateStore(ws).save(state)
    return state


def _success_diagnosis(reason: str = "model converged") -> Diagnosis:
    return Diagnosis(
        status=RunStatus.SUCCESS,
        rule_id="core.ok",
        reason=reason,
        at=utc_now_iso(),
    )


def _write_finalize(
    ws: Workspace, run_id: str, outputs: RealizedOutputs | None
) -> None:
    write_finalize(
        ws,
        FinalizeRecord(
            run_id=run_id,
            diagnosis=_success_diagnosis(),
            postprocess=StepOutcome("postprocess", True, "", 0.1),
            synthesis=StepOutcome("synthesis", True, "", 0.1),
            outputs=outputs,
        ),
    )


def _write_log(ws: Workspace, phase: Phase, job_id: str) -> str:
    path = ws.log_path(phase, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{phase} {job_id} log\n".encode())
    return ws.relative(path)


def _full_realized(ws: Workspace) -> RealizedOutputs:
    (ws.root / "eco_deck.zip").write_bytes(b"eco-bytes")
    (ws.outputs_dir / "deck_processado.zip").write_bytes(b"deck-bytes")
    (ws.outputs_dir / "cortes.zip").write_bytes(b"cortes-bytes")
    (ws.root / "pmo.dat").write_bytes(b"pmo-bytes")
    return RealizedOutputs(
        deck=".hpcmu/outputs/deck_processado.zip",
        archives=(".hpcmu/outputs/cortes.zip",),
        raw=(("pmo.dat", "pmo.dat"),),
    )


def _write_sintese(ws: Workspace, name: str = "x.parquet") -> None:
    sintese = ws.root / "sintese"
    sintese.mkdir(parents=True, exist_ok=True)
    (sintese / name).write_bytes(b"parquet-bytes")


def _status_of(store: RecordingObjectStore, name: str) -> str:
    return store.get_bytes(_s3(name)).decode("ascii")


def _metadata_status(store: RecordingObjectStore, name: str) -> str:
    data = json.loads(store.get_bytes(_s3(name)).decode("ascii"))
    status = data["status"]
    assert isinstance(status, str)
    return status


# ---------------------------------------------------------------------------
# validate_artifacts_prefix -- the prefix validation table (Requirement 1)
# ---------------------------------------------------------------------------


def test_validate_artifacts_prefix_empty_key_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="invalid artifacts prefix"):
        validate_artifacts_prefix(f"s3://{_BUCKET}/")


def test_validate_artifacts_prefix_empty_final_segment_raises_usage_error() -> (
    None
):
    with pytest.raises(UsageError, match="invalid artifacts prefix"):
        validate_artifacts_prefix(f"s3://{_BUCKET}/artifacts//")


def test_validate_artifacts_prefix_valid_hash_strips_trailing_slash() -> None:
    result = validate_artifacts_prefix(_uri())
    assert result == S3Uri(_BUCKET, f"artifacts/{_HASH}")


def test_validate_artifacts_prefix_malformed_uri_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="invalid artifacts prefix"):
        validate_artifacts_prefix("not-an-s3-uri")


# ---------------------------------------------------------------------------
# Amendment rule 1: check_artifacts_path runs before any S3 call or hook
# ---------------------------------------------------------------------------


def test_publish_uri_rejected_by_check_artifacts_path_raises_usage_error_with_no_ops(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=_success_diagnosis())
    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    # accepted by validate_artifacts_prefix (non-empty final segment),
    # rejected by Reporter.check_artifacts_path (not 64 hex chars).
    bad_uri = f"s3://{_BUCKET}/artifacts/short-hash/"

    with pytest.raises(UsageError, match="invalid artifacts uri"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), bad_uri)

    assert store.ops == []
    assert channel.lines == []


# ---------------------------------------------------------------------------
# AC2: a SUCCESS run with realized outputs on a fresh prefix
# ---------------------------------------------------------------------------


def test_publish_success_fresh_prefix_uploads_in_requirement_order(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    model_log = _write_log(ws, Phase.MODEL, "111")
    finalize_log = _write_log(ws, Phase.FINALIZE, "222")
    state = _state(
        ws,
        diagnosis=_success_diagnosis(),
        jobs=(
            JobRecord(Phase.MODEL, "111", utc_now_iso(), model_log),
            JobRecord(Phase.FINALIZE, "222", utc_now_iso(), finalize_log),
        ),
    )
    outputs = _full_realized(ws)
    _write_finalize(ws, state.run_id, outputs)
    _write_sintese(ws)

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _upload_keys(store) == [
        "entradas/eco_deck.zip",
        "entradas/deck_processado.zip",
        "saidas/cortes.zip",
        "saidas/pmo.dat",
        "saidas/logs/model-111.out",
        "saidas/logs/finalize-222.out",
        "sintese/x.parquet",
        "saidas/status.modelops",
        "saidas/run.json",
        "saidas/metadata.modelops",
    ]

    run_json = json.loads(store.get_bytes(_s3("saidas/run.json")))
    metadata_entry = next(
        entry
        for entry in run_json["artifacts"]
        if entry["path"] == "saidas/metadata.modelops"
    )
    assert metadata_entry["bytes"] == len(
        store.get_bytes(_s3("saidas/metadata.modelops"))
    )

    assert _hooks(channel) == [
        Hook("SetExecutionArtifactsPath", (_uri(),)),
        Hook("SetSuccess", ()),
        Hook("SetAnnotation", (_success_diagnosis().annotation(),)),
    ]


def test_publish_synthesis_log_uploads_after_job_logs_before_sintese(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    model_log = _write_log(ws, Phase.MODEL, "111")
    finalize_log = _write_log(ws, Phase.FINALIZE, "222")
    state = _state(
        ws,
        diagnosis=_success_diagnosis(),
        jobs=(
            JobRecord(Phase.MODEL, "111", utc_now_iso(), model_log),
            JobRecord(Phase.FINALIZE, "222", utc_now_iso(), finalize_log),
        ),
    )
    _write_finalize(ws, state.run_id, _full_realized(ws))
    _write_sintese(ws)
    synthesis_log = b"2026-10-04 20:21:31,000 INFO: a\n"
    ws.synthesis_log_path.write_bytes(synthesis_log)

    store = RecordingObjectStore()
    reporter = Reporter(_ListChannel(), enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _upload_keys(store) == [
        "entradas/eco_deck.zip",
        "entradas/deck_processado.zip",
        "saidas/cortes.zip",
        "saidas/pmo.dat",
        "saidas/logs/model-111.out",
        "saidas/logs/finalize-222.out",
        "saidas/logs/synthesis.out",
        "sintese/x.parquet",
        "saidas/status.modelops",
        "saidas/run.json",
        "saidas/metadata.modelops",
    ]
    assert store.get_bytes(_s3("saidas/logs/synthesis.out")) == synthesis_log
    run_json = json.loads(store.get_bytes(_s3("saidas/run.json")))
    entry = next(
        entry
        for entry in run_json["artifacts"]
        if entry["path"] == "saidas/logs/synthesis.out"
    )
    assert entry["bytes"] == len(synthesis_log)


@pytest.mark.parametrize("as_directory", [False, True], ids=["absent", "dir"])
def test_publish_synthesis_log_absent_or_not_a_file_uploads_no_synthesis_key(
    tmp_path: Path, as_directory: bool
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    _write_finalize(ws, state.run_id, None)
    if as_directory:
        ws.synthesis_log_path.mkdir()

    store = RecordingObjectStore()
    reporter = Reporter(_ListChannel(), enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert "saidas/logs/synthesis.out" not in _upload_keys(store)


# ---------------------------------------------------------------------------
# AC3: an upload failure midway on a fresh prefix
# ---------------------------------------------------------------------------


def test_publish_fresh_prefix_upload_failure_raises_storage_error_and_halts(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(
        deck=None,
        # alphabetical, as a real ``realize()`` call would sort them: one
        # PUT succeeds before ``relatorios.zip`` fails, so the best-effort
        # overwrite condition (Requirement 3.1) is satisfied.
        archives=(".hpcmu/outputs/cortes.zip", ".hpcmu/outputs/relatorios.zip"),
        raw=(),
    )
    (ws.outputs_dir / "cortes.zip").write_bytes(b"cortes-bytes")
    (ws.outputs_dir / "relatorios.zip").write_bytes(b"rel-bytes")
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    store.fail_on("relatorios.zip")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with pytest.raises(StorageError, match="relatorios.zip"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    keys = _upload_keys(store)
    assert keys[:2] == ["saidas/cortes.zip", "saidas/relatorios.zip"]
    # nothing past the failure point is uploaded except the best-effort
    # RUNTIME_ERROR overwrite of the commit marker and its status pair.
    assert keys[2:] == ["saidas/metadata.modelops", "saidas/status.modelops"]
    assert (
        _metadata_status(store, "saidas/metadata.modelops") == "RUNTIME_ERROR"
    )
    hooks = _hooks(channel)
    assert hooks == [Hook("SetExecutionArtifactsPath", (_uri(),))]


# ---------------------------------------------------------------------------
# AC4/AC5: a reused prefix
# ---------------------------------------------------------------------------


def _seed_reused_prefix(
    store: RecordingObjectStore, *, previous_run_id: str | None = "old"
) -> None:
    store.seed(_s3("saidas/pmo.dat"), b"old-pmo")
    store.seed(_s3("saidas/metadata.modelops"), b'{"status": "SUCCESS"}')
    if previous_run_id is not None:
        store.seed(
            _s3("saidas/run.json"),
            json.dumps(
                {
                    "kind": RUN_KIND,
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_id": previous_run_id,
                }
            ).encode("utf-8"),
        )
    # reset ops recorded by seeding so tests only see publish()'s own ops.
    store.ops.clear()


def test_publish_reused_prefix_identified_previous_run_overwrites_then_republishes(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    # a real entradas/ upload (eco_deck.zip + the realized deck) so the
    # "both pre-marker PUTs come before any entradas/ PUT" claim (AC4)
    # is actually exercised, not vacuously true.
    (ws.root / "eco_deck.zip").write_bytes(b"eco-bytes")
    (ws.outputs_dir / "deck_processado.zip").write_bytes(b"deck-bytes")
    outputs = RealizedOutputs(
        deck=".hpcmu/outputs/deck_processado.zip", archives=(), raw=()
    )
    _write_finalize(ws, state.run_id, outputs)

    store = _CapturingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    keys = _upload_keys(store)
    assert keys[:2] == ["saidas/metadata.modelops", "saidas/status.modelops"]
    assert keys[-1] == "saidas/metadata.modelops"
    assert _metadata_status(store, "saidas/metadata.modelops") == "SUCCESS"

    # the pre-marker's own two PUTs carried the RUNTIME_ERROR failure
    # pair, both strictly before the first entradas/ PUT.
    uploaded_keys = _uploaded_keys(store)
    assert uploaded_keys[:2] == [
        "saidas/metadata.modelops",
        "saidas/status.modelops",
    ]
    pre_metadata_status = json.loads(store.uploaded[0][1])["status"]
    pre_status_token = store.uploaded[1][1].decode("ascii")
    assert pre_metadata_status == "RUNTIME_ERROR"
    assert pre_status_token == "RUNTIME_ERROR"

    first_entradas_index = next(
        index
        for index, key in enumerate(uploaded_keys)
        if key.startswith("entradas/")
    )
    assert first_entradas_index >= 2
    assert {"entradas/eco_deck.zip", "entradas/deck_processado.zip"} <= set(
        uploaded_keys
    )

    final_key, final_data = store.uploaded[-1]
    assert final_key.endswith("saidas/metadata.modelops")
    assert json.loads(final_data)["status"] == "SUCCESS"

    run_json = json.loads(store.get_bytes(_s3("saidas/run.json")))
    assert run_json["reuse"]["detected"] is True
    assert run_json["reuse"]["previous_run_id"] == "old"

    assert "delete" not in [op for op, _ in store.ops]

    hooks = _hooks(channel)
    annotation_hook = hooks[-1]
    assert annotation_hook.args[0].endswith("(prefix reused; previous run old)")


def test_publish_reused_prefix_upload_failure_keeps_metadata_at_runtime_error(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(
        deck=None, archives=(".hpcmu/outputs/cortes.zip",), raw=()
    )
    (ws.outputs_dir / "cortes.zip").write_bytes(b"cortes-bytes")
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    store.fail_on("cortes.zip")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with pytest.raises(StorageError, match="cortes.zip"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert (
        _metadata_status(store, "saidas/metadata.modelops") == "RUNTIME_ERROR"
    )


def test_publish_reused_prefix_unidentified_previous_run_annotation(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id=None)
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation_hook = _hooks(channel)[-1]
    assert annotation_hook.args[0].endswith(
        "(prefix reused; previous run unidentified)"
    )


def test_publish_no_reuse_annotation_has_no_reuse_suffix(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation_hook = _hooks(channel)[-1]
    assert annotation_hook.args[0] == _success_diagnosis().annotation()
    assert "prefix reused" not in annotation_hook.args[0]


# ---------------------------------------------------------------------------
# Amendment rule 2: the reuse suffix always survives the annotation cap
# ---------------------------------------------------------------------------


def test_publish_reused_prefix_500_char_reason_suffix_survives_cap(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis(reason="x" * 500))
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation = _hooks(channel)[-1].args[0]
    suffix = " (prefix reused; previous run old)"
    assert annotation.endswith(suffix)
    assert len(annotation) <= ANNOTATION_MAX_LENGTH


def test_annotation_max_length_within_platform_annotation_cap() -> None:
    assert ANNOTATION_MAX_LENGTH <= ANNOTATION_CAP


# ---------------------------------------------------------------------------
# ticket-049 AC5: the offline suffix, joined before the reuse suffix
# ---------------------------------------------------------------------------

_OFFLINE_SUFFIX = " (imported offline run, not executed on the cluster)"


def test_publish_offline_state_no_reuse_annotation_ends_with_offline_suffix(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    _state(
        ws,
        diagnosis=_success_diagnosis(),
        execution_source=ExecutionSource.OFFLINE,
    )
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, "run-1", outputs)

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation = _hooks(channel)[-1].args[0]
    assert annotation == _success_diagnosis().annotation() + _OFFLINE_SUFFIX
    assert "prefix reused" not in annotation


def test_publish_offline_state_reused_prefix_annotation_joins_offline_then_reuse_suffix(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    _state(
        ws,
        diagnosis=_success_diagnosis(),
        execution_source=ExecutionSource.OFFLINE,
    )
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, "run-1", outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation = _hooks(channel)[-1].args[0]
    assert annotation == (
        _success_diagnosis().annotation()
        + _OFFLINE_SUFFIX
        + " (prefix reused; previous run old)"
    )


def test_publish_offline_state_500_char_reason_keeps_both_suffixes_within_cap(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    _state(
        ws,
        diagnosis=_success_diagnosis(reason="x" * 500),
        execution_source=ExecutionSource.OFFLINE,
    )
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, "run-1", outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    annotation = _hooks(channel)[-1].args[0]
    suffix = _OFFLINE_SUFFIX + " (prefix reused; previous run old)"
    assert annotation.endswith(suffix)
    assert len(annotation) <= ANNOTATION_MAX_LENGTH


# ---------------------------------------------------------------------------
# Requirement 4: a failing pre-marker PUT stops before any artifact
# ---------------------------------------------------------------------------


def test_publish_reused_prefix_premarker_failure_stops_before_any_artifact(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = _full_realized(ws)
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    store.fail_on("metadata.modelops")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with pytest.raises(StorageError, match="pre-marker"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _upload_keys(store) == ["saidas/metadata.modelops"]
    assert _hooks(channel) == [Hook("SetExecutionArtifactsPath", (_uri(),))]


# ---------------------------------------------------------------------------
# a failing failure-pair PUT is logged and never masks the original error
# ---------------------------------------------------------------------------


def test_publish_failing_best_effort_overwrite_is_logged_and_original_error_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(
        deck=None,
        archives=(".hpcmu/outputs/cortes.zip",),
        raw=(),
    )
    (ws.root / "eco_deck.zip").write_bytes(b"eco-bytes")
    (ws.outputs_dir / "cortes.zip").write_bytes(b"cortes-bytes")
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    store.fail_on("cortes.zip")
    store.fail_on("metadata.modelops")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(StorageError, match="cortes.zip") as excinfo:
            publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert "metadata.modelops" not in str(excinfo.value)
    assert any(
        "metadata.modelops" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Amendment rule 3: an OSError from a vanished raw file is an upload failure
# ---------------------------------------------------------------------------


def test_publish_vanished_raw_file_os_error_gives_failure_pair_then_storage_error(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(
        deck=None, archives=(), raw=(("missing.rv0", "missing.rv0"),)
    )
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    _seed_reused_prefix(store, previous_run_id="old")
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with pytest.raises(StorageError, match="missing.rv0"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert (
        _metadata_status(store, "saidas/metadata.modelops") == "RUNTIME_ERROR"
    )
    assert _hooks(channel) == [Hook("SetExecutionArtifactsPath", (_uri(),))]


# ---------------------------------------------------------------------------
# Epic-02 boundary finding 3: a staging write failure must name the
# artifact it was staging, not whatever the last successful upload was.
# ---------------------------------------------------------------------------


def test_publish_run_json_staging_write_os_error_names_run_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    outputs = RealizedOutputs(deck=None, archives=(), raw=())
    _write_finalize(ws, state.run_id, outputs)

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    original_write_bytes = Path.write_bytes

    def _write_bytes(self: Path, data: bytes) -> int:
        if self.name == "run.json":
            raise OSError("disk full")
        return original_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", _write_bytes)

    with pytest.raises(StorageError, match="saidas/run.json"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())


# ---------------------------------------------------------------------------
# Requirement 2.3: the no-diagnosis fallback
# ---------------------------------------------------------------------------


def test_publish_no_diagnosis_falls_back_to_runtime_error(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=None)

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    reloaded = StateStore(ws).load()
    assert reloaded is not None
    assert reloaded.diagnosis is not None
    assert reloaded.diagnosis.status is RunStatus.RUNTIME_ERROR
    assert reloaded.diagnosis.rule_id == "core.no_diagnosis"
    assert ws.legacy_metadata_path.is_file()
    assert ws.legacy_status_path.read_text(encoding="ascii") == "RUNTIME_ERROR"
    assert (
        _metadata_status(store, "saidas/metadata.modelops") == "RUNTIME_ERROR"
    )
    assert _hooks(channel)[:2] == [
        Hook("SetExecutionArtifactsPath", (_uri(),)),
        Hook("SetRuntimeError", ()),
    ]


# ---------------------------------------------------------------------------
# Testing Requirements: a missing finalize.json still uploads logs/projections
# ---------------------------------------------------------------------------


def test_publish_missing_finalize_json_still_uploads_logs_and_projections(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    model_log = _write_log(ws, Phase.MODEL, "111")
    _state(
        ws,
        diagnosis=Diagnosis(
            status=RunStatus.RUNTIME_ERROR,
            rule_id="core.finalize_crashed",
            reason="finalize crashed",
            at=utc_now_iso(),
        ),
        jobs=(JobRecord(Phase.MODEL, "111", utc_now_iso(), model_log),),
    )
    assert not ws.finalize_path.exists()

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _upload_keys(store) == [
        "saidas/logs/model-111.out",
        "saidas/status.modelops",
        "saidas/run.json",
        "saidas/metadata.modelops",
    ]
    assert _status_of(store, "saidas/status.modelops") == "RUNTIME_ERROR"


# ---------------------------------------------------------------------------
# ticket-038 amendment 2 consistency: an unreadable finalize.json (OSError)
# is treated like a missing/invalid record, not a fatal error
# ---------------------------------------------------------------------------


def test_publish_load_finalize_os_error_treated_as_missing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    model_log = _write_log(ws, Phase.MODEL, "111")
    _state(
        ws,
        diagnosis=_success_diagnosis(),
        jobs=(JobRecord(Phase.MODEL, "111", utc_now_iso(), model_log),),
    )

    def _raise_os_error(ws: Workspace, run_id: str) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(
        "hpc_model_utils.core.lifecycle.publish.load_finalize",
        _raise_os_error,
    )

    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _upload_keys(store) == [
        "saidas/logs/model-111.out",
        "saidas/status.modelops",
        "saidas/run.json",
        "saidas/metadata.modelops",
    ]
    assert _status_of(store, "saidas/status.modelops") == "SUCCESS"
    assert _hooks(channel) == [
        Hook("SetExecutionArtifactsPath", (_uri(),)),
        Hook("SetSuccess", ()),
        Hook("SetAnnotation", (_success_diagnosis().annotation(),)),
    ]


# ---------------------------------------------------------------------------
# A missing state.json is a StateFormatError
# ---------------------------------------------------------------------------


def test_publish_missing_state_raises_state_format_error(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    store = RecordingObjectStore()
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)

    with pytest.raises(StateFormatError, match="state.json"):
        publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert store.ops == []
    assert channel.lines == []


# ---------------------------------------------------------------------------
# ADR-083/ADR-077/ADR-084 (ticket-048): the telemetry export inside publish
# ---------------------------------------------------------------------------

_HEADLINE = (("telemetry.model.cpu_util_mean", "0.5000"),)
_TELEMETRY_KEYS = [f"saidas/telemetry/{name}" for name in EXPORT_FILES]


class _SequencedStore(RecordingObjectStore):
    """Appends each ``exists_any`` and ``upload`` to a list shared with the
    channel and the export spy, so a test can assert their relative order."""

    def __init__(self, events: list[tuple[str, str]]) -> None:
        super().__init__()
        self._events = events

    def exists_any(self, prefix: S3Uri) -> bool:
        self._events.append(("exists_any", str(prefix)))
        return super().exists_any(prefix)

    def upload(self, src: Path, uri: S3Uri) -> None:
        self._events.append(("upload", str(uri)))
        super().upload(src, uri)


class _SequencedChannel(_ListChannel):
    def __init__(self, events: list[tuple[str, str]]) -> None:
        super().__init__()
        self._events = events

    def write_line(self, line: str) -> None:
        self._events.append(("hook", line))
        super().write_line(line)


class _ExportSpy:
    def __init__(
        self, result: ExportResult, events: list[tuple[str, str]]
    ) -> None:
        self.states: list[RunState] = []
        self._result = result
        self._events = events

    def __call__(
        self,
        ws: Workspace,
        state: RunState,
        *,
        interpreter: str = sys.executable,
        timeout_s: float = EXPORT_TIMEOUT_S,
    ) -> ExportResult:
        self._events.append(("export", ""))
        self.states.append(state)
        return self._result


def _fake_export(
    monkeypatch: pytest.MonkeyPatch,
    result: ExportResult,
    events: list[tuple[str, str]] | None = None,
) -> _ExportSpy:
    spy = _ExportSpy(result, [] if events is None else events)
    monkeypatch.setattr(publish_module, "export_telemetry", spy)
    return spy


def _exported(
    ws: Workspace, headline: tuple[tuple[str, str], ...] = _HEADLINE
) -> Exported:
    ws.telemetry_outputs_dir.mkdir(parents=True, exist_ok=True)
    samples, workspace, summary = (
        ws.telemetry_outputs_dir / name for name in EXPORT_FILES
    )
    samples.write_bytes(b"samples-bytes")
    workspace.write_bytes(b"workspace-parquet-bytes")
    summary.write_bytes(b'{"kind": "summary"}')
    return Exported((samples, workspace, summary), headline)


def _jobs_state(ws: Workspace) -> RunState:
    model_log = _write_log(ws, Phase.MODEL, "1000")
    finalize_log = _write_log(ws, Phase.FINALIZE, "1001")
    return _state(
        ws,
        diagnosis=_success_diagnosis(),
        jobs=(
            JobRecord(Phase.MODEL, "1000", utc_now_iso(), model_log),
            JobRecord(Phase.FINALIZE, "1001", utc_now_iso(), finalize_log),
        ),
    )


def _publish_hooks(ws: Workspace, store: RecordingObjectStore) -> list[Hook]:
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())
    return _hooks(channel)


def _timeline(events: list[tuple[str, str]]) -> list[str | Hook]:
    base = _base()
    timeline: list[str | Hook] = []
    for kind, value in events:
        if kind == "upload":
            timeline.append(value[len(base) :])
        elif kind == "hook":
            timeline.extend(parse_hooks(value))
    return timeline


def _metadata_pairs(hooks: list[Hook]) -> list[tuple[str, ...]]:
    return [hook.args for hook in hooks if hook.method == "SetMetadata"]


def test_publish_telemetry_position_uploads_after_sintese_before_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _jobs_state(ws)
    _write_finalize(ws, state.run_id, _full_realized(ws))
    _write_sintese(ws, "a.parquet")
    _write_sintese(ws, "b.parquet")
    exported = _exported(ws)
    _fake_export(monkeypatch, exported)
    store = RecordingObjectStore()

    _publish_hooks(ws, store)

    assert _upload_keys(store) == [
        "entradas/eco_deck.zip",
        "entradas/deck_processado.zip",
        "saidas/cortes.zip",
        "saidas/pmo.dat",
        "saidas/logs/model-1000.out",
        "saidas/logs/finalize-1001.out",
        "sintese/a.parquet",
        "sintese/b.parquet",
        "saidas/telemetry/samples.parquet",
        "saidas/telemetry/workspace.parquet",
        "saidas/telemetry/summary.json",
        "saidas/status.modelops",
        "saidas/run.json",
        "saidas/metadata.modelops",
    ]
    run_json = json.loads(store.get_bytes(_s3("saidas/run.json")))
    sizes = {entry["path"]: entry["bytes"] for entry in run_json["artifacts"]}
    for key, path in zip(_TELEMETRY_KEYS, exported.files, strict=True):
        assert sizes[key] == path.stat().st_size
        assert store.get_bytes(_s3(key)) == path.read_bytes()


def test_publish_telemetry_hooks_follow_the_commit_marker_and_precede_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _jobs_state(ws)
    _write_finalize(ws, state.run_id, None)
    events: list[tuple[str, str]] = []
    _fake_export(monkeypatch, _exported(ws), events)
    store = _SequencedStore(events)
    reporter = Reporter(_SequencedChannel(events), enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    assert _timeline(events)[-5:] == [
        "saidas/metadata.modelops",
        Hook("SetMetadata", ("telemetry.model.cpu_util_mean", "0.5000")),
        Hook("SetExecutionArtifactsPath", (_uri(),)),
        Hook("SetSuccess", ()),
        Hook("SetAnnotation", (_success_diagnosis().annotation(),)),
    ]


def test_publish_telemetry_hooks_emit_exactly_the_headline_pairs_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _jobs_state(ws)
    _write_finalize(ws, state.run_id, None)
    headline = (
        ("telemetry.model.job_mem_peak_bytes_max", "1234"),
        ("telemetry.model.cpu_util_mean", "0.5000"),
    )
    _fake_export(monkeypatch, _exported(ws, headline))

    hooks = _publish_hooks(ws, RecordingObjectStore())

    assert _metadata_pairs(hooks) == [tuple(pair) for pair in headline]


@pytest.mark.parametrize(
    "bad_uri",
    ["not-an-s3-uri", f"s3://{_BUCKET}/artifacts/short-hash/"],
    ids=["malformed", "bad-hash"],
)
def test_publish_telemetry_call_bad_uri_raises_usage_error_without_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_uri: str
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=_success_diagnosis())
    spy = _fake_export(monkeypatch, NothingToExport())
    reporter = Reporter(_ListChannel(), enabled=True)

    with pytest.raises(UsageError, match="invalid artifacts"):
        publish(
            ws,
            FakePlugin(),
            RecordingObjectStore(),
            reporter,
            StateStore(ws),
            bad_uri,
        )

    assert spy.states == []


def test_publish_telemetry_call_missing_state_raises_without_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    spy = _fake_export(monkeypatch, NothingToExport())
    reporter = Reporter(_ListChannel(), enabled=True)

    with pytest.raises(StateFormatError, match="state.json"):
        publish(
            ws,
            FakePlugin(),
            RecordingObjectStore(),
            reporter,
            StateStore(ws),
            _uri(),
        )

    assert spy.states == []


def test_publish_telemetry_call_no_diagnosis_receives_the_fallback_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=None)
    spy = _fake_export(monkeypatch, NothingToExport())

    _publish_hooks(ws, RecordingObjectStore())

    [seen] = spy.states
    assert seen.diagnosis is not None
    assert seen.diagnosis.rule_id == "core.no_diagnosis"


def test_publish_telemetry_call_precedes_the_first_store_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=_success_diagnosis())
    events: list[tuple[str, str]] = []
    _fake_export(monkeypatch, NothingToExport(), events)
    store = _SequencedStore(events)
    reporter = Reporter(_SequencedChannel(events), enabled=True)

    publish(ws, FakePlugin(), store, reporter, StateStore(ws), _uri())

    kinds = [kind for kind, _ in events]
    assert kinds.count("export") == 1
    assert kinds.index("export") < kinds.index("exists_any")


def test_publish_telemetry_call_reused_prefix_still_exports_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    _state(ws, diagnosis=_success_diagnosis())
    spy = _fake_export(monkeypatch, NothingToExport())
    store = RecordingObjectStore()
    _seed_reused_prefix(store)

    _publish_hooks(ws, store)

    assert len(spy.states) == 1


def test_publish_telemetry_skip_export_failed_equals_nothing_to_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _jobs_state(ws)
    _write_finalize(ws, state.run_id, _full_realized(ws))
    _write_sintese(ws)
    outcomes: list[tuple[list[str], bytes, bytes, list[Hook]]] = []
    for result in (NothingToExport(), ExportFailed("x")):
        _fake_export(monkeypatch, result)
        store = RecordingObjectStore()
        hooks = _publish_hooks(ws, store)
        outcomes.append(
            (
                _upload_keys(store),
                store.get_bytes(_s3("saidas/status.modelops")),
                store.get_bytes(_s3("saidas/metadata.modelops")),
                hooks,
            )
        )

    baseline, failed = outcomes
    assert failed == baseline
    keys, _, _, hooks = failed
    assert not [key for key in keys if key.startswith("saidas/telemetry/")]
    assert _metadata_pairs(hooks) == []


def test_publish_telemetry_skip_reused_prefix_keeps_old_objects_and_deletes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _ws(tmp_path)
    state = _state(ws, diagnosis=_success_diagnosis())
    _write_finalize(ws, state.run_id, None)
    _fake_export(monkeypatch, NothingToExport())
    store = RecordingObjectStore()
    store.seed(_s3("saidas/telemetry/summary.json"), b"old-summary")
    _seed_reused_prefix(store)

    _publish_hooks(ws, store)

    assert "delete" not in [op for op, _ in store.ops]
    assert not [
        key
        for key in _upload_keys(store)
        if key.startswith("saidas/telemetry/")
    ]
    assert (
        store.get_bytes(_s3("saidas/telemetry/summary.json")) == b"old-summary"
    )
    run_json = json.loads(store.get_bytes(_s3("saidas/run.json")))
    assert not [
        entry
        for entry in run_json["artifacts"]
        if entry["path"].startswith("saidas/telemetry/")
    ]


def test_publish_telemetry_skip_real_runner_finalize_only_job_exports_finalize_scope(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)
    finalize_log = _write_log(ws, Phase.FINALIZE, "1001")
    state = _state(
        ws,
        diagnosis=_success_diagnosis(),
        jobs=(JobRecord(Phase.FINALIZE, "1001", utc_now_iso(), finalize_log),),
    )
    _write_finalize(ws, state.run_id, None)
    head = header("finalize", "1001")
    write_staging(
        ws.telemetry_dir,
        head,
        [sample(head, seq, cpu_busy_cores=float(seq + 1)) for seq in range(3)],
    )
    store = RecordingObjectStore()

    hooks = _publish_hooks(ws, store)

    assert [
        key
        for key in _upload_keys(store)
        if key.startswith("saidas/telemetry/")
    ] == _TELEMETRY_KEYS
    pairs = _metadata_pairs(hooks)
    assert ("telemetry.finalize.cpu_busy_cores_mean", "2.00") in pairs
    assert not [
        key
        for key, *_ in pairs
        if key.startswith(("telemetry.model.", "telemetry.workspace."))
    ]
