"""ADR-042/ADR-012/ADR-028/R13/R15/R56/R115/R55/R5/R14/R136: the
``publish`` login-side step (v1's ``result_upload``).

ADR-042 makes the upload order a one-way data-integrity contract with
``saidas/metadata.modelops`` as its commit marker: every other object
of the run reaches S3 before a SUCCESS ``metadata.modelops`` can ever
exist there, and ``SetExecutionArtifactsPath``/the terminal hook are
each emitted exactly once, only after every upload has finished. A
reused prefix (R115) is covered twice -- a pre-marker RUNTIME_ERROR
overwrite before the first new artifact (covers a kill mid-upload)
and a best-effort RUNTIME_ERROR overwrite on any later failure (covers
everything else) -- and nothing already there is ever deleted (R5).

ADR-083 places the telemetry export before any upload, uploads its three
files after ``sintese/*`` and before ``saidas/status.modelops``, and emits
its headline hooks after the last upload.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path, PurePosixPath

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
from hpc_model_utils.core.lifecycle.run import StatusReporter
from hpc_model_utils.core.plugin import ModelPlugin
from hpc_model_utils.core.run_record import (
    ArtifactEntry,
    ReuseRecord,
    build_run_record,
    read_previous_run_id,
    render_run_json,
)
from hpc_model_utils.core.state import (
    ExecutionSource,
    RunState,
    StateStore,
    load_finalize,
    render_metadata,
    render_status,
    write_projections,
)
from hpc_model_utils.core.telemetry import Exported, export_telemetry
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.errors import (
    ObjectNotFoundError,
    StorageBackendError,
)
from hpc_model_utils.infra.s3 import ObjectStore, S3Uri

logger = logging.getLogger(__name__)

_UPLOAD_ERRORS = (ObjectNotFoundError, StorageBackendError, OSError)


def validate_artifacts_prefix(uri: str) -> S3Uri:
    """Requirement 1: parse ``uri`` and reject an empty key or an
    empty final ``/``-segment, which ``platform.modelops`` cannot
    catch on its own (ADR-028 -- ``DeleteArtifactsAsync`` deletes
    everything under the stored prefix)."""
    try:
        parsed = S3Uri.parse(uri)
    except ValueError as exc:
        raise UsageError(f"invalid artifacts prefix {uri!r}: {exc}") from exc
    key = parsed.key[:-1] if parsed.key.endswith("/") else parsed.key
    if not key or not key.rsplit("/", 1)[-1]:
        raise UsageError(f"invalid artifacts prefix: {uri!r}")
    return S3Uri(parsed.bucket, key)


def _render_pair(
    state: RunState, *, always_write_parent_path: bool
) -> tuple[bytes, bytes]:
    metadata = render_metadata(
        state, always_write_parent_path=always_write_parent_path
    )
    status = render_status(state)
    # ``state.diagnosis`` is never ``None`` by the time this is called
    # (``publish`` installs the Requirement 2.3 fallback first), so
    # ``metadata_items`` always yields at least ``job_id``/``status``.
    assert metadata is not None
    assert status is not None
    return metadata.encode("ascii"), status.encode("ascii")


def _failure_pair(
    state: RunState, *, always_write_parent_path: bool
) -> tuple[bytes, bytes]:
    """The RUNTIME_ERROR projection of ``state``, used for the reuse
    pre-marker and the best-effort failure overwrite (R115, ADR-042)."""
    diagnosis = state.diagnosis
    assert diagnosis is not None
    failure_state = replace(
        state, diagnosis=replace(diagnosis, status=RunStatus.RUNTIME_ERROR)
    )
    return _render_pair(
        failure_state, always_write_parent_path=always_write_parent_path
    )


def publish(
    ws: Workspace,
    plugin: ModelPlugin,
    store: ObjectStore,
    reporter: StatusReporter,
    state_store: StateStore,
    uri: str,
) -> None:
    prefix = validate_artifacts_prefix(uri)
    try:
        reporter.check_artifacts_path(uri)
    except ValueError as exc:
        raise UsageError(f"invalid artifacts uri {uri!r}: {exc}") from exc

    state = state_store.load()
    if state is None:
        raise StateFormatError("state.json missing before publish")

    if state.diagnosis is None:
        state = replace(
            state,
            diagnosis=Diagnosis(
                RunStatus.RUNTIME_ERROR,
                "core.no_diagnosis",
                "no run diagnosis recorded before result_upload",
                at=utc_now_iso(),
            ),
        )
        state_store.save(state)
        write_projections(
            ws, state, always_write_parent_path=plugin.always_write_parent_path
        )
    diagnosis = state.diagnosis
    assert diagnosis is not None

    telemetry = export_telemetry(ws, state)

    reused = store.exists_any(prefix.join("saidas/"))
    previous: str | None = None
    if reused:
        try:
            previous = read_previous_run_id(
                store.get_bytes(prefix.join("saidas/run.json"))
            )
        except ObjectNotFoundError:
            pass

    try:
        finalize_record = load_finalize(ws, state.run_id)
    except (StateFormatError, OSError) as exc:
        # R43/ticket-038 amendment 2: an unreadable or invalid
        # finalize.json is treated the same as a missing one, so a
        # finalize crash never aborts publish before the logs and
        # projections -- which it needs -- ever reach S3.
        logger.warning(
            "unreadable finalize record at %s: %s", ws.finalize_path, exc
        )
        finalize_record = None
    realized = None if finalize_record is None else finalize_record.outputs

    own_metadata, own_status = _render_pair(
        state, always_write_parent_path=plugin.always_write_parent_path
    )
    failure_metadata, failure_status = _failure_pair(
        state, always_write_parent_path=plugin.always_write_parent_path
    )

    staging = ws.outputs_dir / "publish"
    staging.mkdir(parents=True, exist_ok=True)
    metadata_key = prefix.join("saidas/metadata.modelops")
    status_key = prefix.join("saidas/status.modelops")

    if reused:
        try:
            pre_metadata = staging / "pre_metadata.modelops"
            pre_metadata.write_bytes(failure_metadata)
            store.upload(pre_metadata, metadata_key)
            pre_status = staging / "pre_status.modelops"
            pre_status.write_bytes(failure_status)
            store.upload(pre_status, status_key)
        except _UPLOAD_ERRORS as exc:
            reporter.artifacts_path(uri)
            raise StorageError(f"pre-marker upload failed: {exc}") from exc

    manifest: list[ArtifactEntry] = []
    current = "saidas/metadata.modelops"

    def _upload(path: Path, key: S3Uri, rel: str) -> None:
        nonlocal current
        current = rel
        size = path.stat().st_size
        store.upload(path, key)
        manifest.append(ArtifactEntry(rel, size))

    try:
        eco_deck_path = ws.eco_deck_path
        if eco_deck_path.is_file():
            rel = f"entradas/{eco_deck_path.name}"
            _upload(eco_deck_path, prefix.join(rel), rel)

        if realized is not None and realized.deck is not None:
            rel = "entradas/deck_processado.zip"
            _upload(ws.root / realized.deck, prefix.join(rel), rel)

        if realized is not None:
            for name in realized.archives:
                rel = f"saidas/{PurePosixPath(name).name}"
                _upload(ws.root / name, prefix.join(rel), rel)

            for source, dest in realized.raw:
                rel = f"saidas/{dest}"
                _upload(ws.root / source, prefix.join(rel), rel)

        for job in state.jobs:
            log_path = ws.root / job.log
            if log_path.is_file():
                rel = f"saidas/logs/{job.phase.value}-{job.job_id}.out"
                _upload(log_path, prefix.join(rel), rel)

        if ws.synthesis_log_path.is_file():
            rel = "saidas/logs/synthesis.out"
            _upload(ws.synthesis_log_path, prefix.join(rel), rel)

        sintese_dir = ws.root / "sintese"
        if sintese_dir.is_dir():
            for entry in sorted(sintese_dir.iterdir()):
                if entry.is_file():
                    rel = f"sintese/{entry.name}"
                    _upload(entry, prefix.join(rel), rel)

        if isinstance(telemetry, Exported):
            for path in telemetry.files:
                rel = f"saidas/telemetry/{path.name}"
                _upload(path, prefix.join(rel), rel)

        rel = "saidas/status.modelops"
        current = rel
        status_local = staging / "status.modelops"
        status_local.write_bytes(own_status)
        _upload(status_local, status_key, rel)

        record = build_run_record(
            state,
            artifacts=[
                *manifest,
                ArtifactEntry("saidas/metadata.modelops", len(own_metadata)),
            ],
            reuse=ReuseRecord(reused, previous) if reused else None,
            published_at=utc_now_iso(),
        )
        rel = "saidas/run.json"
        current = rel
        run_json_local = ws.outputs_dir / "run.json"
        run_json_local.write_bytes(render_run_json(record).encode("utf-8"))
        _upload(run_json_local, prefix.join(rel), rel)

        rel = "saidas/metadata.modelops"
        current = rel
        metadata_local = staging / "metadata.modelops"
        metadata_local.write_bytes(own_metadata)
        _upload(metadata_local, metadata_key, rel)
    except _UPLOAD_ERRORS as exc:
        if manifest or reused:
            try:
                overwrite_metadata = staging / "overwrite_metadata.modelops"
                overwrite_metadata.write_bytes(failure_metadata)
                store.upload(overwrite_metadata, metadata_key)
            except _UPLOAD_ERRORS as overwrite_exc:
                logger.warning(
                    "best-effort overwrite of saidas/metadata.modelops "
                    "failed: %s",
                    overwrite_exc,
                )
            try:
                overwrite_status = staging / "overwrite_status.modelops"
                overwrite_status.write_bytes(failure_status)
                store.upload(overwrite_status, status_key)
            except _UPLOAD_ERRORS as overwrite_exc:
                logger.warning(
                    "best-effort overwrite of saidas/status.modelops "
                    "failed: %s",
                    overwrite_exc,
                )
        reporter.artifacts_path(uri)
        raise StorageError(f"upload of {current} failed: {exc}") from exc

    if isinstance(telemetry, Exported):
        for key, value in telemetry.headline:
            reporter.metadata(key, value)

    reporter.artifacts_path(uri)
    annotation = diagnosis.annotation()
    suffixes = ""
    if state.execution_source is ExecutionSource.OFFLINE:
        suffixes += " (imported offline run, not executed on the cluster)"
    if reused:
        suffixes += (
            f" (prefix reused; previous run {previous or 'unidentified'})"
        )
    if suffixes:
        annotation = (
            annotation[: ANNOTATION_MAX_LENGTH - len(suffixes)] + suffixes
        )
    reporter.terminal(diagnosis.status, annotation)
