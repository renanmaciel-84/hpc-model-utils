"""ADR-048/R82: publish a real plugin's golden archive workspace through
v2 ``publish`` (ticket-055).

``publish_golden_run`` rebuilds the v1.1.2 archive-capture workspace,
records the golden run sequence in ``state.json`` and ``finalize.json``
and publishes it to an in-memory store. The contract suites then compare
the uploaded objects with the frozen goldens. Every value is public
fixture data (R104). ``telemetry=True`` first writes builder staging files
for the recorded jobs, so the real export publishes the telemetry files.
Exceptions propagate: a fixture that cannot be published is a test
failure, never a skip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from hpc_model_utils.core.diagnosis import Diagnosis, RunStatus, utc_now_iso
from hpc_model_utils.core.lifecycle.publish import publish
from hpc_model_utils.core.outputs import realize
from hpc_model_utils.core.state import (
    FinalizeRecord,
    InputsInfo,
    JobRecord,
    ModelInfo,
    RunState,
    StateStore,
    current_tool,
    write_finalize,
)
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.models import PLUGINS
from hpc_model_utils.platform.modelops import Reporter
from tests.support.decks import golden_archive_workspace
from tests.support.hooks import Hook, parse_hooks
from tests.support.object_store import RecordingObjectStore
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    header,
    sample,
    walk,
    write_staging,
)

ModelName = Literal["newave", "decomp"]

# The v1 capture used ``artifacts/hash01``, but ``publish`` also enforces
# the Reporter's 64-hex artifacts hash (ADR-028), as ticket-040's tests
# do. The goldens hold keys relative to the prefix, so the hash value
# never reaches a comparison.
PREFIX: Final = "s3://outputs-bucket/artifacts/" + "0" * 62 + "01"

# (model job, finalize job); the finalize id is the golden ``job_id``.
JOB_IDS: Final[Mapping[ModelName, tuple[str, str]]] = MappingProxyType(
    {"newave": ("1000", "1001"), "decomp": ("2000", "2001")}
)

# The ``check_and_fetch_executables`` versions of the ``<model>_no_parent``
# projection goldens.
_MODEL_VERSIONS: Final[Mapping[ModelName, str]] = MappingProxyType(
    {"newave": "30.0.4", "decomp": "32.0"}
)


@dataclass(frozen=True, slots=True)
class PublishedRun:
    ws: Workspace
    store: RecordingObjectStore
    prefix: S3Uri
    hooks: list[Hook]
    state: RunState


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _write_telemetry_staging(ws: Workspace, jobs: Sequence[JobRecord]) -> None:
    for job in jobs:
        head = header(job.phase.value, job.job_id)
        records = [
            sample(head, seq, cpu_busy_cores=1.5 + seq) for seq in range(3)
        ]
        if job.phase is Phase.MODEL:
            records.append(
                walk(head, ts=ORIGIN_MS + 60_000, allocated_bytes=4096)
            )
        write_staging(ws.telemetry_dir, head, records)


def publish_golden_run(
    tmp_path: Path,
    model: ModelName,
    *,
    status: RunStatus = RunStatus.SUCCESS,
    telemetry: bool = False,
) -> PublishedRun:
    ws = golden_archive_workspace(tmp_path, model)
    ws.ensure_layout()
    plugin = PLUGINS[model]

    jobs: list[JobRecord] = []
    for phase, job_id in zip(
        (Phase.MODEL, Phase.FINALIZE), JOB_IDS[model], strict=True
    ):
        log = ws.log_path(phase, job_id)
        log.write_bytes(b"log\n")
        jobs.append(JobRecord(phase, job_id, utc_now_iso(), ws.relative(log)))

    diagnosis = Diagnosis(
        status,
        f"{model}.contract_fixture",
        "contract fixture",
        at=utc_now_iso(),
    )
    state = replace(
        RunState.new(plugin.name, current_tool()),
        model=ModelInfo(plugin.model_name, _MODEL_VERSIONS[model]),
        inputs=InputsInfo(f"s3://inputs-bucket/ingest/deck_{model}.zip", ""),
        study=plugin.study_info(ws),
        jobs=tuple(jobs),
        reported_job_id=JOB_IDS[model][1],
        diagnosis=diagnosis,
    )
    state_store = StateStore(ws)
    state_store.save(state)
    if telemetry:
        _write_telemetry_staging(ws, jobs)

    realized = realize(plugin.outputs(ws), ws, workers=2)
    write_finalize(
        ws,
        FinalizeRecord(
            run_id=state.run_id, diagnosis=diagnosis, outputs=realized
        ),
    )

    sintese = ws.root / "sintese"
    sintese.mkdir()
    (sintese / "example.parquet").write_bytes(b"x\n")

    store = RecordingObjectStore()
    channel = _ListChannel()
    publish(
        ws,
        plugin,
        store,
        Reporter(channel, enabled=True),
        state_store,
        PREFIX,
    )
    return PublishedRun(
        ws=ws,
        store=store,
        prefix=S3Uri.parse(PREFIX),
        hooks=parse_hooks("\n".join(channel.lines)),
        state=state,
    )
