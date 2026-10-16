"""R39/R11/R96/R29/R101/ADR-020/ADR-006: the ``run`` submission step.

``submit`` posts the model job, then the finalize job with
``--dependency=afterany:<model>`` (R39), in that printed order
(ADR-020): the platform's last-match ``jobId`` scrape therefore picks
up the finalize id. Each sbatch call and the matching
``JobLedger.add`` run inside one ``blocked_signals()`` critical
section, so a signal arriving between sbatch returning and the id
being recorded can never lose that id -- D2 (ticket-043) reads the
ledger for exactly this reason.

An offline-ingested run (``execution_source=OFFLINE``) and an
explicit ``--skip`` both submit finalize only, with no dependency
(R96); ``--skip`` is logged as the reason even when both are true,
since it is the caller's own explicit override.

A finalize-submit failure after the model job is already live leaves
that id only in the in-memory ``ledger`` -- by design (R29): there is
one ``store.save``, at the very end, so a failure here leaves
``state.json`` untouched. The orphaned model job stays recoverable
through the ``Submitted batch job <model>`` line already printed to
the platform, through ``cancel_run --job-id``, and later through the
in-process signal handlers (ticket-043); nothing further is needed
here.
"""

from __future__ import annotations

import logging
import signal
import socket
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from hpc_model_utils.core.diagnosis import (
    Diagnosis,
    EvidenceItem,
    RunStatus,
    utc_now_iso,
)
from hpc_model_utils.core.errors import StateFormatError
from hpc_model_utils.core.follow import FollowAborted, LogTail, follow
from hpc_model_utils.core.launch import (
    TELEMETRY_OFF,
    FinalizeInvocation,
    Resources,
    TelemetryConfig,
    Toolchain,
    write_finalize_script,
    write_model_script,
)
from hpc_model_utils.core.lifecycle.finalize import synthesis_status
from hpc_model_utils.core.lifecycle.signals import cancel_on_termination
from hpc_model_utils.core.plugin import ModelPlugin
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import (
    ExecutionSource,
    FinalizeRecord,
    JobRecord,
    RunState,
    StateStore,
    StepRecord,
    load_finalize,
    metadata_items,
    write_projections,
)
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.slurm import Slurm

logger = logging.getLogger(__name__)

_BLOCKED_SIGNALS = {signal.SIGTERM, signal.SIGHUP}


class StatusReporter(Protocol):
    """Structural seam for ``submit``'s ``reporter`` argument (core
    must not import ``platform``). ``platform.modelops.Reporter``
    satisfies this without inheriting from it; ticket-038 and
    ticket-040 import this Protocol from here too."""

    def announce_job(self, job_id: str) -> None: ...

    def terminal(self, status: RunStatus, annotation: str) -> bool: ...

    def metadata(self, key: str, value: str) -> None: ...

    def artifacts_path(self, uri: str) -> None: ...

    def check_artifacts_path(self, uri: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    """``telemetry`` defaults to ``TELEMETRY_OFF`` (ADR-082), so a
    constructor that omits it renders today's job scripts. Revisit: if
    a second production constructor appears, make the field required."""

    resources: Resources
    tools: Toolchain
    skip_model: bool
    synthesis_bin: Path | None
    telemetry: TelemetryConfig = TELEMETRY_OFF


class JobLedger:
    """The in-memory record of every id submitted so far this
    process, created by the caller so the D2 handler (ticket-043) can
    read it even when a later submission fails."""

    def __init__(self) -> None:
        self._ids: list[str] = []

    def add(self, job_id: str) -> None:
        self._ids.append(job_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)


@dataclass(frozen=True, slots=True)
class SubmittedJobs:
    model_id: str | None
    finalize_id: str


@contextmanager
def blocked_signals() -> Iterator[None]:
    """Block ``SIGTERM``/``SIGHUP`` for the *calling thread only* and
    restore the previous mask on exit.

    ``signal.pthread_sigmask`` is per-thread, which is sufficient
    here: ``infra.shell.run``'s reader thread is always created
    inside a block like this one, so it inherits the blocked mask,
    and the CLI process starts no other long-lived thread.
    """
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _BLOCKED_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def submit(
    ws: Workspace,
    plugin: ModelPlugin,
    slurm: Slurm,
    reporter: StatusReporter,
    store: StateStore,
    req: SubmitRequest,
    ledger: JobLedger,
) -> SubmittedJobs:
    state = store.load_or_create(plugin.name)
    ws.ensure_layout()

    finalize_only = (
        req.skip_model or state.execution_source is ExecutionSource.OFFLINE
    )
    if finalize_only:
        reason = "--skip" if req.skip_model else "offline-ingested run"
        logger.info("model job submission skipped: %s", reason)

    model_id: str | None = None
    if not finalize_only:
        spec = plugin.launch(ws, req.resources)
        model_script = write_model_script(
            ws, spec, req.resources, req.tools, telemetry=req.telemetry
        )
        with blocked_signals():
            model_id = slurm.submit(model_script)
            ledger.add(model_id)
        reporter.announce_job(model_id)

    finalize_script = write_finalize_script(
        ws,
        req.resources,
        req.tools,
        FinalizeInvocation(
            plugin.name, model_id, req.resources.cores, req.synthesis_bin
        ),
        telemetry=req.telemetry,
    )
    with blocked_signals():
        finalize_id = slurm.submit(finalize_script, after=model_id)
        ledger.add(finalize_id)
    reporter.announce_job(finalize_id)

    submitted_at = utc_now_iso()
    records: list[JobRecord] = []
    if model_id is not None:
        records.append(
            JobRecord(
                Phase.MODEL,
                model_id,
                submitted_at,
                ws.relative(ws.log_path(Phase.MODEL, model_id)),
            )
        )
    records.append(
        JobRecord(
            Phase.FINALIZE,
            finalize_id,
            submitted_at,
            ws.relative(ws.log_path(Phase.FINALIZE, finalize_id)),
        )
    )
    store.save(replace(state, jobs=state.jobs + tuple(records)))

    return SubmittedJobs(model_id, finalize_id)


def _ingest(
    ws: Workspace,
    slurm: Slurm,
    finalize_id: str,
    run_id: str,
    settings: EngineSettings,
) -> tuple[Diagnosis, FinalizeRecord | None, tuple[StepRecord, ...]]:
    """R31/R43/ADR-006/amendment 2: ingest ``finalize.json`` written by
    the job-side step, or diagnose a finalize crash from ``sacct``
    when the record is absent, unreadable (``OSError``), or stamped
    with a different ``run_id`` (``StateFormatError``, Requirement 3).
    No other exception is caught."""
    error_text: str | None = None
    rec: FinalizeRecord | None
    try:
        rec = load_finalize(ws, run_id)
    except (StateFormatError, OSError) as exc:
        rec = None
        error_text = str(exc)
    if rec is not None:
        return rec.diagnosis, rec, rec.steps

    out = slurm.outcome(
        finalize_id,
        attempts=settings.outcome_attempts,
        backoff=settings.outcome_backoff,
    )
    evidence: tuple[EvidenceItem, ...] = (
        EvidenceItem("slurm", "sacct", out.raw),
    )
    if error_text is not None:
        evidence += (EvidenceItem("guard", "finalize.json", error_text),)
    diagnosis = Diagnosis(
        RunStatus.RUNTIME_ERROR,
        "core.finalize_crashed",
        f"finalize job {finalize_id} ended {out.state} "
        f"(exit {out.exit_code}) without writing its record",
        evidence=evidence,
        job_id=finalize_id,
        at=utc_now_iso(),
    )
    return diagnosis, None, ()


def run(
    ws: Workspace,
    plugin: ModelPlugin,
    slurm: Slurm,
    reporter: StatusReporter,
    store: StateStore,
    req: SubmitRequest,
    ledger: JobLedger,
    *,
    settings: EngineSettings,
    emit: Callable[[str], None],
) -> RunState:
    """R41/R43/R95/R120/R31/ADR-006/ADR-011/ADR-053: submit, follow
    the model log to completion and then the finalize log (R41),
    ingest the finalize outcome, commit it, and re-emit it.

    Every job outcome -- a follow abort or an ingested/diagnosed
    finalize record -- becomes a recorded ``Diagnosis`` and a normal
    return; only a command-level failure (sbatch, an unqueryable
    scheduler, or unwritable state) propagates. This never calls
    ``reporter.terminal``: that belongs to ``result_upload``
    (ticket-040, R15).

    ``submit`` through the finalize ingest run under
    ``cancel_on_termination`` (ticket-043, D2): a SIGTERM, a SIGHUP or
    a broken stdout pipe there scancels ``ledger``'s jobs and exits
    with the matching 128+n code instead of reaching the code below.
    """
    started_at = utc_now_iso()
    entry_monotonic = time.monotonic()

    rec: FinalizeRecord | None = None
    rec_steps: tuple[StepRecord, ...] = ()
    with cancel_on_termination(
        slurm, ledger, store, timeout=settings.cancel_timeout
    ):
        jobs = submit(ws, plugin, slurm, reporter, store, req, ledger)
        state = store.load()
        if state is None:
            raise StateFormatError("state.json missing after submit")

        try:
            if jobs.model_id is not None:
                follow(
                    jobs.model_id,
                    LogTail(ws.log_path(Phase.MODEL, jobs.model_id)),
                    slurm,
                    emit,
                    settings=settings,
                )
            follow(
                jobs.finalize_id,
                LogTail(ws.log_path(Phase.FINALIZE, jobs.finalize_id)),
                slurm,
                emit,
                settings=settings,
            )
        except FollowAborted as exc:
            # Amendment 1: cancel and wait for both jobs to leave the
            # queue. A SchedulerCommandError from either call is a
            # command-level failure and propagates (state.json already
            # lists both jobs, so cancel_run can retry).
            slurm.cancel(ledger.ids)
            gone = slurm.wait_gone(ledger.ids, timeout=settings.cancel_timeout)
            evidence: tuple[EvidenceItem, ...] = ()
            if not gone:
                ids_text = ", ".join(ledger.ids)
                evidence = (
                    EvidenceItem(
                        "slurm",
                        "scancel",
                        f"jobs {ids_text} still queued after "
                        f"{settings.cancel_timeout}s",
                    ),
                )
            diagnosis = Diagnosis(
                RunStatus.RUNTIME_ERROR,
                f"core.{exc.kind}",
                exc.reason,
                evidence=evidence,
                job_id=exc.job_id,
                at=utc_now_iso(),
            )
        else:
            diagnosis, rec, rec_steps = _ingest(
                ws, slurm, jobs.finalize_id, state.run_id, settings
            )

    finished_at = utc_now_iso()
    run_step = StepRecord(
        command="run",
        host=socket.gethostname(),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - entry_monotonic,
        outcome="ok",
    )
    new_state = replace(
        state,
        diagnosis=diagnosis,
        reported_job_id=jobs.finalize_id,
        steps=state.steps + rec_steps + (run_step,),
    )
    store.save(new_state)
    write_projections(
        ws,
        new_state,
        always_write_parent_path=plugin.always_write_parent_path,
    )

    for key, value in metadata_items(
        new_state,
        always_write_parent_path=plugin.always_write_parent_path,
    ):
        if key in ("job_id", "status"):
            reporter.metadata(key, value)
    if rec is not None:
        status = synthesis_status(rec)
        if status is not None:
            reporter.metadata("synthesis_status", status)

    logger.info("run finished: %s [%s]", diagnosis.status, diagnosis.rule_id)
    return new_state
