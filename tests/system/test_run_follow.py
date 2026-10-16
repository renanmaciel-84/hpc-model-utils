"""System tests for the post-submission half of ``run`` (ticket-038,
R41, R43, R95, R120, R31, ADR-006, ADR-011, ADR-053), against the fake
SLURM harness, the real ``platform.modelops.Reporter`` and the
ticket-036 CLI shim.

A few scenarios the fake SLURM cannot express deterministically
(amendment rules, a finalize job that crashes before writing its
record, a run_id mismatch surfacing only at ingest time) are driven
instead by a thin ``Slurm`` subclass that forwards to the real fake
adapter and only overrides the one seam the scenario needs -- the
same "scripted Slurm double" technique the ticket itself sanctions
for the deadline case.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from hpc_model_utils.core.diagnosis import RunStatus
from hpc_model_utils.core.launch import Resources, Toolchain
from hpc_model_utils.core.lifecycle.finalize import legacy_synthesis_bin
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import ExecutionSource, StateStore
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.errors import SchedulerCommandError
from hpc_model_utils.infra.slurm import JobState, Slurm
from hpc_model_utils.platform.modelops import Reporter
from tests.support.cli_shim import write_cli_shim
from tests.support.fake_plugin import FakePlugin, install_fake_model
from tests.support.fake_slurm import FakeSlurm
from tests.support.fake_slurm.shim import create_job, next_job_id
from tests.support.hooks import Hook, parse_hooks

_PLUGIN = FakePlugin()


@pytest.fixture(autouse=True)
def _chdir_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A submission without an explicit #SBATCH --output writes
    # slurm-<id>.out against the process cwd; pin it to tmp_path so it
    # never leaks into the repo.
    monkeypatch.chdir(tmp_path)


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _toolchain(tmp_path: Path, bin_dir: Path) -> Toolchain:
    return Toolchain(
        cli_bin=write_cli_shim(tmp_path),
        mpich_bin=tmp_path / "mpich" / "bin",
        slurm_bin=bin_dir,
    )


def _make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "study-fake"
    root.mkdir()
    return Workspace.at(root)


def _request(
    tools: Toolchain,
    *,
    skip_model: bool = False,
    resources: Resources | None = None,
) -> SubmitRequest:
    return SubmitRequest(
        resources=resources or Resources(queue="batch", cores=2),
        tools=tools,
        skip_model=skip_model,
        synthesis_bin=None,
    )


def _write_legacy_sintetizador(ws: Workspace, *, exit_code: int = 0) -> Path:
    stub = legacy_synthesis_bin(ws, _PLUGIN)
    stub.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "#!/bin/bash\nmkdir -p sintese\n: > sintese/x.parquet\n"
        f"exit {exit_code}\n"
    )
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _hook_lines(channel: _ListChannel) -> list[Hook]:
    return parse_hooks("\n".join(channel.lines))


# ---------------------------------------------------------------------------
# Scripted Slurm doubles for scenarios the fake cannot express by itself.
# ---------------------------------------------------------------------------


def _fabricate_held_job(fake: FakeSlurm, ws: Workspace, script: Path) -> str:
    """A model job created directly in the fake's own job store, held
    and stamped ``PartitionDown`` from birth, with no runner ever
    spawned for it. ``fake_slurm.hold()``/``set_pending_reason()``
    race the fake's own detached runner (it reads ``held`` in a
    polling loop that starts the instant ``sbatch`` returns, and once
    it decides to run, a later hold is too late) -- amendment rule 1
    needs the job to stay queued forever, so this bypasses that race
    entirely rather than trying to win it."""
    job_id = next_job_id(fake.state_dir)
    create_job(
        fake.state_dir,
        job_id,
        {
            "id": job_id,
            "argv": [],
            "directives": {},
            "dependency": None,
            "script": str(script),
            "state": "PENDING",
            "reason": "PartitionDown",
            "held": True,
            "cwd": str(ws.root),
            "submit_dir": str(ws.root),
            "start_time": None,
            "end_time": None,
            "exit_code": None,
            "pid": None,
            "pgid": None,
            "runner_pid": None,
            "configuring_seconds": 0.0,
            "steps": [],
            "step_tasks": [],
            "restart_count": 0,
        },
    )
    return str(job_id)


class _HoldModelPartitionDownSlurm(Slurm):
    """Fabricates the model job as an already-held, never-clearing
    pending job (AC4) instead of submitting it for real."""

    def __init__(self, bin_dir: Path, ws: Workspace, fake: FakeSlurm) -> None:
        super().__init__(bin_dir)
        self._ws = ws
        self._fake = fake

    def submit(self, script: Path, *, after: str | None = None) -> str:
        if script == self._ws.job_script(Phase.MODEL):
            return _fabricate_held_job(self._fake, self._ws, script)
        return super().submit(script, after=after)


class _PartitionDownNeverGoneSlurm(_HoldModelPartitionDownSlurm):
    """Amendment 1: a double whose ``wait_gone`` always reports the
    cancelled jobs as still queued, without any real waiting."""

    def wait_gone(
        self,
        job_ids: Sequence[str],
        *,
        timeout: float,
        poll: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> bool:
        return False


class _PartitionDownWaitGoneErrorSlurm(_HoldModelPartitionDownSlurm):
    """Amendment 1: a double whose ``wait_gone`` raises, pinning down
    that a command-level failure from it propagates out of ``run``."""

    def wait_gone(
        self,
        job_ids: Sequence[str],
        *,
        timeout: float,
        poll: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> bool:
        raise SchedulerCommandError("squeue unreachable")


class _WipeStateOnFirstPollSlurm(Slurm):
    """AC3: the fake's own ``force()`` can only override a job's
    *bookkeeping* once the real finalize process has already exited,
    by which point a well-behaved finalize job has already written
    its record -- it cannot make the process crash before writing it.
    This double instead deletes ``state.json`` on the very first
    queue poll ``run`` makes after ``submit`` returns, so the real
    finalize CLI process genuinely crashes with a ``StateFormatError``
    (the same path ``test_finalize_job_missing_state_writes_fatal_
    line_and_job_failed`` exercises directly) well before it has a
    chance to read its own, still-booting Python interpreter.
    """

    def __init__(self, bin_dir: Path, ws: Workspace) -> None:
        super().__init__(bin_dir)
        self._ws = ws
        self._wiped = False

    def job_state(self, job_id: str) -> JobState | None:
        if not self._wiped:
            self._wiped = True
            self._ws.state_path.unlink(missing_ok=True)
        return super().job_state(job_id)


class _CorruptRunIdOnFirstPollSlurm(Slurm):
    """Requirement 3: stamps ``state.json`` with a different
    ``run_id`` on the very first queue poll ``run`` makes, i.e.
    immediately after ``run``'s own ``store.load()`` has already
    captured the *original* run_id in memory (the value it will later
    pass to ``load_finalize``) but well before the still-booting
    finalize CLI process -- a much heavier Python startup than the
    fake's own lightweight shims -- has read ``state.json`` for
    itself. The finalize job therefore writes its record under the
    *new* run_id, reproducing a mismatch ``load_finalize`` must
    reject, deterministically and without racing that write.
    """

    def __init__(self, bin_dir: Path, ws: Workspace, store: StateStore) -> None:
        super().__init__(bin_dir)
        self._ws = ws
        self._store = store
        self._corrupted = False

    def job_state(self, job_id: str) -> JobState | None:
        if not self._corrupted:
            self._corrupted = True
            state = self._store.load()
            assert state is not None
            self._store.save(replace(state, run_id="0" * 32))
        return super().job_state(job_id)


# ---------------------------------------------------------------------------
# AC2 / Requirement 2 / "exactly once" relay / projections.
# ---------------------------------------------------------------------------


def test_run_success_relays_logs_once_and_emits_expected_hooks(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws, extra_lines=("M1", "M2"))
    _write_legacy_sintetizador(ws)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    ledger = JobLedger()
    emitted: list[str] = []
    settings = EngineSettings(poll_interval=0.02, settle_window=0.05)

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=emitted.append,
    )

    assert emitted.count("M1") == 1
    assert emitted.count("M2") == 1
    m2_idx = emitted.index("M2")
    start_lines = [
        i for i, line in enumerate(emitted) if line.startswith("HPCMU_START")
    ]
    assert any(i > m2_idx for i in start_lines)

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.SUCCESS
    finalize_id = new_state.jobs[-1].job_id
    assert new_state.reported_job_id == finalize_id
    assert ws.legacy_status_path.read_text(encoding="utf-8") == "SUCCESS"

    assert reporter.terminal_emitted is False
    hooks = _hook_lines(channel)
    names = {(h.method, h.args) for h in hooks}
    assert ("SetMetadata", ("job_id", finalize_id)) in names
    assert ("SetMetadata", ("status", "SUCCESS")) in names
    assert ("SetMetadata", ("synthesis_status", "ok")) in names
    assert all(
        h.method
        not in (
            "SetSuccess",
            "SetModelError",
            "SetDataError",
            "SetRuntimeError",
        )
        for h in hooks
    )


def test_run_success_synthesis_failure_emits_synthesis_status_failed(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws)
    _write_legacy_sintetizador(ws, exit_code=1)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(poll_interval=0.02, settle_window=0.05)

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.SUCCESS
    hooks = _hook_lines(channel)
    names = {(h.method, h.args) for h in hooks}
    assert ("SetMetadata", ("status", "SUCCESS")) in names
    assert ("SetMetadata", ("synthesis_status", "failed")) in names


# ---------------------------------------------------------------------------
# Model TIMEOUT ingested from finalize.json; synthesis_status absent.
# ---------------------------------------------------------------------------


def test_run_model_timeout_ingests_finalize_json_without_synthesis_status(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws, sleep=30.0)
    fake_slurm.force("model", state="TIMEOUT")
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    channel = _ListChannel()
    reporter = Reporter(channel, enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(poll_interval=0.02, settle_window=0.05)

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.TIMEOUT
    hooks = _hook_lines(channel)
    names = {(h.method, h.args) for h in hooks}
    assert ("SetMetadata", ("status", "TIMEOUT")) in names
    assert not any(
        h.method == "SetMetadata" and h.args[0] == "synthesis_status"
        for h in hooks
    )


# ---------------------------------------------------------------------------
# AC3: finalize crash before writing its record.
# ---------------------------------------------------------------------------


def test_run_finalize_crash_before_record_diagnoses_finalize_crashed(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = _WipeStateOnFirstPollSlurm(fake_slurm.bin_dir, ws)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02, settle_window=0.05, outcome_attempts=1
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools, skip_model=True),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.RUNTIME_ERROR
    assert new_state.diagnosis.rule_id == "core.finalize_crashed"
    assert "ended FAILED" in new_state.diagnosis.reason
    assert ws.legacy_status_path.read_text(encoding="utf-8") == "RUNTIME_ERROR"


# ---------------------------------------------------------------------------
# Requirement 3: a run_id mismatch is treated as a finalize crash.
# ---------------------------------------------------------------------------


def test_run_id_mismatch_ingests_as_finalize_crashed(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    store = StateStore(ws)
    slurm = _CorruptRunIdOnFirstPollSlurm(fake_slurm.bin_dir, ws, store)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02, settle_window=0.05, outcome_attempts=1
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools, skip_model=True),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.rule_id == "core.finalize_crashed"
    assert any(
        item.layer == "guard" and "run_id" in item.detail
        for item in new_state.diagnosis.evidence
    )


def test_run_load_finalize_os_error_diagnoses_finalize_crashed(
    fake_slurm: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment 2: an ``OSError`` reading ``finalize.json`` goes to
    the same ``core.finalize_crashed`` path as an absent or
    run_id-mismatched record, with the error text as evidence."""
    ws = _make_workspace(tmp_path)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02, settle_window=0.05, outcome_attempts=1
    )

    def _raise_os_error(ws: Workspace, run_id: str) -> NoReturn:
        raise OSError("disk read error")

    monkeypatch.setattr(
        "hpc_model_utils.core.lifecycle.run.load_finalize", _raise_os_error
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools, skip_model=True),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.RUNTIME_ERROR
    assert new_state.diagnosis.rule_id == "core.finalize_crashed"
    assert any(
        item.layer == "guard" and "disk read error" in item.detail
        for item in new_state.diagnosis.evidence
    )


# ---------------------------------------------------------------------------
# AC4: a never-clearing pending reason cancels both jobs.
# ---------------------------------------------------------------------------


def test_run_partition_down_cancels_both_jobs_and_diagnoses_pending_reason(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = _HoldModelPartitionDownSlurm(fake_slurm.bin_dir, ws, fake_slurm)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02, settle_window=0.05, cancel_timeout=2.0
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.RUNTIME_ERROR
    assert new_state.diagnosis.rule_id == "core.pending_reason"
    assert len(ledger.ids) == 2
    for job_id in ledger.ids:
        assert fake_slurm.job(int(job_id))["state"] == "CANCELLED"
    assert ws.legacy_status_path.read_text(encoding="utf-8") == "RUNTIME_ERROR"


# ---------------------------------------------------------------------------
# Amendment 1: wait_gone False still records the diagnosis and exits
# normally; a SchedulerCommandError from wait_gone propagates.
# ---------------------------------------------------------------------------


def test_run_wait_gone_false_still_records_diagnosis_and_returns(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = _PartitionDownNeverGoneSlurm(fake_slurm.bin_dir, ws, fake_slurm)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02, settle_window=0.05, cancel_timeout=0.3
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert new_state.diagnosis is not None
    assert new_state.diagnosis.rule_id == "core.pending_reason"
    ids_text = ", ".join(ledger.ids)
    expected = f"jobs {ids_text} still queued after 0.3s"
    evidence = new_state.diagnosis.evidence
    assert any(
        item.layer == "slurm"
        and item.source == "scancel"
        and item.detail == expected
        for item in evidence
    )


def test_run_wait_gone_scheduler_error_propagates(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = _PartitionDownWaitGoneErrorSlurm(fake_slurm.bin_dir, ws, fake_slurm)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(poll_interval=0.02, settle_window=0.05)

    with pytest.raises(SchedulerCommandError, match="squeue unreachable"):
        run(
            ws,
            _PLUGIN,
            slurm,
            reporter,
            store,
            _request(tools),
            ledger,
            settings=settings,
            emit=lambda _line: None,
        )


# ---------------------------------------------------------------------------
# Deadline abort with a small margin (ticket-042's "cancel mid-run" case).
#
# The fake SLURM does not enforce `--time` itself; `follow()` derives the
# deadline purely from the `%l` time-limit field it reads back through
# squeue plus `settings.follow_margin`. A zero-hour time limit (a valid,
# if unrealistic, `Resources.time_limit_hours`) combined with a tiny
# `follow_margin` reproduces a real "job overstayed its time limit"
# deadline within milliseconds, against a model that sleeps far longer
# -- no fake-SLURM feature is missing here, so no scripted double is
# needed, unlike the finalize-crash and run_id-mismatch scenarios above.
#
# Both the model and the (dependency-waiting) finalize job must end up
# CANCELLED. This used to be flaky under heavy host load because of a
# fidelity bug in the fake's own `_scancel` (it signalled the running
# process before recording CANCELLED, so a process that died in that
# window raced `_finalize` into writing COMPLETED/FAILED instead); that
# is now fixed in `tests/support/fake_slurm/shim.py` (and pinned down by
# `test_scancel_running_job_exiting_immediately_on_sigterm_ends_cancelled`
# in `test_fake_slurm_queue.py`), so this assertion is race-free.
# ---------------------------------------------------------------------------


def test_run_deadline_abort_cancels_both_jobs(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    install_fake_model(ws, sleep=5.0)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    settings = EngineSettings(
        poll_interval=0.02,
        settle_window=0.05,
        follow_margin=0.1,
        cancel_timeout=5.0,
    )

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(
            tools,
            resources=Resources(queue="batch", cores=2, time_limit_hours=0),
        ),
        ledger,
        settings=settings,
        emit=lambda _line: None,
    )

    assert len(ledger.ids) == 2
    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.RUNTIME_ERROR
    assert new_state.diagnosis.rule_id == "core.deadline"
    assert new_state.reported_job_id == ledger.ids[-1]
    for job_id in ledger.ids:
        assert fake_slurm.job(int(job_id))["state"] == "CANCELLED"


# ---------------------------------------------------------------------------
# OFFLINE finalize-only run follows a single log.
# ---------------------------------------------------------------------------


def test_run_offline_finalize_only_follows_single_log(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws = _make_workspace(tmp_path)
    tools = _toolchain(tmp_path, fake_slurm.bin_dir)
    slurm = Slurm(fake_slurm.bin_dir)
    store = StateStore(ws)
    state = store.load_or_create(_PLUGIN.name)
    store.save(replace(state, execution_source=ExecutionSource.OFFLINE))
    (ws.root / "fake.out").write_text("SUCCESS\n", encoding="utf-8")
    (ws.root / "model.fake").write_bytes(b"")
    _write_legacy_sintetizador(ws)
    reporter = Reporter(_ListChannel(), enabled=True)
    ledger = JobLedger()
    emitted: list[str] = []
    settings = EngineSettings(poll_interval=0.02, settle_window=0.05)

    new_state = run(
        ws,
        _PLUGIN,
        slurm,
        reporter,
        store,
        _request(tools),
        ledger,
        settings=settings,
        emit=emitted.append,
    )

    assert len(new_state.jobs) == 1
    assert new_state.jobs[0].phase is Phase.FINALIZE
    assert new_state.diagnosis is not None
    assert new_state.diagnosis.status is RunStatus.SUCCESS
    assert any(line.startswith("HPCMU_START") for line in emitted)
