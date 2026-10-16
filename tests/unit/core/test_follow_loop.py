"""ADR-046/ADR-025/R38/R46/R95/R120/R136: the job follow loop and its
``EngineSettings`` test seams."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hpc_model_utils.core.follow import (
    NEVER_CLEARING,
    PENDING_LIKE,
    FollowAborted,
    FollowResult,
    LogTail,
    follow,
)
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.infra.errors import SchedulerCommandError
from hpc_model_utils.infra.slurm import JobState

_ID = "1"


def _js(
    *,
    state: str,
    reason: str = "",
    time_limit: timedelta | None = None,
    start_time: datetime | None = None,
) -> JobState:
    return JobState(
        job_id=_ID,
        state=state,
        reason=reason,
        time_limit=time_limit,
        start_time=start_time,
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeTail(LogTail):
    """A ``LogTail`` subtype that never touches the filesystem, so
    mypy accepts it where ``follow`` requires a ``LogTail``."""

    def __init__(
        self,
        polls: dict[int, list[str]] | None = None,
        close_lines: list[str] | None = None,
    ) -> None:
        self._polls = dict(polls or {})
        self._count = 0
        self._close_lines = close_lines or []
        self._ever_opened = True

    def poll(self) -> list[str]:
        lines = self._polls.get(self._count, [])
        self._count += 1
        return lines

    def close(self) -> list[str]:
        return self._close_lines


_ScheduleItem = JobState | None | SchedulerCommandError
_Schedule = list[tuple[float, _ScheduleItem]]


class TimedSlurm:
    """Returns the scripted item whose window the fake clock's
    current time falls into; the last entry holds forever."""

    def __init__(self, now: Callable[[], float], schedule: _Schedule) -> None:
        self._now = now
        self._schedule = schedule

    def job_state(self, job_id: str) -> JobState | None:
        t = self._now()
        for until, item in self._schedule:
            if t < until:
                return _resolve(item)
        return _resolve(self._schedule[-1][1])


def _resolve(item: _ScheduleItem) -> JobState | None:
    if isinstance(item, SchedulerCommandError):
        raise item
    return item


class ScriptedSlurm:
    """Returns one scripted item per call; the last entry repeats."""

    def __init__(self, script: list[_ScheduleItem]) -> None:
        self._script = list(script)
        self._i = 0

    def job_state(self, job_id: str) -> JobState | None:
        idx = min(self._i, len(self._script) - 1)
        self._i += 1
        return _resolve(self._script[idx])


class NoOtherMethodsSlurm:
    """Raises if ``follow`` calls anything on ``slurm`` besides
    ``job_state`` -- pins down "follow never scancels"."""

    def __init__(self, now: Callable[[], float], schedule: _Schedule) -> None:
        self._timed = TimedSlurm(now, schedule)

    def job_state(self, job_id: str) -> JobState | None:
        return self._timed.job_state(job_id)

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"follow() must never call Slurm.{name}")


def test_never_clearing_and_pending_like_constants_match_spec() -> None:
    assert NEVER_CLEARING == (
        "PartitionDown",
        "PartitionInactive",
        "BadConstraints",
    )
    assert PENDING_LIKE == frozenset(
        {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}
    )


def test_follow_running_silently_emits_exactly_two_heartbeats_before_gone() -> (
    None
):
    clock = FakeClock()
    job = _js(
        state="RUNNING",
        time_limit=timedelta(hours=2),
        start_time=datetime(2026, 1, 1),
    )
    slurm = TimedSlurm(clock.now, [(1500.0, job), (float("inf"), None)])
    emitted: list[str] = []
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        emitted.append,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    heartbeats = [
        line for line in emitted if line.startswith("[hpcmu] heartbeat")
    ]
    assert len(heartbeats) == 2
    assert all(
        h.startswith(f"[hpcmu] heartbeat: job {_ID} RUNNING")
        for h in heartbeats
    )
    assert result == FollowResult(_ID, None, 1)


def test_follow_heartbeat_resets_after_tail_output_fires_again_later() -> None:
    clock = FakeClock()
    job = _js(
        state="RUNNING",
        time_limit=timedelta(hours=2),
        start_time=datetime(2026, 1, 1),
    )
    slurm = TimedSlurm(clock.now, [(1400.0, job), (float("inf"), None)])
    tail = FakeTail(polls={150: ["out"]})
    emitted: list[str] = []
    follow(
        _ID,
        tail,
        slurm,
        emitted.append,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    heartbeats = [
        line for line in emitted if line.startswith("[hpcmu] heartbeat")
    ]
    assert len(heartbeats) == 1
    assert "out" in emitted
    assert emitted.index("out") < emitted.index(heartbeats[0])


@pytest.mark.parametrize(
    "reason",
    [
        "PartitionDown",
        "PartitionInactive",
        "BadConstraints",
        "ReqNodeNotAvail",
        "ReqNodeNotAvail, UnavailableNodes:",
    ],
)
def test_follow_never_clearing_pending_reason_raises_follow_aborted(
    reason: str,
) -> None:
    clock = FakeClock()
    job = _js(state="PENDING", reason=reason)
    slurm = TimedSlurm(clock.now, [(float("inf"), job)])
    with pytest.raises(FollowAborted, match="pending_reason") as exc_info:
        follow(
            _ID,
            FakeTail(),
            slurm,
            lambda _: None,
            settings=EngineSettings(),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert exc_info.value.kind == "pending_reason"
    assert exc_info.value.job_id == _ID


def test_follow_req_node_not_avail_with_nodes_is_clearing_and_completes() -> (
    None
):
    clock = FakeClock()
    pending = _js(
        state="PENDING",
        reason="ReqNodeNotAvail, UnavailableNodes:node[1-2]",
    )
    schedule: _Schedule = [(100.0, pending), (float("inf"), None)]
    slurm = TimedSlurm(clock.now, schedule)
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.last_state is None


def test_follow_pending_cap_exceeded_raises_pending_cap() -> None:
    clock = FakeClock()
    pending = _js(state="PENDING", reason="Resources")
    slurm = TimedSlurm(clock.now, [(float("inf"), pending)])
    with pytest.raises(FollowAborted, match="pending_cap") as exc_info:
        follow(
            _ID,
            FakeTail(),
            slurm,
            lambda _: None,
            settings=EngineSettings(),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert exc_info.value.kind == "pending_cap"


def test_follow_deadline_exceeded_after_queue_wait_raises_deadline() -> None:
    clock = FakeClock()
    pending = _js(state="PENDING", reason="Resources")
    running = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1),
    )
    schedule: _Schedule = [(36000.0, pending), (41462.0, running)]
    slurm = TimedSlurm(clock.now, schedule)
    with pytest.raises(FollowAborted, match="deadline") as exc_info:
        follow(
            _ID,
            FakeTail(),
            slurm,
            lambda _: None,
            settings=EngineSettings(),
            now=clock.now,
            sleep=clock.sleep,
        )
    assert exc_info.value.kind == "deadline"


def test_follow_deadline_not_exceeded_job_leaves_before_margin() -> None:
    clock = FakeClock()
    pending = _js(state="PENDING", reason="Resources")
    running = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1),
    )
    schedule: _Schedule = [
        (36000.0, pending),
        (41340.0, running),
        (float("inf"), None),
    ]
    slurm = TimedSlurm(clock.now, schedule)
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.last_state is None
    assert result.attempts == 1


def test_follow_requeue_through_pending_resets_attempt_and_emits_marker() -> (
    None
):
    clock = FakeClock()
    running1 = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1, 0, 0, 0),
    )
    requeued = _js(state="REQUEUED", reason="")
    pending = _js(state="PENDING", reason="Resources")
    running2 = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1, 1, 0, 0),
    )
    schedule: _Schedule = [
        (3000.0, running1),
        (3002.0, requeued),
        (75002.0, pending),
        (78002.0, running2),
        (float("inf"), None),
    ]
    slurm = TimedSlurm(clock.now, schedule)
    emitted: list[str] = []
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        emitted.append,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.attempts == 2
    assert any(
        "requeued (REQUEUED): deadline cleared" in line for line in emitted
    )


def test_follow_requeue_detected_by_start_time_change_without_pending() -> None:
    clock = FakeClock()
    running1 = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1, 0, 0, 0),
    )
    running2 = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1, 1, 0, 0),
    )
    schedule: _Schedule = [
        (3000.0, running1),
        (6000.0, running2),
        (float("inf"), None),
    ]
    slurm = TimedSlurm(clock.now, schedule)
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.attempts == 2


def test_follow_configuring_boot_is_same_attempt_not_requeue() -> None:
    clock = FakeClock()
    configuring = _js(
        state="CONFIGURING", time_limit=timedelta(hours=1), start_time=None
    )
    running = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1, 1, 0, 0),
    )
    schedule: _Schedule = [
        (100.0, configuring),
        (5450.0, running),
        (float("inf"), None),
    ]
    slurm = TimedSlurm(clock.now, schedule)
    emitted: list[str] = []
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        emitted.append,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.attempts == 1
    assert not any("restarted" in line for line in emitted)


def test_follow_unlimited_time_limit_never_raises_deadline() -> None:
    clock = FakeClock()
    running = _js(
        state="RUNNING", time_limit=None, start_time=datetime(2026, 1, 1)
    )
    schedule: _Schedule = [(200000.0, running), (float("inf"), None)]
    slurm = TimedSlurm(clock.now, schedule)
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.last_state is None


def test_follow_terminal_state_counts_as_gone_and_settles_before_close() -> (
    None
):
    clock = FakeClock()
    completed = _js(state="COMPLETED")
    slurm = TimedSlurm(clock.now, [(float("inf"), completed)])
    tail = FakeTail(polls={1: [], 2: ["late"]}, close_lines=["[hpcmu] drained"])
    emitted: list[str] = []
    result = follow(
        _ID,
        tail,
        slurm,
        emitted.append,
        settings=EngineSettings(settle_window=5.0, poll_interval=2.0),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert "late" in emitted
    assert emitted.index("late") < emitted.index("[hpcmu] drained")
    assert clock.t >= 5.0
    assert result.last_state == "COMPLETED"


def test_follow_settle_window_with_real_log_tail_drains_late_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-1.out"
    path.write_text("L1\n")
    tail = LogTail(path)
    clock = FakeClock()
    completed = _js(state="COMPLETED")
    slurm = TimedSlurm(clock.now, [(float("inf"), completed)])
    calls = {"n": 0}

    def sleep(seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            with path.open("a", encoding="utf-8") as fh:
                fh.write("LATE\n")
        clock.sleep(seconds)

    emitted: list[str] = []
    result = follow(
        _ID,
        tail,
        slurm,
        emitted.append,
        settings=EngineSettings(settle_window=5.0, poll_interval=1.0),
        now=clock.now,
        sleep=sleep,
    )
    assert "LATE" in emitted
    assert result.last_state == "COMPLETED"


def test_follow_log_opened_before_job_ends_settles_for_settle_window_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-1.out"
    path.write_text("L1\n")
    clock = FakeClock()
    slurm = TimedSlurm(clock.now, [(float("inf"), _js(state="COMPLETED"))])
    settings = EngineSettings(settle_window=5.0, poll_interval=1.0)
    emitted: list[str] = []
    follow(
        _ID,
        LogTail(path),
        slurm,
        emitted.append,
        settings=settings,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert emitted == ["L1"]
    assert (
        settings.settle_window
        <= clock.t
        <= settings.settle_window + settings.poll_interval
    )


def test_follow_log_visible_after_settle_window_within_grace_is_relayed_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-1.out"
    clock = FakeClock()
    slurm = TimedSlurm(clock.now, [(float("inf"), _js(state="COMPLETED"))])
    settings = EngineSettings(
        settle_window=10.0, poll_interval=2.0, missing_log_grace=90.0
    )
    visible_at = settings.settle_window + 30.0
    assert visible_at < settings.missing_log_grace

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.t >= visible_at and not path.exists():
            path.write_text("L1\nL2\nL3\n")

    emitted: list[str] = []
    follow(
        _ID,
        LogTail(path),
        slurm,
        emitted.append,
        settings=settings,
        now=clock.now,
        sleep=sleep,
    )
    assert emitted == ["L1", "L2", "L3"]


def test_follow_log_never_appearing_emits_one_marker_after_the_grace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-1.out"
    clock = FakeClock()
    slurm = TimedSlurm(clock.now, [(float("inf"), _js(state="COMPLETED"))])
    settings = EngineSettings(
        settle_window=10.0, poll_interval=2.0, missing_log_grace=90.0
    )
    emitted: list[str] = []
    result = follow(
        _ID,
        LogTail(path),
        slurm,
        emitted.append,
        settings=settings,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert emitted == ["[hpcmu] log never appeared: model-1.out"]
    assert (
        settings.missing_log_grace
        <= clock.t
        <= settings.missing_log_grace + settings.poll_interval
    )
    assert result.last_state == "COMPLETED"


def test_follow_pending_then_cancelled_without_log_skips_the_grace(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    pending = _js(state="PENDING", reason="Dependency")
    cancelled_at = 6.0
    slurm = TimedSlurm(
        clock.now,
        [(cancelled_at, pending), (float("inf"), _js(state="CANCELLED"))],
    )
    settings = EngineSettings(
        settle_window=10.0, poll_interval=2.0, missing_log_grace=90.0
    )
    emitted: list[str] = []
    result = follow(
        _ID,
        LogTail(tmp_path / "finalize-1.out"),
        slurm,
        emitted.append,
        settings=settings,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert emitted == ["[hpcmu] log never appeared: finalize-1.out"]
    assert (
        settings.settle_window
        <= clock.t - cancelled_at
        <= settings.settle_window + settings.poll_interval
    )
    assert result.last_state == "CANCELLED"


@pytest.mark.parametrize(
    ("early", "terminal"),
    [
        pytest.param(
            _js(
                state="RUNNING",
                time_limit=timedelta(hours=2),
                start_time=datetime(2026, 1, 1),
            ),
            _js(state="CANCELLED"),
            id="running-then-cancelled",
        ),
        pytest.param(
            _js(state="PENDING", reason="Dependency"),
            _js(state="FAILED"),
            id="pending-then-failed",
        ),
        pytest.param(
            _js(state="PENDING", reason="Dependency"),
            None,
            id="pending-then-gone",
        ),
    ],
)
def test_follow_log_never_appearing_keeps_the_grace_unless_cancelled_unstarted(
    tmp_path: Path, early: JobState, terminal: JobState | None
) -> None:
    clock = FakeClock()
    ended_at = 4.0
    slurm = TimedSlurm(clock.now, [(ended_at, early), (float("inf"), terminal)])
    settings = EngineSettings(
        settle_window=10.0, poll_interval=2.0, missing_log_grace=90.0
    )
    emitted: list[str] = []
    follow(
        _ID,
        LogTail(tmp_path / "model-1.out"),
        slurm,
        emitted.append,
        settings=settings,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert emitted == ["[hpcmu] log never appeared: model-1.out"]
    assert (
        settings.missing_log_grace
        <= clock.t - ended_at
        <= settings.missing_log_grace + settings.poll_interval
    )


def test_follow_none_job_state_counts_as_gone_with_last_state_none() -> None:
    clock = FakeClock()
    slurm = TimedSlurm(clock.now, [(float("inf"), None)])
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result == FollowResult(_ID, None, 0)


def test_follow_squeue_failures_absorbed_up_to_budget_and_reset() -> None:
    clock = FakeClock()
    running = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1),
    )
    script: list[_ScheduleItem] = [
        SchedulerCommandError("boom") for _ in range(5)
    ] + [running, None]
    slurm = ScriptedSlurm(script)
    emitted: list[str] = []
    follow(
        _ID,
        FakeTail(),
        slurm,
        emitted.append,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    failures = [
        line for line in emitted if line.startswith("[hpcmu] squeue failed")
    ]
    assert len(failures) == 5
    for n, line in enumerate(failures, start=1):
        assert line.startswith(f"[hpcmu] squeue failed ({n}/5)")


def test_follow_squeue_failures_exceeding_budget_propagate() -> None:
    clock = FakeClock()
    slurm = ScriptedSlurm([SchedulerCommandError("boom")])
    with pytest.raises(SchedulerCommandError, match="boom"):
        follow(
            _ID,
            FakeTail(),
            slurm,
            lambda _: None,
            settings=EngineSettings(),
            now=clock.now,
            sleep=clock.sleep,
        )


def test_follow_never_calls_unexpected_scheduler_method() -> None:
    clock = FakeClock()
    running = _js(
        state="RUNNING",
        time_limit=timedelta(hours=1),
        start_time=datetime(2026, 1, 1),
    )
    schedule: _Schedule = [(100.0, running), (float("inf"), None)]
    slurm = NoOtherMethodsSlurm(clock.now, schedule)
    result = follow(
        _ID,
        FakeTail(),
        slurm,
        lambda _: None,
        settings=EngineSettings(),
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result.last_state is None
