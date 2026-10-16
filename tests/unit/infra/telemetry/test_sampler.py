"""ADR-069/ADR-079: the staging writer, the tick loop and the entry point."""

from __future__ import annotations

import errno
import json
import os
import re
import resource
import signal
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hpc_model_utils.infra.telemetry import sampler
from hpc_model_utils.infra.telemetry.collect import (
    COUNTER_SOURCES,
    Collector,
    Errors,
    Values,
)
from hpc_model_utils.infra.telemetry.procfs import StaticFacts
from hpc_model_utils.infra.telemetry.sampler import (
    ERROR_LINE_CAP,
    FSYNC_INTERVAL_S,
    MAX_CONSECUTIVE_FAILURES,
    MESSAGE_CAP,
    FileOps,
    Identity,
    JsonValue,
    LoopClock,
    StagingWriter,
    build_header,
    main,
    next_tick_ms,
    run_loop,
)
from hpc_model_utils.infra.telemetry.schema import (
    ERROR_FIELDS,
    ERROR_KIND,
    HEADER_FIELDS,
    HEADER_KIND,
    SAMPLE_COLUMNS,
    SOURCES,
    WALK_KIND,
    WORKSPACE_COLUMNS,
)
from tests.support.telemetry_tree import load_bundle, materialize

_NAME = "model-4711-r0-0.ndjson"
_HEADER: dict[str, JsonValue] = {
    "kind": HEADER_KIND,
    "schema_version": 1,
    "job_id": "4711",
}
_WALK: dict[str, JsonValue] = {"kind": WALK_KIND, "ts": 60000, "files": 3}


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _writer(
    tmp_path: Path, clock: _Clock | None = None, ops: FileOps | None = None
) -> StagingWriter:
    return StagingWriter.create(
        tmp_path, "model", "4711", 0, 0, clock or _Clock(), ops
    )


def _recorded(log: list[str]) -> FileOps:
    def open_(path: Path, flags: int, mode: int) -> int:
        log.append("open")
        return os.open(path, flags, mode)

    def write(fd: int, data: bytes) -> int:
        log.append("write")
        return os.write(fd, data)

    def fsync(fd: int) -> None:
        log.append("fsync")
        os.fsync(fd)

    def close(fd: int) -> None:
        log.append("close")
        os.close(fd)

    return FileOps(open_, write, fsync, close)


def _lines(writer: StagingWriter) -> list[bytes]:
    return writer.path.read_bytes().splitlines()


def _kinds(writer: StagingWriter) -> list[str | None]:
    return [json.loads(line).get("kind") for line in _lines(writer)]


def _eio_open(path: Path, flags: int, mode: int) -> int:
    raise OSError(errno.EIO, "injected")


def _eio_fd(fd: int) -> None:
    raise OSError(errno.EIO, "injected")


def _error_class(index: int) -> type[Exception]:
    return type(f"Error{index}", (Exception,), {})


def test_file_ops_system_is_the_os_functions() -> None:
    assert FileOps.system() == FileOps(os.open, os.write, os.fsync, os.close)


def test_create_names_the_file_and_opens_nothing(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    assert writer.path == tmp_path / _NAME
    assert not writer.path.exists()
    assert writer.consecutive_failures == 0
    assert writer.last_failure is None


@pytest.mark.parametrize(
    ("phase", "job_id", "restart", "node_index"),
    [("run", "4711", 0, 0), ("model", "x1", 0, 0), ("model", "4711", -1, 0)],
)
def test_create_invalid_identity_raises_value_error(
    tmp_path: Path, phase: str, job_id: str, restart: int, node_index: int
) -> None:
    with pytest.raises(ValueError):
        StagingWriter.create(
            tmp_path, phase, job_id, restart, node_index, _Clock()
        )


def test_header_then_samples_round_trip_with_header_fsync_first(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))

    assert writer.write_header(_HEADER) is True
    assert writer.write_sample({"ts": 1000, "cpu_util": 0.5}) is True
    assert writer.write_sample({"ts": 2000, "cpu_util": None}) is True

    lines = _lines(writer)
    assert writer.path.name == _NAME
    assert len(lines) == 3
    assert json.loads(lines[0]) == _HEADER
    assert lines[1] == b'{"ts":1000,"cpu_util":0.5}'
    assert lines[2] == b'{"ts":2000,"cpu_util":null}'
    assert log == ["open", "write", "fsync", "write", "write"]
    assert writer.consecutive_failures == 0


def test_open_uses_exclusive_append_flags_and_mode(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    def open_(path: Path, flags: int, mode: int) -> int:
        seen.append((flags, mode))
        return os.open(path, flags, mode)

    writer = _writer(tmp_path, ops=FileOps.system()._replace(open=open_))
    writer.write_sample({"ts": 1})
    writer.write_sample({"ts": 2})

    assert seen == [(os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)]
    assert writer.path.stat().st_mode & 0o777 == 0o644 & ~_umask()


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_second_writer_on_an_existing_file_fails_the_header(
    tmp_path: Path,
) -> None:
    first = _writer(tmp_path)
    first.write_header(_HEADER)
    before = first.path.read_bytes()
    second = _writer(tmp_path)

    assert second.write_header(_HEADER) is False

    assert second.consecutive_failures == 1
    assert isinstance(second.last_failure, FileExistsError)
    assert first.path.read_bytes() == before


def test_open_is_retried_by_the_next_record(tmp_path: Path) -> None:
    blocker = tmp_path / _NAME
    blocker.write_bytes(b"")
    writer = _writer(tmp_path)

    assert writer.write_sample({"ts": 1}) is False
    blocker.unlink()
    assert writer.write_sample({"ts": 2}) is True

    assert writer.consecutive_failures == 0
    assert _lines(writer) == [b'{"ts":2}']


def test_failed_open_does_not_prefix_the_next_record_with_a_newline(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, ops=_eio_first_opens([], 1))
    writer.write_sample({"ts": 1})
    writer.write_sample({"ts": 2})

    assert writer.path.read_bytes() == b'{"ts":2}\n'


def test_torn_write_is_resynced_onto_its_own_line(tmp_path: Path) -> None:
    torn: list[bytes] = []

    def write(fd: int, data: bytes) -> int:
        if not torn:
            torn.append(data[: len(data) // 2])
            os.write(fd, torn[0])
            raise OSError(errno.ENOSPC, "injected")
        return os.write(fd, data)

    writer = _writer(tmp_path, ops=FileOps.system()._replace(write=write))

    first = writer.write_sample({"ts": 1000, "cpu_util": 0.5})
    second = writer.write_sample({"ts": 2000, "cpu_util": 0.25})

    lines = _lines(writer)
    assert (first, second) == (False, True)
    assert lines == [torn[0], b'{"ts":2000,"cpu_util":0.25}']
    assert json.loads(lines[-1]) == {"ts": 2000, "cpu_util": 0.25}
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])
    assert writer.consecutive_failures == 0
    assert isinstance(writer.last_failure, OSError)


def test_short_writes_are_looped_until_the_record_is_complete(
    tmp_path: Path,
) -> None:
    sizes: list[int] = []

    def write(fd: int, data: bytes) -> int:
        sizes.append(os.write(fd, data[:3]))
        return sizes[-1]

    writer = _writer(tmp_path, ops=FileOps.system()._replace(write=write))

    assert writer.write_sample({"ts": 1000}) is True

    assert len(sizes) == 4
    assert _lines(writer) == [b'{"ts":1000}']
    assert b"\n\n" not in writer.path.read_bytes()


def test_write_without_progress_is_a_failure_not_a_hang(
    tmp_path: Path,
) -> None:
    writer = _writer(
        tmp_path, ops=FileOps.system()._replace(write=lambda fd, data: 0)
    )

    assert writer.write_sample({"ts": 1}) is False

    assert writer.consecutive_failures == 1
    assert writer.last_failure is not None
    assert writer.last_failure.errno == errno.EIO


def test_each_record_is_one_write(tmp_path: Path) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))

    writer.write_header(_HEADER)
    writer.write_sample({"ts": 1})
    writer.write_walk(_WALK)
    writer.write_error(1, "stat", OSError("x"))

    assert log.count("write") == 4


def test_pending_header_is_retried_before_the_next_record(
    tmp_path: Path,
) -> None:
    attempts: list[int] = []

    def open_(path: Path, flags: int, mode: int) -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError(errno.EIO, "injected")
        return os.open(path, flags, mode)

    log: list[str] = []
    ops = _recorded(log)._replace(open=open_)
    writer = _writer(tmp_path, ops=ops)

    assert writer.write_header(_HEADER) is False
    assert writer.consecutive_failures == 1
    assert writer.write_sample({"ts": 1}) is True

    assert _kinds(writer) == [HEADER_KIND, None]
    assert log == ["write", "fsync", "write"]
    assert writer.consecutive_failures == 0


def test_pending_header_failure_blocks_the_later_record(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, ops=FileOps.system()._replace(open=_eio_open))
    writer.write_header(_HEADER)

    assert writer.write_sample({"ts": 1}) is False
    assert writer.write_walk(_WALK) is False
    assert writer.write_error(1, "stat", OSError("x")) is False

    assert writer.consecutive_failures == 4
    assert not writer.path.exists()


def test_header_written_is_not_written_again(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_header(_HEADER)
    writer.write_sample({"ts": 1})
    writer.write_sample({"ts": 2})

    assert _kinds(writer) == [HEADER_KIND, None, None]


def test_walk_line_is_written_with_its_kind(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    assert writer.write_walk(_WALK) is True

    assert _lines(writer) == [json.dumps(_WALK, separators=(",", ":")).encode()]


def test_error_line_has_the_error_fields_in_order(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    assert writer.write_error(1234, "stat", PermissionError("nope")) is True

    record = json.loads(_lines(writer)[0])
    assert tuple(record) == ERROR_FIELDS
    assert record == {
        "kind": ERROR_KIND,
        "ts": 1234,
        "source": "stat",
        "error": "PermissionError",
        "message": "nope",
    }


def test_error_is_written_once_per_source_and_class(tmp_path: Path) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))

    results = [
        writer.write_error(1, "stat", OSError("a")),
        writer.write_error(2, "stat", OSError("b")),
        writer.write_error(3, "meminfo", OSError("c")),
        writer.write_error(4, "stat", ValueError("d")),
    ]

    errors = [json.loads(line) for line in _lines(writer)]
    assert results == [True, True, True, True]
    assert [(e["source"], e["error"]) for e in errors] == [
        ("stat", "OSError"),
        ("meminfo", "OSError"),
        ("stat", "ValueError"),
    ]
    assert log.count("fsync") == 3


def test_suppressed_duplicate_leaves_the_failure_counter_alone(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.write_error(1, "stat", OSError("a"))
    writer.consecutive_failures = 3

    assert writer.write_error(2, "stat", OSError("b")) is True

    assert writer.consecutive_failures == 3


def test_error_key_counts_only_after_a_successful_write(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, ops=_eio_first_opens([], 1))

    assert writer.write_error(1, "stat", OSError("a")) is False
    assert writer.write_error(2, "stat", OSError("a")) is True
    assert writer.write_error(3, "stat", OSError("a")) is True

    assert [json.loads(line)["ts"] for line in _lines(writer)] == [2]


def test_error_lines_are_capped_with_one_marker_line(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    sources = sorted(SOURCES)
    keys = [
        (sources[i % len(sources)], _error_class(i // len(sources)))
        for i in range(ERROR_LINE_CAP + 2)
    ]

    results = [
        writer.write_error(1000 + i, source, cls("m"))
        for i, (source, cls) in enumerate(keys)
    ]
    results.append(writer.write_error(9999, keys[0][0], keys[0][1]("m")))

    errors = [json.loads(line) for line in _lines(writer)]
    capped = [e for e in errors if e["error"] == "error_lines_capped"]
    assert all(results)
    assert len(errors) == ERROR_LINE_CAP + 1
    assert len([e for e in errors if e["error"] != "error_lines_capped"]) == 64
    assert capped == [
        {
            "kind": ERROR_KIND,
            "ts": 1000 + ERROR_LINE_CAP,
            "source": "sampler",
            "error": "error_lines_capped",
            "message": "error lines capped at 64",
        }
    ]
    assert errors[-1] == capped[0]


def test_capped_marker_is_retried_after_a_failed_write(tmp_path: Path) -> None:
    fail: list[bool] = []

    def write(fd: int, data: bytes) -> int:
        if fail and b"error_lines_capped" in data:
            fail.clear()
            raise OSError(errno.EIO, "injected")
        return os.write(fd, data)

    writer = _writer(tmp_path, ops=FileOps.system()._replace(write=write))
    for i in range(ERROR_LINE_CAP):
        writer.write_error(i, "stat", _error_class(i)("m"))
    fail.append(True)

    assert writer.write_error(100, "stat", _error_class(900)("m")) is False
    assert writer.write_error(101, "stat", _error_class(901)("m")) is True
    assert writer.write_error(102, "stat", _error_class(902)("m")) is True

    lines = _lines(writer)
    errors = [json.loads(line) for line in lines if line]
    assert lines.count(b"") == 1
    assert [e["error"] for e in errors].count("error_lines_capped") == 1
    assert errors[-1]["ts"] == 101


def test_error_message_is_truncated_and_ascii(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    writer.write_error(1, "stat", OSError("x" * (MESSAGE_CAP + 100)))
    writer.write_error(2, "meminfo", OSError("café ☃"))

    raw = writer.path.read_bytes()
    first, second = (json.loads(line) for line in raw.splitlines())
    assert first["message"] == "x" * MESSAGE_CAP
    assert second["message"] == "café ☃"
    assert raw.isascii()


def test_fsync_follows_the_interval_on_the_injected_clock(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    clock = _Clock(100.0)
    writer = _writer(tmp_path, clock, _recorded(log))
    writer.write_header(_HEADER)

    clock.now = 100.0 + FSYNC_INTERVAL_S - 0.1
    writer.write_sample({"ts": 1})
    writer.write_walk(_WALK)
    assert log.count("fsync") == 1

    clock.now = 100.0 + FSYNC_INTERVAL_S
    writer.write_sample({"ts": 2})
    assert log.count("fsync") == 2

    clock.now = 100.0 + FSYNC_INTERVAL_S + 1.0
    writer.write_sample({"ts": 3})
    assert log.count("fsync") == 2

    clock.now = 100.0 + 2 * FSYNC_INTERVAL_S
    writer.write_walk(_WALK)
    assert log.count("fsync") == 3


def test_header_and_error_lines_fsync_even_when_the_interval_is_not_due(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    clock = _Clock(100.0)
    writer = _writer(tmp_path, clock, _recorded(log))
    writer.write_sample({"ts": 1})
    assert log.count("fsync") == 1

    clock.now = 101.0
    writer.write_header(_HEADER)
    assert log.count("fsync") == 2
    writer.write_error(1, "stat", OSError("a"))
    assert log.count("fsync") == 3


def test_sync_fsyncs_at_once_and_a_closed_writer_has_nothing_to_sync(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))

    assert writer.sync() is True
    assert log == []

    writer.write_header(_HEADER)
    writer.write_sample({"ts": 1})
    assert writer.sync() is True

    assert log.count("fsync") == 2


def test_close_fsyncs_then_closes_once(tmp_path: Path) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))
    writer.write_sample({"ts": 1})
    del log[:]

    writer.close()
    writer.close()

    assert log == ["fsync", "close"]


def test_close_without_an_open_file_does_nothing(tmp_path: Path) -> None:
    log: list[str] = []
    writer = _writer(tmp_path, ops=_recorded(log))

    writer.close()

    assert log == []


def test_close_failures_are_counted_not_raised(tmp_path: Path) -> None:
    ops = FileOps.system()._replace(fsync=_eio_fd, close=_eio_fd)
    writer = _writer(tmp_path, ops=ops)
    writer.write_sample({"ts": 1})
    before = writer.consecutive_failures

    writer.close()

    assert writer.consecutive_failures == before + 2
    assert isinstance(writer.last_failure, OSError)


def test_failed_fsync_is_counted_and_the_record_stays_written(
    tmp_path: Path,
) -> None:
    ops = FileOps.system()._replace(fsync=_eio_fd)
    writer = _writer(tmp_path, ops=ops)

    assert writer.write_header(_HEADER) is True
    assert writer.write_sample({"ts": 1}) is True

    assert _kinds(writer) == [HEADER_KIND, None]
    assert writer.consecutive_failures == 2
    assert isinstance(writer.last_failure, OSError)


def test_failed_fsync_is_retried_by_the_next_sample_line(
    tmp_path: Path,
) -> None:
    broken = [True]
    log: list[str] = []

    def fsync(fd: int) -> None:
        log.append("fsync")
        if broken:
            raise OSError(errno.EIO, "injected")
        os.fsync(fd)

    writer = _writer(tmp_path, ops=FileOps.system()._replace(fsync=fsync))
    writer.write_header(_HEADER)
    assert writer.consecutive_failures == 1

    broken.clear()
    assert writer.write_sample({"ts": 1}) is True

    assert log == ["fsync", "fsync"]
    assert writer.consecutive_failures == 0


def test_failure_counter_counts_five_failing_opens_and_a_success_resets_it(
    tmp_path: Path,
) -> None:
    fail = [True]

    def open_(path: Path, flags: int, mode: int) -> int:
        if fail:
            raise OSError(errno.EIO, "injected")
        return os.open(path, flags, mode)

    writer = _writer(tmp_path, ops=FileOps.system()._replace(open=open_))

    for _ in range(5):
        assert writer.write_sample({"ts": 1}) is False
    assert writer.consecutive_failures == 5
    last = writer.last_failure
    assert isinstance(last, OSError)

    fail.clear()
    assert writer.write_sample({"ts": 2}) is True
    assert writer.consecutive_failures == 0
    assert writer.last_failure is last


def test_invalid_records_raise_before_any_state_changes(
    tmp_path: Path,
) -> None:
    ops = FileOps.system()._replace(open=_eio_open)
    writer = _writer(tmp_path, ops=ops)

    with pytest.raises(ValueError, match="JSON"):
        writer.write_sample({"cpu_util": float("nan")})
    with pytest.raises(ValueError, match="JSON"):
        writer.write_walk({"kind": WALK_KIND, "walk_s": float("inf")})
    with pytest.raises(ValueError, match="header kind"):
        writer.write_header({"job_id": "4711"})
    with pytest.raises(ValueError, match="header kind"):
        writer.write_header({"kind": WALK_KIND})
    with pytest.raises(ValueError, match="sample row"):
        writer.write_sample({"kind": WALK_KIND, "ts": 1})
    with pytest.raises(ValueError, match="walk kind"):
        writer.write_walk({"ts": 1})
    assert writer.consecutive_failures == 0
    assert writer.last_failure is None

    for _ in range(5):
        assert writer.write_sample({"ts": 1}) is False
    assert writer.consecutive_failures == 5


def test_nan_in_a_header_leaves_no_pending_header(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    with pytest.raises(ValueError, match="JSON"):
        writer.write_header({"kind": HEADER_KIND, "x": float("nan")})
    writer.write_sample({"ts": 1})

    assert _kinds(writer) == [None]


_SIGTERM = int(signal.SIGTERM)
_SIGINT = int(signal.SIGINT)
_IDENTITY = Identity("model", "4711", 0, "n001", 0)
_SAMPLE_NAMES = [c.name for c in SAMPLE_COLUMNS]
_WALK_NAMES = [c.name for c in WORKSPACE_COLUMNS]
_SAMPLE_VALUES: Values = dict.fromkeys(
    c.name for c in SAMPLE_COLUMNS if c.source in COUNTER_SOURCES
)
_SAMPLE_VALUES.update({"cpu_util": 0.5, "procs_running": 3})
_WALK_VALUES: Values = {
    "apparent_bytes": 1000,
    "allocated_bytes": 4096,
    "files": 3,
    "dirs": 2,
    "walk_s": 0.125,
    "fs_used_bytes": 10,
    "fs_avail_bytes": 90,
}


@dataclass(frozen=True)
class _Wake:
    """One scripted `wait`: `after` seconds slept (`None` is the whole
    timeout), the signal it returns, and a jump of the wall clock alone."""

    after: float | None = None
    sig: int | None = None
    wall_step: float = 0.0


_TICK = _Wake()
_STOP = _Wake(after=0.0, sig=_SIGTERM)


class _FakeClock:
    """Wall, monotonic, wait and CPU time driven by a script of wakes.

    The wall clock starts at `start` and the monotonic clock at 500; every
    wait adds `mono_drift` to the monotonic clock only. A wait past the end
    of the script fails the test instead of looping.
    """

    def __init__(
        self, start: float, wakes: list[_Wake], *, mono_drift: float = 0.0
    ) -> None:
        self.wall_s = start
        self.mono_s = 500.0
        self.cpu_s = 100.0
        self.maxrss_kib = 4096
        self.timeouts: list[float] = []
        self._wakes = iter(wakes)
        self._mono_drift = mono_drift

    def wall(self) -> float:
        return self.wall_s

    def monotonic(self) -> float:
        return self.mono_s

    def cpu_time(self) -> tuple[float, int]:
        return self.cpu_s, self.maxrss_kib

    def advance(self, seconds: float) -> None:
        self.wall_s += seconds
        self.mono_s += seconds

    def wait(self, timeout_s: float) -> int | None:
        self.timeouts.append(timeout_s)
        wake = next(self._wakes, None)
        if wake is None:
            raise AssertionError("the loop waited past its script")
        self.advance(timeout_s if wake.after is None else wake.after)
        self.mono_s += self._mono_drift
        self.wall_s += wake.wall_step
        return wake.sig

    def loop_clock(self) -> LoopClock:
        return LoopClock(self.wall, self.monotonic, self.wait, self.cpu_time)


class _FakeCollector(Collector):
    """A `Collector` that returns fixed values and records its calls."""

    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.baseline_wall_s = 0.0
        self.baseline_errors: Errors = []
        self.sample_errors: Errors = []
        self.sample_error: Exception | None = None
        self.sample_cpu_s = 0.0
        self.maxrss_kib = 4096
        self.received: list[float] = []
        self.sampled_at: list[float] = []
        self.walks = 0
        self.walks_to_stop = 0
        self.walk_values: Values = dict(_WALK_VALUES)
        self.walk_errors: Errors = []
        self.walk_cpu_s = 0.0

    def baseline(self) -> Errors:
        self.clock.advance(self.baseline_wall_s)
        return self.baseline_errors

    def sample(self, dt_s: float) -> tuple[Values, Errors]:
        if self.sample_error is not None:
            raise self.sample_error
        self.received.append(dt_s)
        self.sampled_at.append(self.clock.mono_s)
        self.clock.cpu_s += self.sample_cpu_s
        self.clock.maxrss_kib = self.maxrss_kib
        return dict(_SAMPLE_VALUES), self.sample_errors

    def walk(self) -> tuple[Values, Errors] | None:
        self.walks += 1
        self.clock.cpu_s += self.walk_cpu_s
        if self.walks_to_stop > 0:
            self.walks_to_stop -= 1
            return None
        return dict(self.walk_values), self.walk_errors


def _loop_writer(
    tmp_path: Path,
    clock: _FakeClock,
    ops: FileOps | None = None,
    identity: Identity = _IDENTITY,
) -> StagingWriter:
    return StagingWriter.create(
        tmp_path,
        identity.phase,
        identity.job_id,
        identity.restart,
        identity.node_index,
        clock.monotonic,
        ops,
    )


def _run(
    writer: StagingWriter,
    clock: _FakeClock,
    collector: Collector,
    *,
    walk: bool = False,
    workspace_interval_s: int = 60,
    identity: Identity = _IDENTITY,
) -> int:
    return run_loop(
        collector,
        writer,
        identity,
        interval_s=5,
        workspace_interval_s=workspace_interval_s,
        walk_enabled=walk,
        clock=clock.loop_clock(),
    )


def _records(writer: StagingWriter) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = [
        json.loads(line) for line in _lines(writer)
    ]
    return records


def _samples(writer: StagingWriter) -> list[dict[str, JsonValue]]:
    return [r for r in _records(writer) if "kind" not in r]


def _of_kind(writer: StagingWriter, kind: str) -> list[dict[str, JsonValue]]:
    return [r for r in _records(writer) if r.get("kind") == kind]


def _eio_open_counted(attempts: list[int]) -> FileOps:
    def open_(path: Path, flags: int, mode: int) -> int:
        attempts.append(1)
        raise OSError(errno.EIO, "injected")

    return FileOps.system()._replace(open=open_)


def _eio_first_opens(attempts: list[int], failures: int) -> FileOps:
    def open_(path: Path, flags: int, mode: int) -> int:
        attempts.append(1)
        if len(attempts) <= failures:
            raise OSError(errno.EIO, "injected")
        return os.open(path, flags, mode)

    return FileOps.system()._replace(open=open_)


def _os_errors(count: int) -> Errors:
    return [(source, OSError("x")) for source in COUNTER_SOURCES[:count]]


@pytest.mark.parametrize(
    ("now_ms", "interval_s", "expected"),
    [
        (10_000, 5, 15_000),
        (12_345, 5, 15_000),
        (15_000, 5, 20_000),
        (0, 5, 5_000),
        (14_999, 5, 15_000),
        (1_999, 1, 2_000),
        (1_700_000_000_123, 30, 1_700_000_010_000),
    ],
)
def test_next_tick_ms_known_instants_return_the_next_grid_multiple(
    now_ms: int, interval_s: int, expected: int
) -> None:
    assert next_tick_ms(now_ms, interval_s) == expected


@settings(deadline=None, max_examples=200)
@given(
    st.integers(min_value=0, max_value=4_000_000_000_000),
    st.integers(min_value=1, max_value=3600),
)
def test_next_tick_ms_any_instant_is_the_first_multiple_strictly_after(
    now_ms: int, interval_s: int
) -> None:
    step_ms = interval_s * 1000
    tick = next_tick_ms(now_ms, interval_s)

    assert tick % step_ms == 0
    assert now_ms < tick <= now_ms + step_ms


def test_loop_clock_system_wires_wall_monotonic_and_cpu_time() -> None:
    clock = LoopClock.system()

    cpu_s, maxrss_kib = clock.cpu_time()

    assert clock.wall is time.time
    assert clock.monotonic is time.monotonic
    assert cpu_s > 0.0
    assert maxrss_kib > 0


def test_loop_clock_system_cpu_time_sums_user_and_system_with_maxrss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = SimpleNamespace(ru_utime=1.5, ru_stime=0.25, ru_maxrss=2048)
    monkeypatch.setattr(resource, "getrusage", lambda who: usage)

    assert LoopClock.system().cpu_time() == (1.75, 2048)


def test_loop_clock_system_wait_without_a_pending_signal_returns_none() -> None:
    assert LoopClock.system().wait(0.0) is None


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_loop_clock_system_wait_returns_the_pending_signal_number(
    sig: signal.Signals,
) -> None:
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {sig})
    try:
        signal.raise_signal(sig)
        assert LoopClock.system().wait(0.0) == int(sig)
    finally:
        signal.sigtimedwait({sig}, 0)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def test_run_loop_sigterm_after_three_ticks_writes_a_final_sample_and_returns_zero(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _TICK, _Wake(1.5, _SIGTERM)])
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, _FakeCollector(clock))

    records = _records(writer)
    assert code == 0
    assert [r["ts"] for r in records] == [15000, 20000, 25000, 26500]
    assert [r["seq"] for r in records] == [0, 1, 2, 3]
    assert len(_SAMPLE_NAMES) == 64
    assert all(list(r) == _SAMPLE_NAMES for r in records)
    assert clock.timeouts == [3.0, 5.0, 5.0, 5.0]


def test_run_loop_sigint_is_a_stop_like_sigterm(tmp_path: Path) -> None:
    clock = _FakeClock(12.0, [_TICK, _Wake(0.5, _SIGINT)])
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, _FakeCollector(clock))

    assert code == 0
    assert [r["ts"] for r in _samples(writer)] == [15000, 15500]


def test_run_loop_stop_closes_the_writer_after_the_final_sample(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    clock = _FakeClock(12.0, [_TICK, _STOP])
    writer = _loop_writer(tmp_path, clock, _recorded(log))

    _run(writer, clock, _FakeCollector(clock))

    assert len(_samples(writer)) == 2
    assert log[-2:] == ["fsync", "close"]
    assert log.count("close") == 1


def test_run_loop_stop_at_the_last_tick_instant_still_increases_ts(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _STOP])
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, _FakeCollector(clock))

    assert [r["ts"] for r in _samples(writer)] == [15000, 15001]


def test_run_loop_stop_before_any_tick_samples_once_at_the_wake_instant(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_Wake(1.25, _SIGTERM)])
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, _FakeCollector(clock))

    assert code == 0
    assert [(r["ts"], r["seq"]) for r in _samples(writer)] == [(13250, 0)]


def test_run_loop_late_tick_is_skipped_and_dt_s_spans_the_gap(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(
        12.0, [_TICK, _Wake(after=11.0), _TICK, _STOP], mono_drift=0.25
    )
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    samples = _samples(writer)
    gap = collector.sampled_at[1] - collector.sampled_at[0]
    assert [r["ts"] for r in samples] == [15000, 30000, 30001]
    assert 20000 not in [r["ts"] for r in samples]
    assert samples[1]["dt_s"] == gap == 15.5
    assert clock.timeouts == [3.0, 5.0, 4.0, 5.0]


@pytest.mark.parametrize(
    ("after_s", "expected"),
    [(9.75, [15000, 20000, 24750]), (10.0, [15000, 25000])],
)
def test_run_loop_tick_is_late_only_from_one_whole_interval_after_it(
    tmp_path: Path, after_s: float, expected: list[int]
) -> None:
    clock = _FakeClock(12.0, [_TICK, _Wake(after=after_s), _STOP])
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, _FakeCollector(clock))

    assert [r["ts"] for r in _samples(writer)] == expected


def test_run_loop_slightly_late_wake_keeps_the_tick_as_sample_and_walk_ts(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_Wake(after=3.25), _STOP])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True)

    assert [r["ts"] for r in _samples(writer)] == [15000, 15250]
    assert [r["ts"] for r in _of_kind(writer, WALK_KIND)] == [15000]


def test_run_loop_early_wake_waits_again_without_sampling(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_Wake(after=2.0), _TICK, _STOP])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    assert clock.timeouts == [3.0, 1.0, 5.0]
    assert [r["ts"] for r in _samples(writer)] == [15000, 15001]
    assert collector.received == [3.0, 0.0]


def test_run_loop_wall_clock_step_back_never_repeats_a_ts(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(
        12.0,
        [_TICK, _Wake(after=5.0, wall_step=-7.0), _TICK, _TICK, _STOP],
    )
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, _FakeCollector(clock))

    assert clock.timeouts == [3.0, 5.0, 2.0, 5.0, 5.0]
    assert [r["ts"] for r in _samples(writer)] == [15000, 20000, 20001]


def test_run_loop_sample_lines_carry_the_dt_s_passed_to_sample(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _Wake(after=2.0, sig=_SIGTERM)])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    assert collector.received == [3.0, 5.0, 2.0]
    assert [r["dt_s"] for r in _samples(writer)] == collector.received


def test_run_loop_dt_s_is_the_monotonic_gap_not_the_wall_gap(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP], mono_drift=0.25)
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    assert collector.received == [3.25, 5.25, 0.25]


def test_run_loop_first_dt_s_counts_from_the_end_of_the_baseline(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.baseline_wall_s = 1.0
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    assert clock.timeouts == [2.0, 5.0]
    assert collector.received == [2.0, 0.0]


def test_run_loop_row_carries_identity_and_collector_values_in_column_order(
    tmp_path: Path,
) -> None:
    identity = Identity("finalize", "99", 2, "n042", 3)
    clock = _FakeClock(12.0, [_TICK, _STOP])
    writer = _loop_writer(tmp_path, clock, identity=identity)

    _run(writer, clock, _FakeCollector(clock), identity=identity)

    row = _samples(writer)[0]
    assert list(row) == _SAMPLE_NAMES
    assert [row[k] for k in _SAMPLE_NAMES[:7]] == [
        15000,
        "finalize",
        "99",
        2,
        "n042",
        3,
        0,
    ]
    assert row["cpu_util"] == 0.5
    assert row["procs_running"] == 3
    assert row["load1"] is None
    assert writer.path.name == "finalize-99-r2-3.ndjson"


def test_run_loop_sample_cost_excludes_the_walk_and_maxrss_follows_cpu_time(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.sample_cpu_s = 0.25
    collector.walk_cpu_s = 8.0
    collector.maxrss_kib = 2048
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True, workspace_interval_s=5)

    samples = _samples(writer)
    assert collector.walks == 2
    assert [r["sample_cost_ms"] for r in samples] == [250.0, 250.0, 250.0]
    assert [r["sampler_maxrss_bytes"] for r in samples] == [2048 * 1024] * 3


def test_run_loop_baseline_errors_are_written_before_the_first_sample(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.baseline_wall_s = 1.0
    collector.baseline_errors = [
        ("stat", OSError("gone")),
        ("meminfo", ValueError("bad")),
    ]
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    records = _records(writer)
    assert [r.get("kind") for r in records] == [
        ERROR_KIND,
        ERROR_KIND,
        None,
        None,
    ]
    assert [(r["ts"], r["source"], r["error"]) for r in records[:2]] == [
        (13000, "stat", "OSError"),
        (13000, "meminfo", "ValueError"),
    ]


def test_run_loop_sample_errors_are_written_after_the_sample_with_its_ts(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.sample_errors = [("vmstat", OSError("unreadable"))]
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector)

    records = _records(writer)
    assert [r.get("kind") for r in records] == [None, ERROR_KIND, None, None]
    assert records[1] == {
        "kind": ERROR_KIND,
        "ts": 15000,
        "source": "vmstat",
        "error": "OSError",
        "message": "unreadable",
    }


def test_run_loop_walk_lines_follow_the_workspace_interval(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK] * 13 + [_STOP])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, collector, walk=True)

    assert code == 0
    assert collector.walks == 2
    assert [r["ts"] for r in _of_kind(writer, WALK_KIND)] == [15000, 75000]
    assert _records(writer)[1]["kind"] == WALK_KIND


def test_run_loop_walk_disabled_writes_no_walk_line(tmp_path: Path) -> None:
    clock = _FakeClock(12.0, [_TICK] * 13 + [_STOP])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=False)

    assert collector.walks == 0
    assert _of_kind(writer, WALK_KIND) == []


def test_run_loop_slow_walk_backs_the_next_walk_off(tmp_path: Path) -> None:
    clock = _FakeClock(12.0, [_TICK] * 49 + [_STOP])
    collector = _FakeCollector(clock)
    collector.walk_values["walk_s"] = 12.0
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True)

    assert [r["ts"] for r in _of_kind(writer, WALK_KIND)] == [15000, 255000]


def test_run_loop_walk_that_failed_to_time_keeps_the_workspace_interval(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK] * 13 + [_STOP])
    collector = _FakeCollector(clock)
    collector.walk_values["walk_s"] = None
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True)

    assert [r["ts"] for r in _of_kind(writer, WALK_KIND)] == [15000, 75000]


def test_run_loop_walk_line_has_kind_identity_and_walk_values(
    tmp_path: Path,
) -> None:
    identity = Identity("finalize", "99", 2, "n042", 0)
    clock = _FakeClock(12.0, [_TICK, _STOP])
    writer = _loop_writer(tmp_path, clock, identity=identity)

    _run(writer, clock, _FakeCollector(clock), walk=True, identity=identity)

    (walk,) = _of_kind(writer, WALK_KIND)
    assert list(walk) == ["kind", *_WALK_NAMES]
    assert "node_index" not in walk
    assert walk == {
        "kind": WALK_KIND,
        "ts": 15000,
        "phase": "finalize",
        "job_id": "99",
        "restart": 2,
        "node": "n042",
        **_WALK_VALUES,
    }


def test_run_loop_walk_errors_are_written_after_the_walk_with_the_tick(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.walk_errors = [("walk", PermissionError("denied"))]
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True)

    records = _records(writer)
    assert [r.get("kind") for r in records] == [
        None,
        WALK_KIND,
        ERROR_KIND,
        None,
    ]
    assert records[2]["ts"] == 15000
    assert records[2]["source"] == "walk"
    assert records[2]["error"] == "PermissionError"


def test_run_loop_final_sample_never_walks_even_when_a_walk_is_due(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True, workspace_interval_s=5)

    assert collector.walks == 2
    assert _kinds(writer) == [None, WALK_KIND, None, WALK_KIND, None]


def test_run_loop_sigterm_after_two_ticks_writes_one_walk_then_the_final_sample(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _Wake(1.5, _SIGTERM)])
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, collector, walk=True)

    records = _records(writer)
    assert code == 0
    assert collector.walks == 1
    assert [r.get("kind") for r in records] == [None, WALK_KIND, None, None]
    assert records[-1]["ts"] == 21500


def test_run_loop_walk_stopped_by_a_signal_leaves_no_line_and_the_final_sample_last(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.walks_to_stop = 99
    collector.walk_errors = [("walk", PermissionError("denied"))]
    writer = _loop_writer(tmp_path, clock)

    code = _run(writer, clock, collector, walk=True)

    assert code == 0
    assert collector.walks == 1
    assert _kinds(writer) == [None, None]
    assert [r["ts"] for r in _samples(writer)] == [15000, 15001]


def test_run_loop_stopped_walk_is_retried_on_the_next_tick(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.walks_to_stop = 1
    writer = _loop_writer(tmp_path, clock)

    _run(writer, clock, collector, walk=True)

    assert collector.walks == 2
    assert _kinds(writer) == [None, None, WALK_KIND, None]
    assert [r["ts"] for r in _of_kind(writer, WALK_KIND)] == [20000]


def test_run_loop_five_failed_ticks_return_one_after_exactly_five_ticks(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK] * 10)
    collector = _FakeCollector(clock)
    ops = FileOps.system()._replace(open=_eio_open)
    writer = _loop_writer(tmp_path, clock, ops)

    code = _run(writer, clock, collector, walk=False)

    assert code == 1
    assert MAX_CONSECUTIVE_FAILURES == 5
    assert len(clock.timeouts) == 5
    assert len(collector.received) == 5
    assert writer.consecutive_failures == 5
    assert not writer.path.exists()


def test_run_loop_failed_walk_write_counts_toward_the_exit(
    tmp_path: Path,
) -> None:
    attempts: list[int] = []
    clock = _FakeClock(12.0, [_TICK] * 10)
    collector = _FakeCollector(clock)
    writer = _loop_writer(tmp_path, clock, _eio_open_counted(attempts))

    code = _run(writer, clock, collector, walk=True)

    assert code == 1
    assert collector.walks == 1
    assert len(clock.timeouts) == 4
    assert len(attempts) == 5


def test_run_loop_failed_sample_that_reaches_the_limit_skips_the_due_walk(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK] * 3)
    collector = _FakeCollector(clock)
    ops = FileOps.system()._replace(open=_eio_open)
    writer = _loop_writer(tmp_path, clock, ops)
    writer.consecutive_failures = MAX_CONSECUTIVE_FAILURES - 1

    code = _run(writer, clock, collector, walk=True)

    assert code == 1
    assert collector.walks == 0
    assert len(collector.received) == 1


def test_run_loop_baseline_failures_return_one_before_any_wait(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK])
    collector = _FakeCollector(clock)
    collector.baseline_errors = [
        (source, OSError("x")) for source in COUNTER_SOURCES[:5]
    ]
    ops = FileOps.system()._replace(open=_eio_open)
    writer = _loop_writer(tmp_path, clock, ops)

    code = _run(writer, clock, collector)

    assert code == 1
    assert clock.timeouts == []
    assert collector.received == []


@pytest.mark.parametrize(
    ("baseline", "sample", "walk", "walk_errors"),
    [
        pytest.param(6, 0, False, 0, id="baseline-errors"),
        pytest.param(0, 5, False, 0, id="sample-errors"),
        pytest.param(0, 3, True, 1, id="walk-line"),
        pytest.param(0, 1, True, 3, id="walk-errors"),
    ],
)
def test_run_loop_limit_reached_mid_group_returns_one_before_the_later_lines(
    tmp_path: Path, baseline: int, sample: int, walk: bool, walk_errors: int
) -> None:
    # The first five opens fail and the sixth would succeed and reset the
    # counter, so the loop must stop after the fifth write of the group.
    attempts: list[int] = []
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    collector = _FakeCollector(clock)
    collector.baseline_errors = _os_errors(baseline)
    collector.sample_errors = _os_errors(sample)
    collector.walk_errors = _os_errors(walk_errors)
    writer = _loop_writer(tmp_path, clock, _eio_first_opens(attempts, 5))

    code = _run(writer, clock, collector, walk=walk)

    assert code == 1
    assert len(attempts) == 5
    assert writer.consecutive_failures == MAX_CONSECUTIVE_FAILURES
    assert collector.walks == int(walk)
    assert not writer.path.exists()


def test_run_loop_failed_writes_still_advance_seq(tmp_path: Path) -> None:
    clock = _FakeClock(12.0, [_TICK] * 3 + [_Wake(1.5, _SIGTERM)])
    writer = _loop_writer(tmp_path, clock, _eio_first_opens([], 2))

    code = _run(writer, clock, _FakeCollector(clock))

    assert code == 0
    assert [(r["seq"], r["ts"]) for r in _samples(writer)] == [
        (2, 25000),
        (3, 26500),
    ]


def test_run_loop_stop_returns_zero_even_when_the_final_write_is_the_fifth_failure(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK] * 4 + [_STOP])
    collector = _FakeCollector(clock)
    ops = FileOps.system()._replace(open=_eio_open)
    writer = _loop_writer(tmp_path, clock, ops)

    code = _run(writer, clock, collector)

    assert code == 0
    assert len(collector.received) == 5
    assert writer.consecutive_failures == MAX_CONSECUTIVE_FAILURES


def test_run_loop_after_a_header_writes_it_once_per_file(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK, _TICK, _STOP])
    writer = _loop_writer(tmp_path, clock)
    writer.write_header({"kind": HEADER_KIND, "schema_version": 1})

    _run(writer, clock, _FakeCollector(clock))

    assert _kinds(writer) == [HEADER_KIND, None, None, None]


def test_run_loop_unexpected_collector_error_propagates(
    tmp_path: Path,
) -> None:
    clock = _FakeClock(12.0, [_TICK])
    collector = _FakeCollector(clock)
    collector.sample_error = RuntimeError("boom")
    writer = _loop_writer(tmp_path, clock)

    with pytest.raises(RuntimeError, match="boom"):
        _run(writer, clock, collector)


_SYNTHETIC = load_bundle("synthetic")
_WORKSPACE = Path("/shared-zfs/ws-test")
_STOP_SIGNALS = {signal.SIGTERM, signal.SIGINT}
_SYNTHETIC_FACTS = StaticFacts(
    "r7a.48xlarge",
    "Synthetic CPU 9R14",
    4,
    16384000000,
    "6.8.0-1021-synthetic",
    "newave",
    2,
    192,
)
_JOB_CGROUP = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e"
_STARTED_AT_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z"
_START_S = 1_700_000_000.25
_SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_NODEID",
    "SLURM_RESTART_COUNT",
    "SLURMD_NODENAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NUM_NODES",
    "SLURM_JOB_CPUS_PER_NODE",
)
_in_process = pytest.mark.usefixtures("signal_mask", "slurm_env", "nice_calls")


@pytest.fixture
def signal_mask() -> Iterator[None]:
    saved = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    yield
    signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    while signal.sigtimedwait(_STOP_SIGNALS, 0) is not None:
        pass
    signal.pthread_sigmask(signal.SIG_SETMASK, saved)


@pytest.fixture
def slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SLURM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "n001")


@pytest.fixture
def nice_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def nice(increment: int) -> int:
        calls.append(increment)
        return increment

    monkeypatch.setattr(os, "nice", nice)
    return calls


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    path = tmp_path / "staging"
    path.mkdir()
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return materialize(_SYNTHETIC, 0, tmp_path / "root")


def _argv(staging: Path, tree: Path) -> list[str]:
    return [
        "--staging",
        str(staging),
        "--phase",
        "model",
        "--workspace",
        str(_WORKSPACE),
        "--interval",
        "5",
        "--workspace-interval",
        "60",
        "--tool-version",
        "2.2.2",
        "--root",
        str(tree),
    ]


def _with(argv: list[str], flag: str, value: str | None) -> list[str]:
    at = argv.index(flag)
    if value is None:
        return argv[:at] + argv[at + 2 :]
    return argv[: at + 1] + [value] + argv[at + 2 :]


def _queue_sigterm() -> None:
    ident = threading.main_thread().ident
    assert ident is not None
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    signal.pthread_kill(ident, signal.SIGTERM)


def _staged(staging: Path, name: str = _NAME) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = [
        json.loads(line) for line in (staging / name).read_bytes().splitlines()
    ]
    return records


def _header_for(
    tree: Path, started_at: float = _START_S
) -> dict[str, JsonValue]:
    return build_header(
        _IDENTITY,
        _SYNTHETIC_FACTS,
        Collector(tree, _WORKSPACE),
        interval_s=5,
        workspace_interval_s=60,
        tool_version="2.2.2",
        started_at=started_at,
    )


def test_build_header_synthetic_inputs_return_every_field_in_order(
    tree: Path,
) -> None:
    header = _header_for(tree)

    assert list(header) == [f.name for f in HEADER_FIELDS]
    assert header == {
        "kind": HEADER_KIND,
        "schema_version": 1,
        "tool_version": "2.2.2",
        "phase": "model",
        "job_id": "4711",
        "restart": 0,
        "node": "n001",
        "node_index": 0,
        "nodes": 2,
        "partition": "newave",
        "cpus_on_node": 192,
        "instance_type": "r7a.48xlarge",
        "cpu_model": "Synthetic CPU 9R14",
        "ncpus": 4,
        "mem_total_bytes": 16384000000,
        "kernel": "6.8.0-1021-synthetic",
        "job_cgroup": _JOB_CGROUP,
        "own_step_cgroup": f"{_JOB_CGROUP}/step_0",
        "workspace_mount": {"mountpoint": "/shared-zfs", "fstype": "nfs4"},
        "interval_s": 5,
        "workspace_interval_s": 60,
        "started_at": "2023-11-14T22:13:20.250Z",
    }
    assert re.fullmatch(_STARTED_AT_RE, str(header["started_at"]))
    json.dumps(header, allow_nan=False)


def test_build_header_unknown_facts_and_cgroups_are_null(
    tmp_path: Path,
) -> None:
    tree = materialize(
        _SYNTHETIC, 0, tmp_path, {"proc/self/cgroup": "0::/user.slice\n"}
    )
    header = build_header(
        _IDENTITY,
        StaticFacts(None, None, None, None, None, None, None, None),
        Collector(tree, Path("/nowhere")),
        interval_s=5,
        workspace_interval_s=60,
        tool_version="2.2.2",
        started_at=_START_S,
    )

    nulls = {name for name, value in header.items() if value is None}
    assert nulls == {f.name for f in HEADER_FIELDS if f.nullable} - {
        "workspace_mount"
    }
    assert header["workspace_mount"] == {"mountpoint": "/", "fstype": "ext4"}
    json.dumps(header, allow_nan=False)


@pytest.mark.parametrize(
    ("started_at", "expected"),
    [
        (0.0, "1970-01-01T00:00:00.000Z"),
        (1_700_000_000.1234, "2023-11-14T22:13:20.123Z"),
        (1_700_000_000.9999, "2023-11-14T22:13:20.999Z"),
    ],
)
def test_build_header_started_at_is_utc_with_milliseconds_and_z(
    tree: Path, started_at: float, expected: str
) -> None:
    assert _header_for(tree, started_at)["started_at"] == expected


@_in_process
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SLURM_JOB_ID", None),
        ("SLURM_JOB_ID", "47a"),
        ("SLURM_JOB_ID", ""),
        ("SLURM_NODEID", None),
        ("SLURM_NODEID", "x"),
        ("SLURM_NODEID", "-1"),
        ("SLURM_RESTART_COUNT", "x"),
        ("SLURM_RESTART_COUNT", "-1"),
    ],
)
def test_main_bad_environment_returns_two_names_the_variable_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staging: Path,
    tree: Path,
    name: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    code = main(_argv(staging, tree))

    assert code == 2
    assert capsys.readouterr().err.startswith(f"hpcmu-telemetry: {name}: ")
    assert list(staging.iterdir()) == []


@_in_process
@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--interval", "0"),
        ("--interval", "-5"),
        ("--interval", "x"),
        ("--interval", None),
        ("--workspace-interval", "0"),
        ("--workspace-interval", None),
        ("--phase", "run"),
        ("--phase", None),
        ("--staging", None),
        ("--workspace", None),
        ("--tool-version", None),
    ],
)
def test_main_bad_argument_exits_two_through_argparse(
    capsys: pytest.CaptureFixture[str],
    staging: Path,
    tree: Path,
    flag: str,
    value: str | None,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(_with(_argv(staging, tree), flag, value))

    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err
    assert list(staging.iterdir()) == []


@_in_process
def test_main_help_exits_zero_and_hides_the_root_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "--workspace-interval" in out
    assert "--root" not in out


@_in_process
def test_main_blocks_the_stop_signals_before_it_parses_the_arguments() -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    assert _STOP_SIGNALS <= set(blocked)


@_in_process
def test_main_pending_sigterm_with_failed_nice_writes_header_error_and_final_sample(
    monkeypatch: pytest.MonkeyPatch, staging: Path, tree: Path
) -> None:
    def nice(increment: int) -> int:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "nice", nice)
    _queue_sigterm()

    code = main(_argv(staging, tree))

    records = _staged(staging)
    errors = [r for r in records if r.get("kind") == ERROR_KIND]
    samples = [r for r in records if "kind" not in r]
    assert code == 0
    assert signal.SIGTERM not in signal.sigpending()
    assert records[0]["kind"] == HEADER_KIND
    assert list(records[0]) == [f.name for f in HEADER_FIELDS]
    assert records[0]["job_cgroup"] == _JOB_CGROUP
    assert [e["source"] for e in errors] == ["sampler"]
    assert errors[0]["error"] == "PermissionError"
    assert samples == [records[-1]]
    assert samples[0]["seq"] == 0
    assert [p.name for p in staging.iterdir()] == [_NAME]


@_in_process
def test_main_nice_success_asks_for_ten_and_records_no_error(
    nice_calls: list[int], staging: Path, tree: Path
) -> None:
    _queue_sigterm()

    code = main(_argv(staging, tree))

    assert code == 0
    assert nice_calls == [10]
    assert [r for r in _staged(staging) if r.get("kind") == ERROR_KIND] == []


@_in_process
@pytest.mark.parametrize(
    ("restart_count", "restart"), [(None, 0), ("0", 0), ("2", 2)]
)
def test_main_restart_count_defaults_to_zero_and_names_the_file(
    monkeypatch: pytest.MonkeyPatch,
    staging: Path,
    tree: Path,
    restart_count: str | None,
    restart: int,
) -> None:
    if restart_count is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart_count)
    _queue_sigterm()

    code = main(_argv(staging, tree))

    name = f"model-4711-r{restart}-0.ndjson"
    assert code == 0
    assert [p.name for p in staging.iterdir()] == [name]
    assert _staged(staging, name)[0]["restart"] == restart


@_in_process
@pytest.mark.parametrize("nodename", ["n001", None])
def test_main_node_name_falls_back_to_the_uname_nodename(
    monkeypatch: pytest.MonkeyPatch,
    staging: Path,
    tree: Path,
    nodename: str | None,
) -> None:
    if nodename is None:
        monkeypatch.delenv("SLURMD_NODENAME")
    _queue_sigterm()

    main(_argv(staging, tree))

    header = _staged(staging)[0]
    assert header["node"] == (nodename or os.uname().nodename)


@_in_process
def test_main_start_up_errors_follow_nice_static_facts_and_collector(
    monkeypatch: pytest.MonkeyPatch, staging: Path, tmp_path: Path
) -> None:
    def nice(increment: int) -> int:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "nice", nice)
    monkeypatch.setattr(time, "time", lambda: _START_S)
    tree = materialize(
        _SYNTHETIC,
        0,
        tmp_path / "root",
        {
            "sys/devices/virtual/dmi/id/product_name": None,
            "proc/self/cgroup": None,
        },
    )
    _queue_sigterm()

    main(_argv(staging, tree))

    records = _staged(staging)
    errors = [r for r in records if r.get("kind") == ERROR_KIND]
    assert [e["source"] for e in errors] == ["sampler", "header", "cgroup"]
    assert {e["ts"] for e in errors} == {1_700_000_000_250}
    assert records[0]["started_at"] == "2023-11-14T22:13:20.250Z"
    assert records[0]["instance_type"] is None
    assert records[0]["job_cgroup"] is None


@_in_process
@pytest.mark.parametrize(
    ("node_id", "walks"), [("0", True), ("1", False), ("12", False)]
)
def test_main_runs_the_loop_with_the_arguments_and_walks_on_node_zero_only(
    monkeypatch: pytest.MonkeyPatch,
    staging: Path,
    tree: Path,
    node_id: str,
    walks: bool,
) -> None:
    monkeypatch.setenv("SLURM_NODEID", node_id)
    seen: list[tuple[Identity, int, int, bool, LoopClock]] = []

    def fake_loop(
        collector: Collector,
        writer: StagingWriter,
        identity: Identity,
        *,
        interval_s: int,
        workspace_interval_s: int,
        walk_enabled: bool,
        clock: LoopClock,
    ) -> int:
        seen.append(
            (identity, interval_s, workspace_interval_s, walk_enabled, clock)
        )
        return 0

    monkeypatch.setattr(sampler, "run_loop", fake_loop)

    code = main(_argv(staging, tree))

    identity, interval_s, workspace_interval_s, walk_enabled, clock = seen[0]
    assert code == 0
    assert identity == Identity("model", "4711", 0, "n001", int(node_id))
    assert (interval_s, workspace_interval_s) == (5, 60)
    assert walk_enabled is walks
    assert clock.monotonic is time.monotonic
    assert [p.name for p in staging.iterdir()] == [
        f"model-4711-r0-{node_id}.ndjson"
    ]


@_in_process
@pytest.mark.parametrize(
    ("code", "message"),
    [
        (0, ""),
        (
            1,
            "hpcmu-telemetry: 5 consecutive staging failures: "
            "[Errno 5] injected\n",
        ),
    ],
)
def test_main_exit_one_prints_the_last_staging_failure_and_zero_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staging: Path,
    tree: Path,
    code: int,
    message: str,
) -> None:
    def fake_loop(
        collector: Collector,
        writer: StagingWriter,
        identity: Identity,
        *,
        interval_s: int,
        workspace_interval_s: int,
        walk_enabled: bool,
        clock: LoopClock,
    ) -> int:
        writer.last_failure = OSError(errno.EIO, "injected")
        return code

    monkeypatch.setattr(sampler, "run_loop", fake_loop)

    assert main(_argv(staging, tree)) == code
    assert capsys.readouterr().err == message


@_in_process
def test_main_missing_staging_directory_fails_the_stop_path_to_zero(
    tmp_path: Path, tree: Path
) -> None:
    _queue_sigterm()

    code = main(_argv(tmp_path / "absent", tree))

    assert code == 0
    assert not (tmp_path / "absent").exists()
