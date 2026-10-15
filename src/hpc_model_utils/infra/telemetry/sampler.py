"""ADR-069/ADR-079: the node sampler's staging writer and tick loop.

One staging file is `<phase>-<job_id>-r<restart>-<node_index>.ndjson`, created
with `O_EXCL` so a second writer fails instead of sharing it (R62). Its lines
are ASCII JSON with `allow_nan=False`, of four kinds: the header (line 1), the
samples (no `kind`), the workspace walks (node 0 only) and the errors.

The file is a raw descriptor, never a buffered file object, so a record is one
logical write (R34): `FileOps.write` is looped over a short write until every
byte is out. After a failed or partial write the next record starts with a
newline, so the damage stays inside one line. Header and error lines are
fsynced at once; sample and walk lines when `FSYNC_INTERVAL_S` has passed.

`OSError` from a file operation is counted in `consecutive_failures` and kept
in `last_failure`, never raised: the caller reads the counter and exits 1 at 5
(ADR-079). A full record, fsync included, resets the counter. A record is
written once its bytes are in the file, so a failed fsync is counted but does
not repeat the record. `ValueError` for a bad kind, a NaN or an infinity
propagates, because it is a programming error.

`run_loop` is the single-threaded tick loop (ADR-079). SIGTERM and SIGINT are
blocked by the caller and `LoopClock.wait` is the only sleep, so a stop is seen
at once and never in the middle of a write. Ticks fall on wall-clock multiples
of the interval (R30), counted in integer milliseconds so the grid does not
drift. A tick the loop wakes for a whole interval late is skipped, never
bunched, and `dt_s` is the real monotonic gap since the previous counter read.
The final sample of a stop is off the grid and never walks the workspace.

`main` is the entry point of `python -I -m
hpc_model_utils.infra.telemetry.sampler` (R26). It blocks SIGTERM and SIGINT
before anything else, so a stop that arrives during start-up stays pending and
the first wait takes it, reads the Slurm environment, writes the header and
the start-up errors, and returns `run_loop`'s code. Exit codes: 0 on a stop, 1
after `MAX_CONSECUTIVE_FAILURES` staging failures, 2 on a usage error, which
is an argparse error or a bad `SLURM_*` variable. The argv is an internal
contract with the sidecar line of the sbatch template; `--root`, the tree
read in place of `/`, is hidden because only the tests use it.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import resource
import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

from hpc_model_utils.infra.telemetry.collect import (
    Collector,
    Errors,
    next_walk_interval,
)
from hpc_model_utils.infra.telemetry.procfs import (
    StaticFacts,
    read_static_facts,
)
from hpc_model_utils.infra.telemetry.schema import (
    ERROR_KIND,
    HEADER_KIND,
    PHASES,
    SAMPLE_COLUMNS,
    SCHEMA_VERSION,
    WALK_KIND,
    WORKSPACE_COLUMNS,
    Column,
    staging_file_name,
)

ERROR_LINE_CAP: Final = 64
MESSAGE_CAP: Final = 500
FSYNC_INTERVAL_S: Final = 30.0
MAX_CONSECUTIVE_FAILURES: Final = 5

_STOP_SIGNALS: Final = frozenset({signal.SIGTERM, signal.SIGINT})
_DIGITS: Final = re.compile("[0-9]+")
_NICE_INCREMENT: Final = 10

_OPEN_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
_FILE_MODE: Final = 0o644

type JsonValue = str | int | float | None | Mapping[str, JsonValue]


class FileOps(NamedTuple):
    """The four `os` functions the writer calls, replaceable in tests."""

    open: Callable[[Path, int, int], int]
    write: Callable[[int, bytes], int]
    fsync: Callable[[int], None]
    close: Callable[[int], None]

    @classmethod
    def system(cls) -> FileOps:
        return cls(os.open, os.write, os.fsync, os.close)


def _encode(record: Mapping[str, JsonValue]) -> bytes:
    text = json.dumps(
        record, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    )
    return (text + "\n").encode("ascii")


def _error_line(ts_ms: int, source: str, error: str, message: str) -> bytes:
    return _encode(
        {
            "kind": ERROR_KIND,
            "ts": ts_ms,
            "source": source,
            "error": error,
            "message": message,
        }
    )


class StagingWriter:
    """Append the records of one staging file; see the module docstring.

    Nothing is opened until the first record, which opens the file with
    `O_WRONLY | O_CREAT | O_EXCL | O_APPEND` and mode `0o644`; a failed open,
    `FileExistsError` included, is retried by the next record. Every record
    method returns `True` when the record is fully written and `False` when an
    open or a write failed.

    `write_header` keeps the header pending until it is written, and every
    later record method writes it first and returns `False` if that fails.
    `write_error` writes the first error of each `(source, class)` only, and
    after `ERROR_LINE_CAP` of them one `error_lines_capped` line; a suppressed
    error returns `True` and touches no counter. `monotonic` is the clock of
    the fsync cadence. `sync()` fsyncs at once, and `close()` fsyncs and closes
    an open file.
    """

    def __init__(
        self, path: Path, monotonic: Callable[[], float], ops: FileOps
    ) -> None:
        self.path = path
        self.consecutive_failures = 0
        self.last_failure: OSError | None = None
        self._monotonic = monotonic
        self._ops = ops
        self._fd: int | None = None
        self._pending_header: bytes | None = None
        self._resync = False
        self._error_keys: set[tuple[str, str]] = set()
        self._capped = False
        self._last_fsync = -math.inf

    @classmethod
    def create(
        cls,
        staging_dir: Path,
        phase: str,
        job_id: str,
        restart: int,
        node_index: int,
        monotonic: Callable[[], float],
        ops: FileOps | None = None,
    ) -> StagingWriter:
        name = staging_file_name(phase, job_id, restart, node_index)
        return cls(
            staging_dir / name,
            monotonic,
            FileOps.system() if ops is None else ops,
        )

    def write_header(self, header: Mapping[str, JsonValue]) -> bool:
        if header.get("kind") != HEADER_KIND:
            raise ValueError(f"header kind must be {HEADER_KIND!r}")
        self._pending_header = _encode(header)
        return self._flush_header()

    def write_sample(self, row: Mapping[str, JsonValue]) -> bool:
        if "kind" in row:
            raise ValueError("a sample row has no 'kind'")
        return self._emit(_encode(row), sync_now=False)

    def write_walk(self, row: Mapping[str, JsonValue]) -> bool:
        if row.get("kind") != WALK_KIND:
            raise ValueError(f"walk kind must be {WALK_KIND!r}")
        return self._emit(_encode(row), sync_now=False)

    def write_error(self, ts_ms: int, source: str, exc: Exception) -> bool:
        key = (source, type(exc).__name__)
        if self._capped or key in self._error_keys:
            return True
        if len(self._error_keys) >= ERROR_LINE_CAP:
            line = _error_line(
                ts_ms,
                "sampler",
                "error_lines_capped",
                f"error lines capped at {ERROR_LINE_CAP}",
            )
            self._capped = self._emit(line, sync_now=True)
            return self._capped
        line = _error_line(ts_ms, source, key[1], str(exc)[:MESSAGE_CAP])
        written = self._emit(line, sync_now=True)
        if written:
            self._error_keys.add(key)
        return written

    def sync(self) -> bool:
        return self._sync(force=True)

    def close(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._sync(force=True)
        self._fd = None
        try:
            self._ops.close(fd)
        except OSError as exc:
            self._fail(exc)

    def _flush_header(self) -> bool:
        header = self._pending_header
        if header is None:
            return True
        if not self._append(header, sync_now=True):
            return False
        self._pending_header = None
        return True

    def _emit(self, data: bytes, *, sync_now: bool) -> bool:
        return self._flush_header() and self._append(data, sync_now=sync_now)

    def _append(self, data: bytes, *, sync_now: bool) -> bool:
        if not self._write(data):
            return False
        if self._sync(force=sync_now):
            self.consecutive_failures = 0
        return True

    def _write(self, data: bytes) -> bool:
        fd = self._fd
        if fd is None:
            try:
                fd = self._ops.open(self.path, _OPEN_FLAGS, _FILE_MODE)
            except OSError as exc:
                return self._fail(exc)
            self._fd = fd
        if self._resync:
            data = b"\n" + data
        try:
            while data:
                count = self._ops.write(fd, data)
                if count <= 0:
                    raise OSError(errno.EIO, "write made no progress")
                data = data[count:]
        except OSError as exc:
            self._resync = True
            return self._fail(exc)
        self._resync = False
        return True

    def _sync(self, *, force: bool) -> bool:
        fd = self._fd
        if fd is None:
            return True
        now = self._monotonic()
        if not force and now - self._last_fsync < FSYNC_INTERVAL_S:
            return True
        try:
            self._ops.fsync(fd)
        except OSError as exc:
            return self._fail(exc)
        self._last_fsync = now
        return True

    def _fail(self, exc: OSError) -> bool:
        self.consecutive_failures += 1
        self.last_failure = exc
        return False


def _wait_for_stop(timeout_s: float) -> int | None:
    info = signal.sigtimedwait(_STOP_SIGNALS, timeout_s)
    return None if info is None else info.si_signo


def _cpu_time() -> tuple[float, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime, usage.ru_maxrss


@dataclass(frozen=True, slots=True)
class LoopClock:
    """The four time and signal sources of `run_loop`, replaceable in tests.

    `wait(timeout_s)` returns the number of a pending SIGTERM or SIGINT, or
    `None` on timeout; `cpu_time()` returns user + system seconds and
    `ru_maxrss` in KiB.
    """

    wall: Callable[[], float]
    monotonic: Callable[[], float]
    wait: Callable[[float], int | None]
    cpu_time: Callable[[], tuple[float, int]]

    @classmethod
    def system(cls) -> LoopClock:
        return cls(time.time, time.monotonic, _wait_for_stop, _cpu_time)


@dataclass(frozen=True, slots=True)
class Identity:
    phase: str
    job_id: str
    restart: int
    node: str
    node_index: int


def next_tick_ms(now_ms: int, interval_s: int) -> int:
    step_ms = interval_s * 1000
    return (now_ms // step_ms + 1) * step_ms


def _wall_ms(clock: LoopClock) -> int:
    return int(clock.wall() * 1000)


def _row(
    columns: Iterable[Column], known: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    return {column.name: known[column.name] for column in columns}


def _limit_reached(writer: StagingWriter) -> bool:
    return writer.consecutive_failures >= MAX_CONSECUTIVE_FAILURES


def _write_errors(writer: StagingWriter, ts_ms: int, errors: Errors) -> bool:
    """Write error lines until the failure limit is reached; True if it was.

    The limit is read first, so a record the caller just wrote counts too, and
    again after each line; the lines after the one that reaches it are not
    written (R35).
    """
    for source, exc in errors:
        if _limit_reached(writer):
            return True
        writer.write_error(ts_ms, source, exc)
    return _limit_reached(writer)


def run_loop(
    collector: Collector,
    writer: StagingWriter,
    identity: Identity,
    *,
    interval_s: int,
    workspace_interval_s: int,
    walk_enabled: bool,
    clock: LoopClock,
) -> int:
    """Sample on the wall-clock grid until a stop signal; return the exit code.

    Reads and drops one baseline first, writing its errors. Each tick writes
    one sample line, then its error lines, and, on node 0 with `walk_enabled`,
    a workspace walk when `next_walk_interval` says it is due. A walk the
    collector abandons for a pending signal (`None`) writes nothing, and the
    next wait returns that signal. A signal takes one final sample off the
    grid, closes the writer and returns 0 whatever the writer's state (R36);
    that sample never walks. Otherwise the loop returns 1 right after any
    write that leaves `writer.consecutive_failures` at or above
    `MAX_CONSECUTIVE_FAILURES`, a baseline, sample, walk or error line, and
    writes nothing more (R35). Any other exception is a programming error and
    propagates (ADR-079).
    """
    step_ms = interval_s * 1000
    who: dict[str, JsonValue] = {
        "phase": identity.phase,
        "job_id": identity.job_id,
        "restart": identity.restart,
        "node": identity.node,
        "node_index": identity.node_index,
    }
    baseline_errors = collector.baseline()
    if _write_errors(writer, _wall_ms(clock), baseline_errors):
        return 1
    last_read = clock.monotonic()
    last_ts = -1
    seq = 0
    next_walk_ms = 0
    while True:
        now_ms = _wall_ms(clock)
        tick = next_tick_ms(now_ms, interval_s)
        sig = clock.wait((tick - now_ms) / 1000)
        now_ms = _wall_ms(clock)
        stop = sig is not None
        if not stop and (
            now_ms < tick or now_ms >= tick + step_ms or tick <= last_ts
        ):
            continue
        ts = max(now_ms, last_ts + 1) if stop else tick
        read_at = clock.monotonic()
        dt_s = read_at - last_read
        last_read = read_at
        cpu_before, _ = clock.cpu_time()
        values, errors = collector.sample(dt_s)
        cpu_after, maxrss_kib = clock.cpu_time()
        last_ts = ts
        writer.write_sample(
            _row(
                SAMPLE_COLUMNS,
                {
                    **who,
                    "ts": ts,
                    "seq": seq,
                    "dt_s": dt_s,
                    "sample_cost_ms": (cpu_after - cpu_before) * 1000,
                    "sampler_maxrss_bytes": maxrss_kib * 1024,
                    **values,
                },
            )
        )
        seq += 1
        if stop:
            for source, exc in errors:
                writer.write_error(ts, source, exc)
            writer.close()
            return 0
        if _write_errors(writer, ts, errors):
            return 1
        if walk_enabled and tick >= next_walk_ms:
            walked = collector.walk()
            if walked is not None:
                walk_values, walk_errors = walked
                writer.write_walk(
                    {
                        "kind": WALK_KIND,
                        **_row(
                            WORKSPACE_COLUMNS,
                            {**who, "ts": tick, **walk_values},
                        ),
                    }
                )
                if _write_errors(writer, tick, walk_errors):
                    return 1
                pause_s = next_walk_interval(
                    walk_values["walk_s"], workspace_interval_s
                )
                next_walk_ms = tick + math.ceil(pause_s * 1000)


def build_header(
    identity: Identity,
    facts: StaticFacts,
    collector: Collector,
    *,
    interval_s: int,
    workspace_interval_s: int,
    tool_version: str,
    started_at: float,
) -> dict[str, JsonValue]:
    """Return the header line, with the `HEADER_FIELDS` names in order.

    `started_at` is the start in seconds since the epoch; the header carries it
    as an ISO 8601 UTC string with milliseconds and a `Z`.
    """
    started = datetime.fromtimestamp(started_at, UTC).isoformat(
        timespec="milliseconds"
    )
    return {
        "kind": HEADER_KIND,
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version,
        "phase": identity.phase,
        "job_id": identity.job_id,
        "restart": identity.restart,
        "node": identity.node,
        "node_index": identity.node_index,
        "nodes": facts.nodes,
        "partition": facts.partition,
        "cpus_on_node": facts.cpus_on_node,
        "instance_type": facts.instance_type,
        "cpu_model": facts.cpu_model,
        "ncpus": facts.ncpus,
        "mem_total_bytes": facts.mem_total_bytes,
        "kernel": facts.kernel,
        "job_cgroup": collector.job_cgroup,
        "own_step_cgroup": collector.own_step_cgroup,
        "workspace_mount": collector.workspace_mount,
        "interval_s": interval_s,
        "workspace_interval_s": workspace_interval_s,
        "started_at": started.replace("+00:00", "Z"),
    }


def _positive_int(text: str) -> int:
    if _DIGITS.fullmatch(text) is None or int(text) < 1:
        raise argparse.ArgumentTypeError(
            f"must be an integer >= 1, got {text!r}"
        )
    return int(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpcmu-telemetry",
        description="Sample the resource use of one node into a staging file.",
    )
    parser.add_argument(
        "--staging", type=Path, required=True, help="staging directory"
    )
    parser.add_argument(
        "--phase", choices=PHASES, required=True, help="job phase"
    )
    parser.add_argument(
        "--workspace", type=Path, required=True, help="workspace directory"
    )
    parser.add_argument(
        "--interval",
        type=_positive_int,
        required=True,
        help="seconds between samples",
    )
    parser.add_argument(
        "--workspace-interval",
        type=_positive_int,
        required=True,
        help="seconds between workspace walks",
    )
    parser.add_argument(
        "--tool-version",
        required=True,
        help="hpc-model-utils version to record in the header",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/"), help=argparse.SUPPRESS
    )
    return parser


def _read_env(environ: Mapping[str, str], phase: str) -> Identity:
    """Return the identity of this attempt; `ValueError` names the variable."""
    numbers: dict[str, str] = {}
    for name, default in (
        ("SLURM_JOB_ID", None),
        ("SLURM_NODEID", None),
        ("SLURM_RESTART_COUNT", "0"),
    ):
        text = environ.get(name, default)
        if text is None:
            raise ValueError(f"{name}: not set")
        if _DIGITS.fullmatch(text) is None:
            raise ValueError(
                f"{name}: must be a non-negative integer in ASCII digits, "
                f"got {text!r}"
            )
        numbers[name] = text
    return Identity(
        phase,
        numbers["SLURM_JOB_ID"],
        int(numbers["SLURM_RESTART_COUNT"]),
        environ.get("SLURMD_NODENAME", os.uname().nodename),
        int(numbers["SLURM_NODEID"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    # First, so a stop during start-up stays pending for the first wait.
    signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    args = _build_parser().parse_args(argv)
    try:
        identity = _read_env(os.environ, args.phase)
    except ValueError as exc:
        sys.stderr.write(f"hpcmu-telemetry: {exc}\n")
        return 2
    started_at = time.time()
    errors: Errors = []
    try:
        os.nice(_NICE_INCREMENT)
    except OSError as exc:
        errors.append(("sampler", exc))
    facts, fact_errors = read_static_facts(
        args.root, os.environ, identity.node_index
    )
    collector = Collector(args.root, args.workspace)
    errors.extend(fact_errors)
    errors.extend(collector.startup_errors)
    writer = StagingWriter.create(
        args.staging,
        identity.phase,
        identity.job_id,
        identity.restart,
        identity.node_index,
        monotonic=time.monotonic,
    )
    writer.write_header(
        build_header(
            identity,
            facts,
            collector,
            interval_s=args.interval,
            workspace_interval_s=args.workspace_interval,
            tool_version=args.tool_version,
            started_at=started_at,
        )
    )
    # A limit reached here is read again by the first check of run_loop.
    _write_errors(writer, int(started_at * 1000), errors)
    code = run_loop(
        collector,
        writer,
        identity,
        interval_s=args.interval,
        workspace_interval_s=args.workspace_interval,
        walk_enabled=identity.node_index == 0,
        clock=LoopClock.system(),
    )
    if code == 1:
        sys.stderr.write(
            f"hpcmu-telemetry: {MAX_CONSECUTIVE_FAILURES} consecutive "
            f"staging failures: {writer.last_failure}\n"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
