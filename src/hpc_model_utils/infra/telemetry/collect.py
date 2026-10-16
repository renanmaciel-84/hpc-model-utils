"""ADR-093/ADR-095: derivations from two raw readings, and the collector.

Busy is user + nice + system + irq + softirq; steal is not busy. Idle and
iowait are one counter, because the kernel moves an idle period between them
at read time: only their sum is monotonic, and it can step back by one tick.
A negative iowait or idle delta is therefore never clamped on its own; only
the sum is, through `max(w, 0)`.

The `job_*` values subtract the sampler's own step from the job cgroup
(ADR-063), clamped at 0; the clamp is part of the meaning. The peak and the
pid count are never subtracted. The pid count is null, with no error line,
when the job cgroup has no `pids.current` (the controller is not enabled).
A rate is null on a missing baseline, a negative delta or `dt_s <= 0`.

The disk, interface and EFA rates sum over the devices present in both reads
and are null on an empty set. `disk_busy_max` is clamped to [0, 1], because
`io_ticks` is counted in jiffies and read on the sampler's clock. An NFS RTT
is latency per op, not a rate: it never divides by `dt_s`, and it is null
when its own op delta is 0, so the other NFS columns survive.

`Collector` composes the parsers and the derivations behind the ADR-079
boundary (R32, R35). It keeps the previous raw readings and writes final
values; one `dt_s` reaches every rate. Each source runs inside its own catch
of `COLLECTOR_ERRORS`, and a failure drops that source's previous reading, so
the next interval is null rather than a rate across the gap. A float that is
not finite becomes null (ADR-070), and the counters it came from are kept.

The workspace walk (R31, R53) is bounded by its own rule instead of the sample
budget. It lstat-s every entry below the workspace and follows no symlink; the
`.hpcmu/telemetry` directory, which the sampler itself writes, is not counted.
A walk that takes more than 5 % of its interval grows the next interval to 20
times the walk. The walk looks for a pending SIGTERM or SIGINT before it opens
each directory, because the sampler blocks both and a signal would otherwise
wait for the whole walk (ADR-079); a stopped walk returns no totals at all.
"""

from __future__ import annotations

import math
import os
import signal
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Final, NamedTuple

from hpc_model_utils.infra.telemetry.procfs import (
    COLLECTOR_ERRORS,
    CpuTimes,
    DiskCounters,
    EfaCounters,
    NetCounters,
    NfsCounters,
    PressureTotals,
    ProcStat,
    job_cgroup_paths,
    list_efa_ports,
    list_physical_netdevs,
    list_whole_disks,
    parse_diskstats,
    parse_flat_keyed,
    parse_loadavg,
    parse_meminfo,
    parse_mountstats,
    parse_net_dev,
    parse_nfs_counters,
    parse_pressure,
    parse_proc_self_cgroup,
    parse_proc_stat,
    parse_single_int,
    parse_vmstat,
    read_efa_port,
    read_text,
    select_mount,
)
from hpc_model_utils.infra.telemetry.schema import (
    SAMPLE_COLUMNS,
    WORKSPACE_COLUMNS,
)

Values = dict[str, float | int | None]
Errors = list[tuple[str, Exception]]

_CPU_KEYS: Final = (
    "cpu_busy_cores",
    "cpu_util",
    "cpu_user_cores",
    "cpu_system_cores",
    "cpu_iowait_cores",
    "cpu_steal_cores",
    "cpu_core_util_min",
    "cpu_core_util_p50",
    "cpu_core_util_max",
    "cpu_active_cores",
    "procs_running",
    "procs_blocked",
)
_VMSTAT_KEYS: Final = ("major_faults_per_s", "oom_kills")
_CGROUP_KEYS: Final = (
    "job_cpu_cores",
    "job_mem_bytes",
    "job_mem_anon_bytes",
    "job_mem_file_bytes",
    "job_mem_shmem_bytes",
    "job_mem_peak_bytes",
    "job_pids",
    "job_oom_kills",
)
_PRESSURE_KEYS: Final = (
    "psi_cpu_some",
    "psi_mem_some",
    "psi_mem_full",
    "psi_io_some",
    "psi_io_full",
)
_DISK_KEYS: Final = (
    "disk_read_bytes_per_s",
    "disk_write_bytes_per_s",
    "disk_busy_max",
)
_NFS_KEYS: Final = (
    "nfs_read_bytes_per_s",
    "nfs_write_bytes_per_s",
    "nfs_server_read_bytes_per_s",
    "nfs_server_write_bytes_per_s",
    "nfs_read_ops_per_s",
    "nfs_write_ops_per_s",
    "nfs_ops_per_s",
    "nfs_read_rtt_ms",
    "nfs_write_rtt_ms",
)
_NET_KEYS: Final = (
    "net_rx_bytes_per_s",
    "net_tx_bytes_per_s",
    "net_errors_per_s",
)
_EFA_KEYS: Final = ("efa_rx_bytes_per_s", "efa_tx_bytes_per_s")
_SECTOR_BYTES: Final = 512
_CGROUP_MOUNT: Final = "/sys/fs/cgroup"
_NFS_FSTYPES: Final = ("nfs", "nfs4")
_PRESSURE_RESOURCES: Final = ("cpu", "memory", "io")

COUNTER_SOURCES: Final = (
    "stat",
    "loadavg",
    "meminfo",
    "vmstat",
    "cgroup",
    "pressure",
    "diskstats",
    "mountstats",
    "netdev",
    "efa",
)
_COLUMNS_BY_SOURCE: Final = {
    source: tuple(c.name for c in SAMPLE_COLUMNS if c.source == source)
    for source in COUNTER_SOURCES
}
_COLUMNS: Final = tuple(
    c.name for c in SAMPLE_COLUMNS if c.source in COUNTER_SOURCES
)
_WORKSPACE_KEYS: Final = tuple(
    c.name for c in WORKSPACE_COLUMNS if c.source != "identity"
)
_ST_BLOCK_BYTES: Final = 512
_WALK_SHARE: Final = 0.05
_WALK_BACKOFF: Final = 20


class JobCgroup(NamedTuple):
    usage_usec: int
    memory_current: int
    memory_peak: int
    anon: int
    file: int
    shmem: int
    oom_kill: int
    pids: int | None


class StepCgroup(NamedTuple):
    usage_usec: int
    memory_current: int
    anon: int
    file: int
    shmem: int


class CgroupReading(NamedTuple):
    job: JobCgroup
    step: StepCgroup | None


_NO_STEP: Final = StepCgroup(0, 0, 0, 0, 0)


def _delta(a: CpuTimes, b: CpuTimes) -> CpuTimes:
    return CpuTimes(*(y - x for x, y in zip(a, b, strict=True)))


def _split(d: CpuTimes) -> tuple[int, int] | None:
    busy = d.user + d.nice + d.system + d.irq + d.softirq
    w = d.idle + d.iowait
    if min(d.user, d.nice, d.system, d.irq, d.softirq, d.steal) < 0 or w < -1:
        return None
    total = busy + d.steal + max(w, 0)
    return (busy, total) if total > 0 else None


def derive_cpu(prev: ProcStat | None, cur: ProcStat) -> Values:
    values: Values = dict.fromkeys(_CPU_KEYS)
    values["procs_running"] = cur.procs_running
    values["procs_blocked"] = cur.procs_blocked
    if prev is None:
        return values
    d = _delta(prev.total, cur.total)
    split = _split(d)
    if split is None:
        return values
    busy, t = split
    n = len(cur.per_cpu)
    values["cpu_busy_cores"] = busy / t * n
    values["cpu_util"] = busy / t
    values["cpu_user_cores"] = (d.user + d.nice) / t * n
    values["cpu_system_cores"] = (d.system + d.irq + d.softirq) / t * n
    if d.iowait >= 0 and d.idle >= 0:
        values["cpu_iowait_cores"] = d.iowait / t * n
    values["cpu_steal_cores"] = d.steal / t * n
    splits = (
        _split(_delta(prev.per_cpu[i], cur.per_cpu[i]))
        for i in prev.per_cpu.keys() & cur.per_cpu.keys()
    )
    shares = [b / t_i for b, t_i in filter(None, splits)]
    if shares:
        values["cpu_core_util_min"] = min(shares)
        values["cpu_core_util_p50"] = median(shares)
        values["cpu_core_util_max"] = max(shares)
        values["cpu_active_cores"] = sum(s >= 0.5 for s in shares)
    return values


def _count(prev: int, cur: int) -> int | None:
    d = cur - prev
    return d if d >= 0 else None


def _per(count: int | None, scale: float) -> float | None:
    return count / scale if count is not None and scale > 0 else None


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def derive_memory(meminfo: Mapping[str, int]) -> Values:
    return {
        "mem_used_bytes": meminfo["MemTotal"] - meminfo["MemAvailable"],
        "mem_available_bytes": meminfo["MemAvailable"],
        "mem_anon_bytes": meminfo["AnonPages"],
        "mem_cached_bytes": meminfo["Cached"],
        "mem_shmem_bytes": meminfo["Shmem"],
        "mem_dirty_bytes": meminfo["Dirty"],
        "mem_writeback_bytes": meminfo["Writeback"],
        "mem_slab_bytes": meminfo["Slab"],
        "swap_used_bytes": meminfo["SwapTotal"] - meminfo["SwapFree"],
    }


def derive_vmstat(
    prev: Mapping[str, int] | None, cur: Mapping[str, int], dt_s: float
) -> Values:
    values: Values = dict.fromkeys(_VMSTAT_KEYS)
    if prev is not None:
        values["major_faults_per_s"] = _per(
            _count(prev["pgmajfault"], cur["pgmajfault"]), dt_s
        )
        values["oom_kills"] = _count(prev["oom_kill"], cur["oom_kill"])
    return values


def derive_job(
    prev: CgroupReading | None, cur: CgroupReading | None, dt_s: float
) -> Values:
    values: Values = dict.fromkeys(_CGROUP_KEYS)
    if cur is None:
        return values
    job = cur.job
    own = cur.step or _NO_STEP
    values["job_mem_bytes"] = max(0, job.memory_current - own.memory_current)
    values["job_mem_anon_bytes"] = max(0, job.anon - own.anon)
    values["job_mem_file_bytes"] = max(0, job.file - own.file)
    values["job_mem_shmem_bytes"] = max(0, job.shmem - own.shmem)
    values["job_mem_peak_bytes"] = job.memory_peak
    values["job_pids"] = job.pids
    if prev is None:
        return values
    values["job_oom_kills"] = _count(prev.job.oom_kill, job.oom_kill)
    if (prev.step is None) == (cur.step is None):
        busy = _count(prev.job.usage_usec, job.usage_usec)
        own_busy = _count((prev.step or _NO_STEP).usage_usec, own.usage_usec)
        if busy is not None and own_busy is not None:
            values["job_cpu_cores"] = _per(max(0, busy - own_busy), 1e6 * dt_s)
    return values


def _totals(totals: Mapping[str, PressureTotals]) -> list[int | None]:
    cpu, memory, io = totals["cpu"], totals["memory"], totals["io"]
    return [
        cpu.some_total_us,
        memory.some_total_us,
        memory.full_total_us,
        io.some_total_us,
        io.full_total_us,
    ]


def _fraction(prev: int | None, cur: int | None, scale: float) -> float | None:
    if prev is None or cur is None:
        return None
    rate = _per(_count(prev, cur), scale)
    return None if rate is None else _clamp01(rate)


def derive_pressure(
    prev: Mapping[str, PressureTotals] | None,
    cur: Mapping[str, PressureTotals],
    dt_s: float,
) -> Values:
    if prev is None:
        return dict.fromkeys(_PRESSURE_KEYS)
    scale = 1e6 * dt_s
    return {
        key: _fraction(p, c, scale)
        for key, p, c in zip(
            _PRESSURE_KEYS, _totals(prev), _totals(cur), strict=True
        )
    }


def _counts(
    prev: tuple[int, ...], cur: tuple[int, ...]
) -> tuple[int, ...] | None:
    d = tuple(y - x for x, y in zip(prev, cur, strict=True))
    return d if min(d) >= 0 else None


def _deltas(
    prev: Mapping[str, tuple[int, ...]] | None,
    cur: Mapping[str, tuple[int, ...]],
    names: Iterable[str],
) -> list[tuple[int, ...]] | None:
    if prev is None:
        return None
    rows: list[tuple[int, ...]] = []
    for name in names:
        if name in prev and name in cur:
            row = _counts(prev[name], cur[name])
            if row is None:
                return None
            rows.append(row)
    return rows or None


def derive_disk(
    prev: Mapping[str, DiskCounters] | None,
    cur: Mapping[str, DiskCounters],
    disks: tuple[str, ...],
    dt_s: float,
) -> Values:
    rows = _deltas(prev, cur, disks)
    if rows is None:
        return dict.fromkeys(_DISK_KEYS)
    read, written, ticks = zip(*rows, strict=True)
    busy = _per(max(ticks), 1000 * dt_s)
    return {
        "disk_read_bytes_per_s": _per(sum(read) * _SECTOR_BYTES, dt_s),
        "disk_write_bytes_per_s": _per(sum(written) * _SECTOR_BYTES, dt_s),
        "disk_busy_max": None if busy is None else _clamp01(busy),
    }


def derive_nfs(
    prev: NfsCounters | None, cur: NfsCounters | None, dt_s: float
) -> Values:
    counts: tuple[int, ...] | None = None
    if prev is not None and cur is not None and dt_s > 0:
        counts = _counts(prev, cur)
    if counts is None:
        return dict.fromkeys(_NFS_KEYS)
    d = NfsCounters(*counts)
    return {
        "nfs_read_bytes_per_s": _per(d.normal_read + d.direct_read, dt_s),
        "nfs_write_bytes_per_s": _per(d.normal_write + d.direct_write, dt_s),
        "nfs_server_read_bytes_per_s": _per(d.server_read, dt_s),
        "nfs_server_write_bytes_per_s": _per(d.server_write, dt_s),
        "nfs_read_ops_per_s": _per(d.read_ops, dt_s),
        "nfs_write_ops_per_s": _per(d.write_ops, dt_s),
        "nfs_ops_per_s": _per(d.total_ops, dt_s),
        "nfs_read_rtt_ms": _per(d.read_rtt_ms, d.read_ops),
        "nfs_write_rtt_ms": _per(d.write_rtt_ms, d.write_ops),
    }


def derive_net(
    prev: Mapping[str, NetCounters] | None,
    cur: Mapping[str, NetCounters],
    ifaces: tuple[str, ...],
    dt_s: float,
) -> Values:
    rows = _deltas(prev, cur, ifaces)
    if rows is None:
        return dict.fromkeys(_NET_KEYS)
    rx, tx, errors = zip(*rows, strict=True)
    return {
        "net_rx_bytes_per_s": _per(sum(rx), dt_s),
        "net_tx_bytes_per_s": _per(sum(tx), dt_s),
        "net_errors_per_s": _per(sum(errors), dt_s),
    }


def derive_efa(
    prev: Mapping[str, EfaCounters] | None,
    cur: Mapping[str, EfaCounters],
    dt_s: float,
) -> Values:
    rows = _deltas(prev, cur, cur)
    if rows is None:
        return dict.fromkeys(_EFA_KEYS)
    rx, tx = zip(*rows, strict=True)
    return {
        "efa_rx_bytes_per_s": _per(sum(rx), dt_s),
        "efa_tx_bytes_per_s": _per(sum(tx), dt_s),
    }


def _read_job(directory: Path) -> JobCgroup:
    memory = parse_flat_keyed(read_text(directory, "memory.stat"))
    return JobCgroup(
        usage_usec=parse_flat_keyed(read_text(directory, "cpu.stat"))[
            "usage_usec"
        ],
        memory_current=parse_single_int(read_text(directory, "memory.current")),
        memory_peak=parse_single_int(read_text(directory, "memory.peak")),
        anon=memory["anon"],
        file=memory["file"],
        shmem=memory["shmem"],
        oom_kill=parse_flat_keyed(read_text(directory, "memory.events"))[
            "oom_kill"
        ],
        pids=_read_pids(directory),
    )


def _read_pids(directory: Path) -> int | None:
    try:
        text = read_text(directory, "pids.current")
    except FileNotFoundError:
        return None
    return parse_single_int(text)


def _read_step(directory: Path) -> StepCgroup:
    memory = parse_flat_keyed(read_text(directory, "memory.stat"))
    return StepCgroup(
        usage_usec=parse_flat_keyed(read_text(directory, "cpu.stat"))[
            "usage_usec"
        ],
        memory_current=parse_single_int(read_text(directory, "memory.current")),
        anon=memory["anon"],
        file=memory["file"],
        shmem=memory["shmem"],
    )


class WalkTotals(NamedTuple):
    apparent_bytes: int
    allocated_bytes: int
    files: int
    dirs: int
    walk_s: float


def _stop_pending() -> bool:
    return bool({signal.SIGTERM, signal.SIGINT} & signal.sigpending())


def walk_workspace(
    workspace: Path,
    skip: Path,
    monotonic: Callable[[], float] = time.monotonic,
    stop_pending: Callable[[], bool] | None = None,
) -> WalkTotals | None:
    """Total every entry below `workspace`, `skip` and its content excluded.

    Returns `None`, with no partial totals, when `stop_pending()` is true
    before a directory is opened; it defaults to `_stop_pending`. An entry
    that vanishes (`FileNotFoundError` on its stat) or a subdirectory that
    vanishes before it is listed is skipped. Any other `OSError`, and a
    missing `workspace`, propagates.
    """
    stop = _stop_pending if stop_pending is None else stop_pending
    skip_path = os.fspath(skip)
    root = os.fspath(workspace)
    stack = [root]
    apparent = allocated = files = dirs = 0
    start = monotonic()
    while stack:
        if stop():
            return None
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.path == skip_path:
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    apparent += st.st_size
                    allocated += st.st_blocks * _ST_BLOCK_BYTES
                    if is_dir:
                        dirs += 1
                        stack.append(entry.path)
                    else:
                        files += 1
        except FileNotFoundError:
            if directory == root:
                raise
    return WalkTotals(apparent, allocated, files, dirs, monotonic() - start)


def filesystem_usage(path: Path) -> tuple[int, int]:
    st = os.statvfs(path)
    return (st.f_blocks - st.f_bfree) * st.f_frsize, st.f_bavail * st.f_frsize


def next_walk_interval(walk_s: float | None, workspace_interval: int) -> float:
    if walk_s is None or walk_s <= _WALK_SHARE * workspace_interval:
        return workspace_interval
    return _WALK_BACKOFF * walk_s


def _guard[T](
    errors: Errors, source: str, read: Callable[[], T], default: T
) -> T:
    try:
        return read()
    except COLLECTOR_ERRORS as exc:
        errors.append((source, exc))
        return default


def _find_mount(root: Path, workspace: Path) -> dict[str, str] | None:
    entry = select_mount(
        parse_mountstats(read_text(root, "proc/self/mountstats")),
        os.path.realpath(workspace),
    )
    if entry is None:
        return None
    return {"mountpoint": entry.mountpoint, "fstype": entry.fstype}


_NO_CGROUP: Final[tuple[str | None, str | None]] = (None, None)
_Step = tuple[Callable[[float | None], Values | None], Callable[[], None]]


def _source[R](
    read: Callable[[], R], derive: Callable[[R | None, R, float], Values]
) -> _Step:
    """Bind a reader to its derivation and to the reading they share.

    Returns `(run, reset)`. `run(dt_s)` reads, derives against the previous
    reading unless `dt_s` is `None`, then keeps the new reading; it returns
    `None` when it derived nothing. A raise leaves the old reading in place,
    and `reset()` drops it.
    """
    prev: list[R] = []

    def run(dt_s: float | None) -> Values | None:
        cur = read()
        values = None
        if dt_s is not None:
            values = derive(prev[0] if prev else None, cur, dt_s)
        prev[:] = [cur]
        return values

    return run, prev.clear


class Collector:
    """Read the ten counter sources under `root` and derive their columns.

    `root` is `/` in production and a fixture tree in tests. The constructor
    resolves what is static once: `job_cgroup` and `own_step_cgroup` as
    logical `/sys/fs/cgroup/...` paths or `None` (ADR-063), the device sets,
    and `workspace_mount` as `{"mountpoint": ..., "fstype": ...}` or `None`.
    A failed lookup is recorded in `startup_errors` and leaves its result
    `None` or empty. A source with nothing to read is not read and yields
    null values without an error.

    `baseline()` reads every source and derives nothing. `sample(dt_s)` reads
    and derives every source, in `COUNTER_SOURCES` order, and passes the same
    `dt_s` to every rate. A source that raises one of `COLLECTOR_ERRORS` has
    null values, loses its previous reading and adds `(source, exc)` to the
    errors; any other exception propagates.

    `walk()` is separate from `sample`: it totals `workspace` and reads its
    filesystem usage, and the caller decides when (R31).
    """

    def __init__(self, root: Path, workspace: Path) -> None:
        errors: Errors = []
        job, step = _guard(
            errors,
            "cgroup",
            lambda: job_cgroup_paths(
                parse_proc_self_cgroup(read_text(root, "proc/self/cgroup"))
            ),
            _NO_CGROUP,
        )
        self._root = root
        self._workspace = workspace
        self.job_cgroup = None if job is None else _CGROUP_MOUNT + job
        self.own_step_cgroup = None if step is None else _CGROUP_MOUNT + step
        self._disks: tuple[str, ...] = _guard(
            errors, "diskstats", lambda: list_whole_disks(root), ()
        )
        self._netdevs: tuple[str, ...] = _guard(
            errors, "netdev", lambda: list_physical_netdevs(root), ()
        )
        self._efa_ports: tuple[str, ...] = _guard(
            errors, "efa", lambda: list_efa_ports(root), ()
        )
        self.workspace_mount: dict[str, str] | None = _guard(
            errors, "mountstats", lambda: _find_mount(root, workspace), None
        )
        self.startup_errors = tuple(errors)
        self._steps = self._build()

    def baseline(self) -> Errors:
        return self._cycle(None)[1]

    def sample(self, dt_s: float) -> tuple[Values, Errors]:
        values, errors = self._cycle(dt_s)
        return {name: values[name] for name in _COLUMNS}, errors

    def walk(self) -> tuple[Values, Errors] | None:
        """Return the seven `WORKSPACE_COLUMNS` values and the errors.

        Returns `None`, without reading `statvfs`, for a walk stopped by a
        pending SIGTERM or SIGINT. The `walk` and `statvfs` sources fail
        apart: a failed one has null values and adds `(source, exc)`.
        """
        values: Values = dict.fromkeys(_WORKSPACE_KEYS)
        errors: Errors = []
        skip = self._workspace / ".hpcmu" / "telemetry"
        try:
            totals = walk_workspace(self._workspace, skip)
        except COLLECTOR_ERRORS as exc:
            errors.append(("walk", exc))
        else:
            if totals is None:
                return None
            values.update(totals._asdict())
        try:
            used, avail = filesystem_usage(self._workspace)
        except COLLECTOR_ERRORS as exc:
            errors.append(("statvfs", exc))
        else:
            values["fs_used_bytes"] = used
            values["fs_avail_bytes"] = avail
        return values, errors

    def _cycle(self, dt_s: float | None) -> tuple[Values, Errors]:
        values: Values = {}
        errors: Errors = []
        for source in COUNTER_SOURCES:
            run, reset = self._steps[source]
            part: Values = dict.fromkeys(_COLUMNS_BY_SOURCE[source])
            try:
                derived = run(dt_s)
            except COLLECTOR_ERRORS as exc:
                reset()
                errors.append((source, exc))
            else:
                if derived is not None:
                    part = derived
                bad = [
                    name
                    for name, value in part.items()
                    if isinstance(value, float) and not math.isfinite(value)
                ]
                part.update(dict.fromkeys(bad))
                errors.extend(
                    (source, ValueError(f"non-finite {name}")) for name in bad
                )
            values.update(part)
        return values, errors

    def _build(self) -> dict[str, _Step]:
        return {
            "stat": _source(
                self._read_stat,
                lambda prev, cur, dt_s: derive_cpu(prev, cur),
            ),
            "loadavg": _source(
                self._read_loadavg,
                lambda prev, cur, dt_s: {"load1": cur},
            ),
            "meminfo": _source(
                self._read_meminfo,
                lambda prev, cur, dt_s: derive_memory(cur),
            ),
            "vmstat": _source(
                self._read_vmstat,
                lambda prev, cur, dt_s: derive_vmstat(prev, cur, dt_s),
            ),
            "cgroup": _source(
                self._read_cgroup,
                lambda prev, cur, dt_s: derive_job(prev, cur, dt_s),
            ),
            "pressure": _source(
                self._read_pressure,
                lambda prev, cur, dt_s: derive_pressure(prev, cur, dt_s),
            ),
            "diskstats": _source(
                self._read_diskstats,
                lambda prev, cur, dt_s: derive_disk(
                    prev, cur, self._disks, dt_s
                ),
            ),
            "mountstats": _source(
                self._read_mountstats,
                lambda prev, cur, dt_s: derive_nfs(prev, cur, dt_s),
            ),
            "netdev": _source(
                self._read_netdev,
                lambda prev, cur, dt_s: derive_net(
                    prev, cur, self._netdevs, dt_s
                ),
            ),
            "efa": _source(
                self._read_efa,
                lambda prev, cur, dt_s: derive_efa(prev, cur, dt_s),
            ),
        }

    def _read_stat(self) -> ProcStat:
        return parse_proc_stat(read_text(self._root, "proc/stat"))

    def _read_loadavg(self) -> float:
        return parse_loadavg(read_text(self._root, "proc/loadavg"))

    def _read_meminfo(self) -> dict[str, int]:
        return parse_meminfo(read_text(self._root, "proc/meminfo"))

    def _read_vmstat(self) -> dict[str, int]:
        return parse_vmstat(read_text(self._root, "proc/vmstat"))

    def _read_cgroup(self) -> CgroupReading | None:
        if self.job_cgroup is None:
            return None
        step = self.own_step_cgroup
        return CgroupReading(
            _read_job(self._root / self.job_cgroup.removeprefix("/")),
            None
            if step is None
            else _read_step(self._root / step.removeprefix("/")),
        )

    def _read_pressure(self) -> dict[str, PressureTotals]:
        return {
            resource: parse_pressure(
                read_text(self._root, f"proc/pressure/{resource}")
            )
            for resource in _PRESSURE_RESOURCES
        }

    def _read_diskstats(self) -> dict[str, DiskCounters]:
        if not self._disks:
            return {}
        return parse_diskstats(read_text(self._root, "proc/diskstats"))

    def _read_mountstats(self) -> NfsCounters | None:
        mount = self.workspace_mount
        if mount is None or mount["fstype"] not in _NFS_FSTYPES:
            return None
        entries = parse_mountstats(
            read_text(self._root, "proc/self/mountstats")
        )
        for entry in reversed(entries):
            if entry.mountpoint == mount["mountpoint"]:
                return parse_nfs_counters(entry)
        raise LookupError(f"mountstats: no entry for {mount['mountpoint']}")

    def _read_netdev(self) -> dict[str, NetCounters]:
        if not self._netdevs:
            return {}
        return parse_net_dev(read_text(self._root, "proc/net/dev"))

    def _read_efa(self) -> dict[str, EfaCounters]:
        return {
            port: read_efa_port(self._root, port) for port in self._efa_ports
        }
