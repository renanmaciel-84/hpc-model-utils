"""ADR-093 raw counters parsed from /proc text; ADR-079 error classes.

The parsers are pure: file text in, typed values out, nothing derived and
nothing caught. A malformed file raises one of `COLLECTOR_ERRORS`, and the
collector boundary is the only catch site, apart from `read_static_facts` at
start-up. The `list_*` functions walk the sysfs device tree under a root; the
collector runs them once, at construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Final, NamedTuple

COLLECTOR_ERRORS: Final = (OSError, ValueError, LookupError, ArithmeticError)

_DEVICE_LINE: Final = re.compile(
    r"device (\S+) mounted on (\S+) with fstype (\S+)(?: statvers=\S+)?"
)
_OCTAL_ESCAPE: Final = re.compile(r"\\([0-7]{3})")
_TASK_STEP: Final = re.compile(r"step_[0-9]+")
_CPUS_ITEM: Final = re.compile(r"([0-9]+)(?:\(x([0-9]+)\))?")

NON_TASK_STEPS: Final = frozenset(
    {"step_batch", "step_extern", "step_interactive"}
)


class CpuTimes(NamedTuple):
    user: int
    nice: int
    system: int
    idle: int
    iowait: int
    irq: int
    softirq: int
    steal: int


class ProcStat(NamedTuple):
    total: CpuTimes
    per_cpu: dict[int, CpuTimes]
    procs_running: int
    procs_blocked: int


class DiskCounters(NamedTuple):
    sectors_read: int
    sectors_written: int
    io_ticks_ms: int


class NetCounters(NamedTuple):
    rx_bytes: int
    tx_bytes: int
    errors: int


class EfaCounters(NamedTuple):
    rx_bytes: int
    tx_bytes: int


class MountEntry(NamedTuple):
    device: str
    mountpoint: str
    fstype: str
    lines: tuple[str, ...]


class NfsCounters(NamedTuple):
    normal_read: int
    normal_write: int
    direct_read: int
    direct_write: int
    server_read: int
    server_write: int
    read_ops: int
    read_rtt_ms: int
    write_ops: int
    write_rtt_ms: int
    total_ops: int


class PressureTotals(NamedTuple):
    some_total_us: int
    full_total_us: int | None


class StaticFacts(NamedTuple):
    instance_type: str | None
    cpu_model: str | None
    ncpus: int | None
    mem_total_bytes: int | None
    kernel: str | None
    partition: str | None
    nodes: int | None
    cpus_on_node: int | None


def read_text(root: Path, relpath: str) -> str:
    """Read `root / relpath` as UTF-8.

    `OSError` and `UnicodeDecodeError` propagate.
    """
    return (root / relpath).read_text(encoding="utf-8")


def parse_proc_stat(text: str) -> ProcStat:
    """Parse `/proc/stat`; the `guest` and `guest_nice` fields are ignored.

    Raises `ValueError` for a `cpu` or `cpuN` line with fewer than eight
    numeric fields, or a `procs_*` line without exactly one value; `KeyError`
    for a missing `cpu`, `cpuN`, `procs_running` or `procs_blocked` line.
    """
    total: CpuTimes | None = None
    per_cpu: dict[int, CpuTimes] = {}
    procs: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key = fields[0]
        if key == "cpu" or (key.startswith("cpu") and key[3:].isdecimal()):
            if len(fields) < 9:
                raise ValueError(f"{key}: fewer than 8 numeric fields")
            times = CpuTimes(*map(int, fields[1:9]))
            if key == "cpu":
                total = times
            else:
                per_cpu[int(key[3:])] = times
        elif key in ("procs_running", "procs_blocked"):
            if len(fields) != 2:
                raise ValueError(f"{key}: expected exactly one value")
            procs[key] = int(fields[1])
    if total is None:
        raise KeyError("cpu")
    if not per_cpu:
        raise KeyError("cpuN")
    return ProcStat(
        total, per_cpu, procs["procs_running"], procs["procs_blocked"]
    )


def parse_loadavg(text: str) -> float:
    """Return the 1-minute load average; empty text raises `ValueError`."""
    fields = text.split()
    if not fields:
        raise ValueError("loadavg: empty text")
    return float(fields[0])


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse `/proc/meminfo` into bytes, skipping lines without a `kB` unit.

    Raises `ValueError` for a `kB` line whose value is not an integer.
    """
    result: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.split()
        if len(value) == 2 and value[1] == "kB":
            result[key] = int(value[0]) * 1024
    return result


def parse_vmstat(text: str) -> dict[str, int]:
    """Parse `/proc/vmstat`.

    Raises `ValueError` for a line without exactly two fields or whose value
    is not an integer.
    """
    result: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"vmstat: expected 'key value', got {line!r}")
        result[fields[0]] = int(fields[1])
    return result


def parse_diskstats(text: str) -> dict[str, DiskCounters]:
    """Parse `/proc/diskstats` into counters keyed by device name.

    Raises `ValueError` naming the device for a line with fewer than 14
    fields, or for a counter that is not an integer.
    """
    result: dict[str, DiskCounters] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 14:
            name = fields[2] if len(fields) > 2 else line.strip()
            raise ValueError(f"{name}: fewer than 14 fields")
        result[fields[2]] = DiskCounters(
            int(fields[5]), int(fields[9]), int(fields[12])
        )
    return result


def parse_net_dev(text: str) -> dict[str, NetCounters]:
    """Parse `/proc/net/dev`; `errors` sums rx/tx errs and rx/tx drop.

    Lines without a `:`, the two header lines, are skipped. Raises
    `ValueError` naming the interface for a line with fewer than 16 fields,
    or for a counter that is not an integer.
    """
    result: dict[str, NetCounters] = {}
    for line in text.splitlines():
        name, colon, rest = line.partition(":")
        if not colon:
            continue
        name = name.strip()
        fields = rest.split()
        if len(fields) < 16:
            raise ValueError(f"{name}: fewer than 16 fields")
        v = list(map(int, fields[:16]))
        result[name] = NetCounters(v[0], v[8], v[2] + v[3] + v[10] + v[11])
    return result


def parse_mountstats(text: str) -> list[MountEntry]:
    """Split `/proc/self/mountstats` into one entry per `device` line.

    `device` and `mountpoint` have their octal escapes (`\\040`) decoded;
    `lines` holds the stripped, non-empty body lines of the entry. Raises
    `ValueError` quoting a `device` line that does not match, and for a body
    line before the first `device` line.
    """
    heads: list[tuple[str, str, str]] = []
    bodies: list[list[str]] = []
    for raw in text.splitlines():
        if raw.partition(" ")[0] == "device":
            match = _DEVICE_LINE.fullmatch(raw)
            if match is None:
                raise ValueError(f"mountstats: malformed device line {raw!r}")
            device, mountpoint = (
                _OCTAL_ESCAPE.sub(lambda m: chr(int(m[1], 8)), field)
                for field in match.group(1, 2)
            )
            heads.append((device, mountpoint, match[3]))
            bodies.append([])
        elif line := raw.strip():
            if not bodies:
                raise ValueError(
                    f"mountstats: line before the first device line {raw!r}"
                )
            bodies[-1].append(line)
    return [
        MountEntry(*head, tuple(body))
        for head, body in zip(heads, bodies, strict=True)
    ]


def select_mount(entries: Iterable[MountEntry], path: str) -> MountEntry | None:
    """Return the entry whose mountpoint is the longest path prefix of `path`.

    The prefix is by path component, so `/shared` does not contain
    `/shared-zfs/x`. Of two entries on one mountpoint the later wins, as it
    shadows the earlier. Returns `None` without a match; a relative `path`
    raises `ValueError`.
    """
    target = PurePosixPath(path)
    if not target.is_absolute():
        raise ValueError(f"select_mount: path is not absolute: {path!r}")
    best: MountEntry | None = None
    depth = 0
    for entry in entries:
        mountpoint = PurePosixPath(entry.mountpoint)
        if target.is_relative_to(mountpoint) and len(mountpoint.parts) >= depth:
            best, depth = entry, len(mountpoint.parts)
    return best


def parse_nfs_counters(entry: MountEntry) -> NfsCounters:
    """Read the raw NFS counters of one `nfs` or `nfs4` mount entry.

    The `bytes:` line gives its first six fields; each per-op line, those
    after `per-op statistics`, is `OP: ops trans timeouts bytes_sent
    bytes_recv queue rtt execute` and fields past the eighth are ignored.
    `total_ops` sums `ops` over every per-op line. Raises `ValueError` for
    another fstype, a `bytes:` line with fewer than 6 fields, or a per-op
    line with fewer than 8; `KeyError` for a missing `bytes:`, `READ:` or
    `WRITE:` line.
    """
    if entry.fstype not in ("nfs", "nfs4"):
        raise ValueError(
            f"{entry.mountpoint}: fstype {entry.fstype!r} is not NFS"
        )
    io: list[int] | None = None
    ops: dict[str, tuple[int, int]] = {}
    in_ops = False
    for line in entry.lines:
        fields = line.split()
        if line == "per-op statistics":
            in_ops = True
        elif in_ops and fields[0].endswith(":"):
            if len(fields) < 9:
                raise ValueError(f"{fields[0]} fewer than 8 fields")
            ops[fields[0]] = (int(fields[1]), int(fields[7]))
        elif fields[0] == "bytes:":
            if len(fields) < 7:
                raise ValueError("bytes: fewer than 6 fields")
            io = list(map(int, fields[1:7]))
    if io is None:
        raise KeyError("bytes:")
    return NfsCounters._make(
        (
            *io,
            *ops["READ:"],
            *ops["WRITE:"],
            sum(count for count, _ in ops.values()),
        )
    )


def parse_counter(text: str) -> int:
    """Parse a single-integer sysfs file; anything else raises `ValueError`."""
    return int(text.strip())


def read_efa_port(root: Path, port: str) -> EfaCounters:
    """Read `rx_bytes` and `tx_bytes` in the root-relative `port` directory.

    `port` is an entry of `list_efa_ports`. `OSError` and `ValueError`
    propagate.
    """
    return EfaCounters(
        parse_counter(read_text(root, f"{port}/rx_bytes")),
        parse_counter(read_text(root, f"{port}/tx_bytes")),
    )


def list_whole_disks(root: Path) -> tuple[str, ...]:
    """Return the sorted whole local disks under `root / sys/block`.

    A whole disk has a `device` entry and a `hidden` file that is absent or
    holds 0, which drops partitions, device-mapper and loop devices, and NVMe
    multipath paths. Returns `()` without `sys/block`; a `hidden` file that is
    not an integer raises `ValueError`.
    """
    base = root / "sys/block"
    if not base.is_dir():
        return ()
    names: list[str] = []
    for entry in base.iterdir():
        if not (entry / "device").exists():
            continue
        if (entry / "hidden").exists() and parse_counter(
            read_text(entry, "hidden")
        ):
            continue
        names.append(entry.name)
    return tuple(sorted(names))


def list_physical_netdevs(root: Path) -> tuple[str, ...]:
    """Return the sorted interfaces under `root / sys/class/net` with a device.

    Only physical NICs have a `device` entry. Returns `()` without the
    directory.
    """
    base = root / "sys/class/net"
    if not base.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in base.iterdir()
            if (entry / "device").exists()
        )
    )


def list_efa_ports(root: Path) -> tuple[str, ...]:
    """Return the sorted root-relative `hw_counters` directories of every port.

    Each entry is `sys/class/infiniband/<dev>/ports/<n>/hw_counters` in POSIX
    form. Returns `()` without an InfiniBand device.
    """
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.glob("sys/class/infiniband/*/ports/*/hw_counters")
            if path.is_dir()
        )
    )


def parse_proc_self_cgroup(text: str) -> str:
    """Return the path of the `0::` line of `/proc/self/cgroup`.

    That line is the cgroup v2 unified entry. Raises `KeyError("0::")`
    without one.
    """
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:]
    raise KeyError("0::")


def job_cgroup_paths(path: str) -> tuple[str | None, str | None]:
    """Return `(job, own_step)` from the sampler's own cgroup `path`.

    The job is never found by name, because Slurm 26.05 renames job
    directories. Both results are cgroup-relative and begin with `/`. The
    first component starting with `step_` decides:

    - `step_<n>`: `own_step` is the path through it and `job` is its parent;
    - `step_batch`, `step_extern`, `step_interactive` (`NON_TASK_STEPS`):
      `(parent, None)`;
    - none: `(None, None)`.

    Any other `step_` component, or one directly under `/`, raises
    `ValueError` quoting `path`.
    """
    parts = PurePosixPath(path).parts
    index = next(
        (i for i, part in enumerate(parts) if part.startswith("step_")), None
    )
    if index is None:
        return None, None
    step = parts[index]
    if index < 2 or not (_TASK_STEP.fullmatch(step) or step in NON_TASK_STEPS):
        raise ValueError(f"cgroup path: unusable step component in {path!r}")
    job = str(PurePosixPath(*parts[:index]))
    if step in NON_TASK_STEPS:
        return job, None
    return job, str(PurePosixPath(*parts[: index + 1]))


def parse_flat_keyed(text: str) -> dict[str, int]:
    """Parse `memory.stat`, `cpu.stat` or `memory.events`, all `key value`.

    Raises `ValueError` for a line without exactly two fields or whose value
    is not an integer.
    """
    result: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"cgroup file: expected 'key value', got {line!r}")
        result[fields[0]] = int(fields[1])
    return result


def parse_single_int(text: str) -> int:
    """Parse `memory.current`, `memory.peak` or `pids.current`.

    A value that is not an integer, such as `max`, raises `ValueError`.
    """
    return parse_counter(text)


def parse_pressure(text: str) -> PressureTotals:
    """Read the `total=` field of the `some` and `full` lines of a PSI file.

    A missing `full` line gives `None`. Raises `KeyError` for a missing `some`
    line, and `ValueError` for a `some` or `full` line without an integer
    `total=` field.
    """
    totals: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("some", "full"):
            continue
        stats = {k: v for k, _, v in (f.partition("=") for f in fields[1:])}
        if "total" not in stats:
            raise ValueError(f"pressure: no total= field in {line!r}")
        totals[fields[0]] = int(stats["total"])
    return PressureTotals(totals["some"], totals.get("full"))


def parse_cpu_model(cpuinfo: str) -> str:
    """Return the stripped value of the first `model name` line of cpuinfo.

    Raises `KeyError("model name")` without one.
    """
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    raise KeyError("model name")


def parse_cpus_per_node(value: str, node_index: int) -> int:
    """Return entry `node_index` of a `SLURM_JOB_CPUS_PER_NODE` value.

    The value is Slurm's compressed list: `192(x2)` expands to `[192, 192]`
    and `48,96(x2)` to `[48, 96, 96]`. A malformed item raises `ValueError`
    quoting it; an index outside the expanded list, a negative one included,
    raises `IndexError`.
    """
    expanded: list[int] = []
    for item in value.split(","):
        match = _CPUS_ITEM.fullmatch(item)
        if match is None:
            raise ValueError(f"cpus per node: malformed item {item!r}")
        expanded += [int(match[1])] * int(match[2] or 1)
    if not 0 <= node_index < len(expanded):
        raise IndexError(
            f"cpus per node: index {node_index} outside the {len(expanded)}"
            f" entries of {value!r}"
        )
    return expanded[node_index]


def _fact[T](
    errors: list[tuple[str, Exception]], read: Callable[[], T]
) -> T | None:
    try:
        return read()
    except COLLECTOR_ERRORS as exc:
        errors.append(("header", exc))
        return None


def read_static_facts(
    root: Path, env: Mapping[str, str], node_index: int
) -> tuple[StaticFacts, list[tuple[str, Exception]]]:
    """Read the per-node static facts of the staging header, once.

    Each fact is read independently. A read or parse that raises one of
    `COLLECTOR_ERRORS` leaves that fact `None` and appends `("header", exc)`
    to the returned list; any other exception propagates. `ncpus` counts the
    `cpuN` lines of `proc/stat`, `mem_total_bytes` is `MemTotal` in bytes, and
    the text facts are stripped. `partition`, `nodes` and `cpus_on_node` come
    from `env`, and an unset or empty variable gives `None` with no error.
    `node_index` is the caller's validated `SLURM_NODEID`.
    """
    errors: list[tuple[str, Exception]] = []
    partition = env.get("SLURM_JOB_PARTITION", "").strip() or None
    nodes_text = env.get("SLURM_JOB_NUM_NODES", "")
    cpus_text = env.get("SLURM_JOB_CPUS_PER_NODE", "")
    facts = StaticFacts(
        _fact(
            errors,
            lambda: read_text(
                root, "sys/devices/virtual/dmi/id/product_name"
            ).strip(),
        ),
        _fact(errors, lambda: parse_cpu_model(read_text(root, "proc/cpuinfo"))),
        _fact(
            errors,
            lambda: len(parse_proc_stat(read_text(root, "proc/stat")).per_cpu),
        ),
        _fact(
            errors,
            lambda: parse_meminfo(read_text(root, "proc/meminfo"))["MemTotal"],
        ),
        _fact(
            errors,
            lambda: read_text(root, "proc/sys/kernel/osrelease").strip(),
        ),
        partition,
        _fact(errors, lambda: int(nodes_text)) if nodes_text else None,
        _fact(errors, lambda: parse_cpus_per_node(cpus_text, node_index))
        if cpus_text
        else None,
    )
    return facts, errors
