"""ADR-093/ADR-079: procfs parsers over fixture bundles and malformed text."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from hpc_model_utils.infra.telemetry import procfs
from hpc_model_utils.infra.telemetry.procfs import (
    COLLECTOR_ERRORS,
    NON_TASK_STEPS,
    CpuTimes,
    DiskCounters,
    EfaCounters,
    MountEntry,
    NetCounters,
    NfsCounters,
    PressureTotals,
    ProcStat,
    StaticFacts,
    job_cgroup_paths,
    list_efa_ports,
    list_physical_netdevs,
    list_whole_disks,
    parse_counter,
    parse_cpu_model,
    parse_cpus_per_node,
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
    read_static_facts,
    read_text,
    select_mount,
)
from tests.support.telemetry_tree import load_bundle, materialize

BUNDLES = ("workstation", "synthetic")

_CPU = "cpu  1 2 3 4 5 6 7 8\n"
_CPU0 = "cpu0 1 2 3 4 5 6 7 8\n"
_PROCS = "procs_running 1\nprocs_blocked 0\n"

_SYNTHETIC_STAT = {
    0: ProcStat(
        CpuTimes(4000, 400, 1000, 12000, 200, 100, 100, 200),
        dict.fromkeys(range(4), CpuTimes(1000, 100, 250, 3000, 50, 25, 25, 50)),
        3,
        1,
    ),
    1: ProcStat(
        CpuTimes(4550, 450, 1125, 13150, 250, 125, 125, 225),
        {
            0: CpuTimes(1300, 150, 300, 3050, 50, 50, 50, 50),
            1: CpuTimes(1200, 100, 300, 3200, 75, 25, 25, 75),
            2: CpuTimes(1050, 100, 275, 3400, 75, 25, 25, 50),
            3: CpuTimes(1000, 100, 250, 3500, 50, 25, 25, 50),
        },
        4,
        0,
    ),
}

_SYNTHETIC_PORT = "sys/class/infiniband/efa_0/ports/1/hw_counters"

_DEVICE = "device fsx:/a mounted on /a with fstype nfs4 statvers=1.1\n"
_NFS_LINES = (
    "bytes:\t1 2 3 4 5 6 7 8",
    "events:\t1000 2 3",
    "per-op statistics",
    "NULL: 0 0 0 0 0 0 0 0 0",
    "READ: 10 11 12 13 14 15 16 17 0",
    "WRITE: 20 21 22 23 24 25 26 27 0",
    "COMMIT: 5 0 0 0 0 0 0 0 0",
)


def _text(bundle: str, snapshot: int, relpath: str) -> str:
    return load_bundle(bundle).snapshots[snapshot].files[relpath]


def test_collector_errors_is_the_adr_079_tuple() -> None:
    assert COLLECTOR_ERRORS == (
        OSError,
        ValueError,
        LookupError,
        ArithmeticError,
    )


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_proc_stat_bundle_returns_typed_counters(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/stat")
    stat = parse_proc_stat(text)
    cpu_lines = [
        line
        for line in text.splitlines()
        if line.startswith("cpu") and line[3:4].isdigit()
    ]
    assert isinstance(stat.total, CpuTimes)
    assert len(stat.per_cpu) == len(cpu_lines) > 0
    assert all(isinstance(t, CpuTimes) for t in stat.per_cpu.values())
    assert all(isinstance(v, int) for v in stat.total)
    assert stat.procs_running >= 0
    assert stat.procs_blocked >= 0


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_loadavg_bundle_returns_non_negative_float(
    bundle: str, snapshot: int
) -> None:
    load = parse_loadavg(_text(bundle, snapshot, "proc/loadavg"))
    assert isinstance(load, float)
    assert load >= 0.0


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_meminfo_bundle_returns_bytes_for_kb_lines_only(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/meminfo")
    meminfo = parse_meminfo(text)
    kb_lines = [line for line in text.splitlines() if line.endswith(" kB")]
    assert len(meminfo) == len(kb_lines)
    assert all(value % 1024 == 0 for value in meminfo.values())
    assert 0 < meminfo["MemAvailable"] <= meminfo["MemTotal"]
    assert "HugePages_Total" not in meminfo


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_vmstat_bundle_returns_every_line(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/vmstat")
    vmstat = parse_vmstat(text)
    assert len(vmstat) == len(text.splitlines())
    assert vmstat["pgmajfault"] >= 0
    assert vmstat["oom_kill"] >= 0


@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_proc_stat_synthetic_returns_exact_counters(
    snapshot: int,
) -> None:
    stat = parse_proc_stat(_text("synthetic", snapshot, "proc/stat"))
    assert stat == _SYNTHETIC_STAT[snapshot]


@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_proc_stat_synthetic_total_equals_sum_of_cpus(
    snapshot: int,
) -> None:
    stat = parse_proc_stat(_text("synthetic", snapshot, "proc/stat"))
    assert stat.total == CpuTimes(*map(sum, zip(*stat.per_cpu.values())))


def test_parse_proc_stat_synthetic_snapshot_1_lists_four_cpus() -> None:
    stat = parse_proc_stat(_text("synthetic", 1, "proc/stat"))
    assert sorted(stat.per_cpu) == [0, 1, 2, 3]
    assert (stat.procs_running, stat.procs_blocked) == (4, 0)


@pytest.mark.parametrize(
    ("snapshot", "total", "free"),
    [(0, 16384000000, 2099200000), (1, 16384000000, 2048000000)],
)
def test_parse_meminfo_synthetic_returns_exact_bytes(
    snapshot: int, total: int, free: int
) -> None:
    meminfo = parse_meminfo(_text("synthetic", snapshot, "proc/meminfo"))
    assert meminfo["MemTotal"] == total
    assert meminfo["MemFree"] == free
    assert "HugePages_Total" not in meminfo


@pytest.mark.parametrize(("snapshot", "faults"), [(0, 1000), (1, 1050)])
def test_parse_vmstat_synthetic_returns_exact_counters(
    snapshot: int, faults: int
) -> None:
    vmstat = parse_vmstat(_text("synthetic", snapshot, "proc/vmstat"))
    assert vmstat["pgmajfault"] == faults
    assert vmstat["oom_kill"] == snapshot


@pytest.mark.parametrize(("snapshot", "load"), [(0, 1.4), (1, 1.5)])
def test_parse_loadavg_synthetic_returns_first_field(
    snapshot: int, load: float
) -> None:
    text = _text("synthetic", snapshot, "proc/loadavg")
    assert parse_loadavg(text) == load


@pytest.mark.parametrize("guest", ["", " 9", " 9 10"])
def test_parse_proc_stat_guest_fields_are_ignored(guest: str) -> None:
    text = f"cpu  1 2 3 4 5 6 7 8{guest}\ncpu0 1 2 3 4 5 6 7 8{guest}\n"
    stat = parse_proc_stat(text + _PROCS)
    assert stat.total == CpuTimes(1, 2, 3, 4, 5, 6, 7, 8)
    assert stat.per_cpu == {0: CpuTimes(1, 2, 3, 4, 5, 6, 7, 8)}


def test_parse_proc_stat_other_lines_are_ignored() -> None:
    text = f"{_CPU}intr 0 1\n\nctxt 5\nbtime 1\nsoftirq 1 2\n{_CPU0}{_PROCS}"
    assert parse_proc_stat(text) == ProcStat(
        CpuTimes(1, 2, 3, 4, 5, 6, 7, 8),
        {0: CpuTimes(1, 2, 3, 4, 5, 6, 7, 8)},
        1,
        0,
    )


def test_parse_meminfo_unitless_line_is_skipped() -> None:
    text = (
        "HugePages_Total:       0\n"
        "VmallocTotal:34359738367 kB\n"
        "MemTotal: 2 kB\n"
    )
    assert parse_meminfo(text) == {
        "VmallocTotal": 34359738367 * 1024,
        "MemTotal": 2048,
    }


@pytest.mark.parametrize(
    ("parser", "text", "error", "match"),
    [
        (parse_proc_stat, "cpu 1 2 3\n", ValueError, "cpu: fewer than 8"),
        (
            parse_proc_stat,
            f"{_CPU}cpu1 1 2 3\n",
            ValueError,
            "cpu1: fewer than 8",
        ),
        (
            parse_proc_stat,
            "cpu  1 2 3 4 5 6 7 x\n",
            ValueError,
            "invalid literal",
        ),
        (parse_proc_stat, "intr 0\n", KeyError, "'cpu'"),
        (parse_proc_stat, f"{_CPU}{_PROCS}", KeyError, "cpuN"),
        (
            parse_proc_stat,
            f"{_CPU}{_CPU0}procs_blocked 0\n",
            KeyError,
            "procs_running",
        ),
        (
            parse_proc_stat,
            f"{_CPU}{_CPU0}procs_running 1\n",
            KeyError,
            "procs_blocked",
        ),
        (
            parse_proc_stat,
            f"{_CPU}{_CPU0}procs_running\nprocs_blocked 0\n",
            ValueError,
            "procs_running: expected",
        ),
        (parse_loadavg, "", ValueError, "empty"),
        (parse_loadavg, " \n", ValueError, "empty"),
        (parse_loadavg, "abc 1 2\n", ValueError, "could not convert"),
        (parse_meminfo, "MemTotal: abc kB\n", ValueError, "invalid literal"),
        (parse_vmstat, "pgfault\n", ValueError, "expected 'key value'"),
        (parse_vmstat, "pgfault 1 2\n", ValueError, "expected 'key value'"),
        (parse_vmstat, "pgfault x\n", ValueError, "invalid literal"),
        (
            parse_diskstats,
            "259 0 nvme0n1 0 0 5 0 0 0 9 0 0 12\n",
            ValueError,
            "nvme0n1: fewer than 14",
        ),
        (parse_diskstats, "1 2\n", ValueError, "1 2: fewer than 14"),
        (
            parse_diskstats,
            "259 0 sda 0 0 x 0 0 0 9 0 0 12 0\n",
            ValueError,
            "invalid literal",
        ),
        (parse_net_dev, "eth0: 1 2 3\n", ValueError, "eth0: fewer than 16"),
        (parse_net_dev, "eth0:\n", ValueError, "eth0: fewer than 16"),
        (
            parse_net_dev,
            "eth0: 1 2 x 4 5 6 7 8 9 10 11 12 13 14 15 16\n",
            ValueError,
            "invalid literal",
        ),
        (parse_counter, "x", ValueError, "invalid literal"),
        (parse_counter, "", ValueError, "invalid literal"),
        (parse_counter, "1.5\n", ValueError, "invalid literal"),
        (
            parse_mountstats,
            "device nfs mounted on /m\n",
            ValueError,
            r"malformed device line 'device nfs mounted on /m'",
        ),
        (
            parse_mountstats,
            "device a mounted on /m with fstype\n",
            ValueError,
            "malformed device line",
        ),
        (
            parse_mountstats,
            "device a mounted on /m with fstype ext4 extra\n",
            ValueError,
            "malformed device line",
        ),
        (
            parse_mountstats,
            "device a mounted on /m with fstype nfs4 statvers=\n",
            ValueError,
            "malformed device line",
        ),
        (
            parse_mountstats,
            "device\n",
            ValueError,
            "malformed device line",
        ),
        (
            parse_mountstats,
            f"\topts:\trw\n{_DEVICE}",
            ValueError,
            "before the first device line",
        ),
    ],
)
def test_parsers_malformed_text_raise_collector_error(
    parser: Callable[[str], object],
    text: str,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match) as excinfo:
        parser(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


def test_read_text_returns_the_file_under_root(tmp_path: Path) -> None:
    bundle = load_bundle("synthetic")
    root = materialize(bundle, 1, tmp_path)
    expected = bundle.snapshots[1].files["proc/stat"]
    assert read_text(root, "proc/stat") == expected
    assert parse_proc_stat(read_text(root, "proc/stat")) == _SYNTHETIC_STAT[1]


def test_read_text_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text(tmp_path, "proc/stat")


def test_read_text_invalid_utf8_raises_unicode_decode_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "stat").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError) as excinfo:
        read_text(tmp_path, "stat")
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


def _sysfs_entry(
    root: Path, base: str, name: str, *, link: bool = True, hidden: str = ""
) -> None:
    entry = root / base / name
    entry.mkdir(parents=True)
    if link:
        (entry / "device").mkdir()
    if hidden:
        (entry / "hidden").write_text(hidden, encoding="utf-8")


# [T7] diskstats columns and the bundle shape are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_diskstats_bundle_returns_every_device(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/diskstats")
    disks = parse_diskstats(text)
    assert len(disks) == len(text.splitlines()) > 0
    assert all(isinstance(v, int) for c in disks.values() for v in c)


# [T7] net/dev columns and the bundle shape are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_net_dev_bundle_returns_every_interface(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/net/dev")
    nets = parse_net_dev(text)
    assert len(nets) == sum(":" in line for line in text.splitlines()) > 0
    assert all(name and name == name.strip() for name in nets)
    assert all(v >= 0 for c in nets.values() for v in c)


# [T7] the device-link and hidden-flag filters are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
def test_discovery_bundle_devices_appear_in_the_counters(
    tmp_path: Path, bundle: str
) -> None:
    root = materialize(load_bundle(bundle), 1, tmp_path)
    disks = list_whole_disks(root)
    nets = list_physical_netdevs(root)
    assert disks
    assert nets
    assert set(disks) <= set(parse_diskstats(read_text(root, "proc/diskstats")))
    assert set(nets) <= set(parse_net_dev(read_text(root, "proc/net/dev")))
    assert all(
        v >= 0 for p in list_efa_ports(root) for v in read_efa_port(root, p)
    )


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (0, DiskCounters(200000, 400000, 3000)),
        (1, DiskCounters(210000, 420000, 5500)),
    ],
)
def test_parse_diskstats_synthetic_returns_exact_counters(
    snapshot: int, expected: DiskCounters
) -> None:
    disks = parse_diskstats(_text("synthetic", snapshot, "proc/diskstats"))
    assert set(disks) == {
        "dm-0",
        "nvme0c0n1",
        "nvme0n1",
        "nvme0n1p1",
        "nvme1n1",
    }
    assert disks["nvme0n1"] == expected


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [(0, DiskCounters(1000, 0, 0)), (1, DiskCounters(1000, 10000, 1000))],
)
def test_parse_diskstats_synthetic_second_disk_returns_exact_counters(
    snapshot: int, expected: DiskCounters
) -> None:
    disks = parse_diskstats(_text("synthetic", snapshot, "proc/diskstats"))
    assert disks["nvme1n1"] == expected


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (0, NetCounters(100000000, 50000000, 0)),
        (1, NetCounters(105000000, 52500000, 5)),
    ],
)
def test_parse_net_dev_synthetic_returns_exact_counters(
    snapshot: int, expected: NetCounters
) -> None:
    nets = parse_net_dev(_text("synthetic", snapshot, "proc/net/dev"))
    assert set(nets) == {"lo", "ens5", "docker0"}
    assert nets["ens5"] == expected


# [T7] EFA hw_counters layout is a probe fact
@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (0, EfaCounters(1000000000, 800000000)),
        (1, EfaCounters(1500000000, 1000000000)),
    ],
)
def test_read_efa_port_synthetic_returns_exact_counters(
    tmp_path: Path, snapshot: int, expected: EfaCounters
) -> None:
    root = materialize(load_bundle("synthetic"), snapshot, tmp_path)
    assert read_efa_port(root, _SYNTHETIC_PORT) == expected


# [T7] the device-link and hidden-flag filters are probe facts
@pytest.mark.parametrize("snapshot", (0, 1))
def test_discovery_synthetic_tree_returns_physical_devices_only(
    tmp_path: Path, snapshot: int
) -> None:
    root = materialize(load_bundle("synthetic"), snapshot, tmp_path)
    assert list_whole_disks(root) == ("nvme0n1", "nvme1n1")
    assert list_physical_netdevs(root) == ("ens5",)
    assert list_efa_ports(root) == (_SYNTHETIC_PORT,)


# [T7] the device-link and hidden-flag filters are probe facts
def test_discovery_workstation_tree_drops_loop_and_virtual_devices(
    tmp_path: Path,
) -> None:
    root = materialize(load_bundle("workstation"), 1, tmp_path)
    assert list_whole_disks(root) == ("nvme0n1",)
    assert list_physical_netdevs(root) == ("eno2", "eth0", "wlo1")
    assert list_efa_ports(root) == ()


def test_discovery_empty_root_returns_empty_tuples(tmp_path: Path) -> None:
    assert list_whole_disks(tmp_path) == ()
    assert list_physical_netdevs(tmp_path) == ()
    assert list_efa_ports(tmp_path) == ()


def test_list_whole_disks_hidden_flag_excludes_only_a_nonzero_value(
    tmp_path: Path,
) -> None:
    _sysfs_entry(tmp_path, "sys/block", "sda", hidden="0\n")
    _sysfs_entry(tmp_path, "sys/block", "sdb", hidden="1\n")
    _sysfs_entry(tmp_path, "sys/block", "sdc")
    assert list_whole_disks(tmp_path) == ("sda", "sdc")


def test_list_whole_disks_decides_by_device_link_not_by_name(
    tmp_path: Path,
) -> None:
    _sysfs_entry(tmp_path, "sys/block", "vda")
    _sysfs_entry(tmp_path, "sys/block", "nvme0n1", link=False)
    assert list_whole_disks(tmp_path) == ("vda",)


def test_list_whole_disks_non_integer_hidden_raises_value_error(
    tmp_path: Path,
) -> None:
    _sysfs_entry(tmp_path, "sys/block", "sda", hidden="x\n")
    with pytest.raises(ValueError, match="invalid literal"):
        list_whole_disks(tmp_path)


def test_list_physical_netdevs_returns_sorted_names_with_a_device_link(
    tmp_path: Path,
) -> None:
    _sysfs_entry(tmp_path, "sys/class/net", "eth1")
    _sysfs_entry(tmp_path, "sys/class/net", "veth0", link=False)
    _sysfs_entry(tmp_path, "sys/class/net", "eth0")
    assert list_physical_netdevs(tmp_path) == ("eth0", "eth1")


def test_list_efa_ports_returns_every_port_sorted(tmp_path: Path) -> None:
    base = "sys/class/infiniband"
    for port in ("efa_1/ports/1", "efa_0/ports/2", "efa_0/ports/1"):
        (tmp_path / base / port / "hw_counters").mkdir(parents=True)
    assert list_efa_ports(tmp_path) == (
        f"{base}/efa_0/ports/1/hw_counters",
        f"{base}/efa_0/ports/2/hw_counters",
        f"{base}/efa_1/ports/1/hw_counters",
    )


def test_list_efa_ports_ignores_a_port_without_a_hw_counters_directory(
    tmp_path: Path,
) -> None:
    ports = tmp_path / "sys/class/infiniband/mlx5_0/ports"
    (ports / "1").mkdir(parents=True)
    (ports / "2").mkdir()
    (ports / "2/hw_counters").write_text("", encoding="utf-8")
    assert list_efa_ports(tmp_path) == ()


def test_read_efa_port_missing_port_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_efa_port(tmp_path, _SYNTHETIC_PORT)


def test_read_efa_port_non_integer_counter_raises_value_error(
    tmp_path: Path,
) -> None:
    port = tmp_path / _SYNTHETIC_PORT
    port.mkdir(parents=True)
    (port / "rx_bytes").write_text("x\n", encoding="utf-8")
    (port / "tx_bytes").write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid literal") as excinfo:
        read_efa_port(tmp_path, _SYNTHETIC_PORT)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


@pytest.mark.parametrize(
    ("text", "value"), [("0\n", 0), ("1", 1), (" 42 \n", 42)]
)
def test_parse_counter_integer_text_returns_int(text: str, value: int) -> None:
    assert parse_counter(text) == value


def test_parse_net_dev_no_space_after_colon_splits_at_the_colon() -> None:
    text = "eth0:1234 1 0 0 0 0 0 0 5678 1 0 0 0 0 0 0\n"
    assert parse_net_dev(text) == {"eth0": NetCounters(1234, 5678, 0)}


def test_parse_net_dev_errors_sum_rx_and_tx_errs_and_drops() -> None:
    text = "  eth0: 10 100 2 3 7 7 7 7 20 200 4 5 7 7 7 7\n"
    assert parse_net_dev(text) == {"eth0": NetCounters(10, 20, 14)}


def test_parse_diskstats_fourteen_fields_parses_the_older_kernel_format() -> (
    None
):
    text = "259 0 sda 0 0 5 0 0 0 9 0 0 12 0\n"
    assert parse_diskstats(text) == {"sda": DiskCounters(5, 9, 12)}


def _mounts(bundle: str, snapshot: int) -> list[MountEntry]:
    return parse_mountstats(_text(bundle, snapshot, "proc/self/mountstats"))


def _mount(bundle: str, snapshot: int, path: str) -> MountEntry:
    entry = select_mount(_mounts(bundle, snapshot), path)
    assert entry is not None
    return entry


def _nfs(*lines: str, fstype: str = "nfs4") -> MountEntry:
    return MountEntry("fsx:/a", "/a", fstype, lines)


# [T7] the mountstats entry layout and the bundle shape are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_mountstats_bundle_accounts_for_every_line(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/self/mountstats")
    entries = _mounts(bundle, snapshot)
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(entries) == sum(line.startswith("device ") for line in lines) > 0
    assert sum(1 + len(e.lines) for e in entries) == len(lines)
    assert all(e.device and e.fstype for e in entries)
    assert all(e.mountpoint.startswith("/") for e in entries)
    assert all(line == line.strip() for e in entries for line in e.lines)


@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_mountstats_synthetic_returns_the_four_entries(
    snapshot: int,
) -> None:
    entries = _mounts("synthetic", snapshot)
    assert [(e.device, e.mountpoint, e.fstype) for e in entries] == [
        ("sysfs", "/sys", "sysfs"),
        ("/dev/nvme0n1p1", "/", "ext4"),
        ("fsx.example.invalid:/other", "/shared", "nfs4"),
        ("fsx.example.invalid:/fsx", "/shared-zfs", "nfs4"),
    ]
    assert [len(e.lines) for e in entries] == [0, 0, 16, 16]
    assert entries[3].lines[0].startswith("opts:")
    assert entries[3].lines[-1].startswith("GETATTR:")


def test_parse_mountstats_empty_text_returns_no_entries() -> None:
    assert parse_mountstats("") == []
    assert parse_mountstats("\n\n") == []


@pytest.mark.parametrize(
    ("field", "decoded"),
    [
        (r"/mnt/with\040space", "/mnt/with space"),
        (r"/mnt/a\011b", "/mnt/a\tb"),
        (r"/mnt/a\134b", "/mnt/a\\b"),
        (r"/mnt/a\134040b", "/mnt/a\\040b"),
        (r"/mnt/a\040\040b", "/mnt/a  b"),
        (r"/mnt/a\9b\04", r"/mnt/a\9b\04"),
    ],
)
def test_parse_mountstats_octal_escapes_are_decoded_in_device_and_mountpoint(
    field: str, decoded: str
) -> None:
    text = f"device {field} mounted on {field} with fstype ext4\n"
    assert parse_mountstats(text) == [MountEntry(decoded, decoded, "ext4", ())]


def test_parse_mountstats_fstype_is_not_decoded() -> None:
    text = "device a mounted on /m with fstype fuse.x\\040y\n"
    assert parse_mountstats(text)[0].fstype == "fuse.x\\040y"


@pytest.mark.parametrize("statvers", ["", " statvers=1.1", " statvers=1.0"])
def test_parse_mountstats_statvers_is_optional(statvers: str) -> None:
    text = f"device a mounted on /m with fstype nfs4{statvers}\n"
    assert parse_mountstats(text) == [MountEntry("a", "/m", "nfs4", ())]


def test_parse_mountstats_body_lines_are_stripped_and_blanks_dropped() -> None:
    text = (
        f"{_DEVICE}\topts:\trw,hard \n\n   age:\t5\n"
        "device b mounted on /b with fstype ext4\n"
    )
    assert parse_mountstats(text) == [
        MountEntry("fsx:/a", "/a", "nfs4", ("opts:\trw,hard", "age:\t5")),
        MountEntry("b", "/b", "ext4", ()),
    ]


def test_parse_mountstats_indented_device_text_is_a_body_line() -> None:
    text = f"{_DEVICE}\tdevice: x\n"
    assert parse_mountstats(text)[0].lines == ("device: x",)


@pytest.mark.parametrize("snapshot", (0, 1))
@pytest.mark.parametrize(
    ("path", "mountpoint"),
    [
        ("/shared-zfs/ws-test", "/shared-zfs"),
        ("/shared-zfs", "/shared-zfs"),
        ("/shared/x", "/shared"),
        ("/shared", "/shared"),
        ("/shared-other/x", "/"),
        ("/shared-zfs-two", "/"),
        ("/home/user", "/"),
        ("/", "/"),
        ("/sys/devices", "/sys"),
        ("/sysfs", "/"),
    ],
)
def test_select_mount_synthetic_picks_the_longest_path_prefix(
    snapshot: int, path: str, mountpoint: str
) -> None:
    assert _mount("synthetic", snapshot, path).mountpoint == mountpoint


def test_select_mount_trailing_and_repeated_slashes_still_select() -> None:
    entries = _mounts("synthetic", 1)
    for path in ("/shared-zfs/", "/shared-zfs//ws-test/"):
        selected = select_mount(entries, path)
        assert selected is not None
        assert selected.mountpoint == "/shared-zfs"


def test_select_mount_is_independent_of_entry_order() -> None:
    entries = _mounts("synthetic", 1)
    expected = select_mount(entries, "/shared-zfs/ws-test")
    assert select_mount(entries[::-1], "/shared-zfs/ws-test") == expected


def test_select_mount_duplicate_mountpoint_later_entry_wins() -> None:
    first = MountEntry("systemd-1", "/mnt/data", "autofs", ())
    second = MountEntry("fsx:/fsx", "/mnt/data", "nfs4", ("opts:\trw",))
    assert select_mount([first, second], "/mnt/data/ws") is second
    assert select_mount([second, first], "/mnt/data/ws") is first


def test_select_mount_duplicate_shorter_mountpoint_does_not_shadow_longer() -> (
    None
):
    root = MountEntry("a", "/", "ext4", ())
    inner = MountEntry("b", "/m/n", "nfs4", ())
    later_root = MountEntry("c", "/", "xfs", ())
    assert select_mount([root, inner, later_root], "/m/n/x") is inner


def test_select_mount_without_a_match_returns_none() -> None:
    entries = [MountEntry("a", "/m", "ext4", ())]
    assert select_mount(entries, "/other") is None
    assert select_mount([], "/other") is None


@pytest.mark.parametrize("path", ["ws", "ws/x", "./ws", "", "."])
def test_select_mount_relative_path_raises_value_error(path: str) -> None:
    with pytest.raises(ValueError, match="not absolute") as excinfo:
        select_mount(_mounts("synthetic", 1), path)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


# [T7] the autofs and shadowed-mount entries are probe facts
@pytest.mark.parametrize("snapshot", (0, 1))
def test_select_mount_workstation_resolves_real_entries(snapshot: int) -> None:
    root = _mount("workstation", snapshot, "/home/user/ws")
    assert (root.mountpoint, root.fstype) == ("/", "ext4")
    data = _mount("workstation", snapshot, "/mnt/data/ws")
    assert (data.mountpoint, data.fstype) == ("/mnt/data", "autofs")
    shadow = _mount("workstation", snapshot, "/proc/sys/fs/binfmt_misc/x")
    assert (shadow.device, shadow.fstype) == ("binfmt_misc", "binfmt_misc")


# [T7] the NFS bytes: and per-op layouts are probe facts
@pytest.mark.parametrize(
    ("snapshot", "path", "expected"),
    [
        (
            0,
            "/shared-zfs/ws-test",
            NfsCounters(
                1000000,
                2000000,
                300000,
                400000,
                1500000,
                2500000,
                100,
                400,
                200,
                1000,
                1310,
            ),
        ),
        (
            1,
            "/shared-zfs/ws-test",
            NfsCounters(
                6000000,
                4500000,
                300000,
                400000,
                7000000,
                5500000,
                150,
                650,
                200,
                1000,
                1460,
            ),
        ),
        (
            1,
            "/shared/x",
            NfsCounters(
                950000000,
                850000000,
                0,
                0,
                950000000,
                850000000,
                950,
                9600,
                850,
                8500,
                6905,
            ),
        ),
    ],
)
def test_parse_nfs_counters_synthetic_returns_exact_counters(
    snapshot: int, path: str, expected: NfsCounters
) -> None:
    entry = _mount("synthetic", snapshot, path)
    assert parse_nfs_counters(entry) == expected


def test_parse_nfs_counters_reads_fields_by_position() -> None:
    assert parse_nfs_counters(_nfs(*_NFS_LINES)) == NfsCounters(
        1, 2, 3, 4, 5, 6, 10, 16, 20, 26, 35
    )


@pytest.mark.parametrize("fstype", ["nfs", "nfs4"])
def test_parse_nfs_counters_accepts_both_nfs_fstypes(fstype: str) -> None:
    entry = _nfs(*_NFS_LINES, fstype=fstype)
    assert parse_nfs_counters(entry).total_ops == 35


@pytest.mark.parametrize("extra", ["", " 1", " 1 2 3"])
def test_parse_nfs_counters_ignores_fields_past_the_expected_ones(
    extra: str,
) -> None:
    lines = (
        f"bytes:\t1 2 3 4 5 6{extra}",
        "per-op statistics",
        f"READ: 10 11 12 13 14 15 16 17{extra}",
        f"WRITE: 20 21 22 23 24 25 26 27{extra}",
    )
    assert parse_nfs_counters(_nfs(*lines)) == NfsCounters(
        1, 2, 3, 4, 5, 6, 10, 16, 20, 26, 30
    )


def test_parse_nfs_counters_total_ops_counts_only_per_op_lines() -> None:
    lines = ("bytes:\t1 2 3 4 5 6", "xprt:\t9 9 9", "per-op statistics")
    entry = _nfs(*lines, "READ: 1 0 0 0 0 0 0 0", "WRITE: 2 0 0 0 0 0 0 0")
    assert parse_nfs_counters(entry).total_ops == 3


def test_parse_nfs_counters_ignores_an_unusual_entry_elsewhere() -> None:
    text = _text("synthetic", 1, "proc/self/mountstats")
    text += (
        "device fsx:/stale mounted on /stale with fstype nfs4\n\tbytes:\t1\n"
    )
    entry = select_mount(parse_mountstats(text), "/shared-zfs/ws-test")
    assert entry is not None
    assert parse_nfs_counters(entry).total_ops == 1460
    stale = select_mount(parse_mountstats(text), "/stale/x")
    assert stale is not None
    with pytest.raises(ValueError, match="bytes: fewer than 6"):
        parse_nfs_counters(stale)


@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_nfs_counters_non_nfs_entry_raises_value_error(
    snapshot: int,
) -> None:
    entry = _mount("synthetic", snapshot, "/home/user")
    assert entry.fstype == "ext4"
    with pytest.raises(ValueError, match="not NFS") as excinfo:
        parse_nfs_counters(entry)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


# [T7] the workstation has no NFS mount, so its project mount is autofs
def test_parse_nfs_counters_workstation_project_mount_is_not_nfs() -> None:
    with pytest.raises(ValueError, match="not NFS"):
        parse_nfs_counters(_mount("workstation", 1, "/mnt/data/ws"))


@pytest.mark.parametrize("fstype", ["cifs", "nfsd", "NFS4", "nfs4 ", ""])
def test_parse_nfs_counters_other_fstype_raises_value_error(
    fstype: str,
) -> None:
    with pytest.raises(ValueError, match="not NFS"):
        parse_nfs_counters(_nfs(*_NFS_LINES, fstype=fstype))


@pytest.mark.parametrize(
    ("lines", "error", "match"),
    [
        (_NFS_LINES[:0], KeyError, "bytes:"),
        (_NFS_LINES[2:], KeyError, "bytes:"),
        (
            (*_NFS_LINES[:2], *_NFS_LINES[2:4], *_NFS_LINES[5:]),
            KeyError,
            "READ:",
        ),
        (_NFS_LINES[:5], KeyError, "WRITE:"),
        (_NFS_LINES[:2], KeyError, "READ:"),
        (("bytes:\t1 2 3 4 5", *_NFS_LINES[2:]), ValueError, "bytes: fewer"),
        (("bytes:", *_NFS_LINES[2:]), ValueError, "bytes: fewer than 6"),
        (("bytes:\t1 2 3 4 5 x", *_NFS_LINES[2:]), ValueError, "invalid"),
        (
            (*_NFS_LINES, "GETATTR: 1 2 3 4 5 6 7"),
            ValueError,
            "GETATTR: fewer than 8",
        ),
        (
            (*_NFS_LINES[:4], "READ: 1 2 3 4 5 6 7", _NFS_LINES[5]),
            ValueError,
            "READ: fewer than 8",
        ),
        (
            (*_NFS_LINES[:5], "WRITE: 1 2 3 4 5 6 x 8"),
            ValueError,
            "invalid literal",
        ),
    ],
)
def test_parse_nfs_counters_malformed_entry_raises_collector_error(
    lines: tuple[str, ...], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match) as excinfo:
        parse_nfs_counters(_nfs(*lines))
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


_JOB = "/system.slice/slurmstepd.scope/job_s7f3a9c2e"
_STEP_0 = f"{_JOB}/step_0"
_FLAT_FILES = ("cpu.stat", "memory.events", "memory.stat")
_SINGLE_FILES = ("memory.current", "memory.peak", "pids.current")


def _cgroup_text(snapshot: int, cgroup: str, name: str) -> str:
    return _text("synthetic", snapshot, f"sys/fs/cgroup{cgroup}/{name}")


def _own_path(bundle: str, snapshot: int) -> str:
    return parse_proc_self_cgroup(_text(bundle, snapshot, "proc/self/cgroup"))


# [T1, T2, T3] the cgroup v2 unified entry is a probe fact
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_proc_self_cgroup_bundle_returns_the_unified_path(
    bundle: str, snapshot: int
) -> None:
    path = _own_path(bundle, snapshot)
    assert path.startswith("/")
    assert _text(bundle, snapshot, "proc/self/cgroup") == f"0::{path}\n"


@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_proc_self_cgroup_synthetic_returns_the_step_path(
    snapshot: int,
) -> None:
    assert _own_path("synthetic", snapshot) == f"{_STEP_0}/user/task_0"


@pytest.mark.parametrize(
    "text",
    [
        "0::/a/b",
        "0::/a/b\n",
        "1:name=systemd:/x\n0::/a/b\n",
        "0::/a/b\n2::/c\n",
    ],
)
def test_parse_proc_self_cgroup_picks_the_unified_line(text: str) -> None:
    assert parse_proc_self_cgroup(text) == "/a/b"


@pytest.mark.parametrize(
    "text", ["", "\n", "1:name=systemd:/x\n", "10::/a\n", " 0::/a\n", "0:/a\n"]
)
def test_parse_proc_self_cgroup_without_unified_line_raises_key_error(
    text: str,
) -> None:
    with pytest.raises(KeyError, match="0::") as excinfo:
        parse_proc_self_cgroup(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


def test_non_task_steps_are_the_three_slurm_helper_steps() -> None:
    assert NON_TASK_STEPS == {"step_batch", "step_extern", "step_interactive"}


# [T1, T2, T3] the step component under the job cgroup is a probe fact
@pytest.mark.parametrize("n", [0, 12])
@pytest.mark.parametrize("tail", ["", "/user", "/user/task_0"])
def test_job_cgroup_paths_numeric_step_returns_job_and_own_step(
    n: int, tail: str
) -> None:
    result = job_cgroup_paths(f"{_JOB}/step_{n}{tail}")
    assert result == (_JOB, f"{_JOB}/step_{n}")
    assert "4711" not in "".join(filter(None, result))


def test_job_cgroup_paths_job_directory_name_is_not_read() -> None:
    legacy = "/system.slice/slurmstepd.scope/job_4711"
    assert job_cgroup_paths(f"{legacy}/step_3/user/task_0") == (
        legacy,
        f"{legacy}/step_3",
    )
    assert job_cgroup_paths(f"{legacy}/user/task_0") == (None, None)


@pytest.mark.parametrize("step", sorted(NON_TASK_STEPS))
@pytest.mark.parametrize("tail", ["", "/user/task_0"])
def test_job_cgroup_paths_non_task_step_returns_job_without_own_step(
    step: str, tail: str
) -> None:
    assert job_cgroup_paths(f"{_JOB}/{step}{tail}") == (_JOB, None)


@pytest.mark.parametrize(
    "path",
    [
        "/user.slice/user-1000.slice/session-3.scope",
        "/system.slice/slurmstepd.scope/job_4711/user/task_0",
        "/a/mystep_1/b",
        "/a/step/b",
        "/",
        "",
    ],
)
def test_job_cgroup_paths_without_step_component_returns_none_pair(
    path: str,
) -> None:
    assert job_cgroup_paths(path) == (None, None)


def test_job_cgroup_paths_first_step_component_decides() -> None:
    assert job_cgroup_paths("/a/step_0/step_foo") == ("/a", "/a/step_0")
    assert job_cgroup_paths("/a/step_batch/step_foo") == ("/a", None)
    with pytest.raises(ValueError, match="unusable step"):
        job_cgroup_paths("/a/step_foo/step_0")


@pytest.mark.parametrize(
    "path",
    [
        "/a/step_foo/b",
        "/a/step_/b",
        "/a/step_1x/b",
        "/a/step_-1/b",
        "/a/step_0.1",
        "/a/step_BATCH",
        "/a/step_batch2",
        "/a/step_٣/b",
        "/step_0/user",
        "/step_batch",
        "/step_extern/x",
        "step_0/user",
    ],
)
def test_job_cgroup_paths_unusable_step_component_raises_value_error(
    path: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(repr(path))) as excinfo:
        job_cgroup_paths(path)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


# [T1, T2, T3] a capture made inside a Slurm job must not be the workstation
@pytest.mark.parametrize("snapshot", (0, 1))
def test_job_cgroup_paths_workstation_has_no_attribution(
    snapshot: int,
) -> None:
    assert job_cgroup_paths(_own_path("workstation", snapshot)) == (None, None)


# [T1, T2, T3] the SLUID job directory is a probe fact
@pytest.mark.parametrize("snapshot", (0, 1))
def test_job_cgroup_paths_synthetic_returns_the_job_and_own_step(
    snapshot: int,
) -> None:
    result = job_cgroup_paths(_own_path("synthetic", snapshot))
    assert result == (_JOB, _STEP_0)
    assert "4711" not in "".join(filter(None, result))


# [T1, T2, T3] the cgroup file set and layouts are probe facts
@pytest.mark.parametrize("snapshot", (0, 1))
@pytest.mark.parametrize("cgroup", [_JOB, _STEP_0])
def test_cgroup_parsers_synthetic_cover_every_file_of_the_directory(
    snapshot: int, cgroup: str
) -> None:
    files = load_bundle("synthetic").snapshots[snapshot].files
    base = f"sys/fs/cgroup{cgroup}/"
    names = {
        path[len(base) :]
        for path in files
        if path.startswith(base) and "/" not in path[len(base) :]
    }
    assert names == {*_FLAT_FILES, *_SINGLE_FILES}
    for name in _FLAT_FILES:
        parsed = parse_flat_keyed(_cgroup_text(snapshot, cgroup, name))
        assert parsed
    for name in _SINGLE_FILES:
        assert parse_single_int(_cgroup_text(snapshot, cgroup, name)) >= 0


# [T1, T2, T3] the memory.stat keys are probe facts
def test_parse_flat_keyed_synthetic_memory_stat_returns_every_line() -> None:
    text = _cgroup_text(1, _JOB, "memory.stat")
    assert parse_flat_keyed(text) == {
        "anon": 5050000000,
        "file": 2550000000,
        "kernel_stack": 1010000,
        "pagetables": 2020000,
        "sock": 0,
        "shmem": 400000000,
        "file_mapped": 318750000,
        "anon_thp": 0,
    }


@pytest.mark.parametrize(
    ("snapshot", "cgroup", "name", "key", "value"),
    [
        (1, _JOB, "memory.stat", "anon", 5050000000),
        (0, _JOB, "memory.stat", "anon", 5000000000),
        (1, _JOB, "cpu.stat", "usage_usec", 110000000),
        (0, _JOB, "cpu.stat", "usage_usec", 100000000),
        (1, _STEP_0, "cpu.stat", "usage_usec", 1050000),
        (1, _JOB, "memory.events", "oom_kill", 1),
        (0, _JOB, "memory.events", "oom_kill", 0),
        (1, _STEP_0, "memory.events", "oom_kill", 0),
    ],
)
def test_parse_flat_keyed_synthetic_returns_exact_counters(
    snapshot: int, cgroup: str, name: str, key: str, value: int
) -> None:
    parsed = parse_flat_keyed(_cgroup_text(snapshot, cgroup, name))
    assert parsed[key] == value


@pytest.mark.parametrize(
    ("snapshot", "cgroup", "name", "value"),
    [
        (1, _JOB, "memory.peak", 9100000000),
        (0, _JOB, "memory.peak", 9000000000),
        (1, _JOB, "memory.current", 8100000000),
        (1, _JOB, "pids.current", 42),
        (0, _JOB, "pids.current", 40),
        (1, _STEP_0, "memory.peak", 24000000),
        (1, _STEP_0, "pids.current", 2),
    ],
)
def test_parse_single_int_synthetic_returns_exact_value(
    snapshot: int, cgroup: str, name: str, value: int
) -> None:
    assert parse_single_int(_cgroup_text(snapshot, cgroup, name)) == value


def test_parse_flat_keyed_empty_text_returns_empty_dict() -> None:
    assert parse_flat_keyed("") == {}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("anon\n", "expected 'key value'"),
        ("anon 1 2\n", "expected 'key value'"),
        ("anon 1\nfile\n", r"'file'"),
        ("anon x\n", "invalid literal"),
        ("anon max\n", "invalid literal"),
        ("anon 1.5\n", "invalid literal"),
    ],
)
def test_parse_flat_keyed_malformed_line_raises_value_error(
    text: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        parse_flat_keyed(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


@pytest.mark.parametrize(
    ("text", "value"), [("8100000000\n", 8100000000), ("0", 0), (" 42 \n", 42)]
)
def test_parse_single_int_integer_text_returns_int(
    text: str, value: int
) -> None:
    assert parse_single_int(text) == value


@pytest.mark.parametrize("text", ["max\n", "", "\n", "1 2\n", "1.5\n", "x\n"])
def test_parse_single_int_non_integer_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError, match="invalid literal") as excinfo:
        parse_single_int(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


# [T7] the PSI total= layout and the bundle shape are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
@pytest.mark.parametrize("resource", ["cpu", "io", "memory"])
def test_parse_pressure_bundle_returns_the_total_fields(
    bundle: str, snapshot: int, resource: str
) -> None:
    text = _text(bundle, snapshot, f"proc/pressure/{resource}")
    some, full = text.splitlines()
    assert parse_pressure(text) == PressureTotals(
        int(some.rpartition("total=")[2]), int(full.rpartition("total=")[2])
    )


# [T7] the PSI total= layout is a probe fact
@pytest.mark.parametrize(
    ("snapshot", "resource", "expected"),
    [
        (1, "memory", PressureTotals(450000, 200000)),
        (0, "memory", PressureTotals(200000, 100000)),
        (1, "cpu", PressureTotals(1500000, 0)),
        (0, "cpu", PressureTotals(1000000, 0)),
        (1, "io", PressureTotals(1000000, 500000)),
        (0, "io", PressureTotals(0, 0)),
    ],
)
def test_parse_pressure_synthetic_returns_exact_totals(
    snapshot: int, resource: str, expected: PressureTotals
) -> None:
    text = _text("synthetic", snapshot, f"proc/pressure/{resource}")
    assert parse_pressure(text) == expected


_SOME = "some avg10=0.00 avg60=0.00 avg300=0.00 total=7\n"
_FULL = "full avg10=0.00 avg60=0.00 avg300=0.00 total=3\n"


def test_parse_pressure_without_full_line_returns_none() -> None:
    assert parse_pressure(_SOME) == PressureTotals(7, None)


@pytest.mark.parametrize("text", [_SOME + _FULL, _FULL + _SOME])
def test_parse_pressure_line_order_does_not_matter(text: str) -> None:
    assert parse_pressure(text) == PressureTotals(7, 3)


def test_parse_pressure_reads_total_by_name_and_skips_other_lines() -> None:
    text = f"\nother avg10=1.00\nsome total=5 avg10=1.00\n{_FULL}\n"
    assert parse_pressure(text) == PressureTotals(5, 3)


@pytest.mark.parametrize(
    ("text", "error", "match"),
    [
        ("", KeyError, "some"),
        (_FULL, KeyError, "some"),
        ("other total=9\n", KeyError, "some"),
        ("some avg10=0.00 avg60=0.00 avg300=0.00\n", ValueError, "no total="),
        (_SOME + "full avg10=0.00\n", ValueError, "no total="),
        ("some total=x\n", ValueError, "invalid literal"),
        ("some total=\n", ValueError, "invalid literal"),
        (_SOME + "full total=1.5\n", ValueError, "invalid literal"),
    ],
)
def test_parse_pressure_malformed_text_raises_collector_error(
    text: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match) as excinfo:
        parse_pressure(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


_DMI = "sys/devices/virtual/dmi/id/product_name"
_SLURM_ENV = {
    "SLURM_JOB_PARTITION": "newave",
    "SLURM_JOB_NUM_NODES": "2",
    "SLURM_JOB_CPUS_PER_NODE": "192(x2)",
}
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
_NO_ENV_FACTS = _SYNTHETIC_FACTS._replace(
    partition=None, nodes=None, cpus_on_node=None
)


def _facts(
    tmp_path: Path,
    env: dict[str, str],
    *,
    node_index: int = 1,
    snapshot: int = 0,
    overrides: dict[str, str | None] | None = None,
) -> tuple[StaticFacts, list[tuple[str, Exception]]]:
    root = materialize(load_bundle("synthetic"), snapshot, tmp_path, overrides)
    return read_static_facts(root, env, node_index)


def _only_header_error(
    errors: list[tuple[str, Exception]], error: type[Exception]
) -> None:
    assert [source for source, _ in errors] == ["header"]
    assert type(errors[0][1]) is error


# [T3, T7] the cpuinfo model name line is a probe fact
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_cpu_model_bundle_returns_the_first_model_name_value(
    bundle: str, snapshot: int
) -> None:
    text = _text(bundle, snapshot, "proc/cpuinfo")
    first = re.search(r"^model name\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    assert first is not None
    assert parse_cpu_model(text) == first[1]


# [T3, T7] the cpuinfo model name line is a probe fact
@pytest.mark.parametrize("snapshot", (0, 1))
def test_parse_cpu_model_synthetic_returns_the_exact_model(
    snapshot: int,
) -> None:
    text = _text("synthetic", snapshot, "proc/cpuinfo")
    assert parse_cpu_model(text) == "Synthetic CPU 9R14"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("model name\t:   A  \nmodel name\t: B\n", "A"),
        ("processor\t: 1\nmodel name\t: A\nprocessor\t: 2\n", "A"),
        ("model name:X", "X"),
        ("model name\t: Vendor: CPU @ 2.0GHz\n", "Vendor: CPU @ 2.0GHz"),
        ("model\t\t: 17\nmodel name\t: A\n", "A"),
        ("model name\t: A\r\n", "A"),
    ],
)
def test_parse_cpu_model_takes_the_first_stripped_value(
    text: str, expected: str
) -> None:
    assert parse_cpu_model(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n",
        "processor\t: 0\nvendor_id\t: X\n",
        "model\t\t: 17\n",
        "model names\t: A\n",
        "name\t: model name\n",
        "CPU implementer\t: 0x41\n",
    ],
)
def test_parse_cpu_model_without_a_model_name_line_raises_key_error(
    text: str,
) -> None:
    with pytest.raises(KeyError, match="model name") as excinfo:
        parse_cpu_model(text)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


# [T3, T7] the SLURM_JOB_CPUS_PER_NODE compressed-list shape is a probe fact
@pytest.mark.parametrize(
    ("value", "expanded"),
    [
        ("192(x2)", [192, 192]),
        ("48,96(x2)", [48, 96, 96]),
        ("5", [5]),
        ("0", [0]),
        ("1(x3),2", [1, 1, 1, 2]),
        ("4(x1)", [4]),
        ("96(x2),48(x2)", [96, 96, 48, 48]),
        ("007(x02)", [7, 7]),
        ("1(x0),2", [2]),
    ],
)
def test_parse_cpus_per_node_expands_every_entry_by_index(
    value: str, expanded: list[int]
) -> None:
    assert [parse_cpus_per_node(value, i) for i in range(len(expanded))] == (
        expanded
    )
    with pytest.raises(IndexError):
        parse_cpus_per_node(value, len(expanded))


@pytest.mark.parametrize(
    ("value", "node_index"),
    [
        ("192(x2)", 2),
        ("48,96(x2)", 3),
        ("5", 1),
        ("192(x2)", -1),
        ("5", -1),
        ("1(x0)", 0),
    ],
)
def test_parse_cpus_per_node_index_outside_the_list_raises_index_error(
    value: str, node_index: int
) -> None:
    with pytest.raises(IndexError, match="outside") as excinfo:
        parse_cpus_per_node(value, node_index)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "",
        ",",
        "1,,2",
        "1,",
        ",1",
        "1,abc",
        "(x2)",
        "1(x)",
        "1(2)",
        "1(x2",
        "1x2)",
        "1 (x2)",
        " 1",
        "1(x2) ",
        "1(x2)(x3)",
        "1(x-2)",
        "-1",
        "1.5",
        "1(x2)x",
        "\u0663",
    ],
)
def test_parse_cpus_per_node_malformed_item_raises_value_error(
    value: str,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_cpus_per_node(value, 0)
    assert isinstance(excinfo.value, COLLECTOR_ERRORS)


def test_parse_cpus_per_node_checks_every_item_before_indexing() -> None:
    with pytest.raises(ValueError, match="'x'"):
        parse_cpus_per_node("1,x", 0)


def test_parse_cpus_per_node_malformed_item_is_quoted() -> None:
    with pytest.raises(ValueError, match=re.escape("'abc'")):
        parse_cpus_per_node("48,abc", 0)


# [T3, T7] the cpuinfo, DMI and Slurm variable shapes are probe facts
@pytest.mark.parametrize("snapshot", (0, 1))
def test_read_static_facts_synthetic_returns_every_fact(
    tmp_path: Path, snapshot: int
) -> None:
    facts, errors = _facts(tmp_path, _SLURM_ENV, snapshot=snapshot)
    assert facts == _SYNTHETIC_FACTS
    assert errors == []


# [T3, T7] the cpuinfo, DMI and Slurm variable shapes are probe facts
@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("snapshot", (0, 1))
def test_read_static_facts_bundle_reads_the_files_without_errors(
    tmp_path: Path, bundle: str, snapshot: int
) -> None:
    root = materialize(load_bundle(bundle), snapshot, tmp_path)
    facts, errors = read_static_facts(root, {}, 0)
    stat = _text(bundle, snapshot, "proc/stat")
    meminfo = re.search(
        r"^MemTotal:\s+([0-9]+) kB$",
        _text(bundle, snapshot, "proc/meminfo"),
        re.MULTILINE,
    )
    assert meminfo is not None
    assert errors == []
    assert facts.instance_type == _text(bundle, snapshot, _DMI).strip()
    assert facts.cpu_model == parse_cpu_model(
        _text(bundle, snapshot, "proc/cpuinfo")
    )
    assert facts.ncpus == sum(
        1 for line in stat.splitlines() if re.match(r"cpu[0-9]", line)
    )
    assert facts.mem_total_bytes == int(meminfo[1]) * 1024
    assert (
        facts.kernel
        == _text(bundle, snapshot, "proc/sys/kernel/osrelease").strip()
    )
    assert facts[5:] == (None, None, None)
    assert all(
        isinstance(fact, str) and fact
        for fact in (facts.instance_type, facts.cpu_model, facts.kernel)
    )


# [T3, T7] the DMI product_name file is a probe fact
def test_read_static_facts_missing_dmi_file_nulls_only_that_fact(
    tmp_path: Path,
) -> None:
    facts, errors = _facts(tmp_path, {}, overrides={_DMI: None})
    assert facts == _NO_ENV_FACTS._replace(instance_type=None)
    assert facts.instance_type is None
    assert facts.partition is None
    assert facts.nodes is None
    assert facts.cpus_on_node is None
    _only_header_error(errors, FileNotFoundError)


# [T3, T7] the file set a header fact is read from is a probe fact
@pytest.mark.parametrize(
    ("relpath", "field"),
    [
        (_DMI, "instance_type"),
        ("proc/cpuinfo", "cpu_model"),
        ("proc/stat", "ncpus"),
        ("proc/meminfo", "mem_total_bytes"),
        ("proc/sys/kernel/osrelease", "kernel"),
    ],
)
def test_read_static_facts_each_file_fact_fails_independently(
    tmp_path: Path, relpath: str, field: str
) -> None:
    facts, errors = _facts(tmp_path, _SLURM_ENV, overrides={relpath: None})
    assert facts == _SYNTHETIC_FACTS._replace(**{field: None})
    _only_header_error(errors, FileNotFoundError)


# [T3, T7] the file layouts a header fact is parsed from are probe facts
@pytest.mark.parametrize(
    ("relpath", "text", "field", "error"),
    [
        ("proc/cpuinfo", "processor\t: 0\n", "cpu_model", KeyError),
        ("proc/meminfo", "MemFree: 1 kB\n", "mem_total_bytes", KeyError),
        ("proc/meminfo", "MemTotal: x kB\n", "mem_total_bytes", ValueError),
        ("proc/stat", "cpu  1 2 3 4 5 6 7 8\n", "ncpus", KeyError),
        ("proc/stat", "cpu  1 2\ncpu0 1 2\n", "ncpus", ValueError),
    ],
)
def test_read_static_facts_malformed_file_nulls_only_that_fact(
    tmp_path: Path, relpath: str, text: str, field: str, error: type[Exception]
) -> None:
    root = materialize(load_bundle("synthetic"), 0, tmp_path)
    (root / relpath).write_text(text, encoding="utf-8")
    facts, errors = read_static_facts(root, _SLURM_ENV, 1)
    assert facts == _SYNTHETIC_FACTS._replace(**{field: None})
    _only_header_error(errors, error)


def test_read_static_facts_invalid_utf8_file_nulls_only_that_fact(
    tmp_path: Path,
) -> None:
    root = materialize(load_bundle("synthetic"), 0, tmp_path)
    (root / _DMI).write_bytes(b"\xff\xfe")
    facts, errors = read_static_facts(root, _SLURM_ENV, 1)
    assert facts == _SYNTHETIC_FACTS._replace(instance_type=None)
    _only_header_error(errors, UnicodeDecodeError)


# [T3, T7] the Slurm variable shapes are probe facts
@pytest.mark.parametrize(
    ("name", "value", "field", "error"),
    [
        ("SLURM_JOB_NUM_NODES", "two", "nodes", ValueError),
        ("SLURM_JOB_NUM_NODES", "2.5", "nodes", ValueError),
        ("SLURM_JOB_CPUS_PER_NODE", "abc", "cpus_on_node", ValueError),
        ("SLURM_JOB_CPUS_PER_NODE", "192", "cpus_on_node", IndexError),
        ("SLURM_JOB_CPUS_PER_NODE", "192(x1)", "cpus_on_node", IndexError),
    ],
)
def test_read_static_facts_malformed_variable_nulls_only_that_fact(
    tmp_path: Path,
    name: str,
    value: str,
    field: str,
    error: type[Exception],
) -> None:
    facts, errors = _facts(tmp_path, {**_SLURM_ENV, name: value})
    assert facts == _SYNTHETIC_FACTS._replace(**{field: None})
    _only_header_error(errors, error)


# [T3, T7] the Slurm variable shapes are probe facts
@pytest.mark.parametrize("name", sorted(_SLURM_ENV))
@pytest.mark.parametrize("value", [None, ""])
def test_read_static_facts_unset_or_empty_variable_is_none_without_error(
    tmp_path: Path, name: str, value: str | None
) -> None:
    env = {k: v for k, v in _SLURM_ENV.items() if k != name}
    if value is not None:
        env[name] = value
    field = {
        "SLURM_JOB_PARTITION": "partition",
        "SLURM_JOB_NUM_NODES": "nodes",
        "SLURM_JOB_CPUS_PER_NODE": "cpus_on_node",
    }[name]
    facts, errors = _facts(tmp_path, env)
    assert facts == _SYNTHETIC_FACTS._replace(**{field: None})
    assert errors == []


# [T3, T7] the Slurm variable shapes are probe facts
def test_read_static_facts_empty_env_returns_the_file_facts_only(
    tmp_path: Path,
) -> None:
    facts, errors = _facts(tmp_path, {})
    assert facts == _NO_ENV_FACTS
    assert errors == []


# [T3, T7] the Slurm variable shapes are probe facts
def test_read_static_facts_cpus_on_node_follows_the_node_index(
    tmp_path: Path,
) -> None:
    env = {**_SLURM_ENV, "SLURM_JOB_CPUS_PER_NODE": "48,96(x2)"}
    cpus = [
        _facts(tmp_path, env, node_index=i)[0].cpus_on_node for i in range(3)
    ]
    assert cpus == [48, 96, 96]


# [T3, T7] the Slurm variable shapes are probe facts
def test_read_static_facts_node_index_outside_the_list_nulls_cpus_on_node(
    tmp_path: Path,
) -> None:
    facts, errors = _facts(tmp_path, _SLURM_ENV, node_index=2)
    assert facts == _SYNTHETIC_FACTS._replace(cpus_on_node=None)
    _only_header_error(errors, IndexError)


def test_read_static_facts_text_facts_are_stripped(tmp_path: Path) -> None:
    facts, errors = _facts(
        tmp_path,
        {**_SLURM_ENV, "SLURM_JOB_PARTITION": " newave\t"},
        overrides={
            _DMI: "  r7a.48xlarge \n\n",
            "proc/sys/kernel/osrelease": "\t6.8.0-1021-synthetic  \n",
            "proc/cpuinfo": "model name\t:  Synthetic CPU 9R14   \n",
        },
    )
    assert facts == _SYNTHETIC_FACTS
    assert errors == []


def test_read_static_facts_failures_are_reported_in_field_order(
    tmp_path: Path,
) -> None:
    facts, errors = _facts(
        tmp_path,
        {"SLURM_JOB_NUM_NODES": "x", "SLURM_JOB_CPUS_PER_NODE": "y"},
        overrides={_DMI: None, "proc/meminfo": "", "proc/cpuinfo": None},
    )
    assert facts == StaticFacts(
        None, None, 4, None, "6.8.0-1021-synthetic", None, None, None
    )
    assert [(source, type(exc)) for source, exc in errors] == [
        ("header", FileNotFoundError),
        ("header", FileNotFoundError),
        ("header", KeyError),
        ("header", ValueError),
        ("header", ValueError),
    ]


def test_read_static_facts_reports_the_raised_exception(
    tmp_path: Path,
) -> None:
    _, errors = _facts(tmp_path, {}, overrides={_DMI: None})
    exc = errors[0][1]
    assert isinstance(exc, FileNotFoundError)
    assert exc.filename == str(tmp_path / _DMI)


def test_read_static_facts_ignores_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {**_SLURM_ENV, "SLURM_NODEID": "0"}.items():
        monkeypatch.setenv(name, value)
    facts, errors = _facts(tmp_path, {})
    assert facts == _NO_ENV_FACTS
    assert errors == []


def test_read_static_facts_does_not_read_the_node_id_variable(
    tmp_path: Path,
) -> None:
    env = {
        **_SLURM_ENV,
        "SLURM_JOB_CPUS_PER_NODE": "48,96",
        "SLURM_NODEID": "0",
    }
    facts, _ = _facts(tmp_path, env, node_index=1)
    assert facts.cpus_on_node == 96


def test_read_static_facts_other_exceptions_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(root: Path, relpath: str) -> str:
        raise RuntimeError(relpath)

    root = materialize(load_bundle("synthetic"), 0, tmp_path)
    monkeypatch.setattr(procfs, "read_text", boom)
    with pytest.raises(RuntimeError, match="product_name"):
        read_static_facts(root, _SLURM_ENV, 1)
