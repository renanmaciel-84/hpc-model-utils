"""ADR-079/ADR-063: the snapshot collector over fixture trees."""

from __future__ import annotations

import contextlib
import os
import signal
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from hpc_model_utils.infra.telemetry import collect
from hpc_model_utils.infra.telemetry.collect import (
    COUNTER_SOURCES,
    Collector,
    Values,
    filesystem_usage,
    next_walk_interval,
    walk_workspace,
)
from hpc_model_utils.infra.telemetry.procfs import PressureTotals
from hpc_model_utils.infra.telemetry.schema import (
    SAMPLE_COLUMNS,
    SOURCES,
    WORKSPACE_COLUMNS,
)
from tests.support.telemetry_tree import load_bundle, materialize

_SYNTHETIC = load_bundle("synthetic")
_WORKSPACE = Path("/shared-zfs/ws-test")
_COLLECTED = [
    c.name for c in SAMPLE_COLUMNS if c.source not in ("identity", "sampler")
]
_WORKSPACE_COLUMNS = [
    c.name for c in WORKSPACE_COLUMNS if c.source != "identity"
]
_WALK_COLUMNS = ["apparent_bytes", "allocated_bytes", "files", "dirs", "walk_s"]
_STEP_DIR = "sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e/step_0/"
_JOB_PIDS = (
    "sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e/pids.current"
)
_EXPECTED: dict[str, float | int | None] = {
    "cpu_busy_cores": 1.55,
    "cpu_util": 0.3875,
    "cpu_user_cores": 1.2,
    "cpu_system_cores": 0.35,
    "cpu_iowait_cores": 0.1,
    "cpu_steal_cores": 0.05,
    "cpu_core_util_min": 0.0,
    "cpu_core_util_p50": 0.325,
    "cpu_core_util_max": 0.9,
    "cpu_active_cores": 2,
    "load1": 1.5,
    "procs_running": 4,
    "procs_blocked": 0,
    "mem_used_bytes": 6144000000,
    "mem_available_bytes": 10240000000,
    "mem_anon_bytes": 4096000000,
    "mem_cached_bytes": 6144000000,
    "mem_shmem_bytes": 512000000,
    "mem_dirty_bytes": 1024000,
    "mem_writeback_bytes": 0,
    "mem_slab_bytes": 307200000,
    "swap_used_bytes": 102400000,
    "major_faults_per_s": 10.0,
    "oom_kills": 1,
    "job_cpu_cores": 1.99,
    "job_mem_bytes": 8080000000,
    "job_mem_anon_bytes": 5035000000,
    "job_mem_file_bytes": 2546000000,
    "job_mem_shmem_bytes": 400000000,
    "job_mem_peak_bytes": 9100000000,
    "job_pids": 42,
    "job_oom_kills": 1,
    "psi_cpu_some": 0.1,
    "psi_mem_some": 0.05,
    "psi_mem_full": 0.02,
    "psi_io_some": 0.2,
    "psi_io_full": 0.1,
    "disk_read_bytes_per_s": 1024000.0,
    "disk_write_bytes_per_s": 3072000.0,
    "disk_busy_max": 0.5,
    "nfs_read_bytes_per_s": 1000000.0,
    "nfs_write_bytes_per_s": 500000.0,
    "nfs_server_read_bytes_per_s": 1100000.0,
    "nfs_server_write_bytes_per_s": 600000.0,
    "nfs_read_ops_per_s": 10.0,
    "nfs_write_ops_per_s": 0.0,
    "nfs_ops_per_s": 30.0,
    "nfs_read_rtt_ms": 5.0,
    "nfs_write_rtt_ms": None,
    "net_rx_bytes_per_s": 1000000.0,
    "net_tx_bytes_per_s": 500000.0,
    "net_errors_per_s": 1.0,
    "efa_rx_bytes_per_s": 100000000.0,
    "efa_tx_bytes_per_s": 40000000.0,
}


def _columns(source: str) -> list[str]:
    return [c.name for c in SAMPLE_COLUMNS if c.source == source]


def _assert_values(
    values: Values, expected: Mapping[str, float | int | None]
) -> None:
    for name, want in expected.items():
        if want is None:
            assert values[name] is None, name
        else:
            assert values[name] == pytest.approx(want, abs=1e-9), name


def _all_none(values: Values, source: str) -> bool:
    return all(values[name] is None for name in _columns(source))


def _start(
    root: Path, overrides: Mapping[str, str | None] | None = None
) -> Collector:
    materialize(_SYNTHETIC, 0, root, overrides)
    collector = Collector(root, _WORKSPACE)
    assert collector.baseline() == []
    return collector


@pytest.fixture
def started(tmp_path: Path) -> Collector:
    return _start(tmp_path)


def test_collector_synthetic_root_resolves_readme_attribution(
    started: Collector,
) -> None:
    assert started.startup_errors == ()
    assert started.job_cgroup == (
        "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e"
    )
    assert started.own_step_cgroup == f"/{_STEP_DIR}".rstrip("/")
    assert started.workspace_mount == {
        "mountpoint": "/shared-zfs",
        "fstype": "nfs4",
    }


def test_sample_synthetic_sequence_matches_readme_table(
    started: Collector, tmp_path: Path
) -> None:
    materialize(_SYNTHETIC, 1, tmp_path)
    values, errors = started.sample(5.0)
    assert errors == []
    assert list(values) == _COLLECTED
    assert set(_EXPECTED) == set(_COLLECTED)
    _assert_values(values, _EXPECTED)


@pytest.mark.parametrize(
    ("dt_s", "disk_write", "nfs_read"),
    [(5.0, 3072000.0, 1000000.0), (2.5, 6144000.0, 2000000.0)],
)
def test_sample_dt_s_divides_every_rate(
    started: Collector,
    tmp_path: Path,
    dt_s: float,
    disk_write: float,
    nfs_read: float,
) -> None:
    materialize(_SYNTHETIC, 1, tmp_path)
    values, errors = started.sample(dt_s)
    assert errors == []
    assert values["disk_write_bytes_per_s"] == pytest.approx(disk_write)
    assert values["nfs_read_bytes_per_s"] == pytest.approx(nfs_read)


def test_sample_failed_source_nulls_one_interval_then_recovers(
    started: Collector, tmp_path: Path
) -> None:
    missing = {"proc/pressure/io": None}
    materialize(_SYNTHETIC, 1, tmp_path, missing)
    values, errors = started.sample(5.0)
    assert _all_none(values, "pressure")
    assert [source for source, _ in errors] == ["pressure"]
    assert isinstance(errors[0][1], FileNotFoundError)
    assert values["cpu_util"] == pytest.approx(0.3875)

    materialize(_SYNTHETIC, 1, tmp_path)
    values, errors = started.sample(5.0)
    assert _all_none(values, "pressure")
    assert errors == []

    values, errors = started.sample(5.0)
    assert values["psi_cpu_some"] == 0.0
    assert errors == []


def test_sample_errors_follow_counter_source_order(
    started: Collector, tmp_path: Path
) -> None:
    gone = {"proc/pressure/io": None, "proc/loadavg": None, "proc/vmstat": None}
    materialize(_SYNTHETIC, 1, tmp_path, gone)
    _, errors = started.sample(5.0)
    assert [source for source, _ in errors] == ["loadavg", "vmstat", "pressure"]


def test_sample_non_finite_value_becomes_null_and_keeps_baseline(
    started: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = collect.derive_pressure

    def with_nan(
        prev: Mapping[str, PressureTotals] | None,
        cur: Mapping[str, PressureTotals],
        dt_s: float,
    ) -> Values:
        return {**real(prev, cur, dt_s), "psi_cpu_some": float("nan")}

    monkeypatch.setattr(collect, "derive_pressure", with_nan)
    materialize(_SYNTHETIC, 1, tmp_path)
    values, errors = started.sample(5.0)
    assert values["psi_cpu_some"] is None
    _assert_values(
        values,
        {k: v for k, v in _EXPECTED.items() if k.startswith("psi_")}
        | {"psi_cpu_some": None},
    )
    assert [(source, type(exc)) for source, exc in errors] == [
        ("pressure", ValueError)
    ]
    assert str(errors[0][1]) == "non-finite psi_cpu_some"

    monkeypatch.undo()
    materialize(_SYNTHETIC, 1, tmp_path)
    values, errors = started.sample(5.0)
    assert values["psi_cpu_some"] == 0.0
    assert errors == []


@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_sample_non_finite_loadavg_text_becomes_null(
    started: Collector, tmp_path: Path, text: str
) -> None:
    override = {"proc/loadavg": f"{text} 1.20 1.00 4/500 12345\n"}
    materialize(_SYNTHETIC, 1, tmp_path, override)
    values, errors = started.sample(5.0)
    assert values["load1"] is None
    assert [(source, str(exc)) for source, exc in errors] == [
        ("loadavg", "non-finite load1")
    ]


def test_sample_programming_error_propagates(
    started: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object) -> Values:
        raise TypeError("bug")

    monkeypatch.setattr(collect, "derive_cpu", broken)
    materialize(_SYNTHETIC, 1, tmp_path)
    with pytest.raises(TypeError, match="bug"):
        started.sample(5.0)


def test_baseline_reads_without_deriving_and_returns_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object) -> Values:
        raise AssertionError("baseline derived")

    for name in dir(collect):
        if name.startswith("derive_"):
            monkeypatch.setattr(collect, name, forbidden)
    materialize(_SYNTHETIC, 0, tmp_path, {"proc/vmstat": None})
    collector = Collector(tmp_path, _WORKSPACE)
    errors = collector.baseline()
    assert [(source, type(exc)) for source, exc in errors] == [
        ("vmstat", FileNotFoundError)
    ]


def test_collector_unknown_step_name_records_startup_error(
    tmp_path: Path,
) -> None:
    override = {
        "proc/self/cgroup": "0::/system.slice/slurmstepd.scope/job_x/step_zz\n"
    }
    collector = _start(tmp_path, override)
    assert [(s, type(e)) for s, e in collector.startup_errors] == [
        ("cgroup", ValueError)
    ]
    assert collector.job_cgroup is None
    assert collector.own_step_cgroup is None
    materialize(_SYNTHETIC, 1, tmp_path, override)
    values, errors = collector.sample(5.0)
    assert errors == []
    assert _all_none(values, "cgroup")


def test_sample_step_directory_vanishing_nulls_only_the_cgroup_source(
    started: Collector, tmp_path: Path
) -> None:
    files = _SYNTHETIC.snapshots[1].files
    gone = {rel: None for rel in files if rel.startswith(_STEP_DIR)}
    assert gone
    materialize(_SYNTHETIC, 1, tmp_path, gone)
    values, errors = started.sample(5.0)
    assert [(source, type(exc)) for source, exc in errors] == [
        ("cgroup", FileNotFoundError)
    ]
    assert _all_none(values, "cgroup")
    assert values["cpu_util"] == pytest.approx(0.3875)


def test_sample_workspace_mount_entry_disappearing_raises_lookup_error(
    started: Collector, tmp_path: Path
) -> None:
    shorter = "device sysfs mounted on /sys with fstype sysfs\n"
    materialize(_SYNTHETIC, 1, tmp_path, {"proc/self/mountstats": shorter})
    values, errors = started.sample(5.0)
    assert [(source, type(exc)) for source, exc in errors] == [
        ("mountstats", LookupError)
    ]
    assert _all_none(values, "mountstats")
    assert values["net_rx_bytes_per_s"] == pytest.approx(1000000.0)


def test_sample_workspace_mount_that_is_not_nfs_is_not_read(
    tmp_path: Path,
) -> None:
    local = "device /dev/sdb mounted on /shared-zfs with fstype ext4\n"
    collector = _start(tmp_path, {"proc/self/mountstats": local})
    assert collector.workspace_mount == {
        "mountpoint": "/shared-zfs",
        "fstype": "ext4",
    }
    materialize(_SYNTHETIC, 1, tmp_path, {"proc/self/mountstats": None})
    values, errors = collector.sample(5.0)
    assert errors == []
    assert _all_none(values, "mountstats")


def test_sample_without_efa_port_yields_null_without_error(
    tmp_path: Path,
) -> None:
    files = _SYNTHETIC.snapshots[0].files
    no_efa = {r: None for r in files if r.startswith("sys/class/infiniband/")}
    collector = _start(tmp_path, no_efa)
    materialize(_SYNTHETIC, 1, tmp_path, no_efa)
    values, errors = collector.sample(5.0)
    assert errors == []
    assert _all_none(values, "efa")
    assert values["net_rx_bytes_per_s"] == pytest.approx(1000000.0)


def test_sample_workstation_tree_reports_cpu_and_memory(
    tmp_path: Path,
) -> None:
    bundle = load_bundle("workstation")
    materialize(bundle, 0, tmp_path)
    collector = Collector(tmp_path, tmp_path / "workspace")
    assert collector.job_cgroup is None
    assert collector.own_step_cgroup is None
    collector.baseline()
    materialize(bundle, 1, tmp_path)
    values, errors = collector.sample(5.0)
    assert list(values) == _COLLECTED
    assert values["cpu_util"] is not None
    assert values["mem_used_bytes"] is not None
    assert _all_none(values, "cgroup")
    assert {source for source, _ in errors} <= SOURCES


def test_sample_prd_tree_reads_job_cgroup_without_pids_current(
    tmp_path: Path,
) -> None:
    bundle = load_bundle("prd")
    materialize(bundle, 0, tmp_path)
    collector = Collector(tmp_path, Path("/shared-zfs/ws-prd"))
    assert collector.startup_errors == ()
    assert collector.baseline() == []
    materialize(bundle, 1, tmp_path)
    values, errors = collector.sample(5.31)
    assert errors == []
    assert {k for k, v in values.items() if v is None} == {
        "job_pids",
        "nfs_read_rtt_ms",
    }
    assert values["job_mem_bytes"] == 124329984
    assert values["job_mem_peak_bytes"] == 134823936
    assert values["job_oom_kills"] == 0
    assert values["job_cpu_cores"] is not None
    assert 191.5 < values["job_cpu_cores"] <= 192.0
    assert values["efa_rx_bytes_per_s"] == 0.0
    assert values["efa_tx_bytes_per_s"] == 0.0


def test_baseline_unparseable_pids_current_nulls_the_cgroup_source(
    tmp_path: Path,
) -> None:
    materialize(_SYNTHETIC, 0, tmp_path, {_JOB_PIDS: "max\n"})
    collector = Collector(tmp_path, _WORKSPACE)
    errors = collector.baseline()
    assert [(source, type(exc)) for source, exc in errors] == [
        ("cgroup", ValueError)
    ]


def test_baseline_pids_current_other_than_missing_is_not_swallowed(
    tmp_path: Path,
) -> None:
    materialize(_SYNTHETIC, 0, tmp_path)
    (tmp_path / _JOB_PIDS).unlink()
    (tmp_path / _JOB_PIDS).mkdir()
    collector = Collector(tmp_path, _WORKSPACE)
    errors = collector.baseline()
    assert [(source, type(exc)) for source, exc in errors] == [
        ("cgroup", IsADirectoryError)
    ]


def test_counter_sources_are_schema_sources_with_every_collected_column() -> (
    None
):
    assert len(COUNTER_SOURCES) == 10
    assert set(COUNTER_SOURCES) <= SOURCES
    columns = {name for source in COUNTER_SOURCES for name in _columns(source)}
    assert columns == set(_COLLECTED)


class _Entry:
    def __init__(self, entry: os.DirEntry[str], error: OSError | None) -> None:
        self._entry = entry
        self._error = error
        self.path = entry.path

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        if self._error is not None:
            raise self._error
        return self._entry.stat(follow_symlinks=follow_symlinks)

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self._entry.is_dir(follow_symlinks=follow_symlinks)


def _flaky_scandir(
    monkeypatch: pytest.MonkeyPatch,
    stat_errors: Mapping[str, OSError] | None = None,
    scan_errors: Mapping[str, OSError] | None = None,
) -> None:
    real = os.scandir
    stat_errors = stat_errors or {}
    scan_errors = scan_errors or {}

    @contextlib.contextmanager
    def scandir(path: str) -> Iterator[Iterator[_Entry]]:
        if (error := scan_errors.get(Path(path).name)) is not None:
            raise error
        with real(path) as entries:
            yield (_Entry(e, stat_errors.get(e.name)) for e in entries)

    monkeypatch.setattr(os, "scandir", scandir)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / ".hpcmu" / "telemetry").mkdir(parents=True)
    (ws / "a.txt").write_bytes(b"a" * 10)
    (ws / "sub" / "b.bin").write_bytes(b"b" * 4096)
    (ws / "link").symlink_to("/etc")
    (ws / ".hpcmu" / "telemetry" / "x.ndjson").write_bytes(b"x" * 100)
    return ws


@pytest.fixture
def skip(workspace: Path) -> Path:
    return workspace / ".hpcmu" / "telemetry"


@pytest.fixture
def walker(tmp_path: Path, workspace: Path) -> Collector:
    materialize(_SYNTHETIC, 0, tmp_path / "root")
    return Collector(tmp_path / "root", workspace)


def test_walk_workspace_counts_tree_without_following_symlink_or_skip(
    workspace: Path, skip: Path
) -> None:
    counted = [
        workspace / "a.txt",
        workspace / "sub",
        workspace / "sub" / "b.bin",
        workspace / "link",
        workspace / ".hpcmu",
    ]
    stats = [os.lstat(path) for path in counted]
    totals = walk_workspace(workspace, skip)
    assert totals is not None
    assert (totals.files, totals.dirs) == (3, 2)
    assert totals.apparent_bytes == sum(st.st_size for st in stats)
    assert totals.allocated_bytes == sum(st.st_blocks * 512 for st in stats)
    assert totals.walk_s >= 0


def test_walk_workspace_walk_s_is_end_minus_start_of_the_clock(
    workspace: Path, skip: Path
) -> None:
    ticks = iter([10.0, 12.5])
    totals = walk_workspace(workspace, skip, monotonic=lambda: next(ticks))
    assert totals is not None
    assert totals.walk_s == 2.5


def test_walk_workspace_missing_root_raises_file_not_found(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        walk_workspace(tmp_path / "nowhere", tmp_path / "skip")


def test_walk_workspace_vanished_entry_is_skipped(
    workspace: Path, skip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flaky_scandir(monkeypatch, {"a.txt": FileNotFoundError("gone")})
    totals = walk_workspace(workspace, skip)
    assert totals is not None
    assert totals.files == 2


def test_walk_workspace_vanished_subdirectory_is_skipped(
    workspace: Path, skip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flaky_scandir(monkeypatch, scan_errors={"sub": FileNotFoundError("gone")})
    totals = walk_workspace(workspace, skip)
    assert totals is not None
    assert totals.files == 2


def test_walk_permission_error_nulls_walk_columns_and_keeps_statvfs(
    walker: Collector, monkeypatch: pytest.MonkeyPatch
) -> None:
    _flaky_scandir(monkeypatch, {"a.txt": PermissionError("denied")})
    result = walker.walk()
    assert result is not None
    values, errors = result
    assert [(source, type(exc)) for source, exc in errors] == [
        ("walk", PermissionError)
    ]
    assert all(values[name] is None for name in _WALK_COLUMNS)
    assert values["fs_used_bytes"] is not None
    assert values["fs_avail_bytes"] is not None


def test_walk_workspace_stop_pending_after_first_directory_returns_none(
    workspace: Path, skip: Path
) -> None:
    answers = iter([False, True])
    assert (
        walk_workspace(workspace, skip, stop_pending=lambda: next(answers))
        is None
    )


def test_walk_workspace_stop_pending_before_root_opens_nothing(
    workspace: Path, skip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(path: str) -> None:
        raise AssertionError("scandir called")

    monkeypatch.setattr(os, "scandir", forbidden)
    assert walk_workspace(workspace, skip, stop_pending=lambda: True) is None


def test_walk_workspace_pending_sigterm_returns_none(
    workspace: Path, skip: Path
) -> None:
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        signal.raise_signal(signal.SIGTERM)
        assert walk_workspace(workspace, skip) is None
    finally:
        signal.sigtimedwait({signal.SIGTERM}, 0)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def test_walk_stopped_by_signal_returns_none_without_statvfs(
    walker: Collector, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(path: Path) -> tuple[int, int]:
        raise AssertionError("statvfs read")

    monkeypatch.setattr(collect, "_stop_pending", lambda: True)
    monkeypatch.setattr(collect, "filesystem_usage", forbidden)
    assert walker.walk() is None


def test_walk_returns_the_seven_workspace_values_in_schema_order(
    walker: Collector,
) -> None:
    result = walker.walk()
    assert result is not None
    values, errors = result
    assert errors == []
    assert list(values) == _WORKSPACE_COLUMNS
    assert (values["files"], values["dirs"]) == (3, 2)
    assert isinstance(values["fs_used_bytes"], int)
    assert isinstance(values["fs_avail_bytes"], int)


def test_walk_statvfs_failure_nulls_only_the_statvfs_columns(
    walker: Collector, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> os.statvfs_result:
        raise OSError("statvfs failed")

    monkeypatch.setattr(os, "statvfs", broken)
    result = walker.walk()
    assert result is not None
    values, errors = result
    assert [(source, type(exc)) for source, exc in errors] == [
        ("statvfs", OSError)
    ]
    assert (values["fs_used_bytes"], values["fs_avail_bytes"]) == (None, None)
    assert values["files"] == 3


def test_filesystem_usage_scales_block_counts_by_fragment_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = os.statvfs_result((4096, 512, 1000, 400, 300, 0, 0, 0, 0, 255))
    monkeypatch.setattr(os, "statvfs", lambda path: fake)
    assert filesystem_usage(tmp_path) == (600 * 512, 300 * 512)


def test_filesystem_usage_real_filesystem_is_within_its_capacity(
    tmp_path: Path,
) -> None:
    used, avail = filesystem_usage(tmp_path)
    real = os.statvfs(tmp_path)
    assert min(used, avail) >= 0
    assert used + avail <= real.f_blocks * real.f_frsize


@pytest.mark.parametrize(
    ("walk_s", "expected"), [(None, 60), (3.0, 60), (3.5, 70.0)]
)
def test_next_walk_interval_grows_only_past_five_percent_of_the_interval(
    walk_s: float | None, expected: float
) -> None:
    assert next_walk_interval(walk_s, 60) == expected
