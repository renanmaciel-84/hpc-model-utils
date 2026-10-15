"""ADR-093/ADR-095: derivations over fixtures, edge cases and invariants."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hpc_model_utils.infra.telemetry.collect import (
    CgroupReading,
    JobCgroup,
    StepCgroup,
    Values,
    derive_cpu,
    derive_disk,
    derive_efa,
    derive_job,
    derive_memory,
    derive_net,
    derive_nfs,
    derive_pressure,
    derive_vmstat,
)
from hpc_model_utils.infra.telemetry.procfs import (
    CpuTimes,
    DiskCounters,
    EfaCounters,
    NetCounters,
    NfsCounters,
    PressureTotals,
    ProcStat,
    job_cgroup_paths,
    parse_counter,
    parse_diskstats,
    parse_flat_keyed,
    parse_meminfo,
    parse_mountstats,
    parse_net_dev,
    parse_nfs_counters,
    parse_pressure,
    parse_proc_self_cgroup,
    parse_proc_stat,
    parse_single_int,
    parse_vmstat,
    select_mount,
)
from hpc_model_utils.infra.telemetry.schema import SAMPLE_COLUMNS
from tests.support.telemetry_tree import load_bundle

_STAT_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "stat"]
_MEMINFO_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "meminfo"]
_VMSTAT_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "vmstat"]
_CGROUP_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "cgroup"]
_PRESSURE_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "pressure"]
_DISK_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "diskstats"]
_NFS_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "mountstats"]
_NET_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "netdev"]
_EFA_COLUMNS = [c.name for c in SAMPLE_COLUMNS if c.source == "efa"]
_RATE_KEYS = _STAT_COLUMNS[:10]
_BASE = CpuTimes(*(1000,) * 8)


def _synthetic(snapshot: int) -> ProcStat:
    return parse_proc_stat(_files(snapshot)["proc/stat"])


def _one_cpu(**deltas: int) -> tuple[ProcStat, ProcStat]:
    step = CpuTimes(**{f: deltas.get(f, 0) for f in CpuTimes._fields})
    cur = CpuTimes(*(a + b for a, b in zip(_BASE, step, strict=True)))
    return (
        ProcStat(_BASE, {0: _BASE}, 1, 0),
        ProcStat(cur, {0: cur}, 1, 0),
    )


def _all_null(result: dict[str, float | int | None]) -> bool:
    return all(result[k] is None for k in _RATE_KEYS)


def test_derive_cpu_synthetic_pair_matches_readme_table() -> None:
    result = derive_cpu(_synthetic(0), _synthetic(1))
    assert list(result) == _STAT_COLUMNS
    expected = {
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
        "procs_running": 4,
        "procs_blocked": 0,
    }
    assert result == {
        k: pytest.approx(v, abs=1e-9) for k, v in expected.items()
    }


def test_derive_cpu_reversed_pair_returns_null_rates() -> None:
    first, second = _synthetic(0), _synthetic(1)
    result = derive_cpu(second, first)
    assert _all_null(result)
    assert result["procs_running"] == first.procs_running
    assert result["procs_blocked"] == first.procs_blocked


def test_derive_cpu_missing_baseline_returns_null_rates() -> None:
    cur = _synthetic(1)
    result = derive_cpu(None, cur)
    assert list(result) == _STAT_COLUMNS
    assert _all_null(result)
    assert result["procs_running"] == cur.procs_running


def test_derive_cpu_identical_readings_return_null_rates() -> None:
    stat = _synthetic(1)
    result = derive_cpu(stat, stat)
    assert _all_null(result)
    assert result["procs_blocked"] == stat.procs_blocked


def test_derive_cpu_one_cpu_reset_leaves_the_others() -> None:
    prev, cur = _synthetic(0), _synthetic(1)
    broken = cur.per_cpu[1]._replace(user=0)
    cur = cur._replace(per_cpu={**cur.per_cpu, 1: broken})
    result = derive_cpu(prev, cur)
    assert result["cpu_core_util_min"] == pytest.approx(0.0, abs=1e-9)
    assert result["cpu_core_util_p50"] == pytest.approx(0.15, abs=1e-9)
    assert result["cpu_core_util_max"] == pytest.approx(0.9, abs=1e-9)
    assert result["cpu_active_cores"] == 1
    assert result["cpu_busy_cores"] == pytest.approx(1.55, abs=1e-9)


def test_derive_cpu_every_cpu_reset_nulls_only_per_core_values() -> None:
    prev, cur = _synthetic(0), _synthetic(1)
    cur = cur._replace(
        per_cpu={i: t._replace(user=0) for i, t in cur.per_cpu.items()}
    )
    result = derive_cpu(prev, cur)
    assert result["cpu_busy_cores"] == pytest.approx(1.55, abs=1e-9)
    assert all(result[k] is None for k in _RATE_KEYS[6:])


def test_derive_cpu_hot_plugged_cpu_is_paired_only_with_itself() -> None:
    prev, cur = _synthetic(0), _synthetic(1)
    per_cpu = {k: v for k, v in cur.per_cpu.items() if k != 0}
    per_cpu[7] = CpuTimes(*(10**9,) * 8)
    result = derive_cpu(prev, cur._replace(per_cpu=per_cpu))
    assert result["cpu_core_util_max"] == pytest.approx(0.5, abs=1e-9)
    assert result["cpu_active_cores"] == 1


def test_derive_cpu_iowait_step_back_nulls_only_iowait() -> None:
    prev = CpuTimes(1000, 0, 500, 5000, 300, 0, 0, 0)
    cur = CpuTimes(1060, 0, 540, 5110, 290, 0, 0, 0)
    result = derive_cpu(
        ProcStat(prev, {0: prev}, 1, 0), ProcStat(cur, {0: cur}, 1, 0)
    )
    assert result["cpu_iowait_cores"] is None
    assert result["cpu_busy_cores"] == pytest.approx(0.5, abs=1e-9)
    assert result["cpu_util"] == pytest.approx(0.5, abs=1e-9)
    assert result["cpu_user_cores"] == pytest.approx(0.3, abs=1e-9)
    assert result["cpu_system_cores"] == pytest.approx(0.2, abs=1e-9)
    assert result["cpu_steal_cores"] == pytest.approx(0.0, abs=1e-9)
    for key in ("cpu_core_util_min", "cpu_core_util_p50", "cpu_core_util_max"):
        assert result[key] == pytest.approx(0.5, abs=1e-9)
    assert result["cpu_active_cores"] == 1


def test_derive_cpu_one_tick_idle_step_back_is_clamped() -> None:
    result = derive_cpu(*_one_cpu(user=100, idle=-1))
    assert result["cpu_busy_cores"] == pytest.approx(1.0, abs=1e-9)
    assert result["cpu_util"] == pytest.approx(1.0, abs=1e-9)
    assert result["cpu_iowait_cores"] is None
    assert result["cpu_active_cores"] == 1


def test_derive_cpu_idle_step_back_beyond_one_tick_resets() -> None:
    result = derive_cpu(*_one_cpu(user=100, idle=-2))
    assert _all_null(result)


def test_derive_cpu_steal_decrease_resets() -> None:
    result = derive_cpu(*_one_cpu(user=100, idle=100, steal=-1))
    assert _all_null(result)


_ticks = st.integers(0, 10_000)
_signed = st.integers(-3, 10_000)
_steps = st.builds(
    CpuTimes,
    user=_ticks,
    nice=_ticks,
    system=_ticks,
    idle=_signed,
    iowait=_signed,
    irq=_ticks,
    softirq=_ticks,
    steal=_ticks,
)
_bases = st.builds(CpuTimes, *(st.integers(10_000, 10**9),) * 8)


def _kernel_total(lines: list[CpuTimes]) -> CpuTimes:
    return CpuTimes(*map(sum, zip(*lines, strict=True)))


@st.composite
def _stat_pairs(draw: st.DrawFn) -> tuple[ProcStat, ProcStat]:
    lines = draw(st.lists(st.tuples(_bases, _steps), min_size=1, max_size=8))
    prevs = [p for p, _ in lines]
    curs = [
        CpuTimes(*(a + b for a, b in zip(p, s, strict=True))) for p, s in lines
    ]
    return (
        ProcStat(_kernel_total(prevs), dict(enumerate(prevs)), 2, 1),
        ProcStat(_kernel_total(curs), dict(enumerate(curs)), 7, 3),
    )


@settings(deadline=None, max_examples=200)
@given(_stat_pairs())
def test_derive_cpu_random_pairs_hold_adr_095_invariants(
    pair: tuple[ProcStat, ProcStat],
) -> None:
    prev, cur = pair
    result = derive_cpu(prev, cur)
    n = len(cur.per_cpu)
    d = CpuTimes(*(c - p for p, c in zip(prev.total, cur.total, strict=True)))
    busy = d.user + d.nice + d.system + d.irq + d.softirq
    w = d.idle + d.iowait
    total = busy + d.steal + max(w, 0)
    aggregate_null = w < -1 or total <= 0

    assert list(result) == _STAT_COLUMNS
    assert result["procs_running"] == 7
    assert result["procs_blocked"] == 3
    assert _all_null(result) == aggregate_null
    if aggregate_null:
        return

    for key in _RATE_KEYS:
        value = result[key]
        assert value is None or (
            isinstance(value, int | float) and math.isfinite(value)
        )
    iowait_null = d.iowait < 0 or d.idle < 0
    assert (result["cpu_iowait_cores"] is None) == iowait_null
    assert all(
        result[k] is not None for k in _RATE_KEYS if k != "cpu_iowait_cores"
    )

    cores = [
        k
        for k in _RATE_KEYS
        if k.endswith("_cores") and k != "cpu_active_cores"
    ]
    for key in cores:
        value = result[key]
        assert value is None or 0 <= value <= n + 1e-9
    for key in ("cpu_util", *_RATE_KEYS[6:9]):
        fraction = result[key]
        assert fraction is not None and 0 <= fraction <= 1
    active = result["cpu_active_cores"]
    assert active is not None and 0 <= active <= n

    busy_cores = result["cpu_busy_cores"]
    assert busy_cores is not None
    user, system = result["cpu_user_cores"], result["cpu_system_cores"]
    assert user is not None and system is not None
    assert user + system == pytest.approx(busy_cores, abs=1e-9)
    steal = result["cpu_steal_cores"]
    assert steal is not None
    iowait = result["cpu_iowait_cores"]
    if iowait is None:
        assert busy_cores + steal <= n + 1e-9
    else:
        idle = d.idle / total * n
        assert busy_cores + iowait + steal + idle == pytest.approx(n, abs=1e-9)


def _files(snapshot: int, bundle: str = "synthetic") -> Mapping[str, str]:
    return load_bundle(bundle).snapshots[snapshot].files


def _job_at(files: Mapping[str, str], path: str) -> JobCgroup:
    base = f"sys/fs/cgroup{path}"
    stat = parse_flat_keyed(files[f"{base}/memory.stat"])
    return JobCgroup(
        parse_flat_keyed(files[f"{base}/cpu.stat"])["usage_usec"],
        parse_single_int(files[f"{base}/memory.current"]),
        parse_single_int(files[f"{base}/memory.peak"]),
        stat["anon"],
        stat["file"],
        stat["shmem"],
        parse_flat_keyed(files[f"{base}/memory.events"])["oom_kill"],
        parse_single_int(files[f"{base}/pids.current"]),
    )


def _step_at(files: Mapping[str, str], path: str) -> StepCgroup:
    base = f"sys/fs/cgroup{path}"
    stat = parse_flat_keyed(files[f"{base}/memory.stat"])
    return StepCgroup(
        parse_flat_keyed(files[f"{base}/cpu.stat"])["usage_usec"],
        parse_single_int(files[f"{base}/memory.current"]),
        stat["anon"],
        stat["file"],
        stat["shmem"],
    )


def _read_cgroup(
    snapshot: int, bundle: str = "synthetic"
) -> CgroupReading | None:
    files = _files(snapshot, bundle)
    job, step = job_cgroup_paths(
        parse_proc_self_cgroup(files["proc/self/cgroup"])
    )
    if job is None:
        return None
    return CgroupReading(
        _job_at(files, job), None if step is None else _step_at(files, step)
    )


def _cgroup(snapshot: int) -> CgroupReading:
    reading = _read_cgroup(snapshot)
    assert reading is not None
    return reading


def _pressure(snapshot: int) -> dict[str, PressureTotals]:
    files = _files(snapshot)
    return {
        name: parse_pressure(files[f"proc/pressure/{name}"])
        for name in ("cpu", "memory", "io")
    }


def _all_none(result: dict[str, float | int | None]) -> bool:
    return all(v is None for v in result.values())


@pytest.mark.parametrize("snapshot", [0, 1])
def test_derive_memory_synthetic_snapshot_matches_readme_table(
    snapshot: int,
) -> None:
    result = derive_memory(parse_meminfo(_files(snapshot)["proc/meminfo"]))
    assert list(result) == _MEMINFO_COLUMNS
    available = (10_050_000, 10_000_000)[snapshot] * 1024
    dirty = (800, 1000)[snapshot] * 1024
    assert result == {
        "mem_used_bytes": 16_000_000 * 1024 - available,
        "mem_available_bytes": available,
        "mem_anon_bytes": 4_096_000_000,
        "mem_cached_bytes": 6_144_000_000,
        "mem_shmem_bytes": 512_000_000,
        "mem_dirty_bytes": dirty,
        "mem_writeback_bytes": 0,
        "mem_slab_bytes": 307_200_000,
        "swap_used_bytes": 102_400_000,
    }
    if snapshot == 1:
        assert result["mem_used_bytes"] == 6_144_000_000
        assert result["mem_available_bytes"] == 10_240_000_000
        assert result["mem_dirty_bytes"] == 1_024_000


@pytest.mark.parametrize(
    "key",
    [
        "MemTotal",
        "MemAvailable",
        "AnonPages",
        "Cached",
        "Shmem",
        "Dirty",
        "Writeback",
        "Slab",
        "SwapTotal",
        "SwapFree",
    ],
)
def test_derive_memory_missing_key_raises_key_error(key: str) -> None:
    meminfo = parse_meminfo(_files(1)["proc/meminfo"])
    del meminfo[key]
    with pytest.raises(KeyError, match=key):
        derive_memory(meminfo)


def test_derive_vmstat_synthetic_pair_matches_readme_table() -> None:
    prev = parse_vmstat(_files(0)["proc/vmstat"])
    cur = parse_vmstat(_files(1)["proc/vmstat"])
    result = derive_vmstat(prev, cur, 5.0)
    assert list(result) == _VMSTAT_COLUMNS
    assert result == {"major_faults_per_s": 10.0, "oom_kills": 1}


def test_derive_vmstat_missing_baseline_returns_null_values() -> None:
    cur = parse_vmstat(_files(1)["proc/vmstat"])
    result = derive_vmstat(None, cur, 5.0)
    assert list(result) == _VMSTAT_COLUMNS
    assert _all_none(result)


def test_derive_vmstat_counter_decrease_nulls_only_that_value() -> None:
    prev = parse_vmstat(_files(0)["proc/vmstat"])
    cur = parse_vmstat(_files(1)["proc/vmstat"])
    reset_faults = derive_vmstat({**prev, "pgmajfault": 2000}, cur, 5.0)
    assert reset_faults == {"major_faults_per_s": None, "oom_kills": 1}
    reset_oom = derive_vmstat({**prev, "oom_kill": 5}, cur, 5.0)
    assert reset_oom == {"major_faults_per_s": 10.0, "oom_kills": None}


@pytest.mark.parametrize("dt_s", [0.0, -5.0])
def test_derive_vmstat_non_positive_dt_nulls_only_the_rate(dt_s: float) -> None:
    prev = parse_vmstat(_files(0)["proc/vmstat"])
    cur = parse_vmstat(_files(1)["proc/vmstat"])
    result = derive_vmstat(prev, cur, dt_s)
    assert result == {"major_faults_per_s": None, "oom_kills": 1}


def test_derive_job_synthetic_pair_matches_readme_table() -> None:
    result = derive_job(_cgroup(0), _cgroup(1), 5.0)
    assert list(result) == _CGROUP_COLUMNS
    assert result == {
        "job_cpu_cores": pytest.approx(1.99, abs=1e-9),
        "job_mem_bytes": 8_080_000_000,
        "job_mem_anon_bytes": 5_035_000_000,
        "job_mem_file_bytes": 2_546_000_000,
        "job_mem_shmem_bytes": 400_000_000,
        "job_mem_peak_bytes": 9_100_000_000,
        "job_pids": 42,
        "job_oom_kills": 1,
    }


def test_derive_job_without_step_subtracts_nothing() -> None:
    prev = _cgroup(0)._replace(step=None)
    cur = _cgroup(1)._replace(step=None)
    result = derive_job(prev, cur, 5.0)
    assert result == {
        "job_cpu_cores": 2.0,
        "job_mem_bytes": 8_100_000_000,
        "job_mem_anon_bytes": 5_050_000_000,
        "job_mem_file_bytes": 2_550_000_000,
        "job_mem_shmem_bytes": 400_000_000,
        "job_mem_peak_bytes": 9_100_000_000,
        "job_pids": 42,
        "job_oom_kills": 1,
    }


def test_derive_job_no_job_cgroup_returns_all_null() -> None:
    result = derive_job(_cgroup(0), None, 5.0)
    assert list(result) == _CGROUP_COLUMNS
    assert _all_none(result)
    assert _all_none(derive_job(None, None, 5.0))


def test_derive_job_workstation_bundle_has_no_job_cgroup() -> None:
    prev, cur = _read_cgroup(0, "workstation"), _read_cgroup(1, "workstation")
    assert prev is None and cur is None
    assert _all_none(derive_job(prev, cur, 5.0))


def test_derive_job_step_larger_than_job_clamps_values_to_zero() -> None:
    prev, cur = _cgroup(0), _cgroup(1)
    prev = prev._replace(step=StepCgroup(0, 0, 0, 0, 0))
    cur = cur._replace(
        step=StepCgroup(
            20_000_000, 9_000_000_000, 6_000_000_000, 3_000_000_000, 500_000_000
        )
    )
    result = derive_job(prev, cur, 5.0)
    assert result["job_cpu_cores"] == 0.0
    for key in (
        "job_mem_bytes",
        "job_mem_anon_bytes",
        "job_mem_file_bytes",
        "job_mem_shmem_bytes",
    ):
        assert result[key] == 0
    assert result["job_mem_peak_bytes"] == 9_100_000_000
    assert result["job_pids"] == 42


def test_derive_job_missing_baseline_keeps_gauges_and_nulls_deltas() -> None:
    result = derive_job(None, _cgroup(1), 5.0)
    assert result == {
        "job_cpu_cores": None,
        "job_mem_bytes": 8_080_000_000,
        "job_mem_anon_bytes": 5_035_000_000,
        "job_mem_file_bytes": 2_546_000_000,
        "job_mem_shmem_bytes": 400_000_000,
        "job_mem_peak_bytes": 9_100_000_000,
        "job_pids": 42,
        "job_oom_kills": None,
    }


@pytest.mark.parametrize("prev_has_step", [True, False])
def test_derive_job_step_appearing_or_disappearing_nulls_only_cpu(
    prev_has_step: bool,
) -> None:
    prev, cur = _cgroup(0), _cgroup(1)
    if prev_has_step:
        cur = cur._replace(step=None)
    else:
        prev = prev._replace(step=None)
    result = derive_job(prev, cur, 5.0)
    assert result["job_cpu_cores"] is None
    assert result["job_oom_kills"] == 1
    assert result["job_mem_bytes"] == (
        8_100_000_000 if prev_has_step else 8_080_000_000
    )
    assert result["job_pids"] == 42


def test_derive_job_usage_decrease_nulls_cpu_only() -> None:
    prev, cur = _cgroup(0), _cgroup(1)
    job_reset = prev._replace(job=prev.job._replace(usage_usec=10**9))
    assert derive_job(job_reset, cur, 5.0)["job_cpu_cores"] is None
    assert prev.step is not None
    step_reset = prev._replace(step=prev.step._replace(usage_usec=10**9))
    result = derive_job(step_reset, cur, 5.0)
    assert result["job_cpu_cores"] is None
    assert result["job_oom_kills"] == 1


def test_derive_job_oom_decrease_nulls_oom_only() -> None:
    prev, cur = _cgroup(0), _cgroup(1)
    prev = prev._replace(job=prev.job._replace(oom_kill=5))
    result = derive_job(prev, cur, 5.0)
    assert result["job_oom_kills"] is None
    assert result["job_cpu_cores"] == pytest.approx(1.99, abs=1e-9)


@pytest.mark.parametrize("dt_s", [0.0, -5.0])
def test_derive_job_non_positive_dt_nulls_only_cpu(dt_s: float) -> None:
    result = derive_job(_cgroup(0), _cgroup(1), dt_s)
    assert result["job_cpu_cores"] is None
    assert result["job_oom_kills"] == 1
    assert result["job_mem_bytes"] == 8_080_000_000


def test_derive_pressure_synthetic_pair_matches_readme_table() -> None:
    result = derive_pressure(_pressure(0), _pressure(1), 5.0)
    assert list(result) == _PRESSURE_COLUMNS
    expected = {
        "psi_cpu_some": 0.1,
        "psi_mem_some": 0.05,
        "psi_mem_full": 0.02,
        "psi_io_some": 0.2,
        "psi_io_full": 0.1,
    }
    assert result == {
        k: pytest.approx(v, abs=1e-12) for k, v in expected.items()
    }


def test_derive_pressure_delta_above_the_window_clamps_to_one() -> None:
    prev, cur = _pressure(0), _pressure(1)
    cur = {**cur, "io": PressureTotals(10**9, 10**9)}
    result = derive_pressure(prev, cur, 5.0)
    assert result["psi_io_some"] == 1.0
    assert result["psi_io_full"] == 1.0
    assert result["psi_cpu_some"] == pytest.approx(0.1, abs=1e-12)


def test_derive_pressure_missing_baseline_returns_null_values() -> None:
    result = derive_pressure(None, _pressure(1), 5.0)
    assert list(result) == _PRESSURE_COLUMNS
    assert _all_none(result)


def test_derive_pressure_missing_full_line_nulls_only_full_columns() -> None:
    prev, cur = _pressure(0), _pressure(1)
    prev = {
        **prev,
        "cpu": PressureTotals(prev["cpu"].some_total_us, None),
        "memory": PressureTotals(prev["memory"].some_total_us, None),
    }
    cur = {**cur, "io": PressureTotals(cur["io"].some_total_us, None)}
    result = derive_pressure(prev, cur, 5.0)
    assert result["psi_cpu_some"] == pytest.approx(0.1, abs=1e-12)
    assert result["psi_mem_some"] == pytest.approx(0.05, abs=1e-12)
    assert result["psi_io_some"] == pytest.approx(0.2, abs=1e-12)
    assert result["psi_mem_full"] is None
    assert result["psi_io_full"] is None


def test_derive_pressure_total_decrease_nulls_only_that_column() -> None:
    prev, cur = _pressure(0), _pressure(1)
    prev = {**prev, "memory": PressureTotals(10**9, 100_000)}
    result = derive_pressure(prev, cur, 5.0)
    assert result["psi_mem_some"] is None
    assert result["psi_mem_full"] == pytest.approx(0.02, abs=1e-12)
    assert result["psi_io_some"] == pytest.approx(0.2, abs=1e-12)


@pytest.mark.parametrize("dt_s", [0.0, -5.0])
def test_derive_pressure_non_positive_dt_returns_null_values(
    dt_s: float,
) -> None:
    result = derive_pressure(_pressure(0), _pressure(1), dt_s)
    assert list(result) == _PRESSURE_COLUMNS
    assert _all_none(result)


_counters = st.integers(0, 10**12)
_jobs = st.builds(JobCgroup, *(_counters,) * 8)
_cgroup_steps = st.builds(StepCgroup, *(_counters,) * 5)
_readings = st.builds(CgroupReading, _jobs, st.none() | _cgroup_steps)
_dts = st.one_of(st.floats(1e-3, 1e4), st.floats(-10.0, 0.0))


@settings(deadline=None, max_examples=200)
@given(st.none() | _readings, st.none() | _readings, _dts)
def test_derive_job_random_readings_hold_adr_095_invariants(
    prev: CgroupReading | None, cur: CgroupReading | None, dt_s: float
) -> None:
    result = derive_job(prev, cur, dt_s)
    assert list(result) == _CGROUP_COLUMNS
    for value in result.values():
        assert value is None or (math.isfinite(value) and value >= 0)
    if cur is None:
        assert _all_none(result)
        return

    job, step = cur
    assert result["job_mem_peak_bytes"] == job.memory_peak
    assert result["job_pids"] == job.pids
    gauges = {
        "job_mem_bytes": (job.memory_current, step and step.memory_current),
        "job_mem_anon_bytes": (job.anon, step and step.anon),
        "job_mem_file_bytes": (job.file, step and step.file),
        "job_mem_shmem_bytes": (job.shmem, step and step.shmem),
    }
    for key, (job_value, step_value) in gauges.items():
        value = result[key]
        assert value is not None and 0 <= value <= job_value
        assert value == (
            job_value if step_value is None else max(0, job_value - step_value)
        )

    if prev is None:
        assert result["job_cpu_cores"] is None
        assert result["job_oom_kills"] is None
        return
    d_oom = job.oom_kill - prev.job.oom_kill
    assert result["job_oom_kills"] == (d_oom if d_oom >= 0 else None)
    d_job = job.usage_usec - prev.job.usage_usec
    d_step = (
        0
        if step is None or prev.step is None
        else step.usage_usec - prev.step.usage_usec
    )
    cpu = result["job_cpu_cores"]
    if (
        (step is None) != (prev.step is None)
        or d_job < 0
        or d_step < 0
        or dt_s <= 0
    ):
        assert cpu is None
    else:
        expected = max(0, d_job - d_step) / (1e6 * dt_s)
        assert cpu == pytest.approx(expected, rel=1e-12, abs=0)


_totals = st.builds(PressureTotals, _counters, st.none() | _counters)
_pressures = st.fixed_dictionaries(
    {"cpu": _totals, "memory": _totals, "io": _totals}
)
_PSI_SOURCES = {
    "psi_cpu_some": ("cpu", "some_total_us"),
    "psi_mem_some": ("memory", "some_total_us"),
    "psi_mem_full": ("memory", "full_total_us"),
    "psi_io_some": ("io", "some_total_us"),
    "psi_io_full": ("io", "full_total_us"),
}


@settings(deadline=None, max_examples=200)
@given(st.none() | _pressures, _pressures, _dts)
def test_derive_pressure_random_totals_hold_adr_095_invariants(
    prev: dict[str, PressureTotals] | None,
    cur: dict[str, PressureTotals],
    dt_s: float,
) -> None:
    result = derive_pressure(prev, cur, dt_s)
    assert list(result) == _PRESSURE_COLUMNS
    assert list(_PSI_SOURCES) == _PRESSURE_COLUMNS
    for column, (resource, field) in _PSI_SOURCES.items():
        value = result[column]
        if prev is None:
            assert value is None
            continue
        before = getattr(prev[resource], field)
        after = getattr(cur[resource], field)
        if before is None or after is None or after < before or dt_s <= 0:
            assert value is None
        else:
            assert value is not None and 0.0 <= value <= 1.0
            assert value == pytest.approx(
                min(1.0, (after - before) / (1e6 * dt_s)), rel=1e-12, abs=0
            )


_DISKS = ("nvme0n1", "nvme1n1")
_IFACES = ("ens5",)
_WORKSPACE = "/shared-zfs/ws-test"
_EFA_PORT = "sys/class/infiniband/efa_0/ports/1/hw_counters"
_NFS_BASE = NfsCounters(*(1000,) * 11)


def _disk_reads(snapshot: int) -> dict[str, DiskCounters]:
    return parse_diskstats(_files(snapshot)["proc/diskstats"])


def _net_reads(snapshot: int) -> dict[str, NetCounters]:
    return parse_net_dev(_files(snapshot)["proc/net/dev"])


def _nfs_read(snapshot: int) -> NfsCounters:
    entry = select_mount(
        parse_mountstats(_files(snapshot)["proc/self/mountstats"]), _WORKSPACE
    )
    assert entry is not None
    return parse_nfs_counters(entry)


def _efa_reads(snapshot: int) -> dict[str, EfaCounters]:
    files = _files(snapshot)
    return {
        _EFA_PORT: EfaCounters(
            parse_counter(files[f"{_EFA_PORT}/rx_bytes"]),
            parse_counter(files[f"{_EFA_PORT}/tx_bytes"]),
        )
    }


def _nfs_pair(**deltas: int) -> tuple[NfsCounters, NfsCounters]:
    cur = NfsCounters(
        *(
            value + deltas.get(field, 0)
            for field, value in zip(NfsCounters._fields, _NFS_BASE, strict=True)
        )
    )
    return _NFS_BASE, cur


def _io(dt_s: float) -> tuple[Values, Values, Values, Values]:
    return (
        derive_disk(_disk_reads(0), _disk_reads(1), _DISKS, dt_s),
        derive_nfs(_nfs_read(0), _nfs_read(1), dt_s),
        derive_net(_net_reads(0), _net_reads(1), _IFACES, dt_s),
        derive_efa(_efa_reads(0), _efa_reads(1), dt_s),
    )


def test_derive_disk_synthetic_pair_matches_readme_table() -> None:
    result = _io(5.0)[0]
    assert list(result) == _DISK_COLUMNS
    assert result == {
        "disk_read_bytes_per_s": 1_024_000.0,
        "disk_write_bytes_per_s": 3_072_000.0,
        "disk_busy_max": 0.5,
    }


def test_derive_nfs_synthetic_pair_matches_readme_table() -> None:
    result = _io(5.0)[1]
    assert list(result) == _NFS_COLUMNS
    assert result == {
        "nfs_read_bytes_per_s": 1_000_000.0,
        "nfs_write_bytes_per_s": 500_000.0,
        "nfs_server_read_bytes_per_s": 1_100_000.0,
        "nfs_server_write_bytes_per_s": 600_000.0,
        "nfs_read_ops_per_s": 10.0,
        "nfs_write_ops_per_s": 0.0,
        "nfs_ops_per_s": 30.0,
        "nfs_read_rtt_ms": 5.0,
        "nfs_write_rtt_ms": None,
    }


def test_derive_net_synthetic_pair_matches_readme_table() -> None:
    result = _io(5.0)[2]
    assert list(result) == _NET_COLUMNS
    assert result == {
        "net_rx_bytes_per_s": 1_000_000.0,
        "net_tx_bytes_per_s": 500_000.0,
        "net_errors_per_s": 1.0,
    }


def test_derive_efa_synthetic_pair_matches_readme_table() -> None:
    result = _io(5.0)[3]
    assert list(result) == _EFA_COLUMNS
    assert result == {
        "efa_rx_bytes_per_s": 100_000_000.0,
        "efa_tx_bytes_per_s": 40_000_000.0,
    }


def test_derive_io_half_the_interval_doubles_rates_and_keeps_rtt() -> None:
    disk, nfs, net, efa = _io(2.5)
    assert disk == {
        "disk_read_bytes_per_s": 2_048_000.0,
        "disk_write_bytes_per_s": 6_144_000.0,
        "disk_busy_max": 1.0,
    }
    assert nfs == {
        "nfs_read_bytes_per_s": 2_000_000.0,
        "nfs_write_bytes_per_s": 1_000_000.0,
        "nfs_server_read_bytes_per_s": 2_200_000.0,
        "nfs_server_write_bytes_per_s": 1_200_000.0,
        "nfs_read_ops_per_s": 20.0,
        "nfs_write_ops_per_s": 0.0,
        "nfs_ops_per_s": 60.0,
        "nfs_read_rtt_ms": 5.0,
        "nfs_write_rtt_ms": None,
    }
    assert net == {
        "net_rx_bytes_per_s": 2_000_000.0,
        "net_tx_bytes_per_s": 1_000_000.0,
        "net_errors_per_s": 2.0,
    }
    assert efa == {
        "efa_rx_bytes_per_s": 200_000_000.0,
        "efa_tx_bytes_per_s": 80_000_000.0,
    }


def test_derive_nfs_zero_read_ops_nulls_only_read_rtt() -> None:
    prev, cur = _nfs_pair(
        write_ops=10,
        write_rtt_ms=50,
        total_ops=10,
        normal_write=5000,
        server_write=6000,
    )
    result = derive_nfs(prev, cur, 5.0)
    assert result["nfs_read_rtt_ms"] is None
    assert result["nfs_write_rtt_ms"] == 5.0
    assert [k for k, v in result.items() if v is None] == ["nfs_read_rtt_ms"]
    assert result["nfs_write_ops_per_s"] == 2.0
    assert result["nfs_ops_per_s"] == 2.0
    assert result["nfs_write_bytes_per_s"] == 1000.0
    assert result["nfs_server_write_bytes_per_s"] == 1200.0


def test_derive_nfs_idle_interval_nulls_only_both_rtts() -> None:
    result = derive_nfs(*_nfs_pair(), 5.0)
    assert list(result) == _NFS_COLUMNS
    nulls = [k for k, v in result.items() if v is None]
    assert nulls == ["nfs_read_rtt_ms", "nfs_write_rtt_ms"]
    assert all(v == 0.0 for v in result.values() if v is not None)


@pytest.mark.parametrize("field", NfsCounters._fields)
def test_derive_nfs_any_field_decrease_nulls_all_values(field: str) -> None:
    result = derive_nfs(*_nfs_pair(**{field: -1}), 5.0)
    assert list(result) == _NFS_COLUMNS
    assert _all_none(result)


def test_derive_nfs_missing_reading_returns_null_values() -> None:
    cur = _nfs_read(1)
    assert _all_none(derive_nfs(None, cur, 5.0))
    assert _all_none(derive_nfs(cur, None, 5.0))
    assert list(derive_nfs(None, None, 5.0)) == _NFS_COLUMNS
    assert _all_none(derive_nfs(None, None, 5.0))


def test_derive_disk_busy_above_one_clamps_to_one() -> None:
    prev = {"nvme0n1": DiskCounters(0, 0, 0)}
    cur = {"nvme0n1": DiskCounters(0, 0, 5004)}
    result = derive_disk(prev, cur, ("nvme0n1",), 5.0)
    assert result["disk_busy_max"] == 1.0
    assert result["disk_read_bytes_per_s"] == 0.0
    assert result["disk_write_bytes_per_s"] == 0.0


def test_derive_disk_sectors_read_decrease_nulls_all_values() -> None:
    cur = _disk_reads(1)
    cur["nvme0n1"] = cur["nvme0n1"]._replace(sectors_read=199_999)
    result = derive_disk(_disk_reads(0), cur, _DISKS, 5.0)
    assert list(result) == _DISK_COLUMNS
    assert _all_none(result)


def test_derive_disk_partition_outside_disks_is_ignored() -> None:
    prev, cur = _disk_reads(0), _disk_reads(1)
    assert "nvme0n1p1" in prev and "nvme0n1p1" in cur
    expected = {
        "disk_read_bytes_per_s": 1_024_000.0,
        "disk_write_bytes_per_s": 3_072_000.0,
        "disk_busy_max": 0.5,
    }
    assert derive_disk(prev, cur, _DISKS, 5.0) == expected
    cur["nvme0n1p1"] = DiskCounters(0, 0, 0)
    assert derive_disk(prev, cur, _DISKS, 5.0) == expected


def test_derive_disk_device_in_one_read_is_ignored() -> None:
    prev, cur = _disk_reads(0), _disk_reads(1)
    del prev["nvme0n1"]
    result = derive_disk(prev, cur, _DISKS, 5.0)
    assert result == {
        "disk_read_bytes_per_s": 0.0,
        "disk_write_bytes_per_s": 1_024_000.0,
        "disk_busy_max": 0.2,
    }
    del cur["nvme1n1"]
    assert _all_none(derive_disk(prev, cur, _DISKS, 5.0))


def test_derive_net_sums_the_listed_interfaces() -> None:
    result = derive_net(
        _net_reads(0), _net_reads(1), ("ens5", "lo", "docker0"), 5.0
    )
    assert result == {
        "net_rx_bytes_per_s": 5_000_000.0,
        "net_tx_bytes_per_s": 4_500_000.0,
        "net_errors_per_s": 1.0,
    }


def test_derive_net_interface_in_one_read_is_ignored() -> None:
    prev = {**_net_reads(0), "ens6": NetCounters(10**9, 10**9, 0)}
    cur = {**_net_reads(1), "ens7": NetCounters(1, 1, 1)}
    result = derive_net(prev, cur, ("ens5", "ens6", "ens7"), 5.0)
    assert result == _io(5.0)[2]


def test_derive_net_counter_decrease_nulls_all_values() -> None:
    prev = _net_reads(0)
    prev["ens5"] = prev["ens5"]._replace(errors=6)
    result = derive_net(prev, _net_reads(1), _IFACES, 5.0)
    assert list(result) == _NET_COLUMNS
    assert _all_none(result)


def test_derive_efa_sums_every_port_present_in_both_reads() -> None:
    prev = {"p1": EfaCounters(0, 0), "p2": EfaCounters(100, 200)}
    cur = {
        "p1": EfaCounters(1000, 2000),
        "p2": EfaCounters(600, 700),
        "p3": EfaCounters(10**9, 10**9),
    }
    assert derive_efa(prev, cur, 5.0) == {
        "efa_rx_bytes_per_s": 300.0,
        "efa_tx_bytes_per_s": 500.0,
    }


def test_derive_efa_counter_decrease_nulls_all_values() -> None:
    cur = _efa_reads(1)
    cur[_EFA_PORT] = cur[_EFA_PORT]._replace(tx_bytes=0)
    result = derive_efa(_efa_reads(0), cur, 5.0)
    assert list(result) == _EFA_COLUMNS
    assert _all_none(result)


def test_derive_io_empty_sets_return_null_values() -> None:
    results = (
        derive_disk(_disk_reads(0), _disk_reads(1), (), 5.0),
        derive_disk(_disk_reads(0), _disk_reads(1), ("sdz",), 5.0),
        derive_nfs(_nfs_read(0), None, 5.0),
        derive_net(_net_reads(0), _net_reads(1), (), 5.0),
        derive_efa(_efa_reads(0), {}, 5.0),
    )
    assert all(_all_none(r) for r in results)
    assert [list(r) for r in results] == [
        _DISK_COLUMNS,
        _DISK_COLUMNS,
        _NFS_COLUMNS,
        _NET_COLUMNS,
        _EFA_COLUMNS,
    ]


def test_derive_io_missing_baseline_returns_null_values() -> None:
    results = (
        derive_disk(None, _disk_reads(1), _DISKS, 5.0),
        derive_net(None, _net_reads(1), _IFACES, 5.0),
        derive_efa(None, _efa_reads(1), 5.0),
    )
    assert all(_all_none(r) for r in results)
    assert [list(r) for r in results] == [
        _DISK_COLUMNS,
        _NET_COLUMNS,
        _EFA_COLUMNS,
    ]


@pytest.mark.parametrize("dt_s", [0.0, -5.0])
def test_derive_io_non_positive_dt_returns_null_values(dt_s: float) -> None:
    results = _io(dt_s)
    assert all(_all_none(r) for r in results)
    assert [list(r) for r in results] == [
        _DISK_COLUMNS,
        _NFS_COLUMNS,
        _NET_COLUMNS,
        _EFA_COLUMNS,
    ]


_NAMES = ("sda", "sdb", "nvme0n1", "nvme1n1")
_steps_after = st.integers(-1, 10_000)


@st.composite
def _disk_cases(
    draw: st.DrawFn,
) -> tuple[dict[str, DiskCounters], dict[str, DiskCounters], tuple[str, ...]]:
    names = st.sampled_from(_NAMES)
    prev = draw(
        st.dictionaries(names, st.builds(DiskCounters, *(_counters,) * 3))
    )
    cur = {
        name: DiskCounters(*(v + draw(_steps_after) for v in counters))
        for name, counters in prev.items()
        if draw(st.booleans())
    }
    cur.update(
        draw(st.dictionaries(names, st.builds(DiskCounters, *(_counters,) * 3)))
    )
    disks = tuple(draw(st.lists(names, unique=True)))
    return prev, cur, disks


@settings(deadline=None, max_examples=200)
@given(_disk_cases(), _dts)
def test_derive_disk_random_reads_hold_adr_095_invariants(
    case: tuple[
        dict[str, DiskCounters], dict[str, DiskCounters], tuple[str, ...]
    ],
    dt_s: float,
) -> None:
    prev, cur, disks = case
    result = derive_disk(prev, cur, disks, dt_s)
    assert list(result) == _DISK_COLUMNS
    for value in result.values():
        assert value is None or (math.isfinite(value) and value >= 0)

    shared = [n for n in disks if n in prev and n in cur]
    steps = [
        tuple(c - p for p, c in zip(prev[n], cur[n], strict=True))
        for n in shared
    ]
    if not shared or any(min(s) < 0 for s in steps) or dt_s <= 0:
        assert _all_none(result)
        return
    read, written, ticks = zip(*steps, strict=True)
    assert result["disk_read_bytes_per_s"] == pytest.approx(
        sum(read) * 512 / dt_s, rel=1e-12, abs=0
    )
    assert result["disk_write_bytes_per_s"] == pytest.approx(
        sum(written) * 512 / dt_s, rel=1e-12, abs=0
    )
    busy = result["disk_busy_max"]
    assert busy is not None and 0.0 <= busy <= 1.0
    assert busy == pytest.approx(
        min(1.0, max(ticks) / (1000 * dt_s)), rel=1e-12, abs=0
    )


@st.composite
def _nfs_cases(draw: st.DrawFn) -> tuple[NfsCounters, NfsCounters]:
    prev = draw(st.builds(NfsCounters, *(_counters,) * 11))
    cur = NfsCounters(*(v + draw(_steps_after) for v in prev))
    return prev, cur


@settings(deadline=None, max_examples=200)
@given(_nfs_cases(), _dts)
def test_derive_nfs_random_reads_hold_adr_095_invariants(
    case: tuple[NfsCounters, NfsCounters], dt_s: float
) -> None:
    prev, cur = case
    result = derive_nfs(prev, cur, dt_s)
    assert list(result) == _NFS_COLUMNS
    for value in result.values():
        assert value is None or (math.isfinite(value) and value >= 0)

    d = NfsCounters(*(c - p for p, c in zip(prev, cur, strict=True)))
    if min(d) < 0 or dt_s <= 0:
        assert _all_none(result)
        return
    counts = {
        "nfs_read_bytes_per_s": d.normal_read + d.direct_read,
        "nfs_write_bytes_per_s": d.normal_write + d.direct_write,
        "nfs_server_read_bytes_per_s": d.server_read,
        "nfs_server_write_bytes_per_s": d.server_write,
        "nfs_read_ops_per_s": d.read_ops,
        "nfs_write_ops_per_s": d.write_ops,
        "nfs_ops_per_s": d.total_ops,
    }
    for key, count in counts.items():
        assert result[key] == pytest.approx(count / dt_s, rel=1e-12, abs=0)
    for key, rtt, ops in (
        ("nfs_read_rtt_ms", d.read_rtt_ms, d.read_ops),
        ("nfs_write_rtt_ms", d.write_rtt_ms, d.write_ops),
    ):
        if ops == 0:
            assert result[key] is None
        else:
            assert result[key] == pytest.approx(rtt / ops, rel=1e-12, abs=0)
