"""ADR-068/ADR-070/ADR-093/ADR-075: the telemetry column contract.

It also holds the ADR-071/ADR-076/ADR-077/ADR-085 roll-up specs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final = 1
PHASES: Final = ("model", "finalize")

HEADER_KIND: Final = "hpcmu.telemetry.header"
WALK_KIND: Final = "hpcmu.telemetry.walk"
ERROR_KIND: Final = "hpcmu.telemetry.error"

SAMPLES_KIND: Final = "hpcmu.telemetry.samples"
WORKSPACE_KIND: Final = "hpcmu.telemetry.workspace"
SUMMARY_KIND: Final = "hpcmu.telemetry.summary"

# ADR-085: the files published below saidas/telemetry, in upload order.
SAMPLES_FILE: Final = "samples.parquet"
WORKSPACE_FILE: Final = "workspace.parquet"
SUMMARY_FILE: Final = "summary.json"
EXPORT_FILES: Final = (SAMPLES_FILE, WORKSPACE_FILE, SUMMARY_FILE)


@dataclass(frozen=True, slots=True)
class MetadataKey:
    name: str
    meaning: str


# R52: the key-value metadata of both Parquet files.
PARQUET_METADATA_KEYS: Final[tuple[MetadataKey, ...]] = (
    MetadataKey(
        name="hpcmu.telemetry.schema_version",
        meaning="SCHEMA_VERSION as a decimal string",
    ),
    MetadataKey(
        name="hpcmu.telemetry.kind",
        meaning="SAMPLES_KIND or WORKSPACE_KIND",
    ),
    MetadataKey(
        name="hpcmu.tool_version",
        meaning="the hpc-model-utils version that ran the export",
    ),
    MetadataKey(
        name="hpcmu.run_id",
        meaning="the run_id of state.json",
    ),
    MetadataKey(
        name="hpcmu.model",
        meaning="the plugin name of the run (state.json plugin)",
    ),
    MetadataKey(
        name="hpcmu.headers",
        meaning=(
            "the JSON list of the valid staging headers, in sample sort order"
        ),
    ),
)

DTYPES: Final = frozenset(
    {
        "TIMESTAMP_MS_UTC",
        "STRING",
        "INT16",
        "INT32",
        "INT64",
        "FLOAT32",
        "FLOAT64",
    }
)
UNITS: Final = frozenset(
    {"-", "ms", "s", "cores", "fraction", "count", "1/s", "bytes", "bytes/s"}
)
# Written in canonical order, which a frozenset does not keep. "header" is
# used only by error lines, for static-fact failures.
SOURCES: Final = frozenset(
    {
        "identity",
        "sampler",
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
        "walk",
        "statvfs",
        "header",
    }
)


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    dtype: str
    nullable: bool
    unit: str
    source: str
    meaning: str


@dataclass(frozen=True, slots=True)
class HeaderField:
    name: str
    json_type: str
    nullable: bool
    meaning: str


# Conventions behind the meanings: Δ is the difference between this
# sample's read and the previous one, and online CPUs is the number of cpuN
# lines in /proc/stat. Every rate is null on a counter decrease, on a
# missing baseline, or after a failed read of its source; the /proc/stat
# CPU columns follow the per-line ADR-093 rule written into their meanings.
SAMPLE_COLUMNS: Final[tuple[Column, ...]] = (
    Column(
        name="ts",
        dtype="TIMESTAMP_MS_UTC",
        nullable=False,
        unit="-",
        source="identity",
        meaning=(
            "scheduled sample instant, a multiple of interval_s on the "
            "wall-clock grid, except the off-grid final sample taken at "
            "SIGTERM or SIGINT; strictly increasing per staging file"
        ),
    ),
    Column(
        name="phase",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="job phase, one of PHASES",
    ),
    Column(
        name="job_id",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="Slurm job id (SLURM_JOB_ID)",
    ),
    Column(
        name="restart",
        dtype="INT16",
        nullable=False,
        unit="-",
        source="identity",
        meaning="SLURM_RESTART_COUNT of the attempt, 0 when unset",
    ),
    Column(
        name="node",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="Slurm node name (SLURMD_NODENAME)",
    ),
    Column(
        name="node_index",
        dtype="INT16",
        nullable=False,
        unit="-",
        source="identity",
        meaning="SLURM_NODEID",
    ),
    Column(
        name="seq",
        dtype="INT32",
        nullable=False,
        unit="count",
        source="identity",
        meaning=(
            "0-based sample number within the staging file; a gap means a "
            "dropped line"
        ),
    ),
    Column(
        name="dt_s",
        dtype="FLOAT32",
        nullable=True,
        unit="s",
        source="sampler",
        meaning=(
            "monotonic seconds since the previous counter read; for seq 0 the "
            "previous read is the unwritten start-up baseline"
        ),
    ),
    Column(
        name="sample_cost_ms",
        dtype="FLOAT32",
        nullable=True,
        unit="ms",
        source="sampler",
        meaning=(
            "sampler CPU time (getrusage user + system) spent on this sample, "
            "the workspace walk excluded"
        ),
    ),
    Column(
        name="sampler_maxrss_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="sampler",
        meaning=(
            "sampler peak resident set size so far (ru_maxrss × 1024); "
            "ru_maxrss carries over execve, so it can only over-report"
        ),
    ),
    Column(
        name="cpu_busy_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="stat",
        meaning=(
            "Δbusy / T × online CPUs from the /proc/stat cpu line, with Δbusy "
            "= Δ(user + nice + system + irq + softirq) and T = Δbusy + Δsteal "
            "+ max(Δidle + Δiowait, 0); null when a busy field or steal "
            "decreases, when Δidle + Δiowait < −1, or when T ≤ 0; steal is not "
            "busy"
        ),
    ),
    Column(
        name="cpu_util",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="stat",
        meaning="cpu_busy_cores / online CPUs",
    ),
    Column(
        name="cpu_user_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="stat",
        meaning="Δ(user + nice) / T × online CPUs; null as cpu_busy_cores",
    ),
    Column(
        name="cpu_system_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="stat",
        meaning=(
            "Δ(system + irq + softirq) / T × online CPUs; null as "
            "cpu_busy_cores"
        ),
    ),
    Column(
        name="cpu_iowait_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="stat",
        meaning=(
            "Δiowait / T × online CPUs; null as cpu_busy_cores, and also null "
            "when Δiowait < 0 or Δidle < 0"
        ),
    ),
    Column(
        name="cpu_steal_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="stat",
        meaning="Δsteal / T × online CPUs; null as cpu_busy_cores",
    ),
    Column(
        name="cpu_core_util_min",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="stat",
        meaning=(
            "minimum over the cpuN lines of Δbusy / T of that CPU, under the "
            "same null rule per CPU; a CPU meeting the null rule is left out, "
            "and the value is null when every CPU is"
        ),
    ),
    Column(
        name="cpu_core_util_p50",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="stat",
        meaning=(
            "median over the cpuN lines of Δbusy / T of that CPU, under the "
            "same null rule per CPU; a CPU meeting the null rule is left out, "
            "and the value is null when every CPU is"
        ),
    ),
    Column(
        name="cpu_core_util_max",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="stat",
        meaning=(
            "maximum over the cpuN lines of Δbusy / T of that CPU, under the "
            "same null rule per CPU; a CPU meeting the null rule is left out, "
            "and the value is null when every CPU is"
        ),
    ),
    Column(
        name="cpu_active_cores",
        dtype="INT16",
        nullable=True,
        unit="count",
        source="stat",
        meaning=(
            "number of cpuN lines with Δbusy / T >= 0.5, under the same null "
            "rule per CPU; null when every CPU is left out"
        ),
    ),
    Column(
        name="load1",
        dtype="FLOAT32",
        nullable=True,
        unit="-",
        source="loadavg",
        meaning="first field of /proc/loadavg",
    ),
    Column(
        name="procs_running",
        dtype="INT32",
        nullable=True,
        unit="count",
        source="stat",
        meaning="procs_running of /proc/stat at the sample",
    ),
    Column(
        name="procs_blocked",
        dtype="INT32",
        nullable=True,
        unit="count",
        source="stat",
        meaning="procs_blocked of /proc/stat at the sample",
    ),
    Column(
        name="mem_used_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="(MemTotal − MemAvailable) × 1024",
    ),
    Column(
        name="mem_available_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="MemAvailable × 1024",
    ),
    Column(
        name="mem_anon_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="AnonPages × 1024",
    ),
    Column(
        name="mem_cached_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="Cached × 1024",
    ),
    Column(
        name="mem_shmem_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="Shmem × 1024; includes MPI shared-memory segments in /dev/shm",
    ),
    Column(
        name="mem_dirty_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="Dirty × 1024",
    ),
    Column(
        name="mem_writeback_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="Writeback × 1024",
    ),
    Column(
        name="mem_slab_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="Slab × 1024",
    ),
    Column(
        name="swap_used_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="meminfo",
        meaning="(SwapTotal − SwapFree) × 1024",
    ),
    Column(
        name="major_faults_per_s",
        dtype="FLOAT32",
        nullable=True,
        unit="1/s",
        source="vmstat",
        meaning="Δpgmajfault / dt_s",
    ),
    Column(
        name="oom_kills",
        dtype="INT32",
        nullable=True,
        unit="count",
        source="vmstat",
        meaning="Δoom_kill of /proc/vmstat in this interval",
    ),
    Column(
        name="job_cpu_cores",
        dtype="FLOAT32",
        nullable=True,
        unit="cores",
        source="cgroup",
        meaning=(
            "(Δusage_usec of the job cgroup − Δusage_usec of the sampler's own "
            "step) / (1e6 × dt_s), clamped at 0; no subtraction when "
            "own_step_cgroup is null"
        ),
    ),
    Column(
        name="job_mem_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="cgroup",
        meaning=(
            "memory.current of the job cgroup − memory.current of the own "
            "step, clamped at 0"
        ),
    ),
    Column(
        name="job_mem_anon_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="cgroup",
        meaning=(
            "memory.stat anon of the job cgroup − anon of the own step, "
            "clamped at 0"
        ),
    ),
    Column(
        name="job_mem_file_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="cgroup",
        meaning=(
            "memory.stat file of the job cgroup − file of the own step, "
            "clamped at 0"
        ),
    ),
    Column(
        name="job_mem_shmem_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="cgroup",
        meaning=(
            "memory.stat shmem of the job cgroup − shmem of the own step, "
            "clamped at 0"
        ),
    ),
    Column(
        name="job_mem_peak_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="cgroup",
        meaning=(
            "memory.peak of the job cgroup: the kernel high-water mark since "
            "the cgroup was created, page cache and the sampler included; "
            "never subtracted"
        ),
    ),
    Column(
        name="job_pids",
        dtype="INT32",
        nullable=True,
        unit="count",
        source="cgroup",
        meaning="pids.current of the job cgroup, the sampler step included",
    ),
    Column(
        name="job_oom_kills",
        dtype="INT32",
        nullable=True,
        unit="count",
        source="cgroup",
        meaning="Δoom_kill of the job cgroup's memory.events in this interval",
    ),
    Column(
        name="psi_cpu_some",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="pressure",
        meaning=(
            "Δ some total (µs) of /proc/pressure/cpu / (1e6 × dt_s), clamped "
            "to [0, 1]"
        ),
    ),
    Column(
        name="psi_mem_some",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="pressure",
        meaning=(
            "Δ some total of /proc/pressure/memory / (1e6 × dt_s), clamped to "
            "[0, 1]"
        ),
    ),
    Column(
        name="psi_mem_full",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="pressure",
        meaning=(
            "Δ full total of /proc/pressure/memory / (1e6 × dt_s), clamped to "
            "[0, 1]"
        ),
    ),
    Column(
        name="psi_io_some",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="pressure",
        meaning=(
            "Δ some total of /proc/pressure/io / (1e6 × dt_s), clamped to "
            "[0, 1]"
        ),
    ),
    Column(
        name="psi_io_full",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="pressure",
        meaning=(
            "Δ full total of /proc/pressure/io / (1e6 × dt_s), clamped to "
            "[0, 1]"
        ),
    ),
    Column(
        name="disk_read_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="diskstats",
        meaning=(
            "Σ Δsectors read × 512 / dt_s over whole local disks (a "
            "/sys/block/<dev>/device entry and hidden absent or 0) present in "
            "both reads"
        ),
    ),
    Column(
        name="disk_write_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="diskstats",
        meaning="Σ Δsectors written × 512 / dt_s over the same disks",
    ),
    Column(
        name="disk_busy_max",
        dtype="FLOAT32",
        nullable=True,
        unit="fraction",
        source="diskstats",
        meaning=(
            "maximum over the same disks of Δio_ticks (ms) / (1000 × dt_s), "
            "clamped to [0, 1]"
        ),
    ),
    Column(
        name="nfs_read_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="mountstats",
        meaning=(
            "Δ(normalreadbytes + directreadbytes) / dt_s of the workspace "
            "mount (the mountstats entry whose mountpoint is the longest "
            "prefix of the workspace path)"
        ),
    ),
    Column(
        name="nfs_write_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="mountstats",
        meaning=(
            "Δ(normalwritebytes + directwritebytes) / dt_s of the workspace "
            "mount"
        ),
    ),
    Column(
        name="nfs_server_read_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="mountstats",
        meaning="Δserverreadbytes / dt_s of the workspace mount",
    ),
    Column(
        name="nfs_server_write_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="mountstats",
        meaning="Δserverwritebytes / dt_s of the workspace mount",
    ),
    Column(
        name="nfs_read_ops_per_s",
        dtype="FLOAT32",
        nullable=True,
        unit="1/s",
        source="mountstats",
        meaning="ΔREAD ops / dt_s of the workspace mount",
    ),
    Column(
        name="nfs_write_ops_per_s",
        dtype="FLOAT32",
        nullable=True,
        unit="1/s",
        source="mountstats",
        meaning="ΔWRITE ops / dt_s of the workspace mount",
    ),
    Column(
        name="nfs_ops_per_s",
        dtype="FLOAT32",
        nullable=True,
        unit="1/s",
        source="mountstats",
        meaning=(
            "Δ(ops summed over every per-op line) / dt_s of the workspace "
            "mount; on node 0 it includes the walk's own operations"
        ),
    ),
    Column(
        name="nfs_read_rtt_ms",
        dtype="FLOAT32",
        nullable=True,
        unit="ms",
        source="mountstats",
        meaning="ΔREAD rtt (ms) / ΔREAD ops; null when ΔREAD ops is 0",
    ),
    Column(
        name="nfs_write_rtt_ms",
        dtype="FLOAT32",
        nullable=True,
        unit="ms",
        source="mountstats",
        meaning="ΔWRITE rtt (ms) / ΔWRITE ops; null when ΔWRITE ops is 0",
    ),
    Column(
        name="net_rx_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="netdev",
        meaning=(
            "Σ Δreceive bytes / dt_s over interfaces with a "
            "/sys/class/net/<if>/device entry, present in both reads"
        ),
    ),
    Column(
        name="net_tx_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="netdev",
        meaning="Σ Δtransmit bytes / dt_s over the same interfaces",
    ),
    Column(
        name="net_errors_per_s",
        dtype="FLOAT32",
        nullable=True,
        unit="1/s",
        source="netdev",
        meaning=(
            "Σ Δ(rx errs + rx drop + tx errs + tx drop) / dt_s over the same "
            "interfaces"
        ),
    ),
    Column(
        name="efa_rx_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="efa",
        meaning=(
            "Σ Δhw_counters/rx_bytes / dt_s over "
            "/sys/class/infiniband/*/ports/1; null without an EFA device; MPI "
            "traffic over EFA bypasses net_*"
        ),
    ),
    Column(
        name="efa_tx_bytes_per_s",
        dtype="FLOAT64",
        nullable=True,
        unit="bytes/s",
        source="efa",
        meaning=(
            "Σ Δhw_counters/tx_bytes / dt_s over the same ports; null without "
            "an EFA device"
        ),
    ),
)

WORKSPACE_COLUMNS: Final[tuple[Column, ...]] = (
    Column(
        name="ts",
        dtype="TIMESTAMP_MS_UTC",
        nullable=False,
        unit="-",
        source="identity",
        meaning="tick instant of the sample after which the walk ran",
    ),
    Column(
        name="phase",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="job phase, one of PHASES",
    ),
    Column(
        name="job_id",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="Slurm job id (SLURM_JOB_ID)",
    ),
    Column(
        name="restart",
        dtype="INT16",
        nullable=False,
        unit="-",
        source="identity",
        meaning="SLURM_RESTART_COUNT of the attempt, 0 when unset",
    ),
    Column(
        name="node",
        dtype="STRING",
        nullable=False,
        unit="-",
        source="identity",
        meaning="Slurm node name (SLURMD_NODENAME)",
    ),
    Column(
        name="apparent_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="walk",
        meaning=(
            "Σ lstat st_size over every entry below the workspace root, "
            ".hpcmu/telemetry excluded"
        ),
    ),
    Column(
        name="allocated_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="walk",
        meaning="Σ lstat st_blocks × 512 over the same entries",
    ),
    Column(
        name="files",
        dtype="INT64",
        nullable=True,
        unit="count",
        source="walk",
        meaning="non-directory entries",
    ),
    Column(
        name="dirs",
        dtype="INT64",
        nullable=True,
        unit="count",
        source="walk",
        meaning="directory entries below the root",
    ),
    Column(
        name="walk_s",
        dtype="FLOAT32",
        nullable=True,
        unit="s",
        source="walk",
        meaning="wall seconds the walk took",
    ),
    Column(
        name="fs_used_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="statvfs",
        meaning="(f_blocks − f_bfree) × f_frsize of the workspace filesystem",
    ),
    Column(
        name="fs_avail_bytes",
        dtype="INT64",
        nullable=True,
        unit="bytes",
        source="statvfs",
        meaning="f_bavail × f_frsize",
    ),
)

STAT_AGGREGATES: Final = ("min", "mean", "p95", "p99", "max", "total")


@dataclass(frozen=True, slots=True)
class StatSpec:
    """ADR-071/ADR-076: the summary.json statistics of one column.

    The published name of an aggregate a other than total is
    f"{column}_{a}". The published name of the total is total_name, stored
    literally, and None without a total. A statistic with no contributing
    value is absent, never 0 (ADR-076), so an all-null column has no
    statistic: its Parquet column keeps its declared type and is all null
    (ADR-070), and it emits no headline hook key (ADR-077).
    """

    column: str
    aggregates: tuple[str, ...]
    total_name: str | None


# One row per non-identity column except dt_s, which feeds the gap fields
# instead. Gauges and levels get min, mean, p95, max. Rates (bytes/s, 1/s)
# get mean, p95, max, total, where a total is sum(rate * dt_s). The per-op
# latencies get no mean: a dt-weighted mean of a per-interval average is
# not the mean latency of the operations. High-water marks and Δ counts get
# max only. p99 is only the R57 sampler cost gate.
SAMPLE_STATS: Final[tuple[StatSpec, ...]] = (
    StatSpec(
        column="sample_cost_ms",
        aggregates=("mean", "p95", "p99", "max"),
        total_name=None,
    ),
    StatSpec(
        column="sampler_maxrss_bytes",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="cpu_busy_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_util",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_user_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_system_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_iowait_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_steal_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_core_util_min",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_core_util_p50",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_core_util_max",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="cpu_active_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="load1",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="procs_running",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="procs_blocked",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_used_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_available_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_anon_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_cached_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_shmem_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_dirty_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_writeback_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="mem_slab_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="swap_used_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="major_faults_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="major_faults_total",
    ),
    StatSpec(
        column="oom_kills",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="job_cpu_cores",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_mem_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_mem_anon_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_mem_file_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_mem_shmem_bytes",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_mem_peak_bytes",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="job_pids",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="job_oom_kills",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="psi_cpu_some",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="psi_mem_some",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="psi_mem_full",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="psi_io_some",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="psi_io_full",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="disk_read_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="disk_read_bytes_total",
    ),
    StatSpec(
        column="disk_write_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="disk_write_bytes_total",
    ),
    StatSpec(
        column="disk_busy_max",
        aggregates=("min", "mean", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="nfs_read_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_read_bytes_total",
    ),
    StatSpec(
        column="nfs_write_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_write_bytes_total",
    ),
    StatSpec(
        column="nfs_server_read_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_server_read_bytes_total",
    ),
    StatSpec(
        column="nfs_server_write_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_server_write_bytes_total",
    ),
    StatSpec(
        column="nfs_read_ops_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_read_ops_total",
    ),
    StatSpec(
        column="nfs_write_ops_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_write_ops_total",
    ),
    StatSpec(
        column="nfs_ops_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="nfs_ops_total",
    ),
    StatSpec(
        column="nfs_read_rtt_ms",
        aggregates=("min", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="nfs_write_rtt_ms",
        aggregates=("min", "p95", "max"),
        total_name=None,
    ),
    StatSpec(
        column="net_rx_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="net_rx_bytes_total",
    ),
    StatSpec(
        column="net_tx_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="net_tx_bytes_total",
    ),
    StatSpec(
        column="net_errors_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="net_errors_total",
    ),
    StatSpec(
        column="efa_rx_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="efa_rx_bytes_total",
    ),
    StatSpec(
        column="efa_tx_bytes_per_s",
        aggregates=("mean", "p95", "max", "total"),
        total_name="efa_tx_bytes_total",
    ),
)

# R54: the workspace maxima and the minimum free bytes.
WORKSPACE_STATS: Final[tuple[StatSpec, ...]] = (
    StatSpec(
        column="apparent_bytes",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="allocated_bytes",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="files",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="dirs",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="walk_s",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="fs_used_bytes",
        aggregates=("max",),
        total_name=None,
    ),
    StatSpec(
        column="fs_avail_bytes",
        aggregates=("min",),
        total_name=None,
    ),
)

HEADER_FIELDS: Final[tuple[HeaderField, ...]] = (
    HeaderField(
        name="kind",
        json_type="string",
        nullable=False,
        meaning="line kind, hpcmu.telemetry.header",
    ),
    HeaderField(
        name="schema_version",
        json_type="integer",
        nullable=False,
        meaning="telemetry schema version of the writer (SCHEMA_VERSION)",
    ),
    HeaderField(
        name="tool_version",
        json_type="string",
        nullable=False,
        meaning=(
            "hpc-model-utils version, from the sampler's --tool-version "
            "argument"
        ),
    ),
    HeaderField(
        name="phase",
        json_type="string",
        nullable=False,
        meaning="job phase, one of PHASES",
    ),
    HeaderField(
        name="job_id",
        json_type="string",
        nullable=False,
        meaning="Slurm job id (SLURM_JOB_ID)",
    ),
    HeaderField(
        name="restart",
        json_type="integer",
        nullable=False,
        meaning="SLURM_RESTART_COUNT of the attempt, 0 when unset",
    ),
    HeaderField(
        name="node",
        json_type="string",
        nullable=False,
        meaning="Slurm node name (SLURMD_NODENAME)",
    ),
    HeaderField(
        name="node_index",
        json_type="integer",
        nullable=False,
        meaning="SLURM_NODEID",
    ),
    HeaderField(
        name="nodes",
        json_type="integer",
        nullable=True,
        meaning="SLURM_JOB_NUM_NODES",
    ),
    HeaderField(
        name="partition",
        json_type="string",
        nullable=True,
        meaning="SLURM_JOB_PARTITION",
    ),
    HeaderField(
        name="cpus_on_node",
        json_type="integer",
        nullable=True,
        meaning="SLURM_JOB_CPUS_PER_NODE indexed by SLURM_NODEID",
    ),
    HeaderField(
        name="instance_type",
        json_type="string",
        nullable=True,
        meaning="DMI product_name",
    ),
    HeaderField(
        name="cpu_model",
        json_type="string",
        nullable=True,
        meaning="value of the first model name line of /proc/cpuinfo",
    ),
    HeaderField(
        name="ncpus",
        json_type="integer",
        nullable=True,
        meaning="online CPUs",
    ),
    HeaderField(
        name="mem_total_bytes",
        json_type="integer",
        nullable=True,
        meaning="MemTotal × 1024",
    ),
    HeaderField(
        name="kernel",
        json_type="string",
        nullable=True,
        meaning="/proc/sys/kernel/osrelease",
    ),
    HeaderField(
        name="job_cgroup",
        json_type="string",
        nullable=True,
        meaning=(
            "/sys/fs/cgroup path of the job cgroup, the parent of the step "
            "component of /proc/self/cgroup; null without a step component"
        ),
    ),
    HeaderField(
        name="own_step_cgroup",
        json_type="string",
        nullable=True,
        meaning=(
            "/sys/fs/cgroup path of the sampler's own step, subtracted from "
            "the job values; null when own-step subtraction is off"
        ),
    ),
    HeaderField(
        name="workspace_mount",
        json_type="object",
        nullable=True,
        meaning=(
            "{mountpoint, fstype} of the mountstats entry whose mountpoint is "
            "the longest prefix of the workspace path"
        ),
    ),
    HeaderField(
        name="interval_s",
        json_type="integer",
        nullable=False,
        meaning="sample interval in seconds, the sampler's --interval argument",
    ),
    HeaderField(
        name="workspace_interval_s",
        json_type="integer",
        nullable=False,
        meaning=(
            "configured workspace walk interval in seconds, the sampler's "
            "--workspace-interval argument"
        ),
    ),
    HeaderField(
        name="started_at",
        json_type="string",
        nullable=False,
        meaning="sampler start time, ISO 8601 UTC with milliseconds and a Z",
    ),
)

ERROR_FIELDS: Final = ("kind", "ts", "source", "error", "message")
SAMPLE_SORT_KEYS: Final = ("phase", "job_id", "restart", "node_index", "ts")
WORKSPACE_SORT_KEYS: Final = ("phase", "job_id", "restart", "ts")

HEADLINE_KEY_PREFIX: Final = "telemetry."
WORKSPACE_SCOPE: Final = "workspace"
HEADLINE_FORMATS: Final = frozenset({"bytes", "cores", "fraction"})
# node_rows: the statistic over the pooled rows of every node and restart,
# so the value is a per-node figure. sum_nodes: the sum over the rows of
# every node and restart, a whole-job figure. walks: over the walks of the
# reported jobs.
HEADLINE_POOLINGS: Final = frozenset({"node_rows", "sum_nodes", "walks"})


@dataclass(frozen=True, slots=True)
class HeadlineSpec:
    """ADR-077: one summary.json headline value and its hook.

    scope is a PHASES value or WORKSPACE_SCOPE, and source is SAMPLES_FILE
    or WORKSPACE_FILE. key is the literal summary.json headline key
    <scope>.<stem>, and the hook key is HEADLINE_KEY_PREFIX + key.
    """

    scope: str
    source: str
    column: str
    aggregate: str
    key: str
    pooling: str
    fmt: str


HEADLINE: Final[tuple[HeadlineSpec, ...]] = (
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="job_mem_peak_bytes",
        aggregate="max",
        key="model.job_mem_peak_bytes_max",
        pooling="node_rows",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="mem_available_bytes",
        aggregate="min",
        key="model.mem_available_bytes_min",
        pooling="node_rows",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="cpu_busy_cores",
        aggregate="mean",
        key="model.cpu_busy_cores_mean",
        pooling="node_rows",
        fmt="cores",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="cpu_busy_cores",
        aggregate="p95",
        key="model.cpu_busy_cores_p95",
        pooling="node_rows",
        fmt="cores",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="cpu_util",
        aggregate="mean",
        key="model.cpu_util_mean",
        pooling="node_rows",
        fmt="fraction",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="nfs_read_bytes_per_s",
        aggregate="total",
        key="model.nfs_read_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="nfs_write_bytes_per_s",
        aggregate="total",
        key="model.nfs_write_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="disk_read_bytes_per_s",
        aggregate="total",
        key="model.disk_read_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="model",
        source=SAMPLES_FILE,
        column="disk_write_bytes_per_s",
        aggregate="total",
        key="model.disk_write_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="job_mem_peak_bytes",
        aggregate="max",
        key="finalize.job_mem_peak_bytes_max",
        pooling="node_rows",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="mem_available_bytes",
        aggregate="min",
        key="finalize.mem_available_bytes_min",
        pooling="node_rows",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="cpu_busy_cores",
        aggregate="mean",
        key="finalize.cpu_busy_cores_mean",
        pooling="node_rows",
        fmt="cores",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="cpu_busy_cores",
        aggregate="p95",
        key="finalize.cpu_busy_cores_p95",
        pooling="node_rows",
        fmt="cores",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="cpu_util",
        aggregate="mean",
        key="finalize.cpu_util_mean",
        pooling="node_rows",
        fmt="fraction",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="nfs_read_bytes_per_s",
        aggregate="total",
        key="finalize.nfs_read_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="nfs_write_bytes_per_s",
        aggregate="total",
        key="finalize.nfs_write_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="disk_read_bytes_per_s",
        aggregate="total",
        key="finalize.disk_read_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="finalize",
        source=SAMPLES_FILE,
        column="disk_write_bytes_per_s",
        aggregate="total",
        key="finalize.disk_write_bytes_total",
        pooling="sum_nodes",
        fmt="bytes",
    ),
    HeadlineSpec(
        scope="workspace",
        source=WORKSPACE_FILE,
        column="allocated_bytes",
        aggregate="max",
        key="workspace.allocated_bytes_max",
        pooling="walks",
        fmt="bytes",
    ),
)

# ADR-071: summary.json holds at most WARNINGS_CAP warnings
# {code, phase, job_id, restart, node_index, count, message}.
# STAGING_LINE_LIMIT is the bytes of one staging line, its \n included.
WARNINGS_CAP: Final = 500
STAGING_LINE_LIMIT: Final = 65536


@dataclass(frozen=True, slots=True)
class WarningCode:
    code: str
    meaning: str


WARNING_CODES: Final[tuple[WarningCode, ...]] = (
    WarningCode(
        code="bad_header",
        meaning="the file has no valid header line, and the file is skipped",
    ),
    WarningCode(
        code="truncated_tail",
        meaning="the last line has no newline and is dropped",
    ),
    WarningCode(
        code="bad_line",
        meaning=(
            "a line is not ASCII JSON within the line limit and is dropped"
        ),
    ),
    WarningCode(
        code="bad_record",
        meaning=(
            "a JSON line breaks the record rules of its kind and is dropped"
        ),
    ),
    WarningCode(
        code="identity_mismatch",
        meaning=(
            "a sample or walk names another attempt than its header and is "
            "dropped"
        ),
    ),
    WarningCode(
        code="non_monotonic_ts",
        meaning=(
            "a sample or walk ts is not after the previous kept one and is "
            "dropped"
        ),
    ),
    WarningCode(
        code="sampler_error",
        meaning="an error line written by the sampler",
    ),
    WarningCode(
        code="missing_node",
        meaning="a node of an attempt has no valid staging file",
    ),
    WarningCode(
        code="warnings_capped",
        meaning=(
            "the warning list reached WARNINGS_CAP, and the count holds the "
            "warning occurrences left out of the list"
        ),
    ),
)


def staging_file_name(
    phase: str, job_id: str, restart: int, node_index: int
) -> str:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}: {phase!r}")
    if re.fullmatch("[0-9]+", job_id) is None:
        raise ValueError(f"job_id must be ASCII digits: {job_id!r}")
    if restart < 0:
        raise ValueError(f"restart must be >= 0: {restart}")
    if node_index < 0:
        raise ValueError(f"node_index must be >= 0: {node_index}")
    return f"{phase}-{job_id}-r{restart}-{node_index}.ndjson"


def format_headline_value(fmt: str, value: int | float) -> str:
    """ADR-077: the single producer of headline hook value strings.

    The domain is a non-negative int or float, not a bool, whose float() is
    finite; any other value raises ValueError. A result is plain decimal
    digits, fixed-point with 2 decimals for cores and 4 for fraction: no
    sign, no exponent, no space.
    """
    if isinstance(value, bool):
        raise ValueError(f"value must not be a bool: {value!r}")
    if fmt not in HEADLINE_FORMATS:
        raise ValueError(
            f"fmt must be one of {sorted(HEADLINE_FORMATS)}: {fmt!r}"
        )
    try:
        as_float = float(value)
    except OverflowError as err:
        # Only an int reaches here. Its bit length names it without a
        # decimal conversion, which is capped at 4300 digits.
        raise ValueError(
            "value must be within the float range: an int of "
            f"{int(value).bit_length()} bits"
        ) from err
    if not math.isfinite(as_float):
        raise ValueError(f"value must be finite: {value!r}")
    if value < 0:
        raise ValueError(f"value must be >= 0: {value!r}")
    # -0.0 passes the check above; abs drops its sign bit.
    magnitude = abs(value)
    if fmt == "bytes":
        return str(round(magnitude))
    if fmt == "cores":
        return f"{magnitude:.2f}"
    return f"{magnitude:.4f}"
