"""ADR-068/ADR-070/ADR-093/ADR-075: the telemetry column contract.

It also holds the ADR-071/ADR-076/ADR-077/ADR-085 roll-up specs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
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


MEANING_CONVENTIONS: Final[tuple[str, ...]] = (
    "Δ is the difference between this sample's read and the previous one.",
    "online CPUs is the number of cpuN lines in /proc/stat.",
    "Every rate is null on a counter decrease, on a missing baseline, or "
    "after a failed read of its source.",
    "The /proc/stat CPU columns follow the per-line ADR-093 rule written "
    "into their meanings.",
    "A source whose read fails leaves every column of that source null in "
    "that sample; its rates are also null in the next sample, which has no "
    "baseline.",
    "A source with nothing to read (no step component in the sampler's "
    "cgroup path, a workspace not on an NFS mount, no EFA device) leaves its "
    "columns null and writes no error line.",
    "A float that is not finite is written as null.",
    "Every cgroup column is null when the header's job_cgroup is null.",
    "A header field whose source cannot be read or parsed, a malformed Slurm "
    "variable included, is null.",
)

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
        meaning=(
            "Slurm node name (SLURMD_NODENAME), or the host name (uname "
            "nodename) when SLURMD_NODENAME is unset"
        ),
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
        meaning="cpu_busy_cores / online CPUs; null as cpu_busy_cores",
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
            "step, clamped at 0; no subtraction when own_step_cgroup is null"
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
            "clamped at 0; no subtraction when own_step_cgroup is null"
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
            "clamped at 0; no subtraction when own_step_cgroup is null"
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
            "clamped at 0; no subtraction when own_step_cgroup is null"
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
        meaning=(
            "pids.current of the job cgroup, the sampler step included; null "
            "when the job cgroup has no pids controller (pids.current absent)"
        ),
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
            "/sys/block/<dev>/device entry, and a /sys/block/<dev>/hidden file "
            "that is absent or holds 0) present in both reads"
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
            "prefix of the workspace path, its symlinks resolved)"
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
        meaning=(
            "Slurm node name (SLURMD_NODENAME), or the host name (uname "
            "nodename) when SLURMD_NODENAME is unset"
        ),
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
        meaning=(
            "Slurm node name (SLURMD_NODENAME), or the host name (uname "
            "nodename) when SLURMD_NODENAME is unset"
        ),
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
        meaning="SLURM_JOB_NUM_NODES; null when it is unset or empty",
    ),
    HeaderField(
        name="partition",
        json_type="string",
        nullable=True,
        meaning="SLURM_JOB_PARTITION, stripped; null when it is unset or blank",
    ),
    HeaderField(
        name="cpus_on_node",
        json_type="integer",
        nullable=True,
        meaning=(
            "SLURM_JOB_CPUS_PER_NODE indexed by SLURM_NODEID, after expanding "
            "Slurm's compressed list (192(x2) is 192, 192); null when "
            "SLURM_JOB_CPUS_PER_NODE is unset or empty, or when SLURM_NODEID "
            "is outside the expanded list"
        ),
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
            "the job values; null when the sampler's cgroup has no numbered "
            "step_N component (a step_batch, step_extern or step_interactive "
            "step, or no step), and then nothing is subtracted"
        ),
    ),
    HeaderField(
        name="workspace_mount",
        json_type="object",
        nullable=True,
        meaning=(
            "{mountpoint, fstype} of the mountstats entry whose mountpoint is "
            "the longest prefix of the workspace path, its symlinks resolved; "
            "null when no entry matches"
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


@dataclass(frozen=True, slots=True)
class Term:
    name: str
    meaning: str


HEADLINE_FORMAT_TERMS: Final[tuple[Term, ...]] = (
    Term(
        name="bytes",
        meaning=(
            "an integer number of bytes, rounded half to even, written as "
            "plain digits"
        ),
    ),
    Term(
        name="cores",
        meaning="a fixed-point number with 2 decimals, no sign and no exponent",
    ),
    Term(
        name="fraction",
        meaning="a fixed-point number with 4 decimals, no sign and no exponent",
    ),
)
HEADLINE_FORMATS: Final = frozenset(t.name for t in HEADLINE_FORMAT_TERMS)

HEADLINE_POOLING_TERMS: Final[tuple[Term, ...]] = (
    Term(
        name="node_rows",
        meaning=(
            "the statistic over the rows of every node and restart pooled "
            "together, so the value is a per-node figure: a mean is the mean "
            "of one node and a max is the worst node"
        ),
    ),
    Term(
        name="sum_nodes",
        meaning=(
            "the sum over the rows of every node and restart, a whole-job "
            "figure"
        ),
    ),
    Term(
        name="walks",
        meaning="the statistic over the walks of the reported jobs",
    ),
)
HEADLINE_POOLINGS: Final = frozenset(t.name for t in HEADLINE_POOLING_TERMS)


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

# ADR-071: summary.json holds at most WARNINGS_CAP warnings, whose fields
# are the warnings[] rows of SUMMARY_FIELDS.
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


@dataclass(frozen=True, slots=True)
class SummaryField:
    """ADR-071: one key path of summary.json.

    path joins the keys with ".", and "[]" stands for the elements of an
    array. json_type is object, array, string, integer, number or map, where
    a map is an object whose keys are listed elsewhere in the contract. A
    required path is present in every object of its parent.
    """

    path: str
    json_type: str
    required: bool
    nullable: bool
    meaning: str


SUMMARY_FIELDS: Final[tuple[SummaryField, ...]] = (
    SummaryField(
        path="kind",
        json_type="string",
        required=True,
        nullable=False,
        meaning="the summary kind, hpcmu.telemetry.summary",
    ),
    SummaryField(
        path="schema_version",
        json_type="integer",
        required=True,
        nullable=False,
        meaning="the schema version of the contract",
    ),
    SummaryField(
        path="run",
        json_type="object",
        required=True,
        nullable=False,
        meaning="the run the export covers",
    ),
    SummaryField(
        path="run.run_id",
        json_type="string",
        required=True,
        nullable=False,
        meaning="the run_id of state.json",
    ),
    SummaryField(
        path="run.model",
        json_type="string",
        required=True,
        nullable=False,
        meaning="the plugin name of the run",
    ),
    SummaryField(
        path="run.tool_version",
        json_type="string",
        required=True,
        nullable=False,
        meaning="the hpc-model-utils version that ran the export",
    ),
    SummaryField(
        path="run.interval_s",
        json_type="integer",
        required=True,
        nullable=True,
        meaning=(
            "interval_s of the first valid header; null when no header is valid"
        ),
    ),
    SummaryField(
        path="run.workspace_interval_s",
        json_type="integer",
        required=True,
        nullable=True,
        meaning=(
            "workspace_interval_s of the first valid header; null when no "
            "header is valid"
        ),
    ),
    SummaryField(
        path="jobs",
        json_type="array",
        required=True,
        nullable=False,
        meaning=(
            "one entry per exported job, in the order of the recorded jobs"
        ),
    ),
    SummaryField(
        path="jobs[].phase",
        json_type="string",
        required=True,
        nullable=False,
        meaning="a PHASES value",
    ),
    SummaryField(
        path="jobs[].job_id",
        json_type="string",
        required=True,
        nullable=False,
        meaning="the Slurm job id",
    ),
    SummaryField(
        path="jobs[].nodes",
        json_type="array",
        required=True,
        nullable=False,
        meaning=(
            "one entry per valid staging file of the job, in sample sort order"
        ),
    ),
    SummaryField(
        path="jobs[].nodes[].header",
        json_type="map",
        required=True,
        nullable=False,
        meaning=(
            "the staging header, keys as in the staging header fields table"
        ),
    ),
    SummaryField(
        path="jobs[].nodes[].first_ts",
        json_type="integer",
        required=False,
        nullable=False,
        meaning=(
            "ts of the first kept sample, in ms since the epoch; absent "
            "without a sample"
        ),
    ),
    SummaryField(
        path="jobs[].nodes[].last_ts",
        json_type="integer",
        required=False,
        nullable=False,
        meaning="ts of the last kept sample; absent without a sample",
    ),
    SummaryField(
        path="jobs[].nodes[].samples",
        json_type="integer",
        required=True,
        nullable=False,
        meaning="the number of kept samples",
    ),
    SummaryField(
        path="jobs[].nodes[].gaps",
        json_type="integer",
        required=True,
        nullable=False,
        meaning="the number of samples whose dt_s exceeds 1.5 × interval_s",
    ),
    SummaryField(
        path="jobs[].nodes[].max_gap_s",
        json_type="number",
        required=False,
        nullable=False,
        meaning="the largest non-null dt_s; absent when every dt_s is null",
    ),
    SummaryField(
        path="jobs[].nodes[].stats",
        json_type="map",
        required=True,
        nullable=False,
        meaning=(
            "the samples.parquet statistics of this file, names as in the "
            "statistics table"
        ),
    ),
    SummaryField(
        path="workspace",
        json_type="object",
        required=True,
        nullable=False,
        meaning="the workspace walks of every exported job",
    ),
    SummaryField(
        path="workspace.walks",
        json_type="integer",
        required=True,
        nullable=False,
        meaning="the number of kept walks",
    ),
    SummaryField(
        path="workspace.stats",
        json_type="map",
        required=True,
        nullable=False,
        meaning="the workspace.parquet statistics over every kept walk",
    ),
    SummaryField(
        path="headline",
        json_type="object",
        required=True,
        nullable=False,
        meaning="the headline values and the jobs they come from",
    ),
    SummaryField(
        path="headline.job_ids",
        json_type="map",
        required=True,
        nullable=False,
        meaning=(
            "for each phase, the job id of its last recorded job, the one "
            "the headline values report"
        ),
    ),
    SummaryField(
        path="headline.values",
        json_type="map",
        required=True,
        nullable=False,
        meaning="the headline values, keys as in the headline table",
    ),
    SummaryField(
        path="warnings",
        json_type="array",
        required=True,
        nullable=False,
        meaning=(
            "at most WARNINGS_CAP warnings, in file order then the missing "
            "nodes"
        ),
    ),
    SummaryField(
        path="warnings[].code",
        json_type="string",
        required=True,
        nullable=False,
        meaning="a warning code",
    ),
    SummaryField(
        path="warnings[].phase",
        json_type="string",
        required=True,
        nullable=True,
        meaning="the attempt's phase; null for warnings_capped",
    ),
    SummaryField(
        path="warnings[].job_id",
        json_type="string",
        required=True,
        nullable=True,
        meaning="the attempt's job id; null for warnings_capped",
    ),
    SummaryField(
        path="warnings[].restart",
        json_type="integer",
        required=True,
        nullable=True,
        meaning="the attempt's restart; null for warnings_capped",
    ),
    SummaryField(
        path="warnings[].node_index",
        json_type="integer",
        required=True,
        nullable=True,
        meaning="the attempt's node index; null for warnings_capped",
    ),
    SummaryField(
        path="warnings[].count",
        json_type="integer",
        required=True,
        nullable=False,
        meaning="the occurrences this warning stands for",
    ),
    SummaryField(
        path="warnings[].message",
        json_type="string",
        required=True,
        nullable=False,
        meaning="a human-readable detail",
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


def _code(text: str) -> str:
    return f"`{text}`"


def _escape(text: str) -> str:
    """text with the Markdown specials \\ * < & backslash-escaped.

    The meanings hold no newline, | or backtick (test_schema.py), so only
    these four can change how GitHub renders a cell: <dev> would read as an
    HTML tag and infiniband/*/ports as an emphasis marker.
    """
    for special in ("\\", "*", "<", "&"):
        text = text.replace(special, "\\" + special)
    return text


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _table(
    header: tuple[str, ...], rows: Iterable[tuple[str, ...]]
) -> list[str]:
    lines = [f"| {' | '.join(header)} |", f"|{' --- |' * len(header)}"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


# file -> (kind, the name of its kind constant, sort keys) for EXPORT_FILES:
# the file rows of render_markdown and of contract_snapshot.
_FILE_FACTS: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    SAMPLES_FILE: (SAMPLES_KIND, "SAMPLES_KIND", SAMPLE_SORT_KEYS),
    WORKSPACE_FILE: (WORKSPACE_KIND, "WORKSPACE_KIND", WORKSPACE_SORT_KEYS),
    SUMMARY_FILE: (SUMMARY_KIND, "SUMMARY_KIND", ()),
}
_HEADLINE_SCOPES: Final = (*PHASES, WORKSPACE_SCOPE)


def _file_row(name: str) -> tuple[str, str, str]:
    kind, constant, sort_keys = _FILE_FACTS[name]
    return (
        _code(name),
        f"{_code(kind)} ({_code(constant)})",
        ", ".join(_code(key) for key in sort_keys) or "-",
    )


def _published_names(spec: StatSpec) -> list[str]:
    """The published names of spec in aggregate order.

    total is last in STAT_AGGREGATES, so its literal name goes last.
    """
    names = [f"{spec.column}_{a}" for a in spec.aggregates if a != "total"]
    if spec.total_name is not None:
        names.append(spec.total_name)
    return names


def _stat_names(spec: StatSpec) -> str:
    """The published names of spec in aggregate order, as code spans."""
    return ", ".join(_code(name) for name in _published_names(spec))


def _column_table(columns: tuple[Column, ...]) -> list[str]:
    return _table(
        ("column", "type", "nullable", "unit", "source", "meaning"),
        (
            (
                _code(c.name),
                _code(c.dtype),
                _yes_no(c.nullable),
                _code(c.unit),
                _code(c.source),
                _escape(c.meaning),
            )
            for c in columns
        ),
    )


def _term_table(title: str, terms: tuple[Term, ...]) -> list[str]:
    return _table(
        (title, "meaning"),
        ((_code(t.name), _escape(t.meaning)) for t in terms),
    )


def render_markdown() -> str:
    """ADR-072/R56: the generated reference block of docs/telemetry.md.

    A pure function of this module's records: the text starts with the
    first ### heading and ends with one newline. Names, keys, paths, types,
    units and sources are code spans; meanings, conventions and terms go
    through _escape.
    """
    scopes = ", ".join(_code(scope) for scope in _HEADLINE_SCOPES)
    stat_maps = (
        (SAMPLE_STATS, SAMPLES_FILE, "jobs[].nodes[].stats"),
        (WORKSPACE_STATS, WORKSPACE_FILE, "workspace.stats"),
    )
    sections: list[list[str]] = [
        [
            "### Version and files",
            "",
            f"Schema version: {_code(str(SCHEMA_VERSION))} "
            f"({_code('SCHEMA_VERSION')}).",
            "",
            f"Phases: {', '.join(_code(phase) for phase in PHASES)} "
            f"({_code('PHASES')}).",
            "",
            *_table(
                ("file", "kind", "sort order"),
                (_file_row(name) for name in EXPORT_FILES),
            ),
        ],
        [
            "### Parquet key-value metadata",
            "",
            "Both Parquet files carry these keys in their key-value metadata.",
            "",
            *_table(
                ("key", "meaning"),
                (
                    (_code(k.name), _escape(k.meaning))
                    for k in PARQUET_METADATA_KEYS
                ),
            ),
        ],
        [
            "### Conventions",
            "",
            *(f"- {_escape(text)}" for text in MEANING_CONVENTIONS),
        ],
        [
            f"### {SAMPLES_FILE} columns",
            "",
            *_column_table(SAMPLE_COLUMNS),
        ],
        [
            f"### {WORKSPACE_FILE} columns",
            "",
            *_column_table(WORKSPACE_COLUMNS),
        ],
        [
            "### Staging header fields",
            "",
            "Each field is published verbatim in the "
            f"{_code('hpcmu.headers')} metadata key and in "
            f"{_code(SUMMARY_FILE)} {_code('jobs[].nodes[].header')}.",
            "",
            *_table(
                ("field", "JSON type", "nullable", "meaning"),
                (
                    (
                        _code(f.name),
                        _code(f.json_type),
                        _yes_no(f.nullable),
                        _escape(f.meaning),
                    )
                    for f in HEADER_FIELDS
                ),
            ),
        ],
        [
            f"### {SUMMARY_FILE} layout",
            "",
            "A path joins the keys with `.`, and `[]` stands for the "
            "elements of an array. A `map` is an object whose keys are "
            "listed in another table. A required path is present in every "
            "object of its parent.",
            "",
            *_table(
                ("path", "JSON type", "required", "nullable", "meaning"),
                (
                    (
                        _code(f.path),
                        _code(f.json_type),
                        _yes_no(f.required),
                        _yes_no(f.nullable),
                        _escape(f.meaning),
                    )
                    for f in SUMMARY_FIELDS
                ),
            ),
        ],
        [
            f"### {SUMMARY_FILE} statistics",
            "",
            "Every aggregate except `total` is published as "
            "`<column>_<aggregate>`, and the total under its listed name. "
            "The aggregates are "
            f"{', '.join(_code(a) for a in STAT_AGGREGATES)}, in this order. "
            "A statistic with no contributing value is absent, never 0.",
            "",
            *_table(
                ("column", "file", f"{SUMMARY_FILE} map", "names"),
                (
                    (
                        _code(spec.column),
                        _code(file),
                        _code(stat_map),
                        _stat_names(spec),
                    )
                    for specs, file, stat_map in stat_maps
                    for spec in specs
                ),
            ),
        ],
        [
            "### Headline values and hooks",
            "",
            "`headline.values` holds `<scope>.<statistic>`, and the ModelOps "
            "metadata hook of a value is "
            f"{_code(HEADLINE_KEY_PREFIX + '<scope>.<statistic>')} "
            f"({_code('HEADLINE_KEY_PREFIX')}). `<statistic>` is a published "
            f"statistic name, and the scopes are {scopes}.",
            "",
            *_term_table("pooling", HEADLINE_POOLING_TERMS),
            "",
            *_term_table("format", HEADLINE_FORMAT_TERMS),
            "",
            *_table(
                (
                    "hook key",
                    "scope",
                    "file",
                    "column",
                    "aggregate",
                    "pooling",
                    "format",
                ),
                (
                    (
                        _code(HEADLINE_KEY_PREFIX + h.key),
                        _code(h.scope),
                        _code(h.source),
                        _code(h.column),
                        _code(h.aggregate),
                        _code(h.pooling),
                        _code(h.fmt),
                    )
                    for h in HEADLINE
                ),
            ),
        ],
        [
            "### Warning codes",
            "",
            *_table(
                ("code", "meaning"),
                ((_code(w.code), _escape(w.meaning)) for w in WARNING_CODES),
            ),
            "",
            f"The list holds at most {_code(str(WARNINGS_CAP))} warnings "
            f"({_code('WARNINGS_CAP')}).",
        ],
    ]
    lines: list[str] = []
    for section in sections:
        if lines:
            lines.append("")
        lines.extend(section)
    return "\n".join(lines) + "\n"


type SnapshotValue = (
    str | int | bool | None | list[SnapshotValue] | dict[str, SnapshotValue]
)


def _texts(items: Iterable[str]) -> list[SnapshotValue]:
    return list(items)


def _file_entry(name: str) -> dict[str, SnapshotValue]:
    kind, _, sort_keys = _FILE_FACTS[name]
    return {"name": name, "kind": kind, "sort_keys": _texts(sort_keys)}


def _metadata_entry(key: MetadataKey) -> dict[str, SnapshotValue]:
    return {"name": key.name, "meaning": key.meaning}


def _column_entry(column: Column) -> dict[str, SnapshotValue]:
    return {
        "name": column.name,
        "dtype": column.dtype,
        "nullable": column.nullable,
        "unit": column.unit,
        "source": column.source,
        "meaning": column.meaning,
    }


def _header_entry(field: HeaderField) -> dict[str, SnapshotValue]:
    return {
        "name": field.name,
        "json_type": field.json_type,
        "nullable": field.nullable,
        "meaning": field.meaning,
    }


def _summary_entry(field: SummaryField) -> dict[str, SnapshotValue]:
    return {
        "path": field.path,
        "json_type": field.json_type,
        "required": field.required,
        "nullable": field.nullable,
        "meaning": field.meaning,
    }


def _stat_entry(spec: StatSpec) -> dict[str, SnapshotValue]:
    return {
        "column": spec.column,
        "aggregates": _texts(spec.aggregates),
        "total_name": spec.total_name,
        "names": _texts(_published_names(spec)),
    }


def _term_entry(term: Term) -> dict[str, SnapshotValue]:
    return {"name": term.name, "meaning": term.meaning}


def _headline_entry(spec: HeadlineSpec) -> dict[str, SnapshotValue]:
    return {
        "scope": spec.scope,
        "source": spec.source,
        "column": spec.column,
        "aggregate": spec.aggregate,
        "key": spec.key,
        "pooling": spec.pooling,
        "fmt": spec.fmt,
    }


def _warning_entry(warning: WarningCode) -> dict[str, SnapshotValue]:
    return {"code": warning.code, "meaning": warning.meaning}


def contract_snapshot() -> dict[str, SnapshotValue]:
    """ADR-072/ADR-093: the published contract as plain JSON data.

    tests/goldens/telemetry/schema-v<SCHEMA_VERSION>.json freezes it, and
    a contract test holds every later version to its survival rules. Tuples
    become lists in tuple order, and each call builds a fresh object. The
    staging-only items stay out (ADR-069): the walk and error kinds, the
    error fields, the line limit and the staging file name, and the token
    sets, whose used members the columns carry.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kinds": {
            "header": HEADER_KIND,
            "samples": SAMPLES_KIND,
            "workspace": WORKSPACE_KIND,
            "summary": SUMMARY_KIND,
        },
        "phases": _texts(PHASES),
        "files": [_file_entry(name) for name in EXPORT_FILES],
        "parquet_metadata": [_metadata_entry(k) for k in PARQUET_METADATA_KEYS],
        "meaning_conventions": _texts(MEANING_CONVENTIONS),
        "sample_columns": [_column_entry(c) for c in SAMPLE_COLUMNS],
        "workspace_columns": [_column_entry(c) for c in WORKSPACE_COLUMNS],
        "header_fields": [_header_entry(f) for f in HEADER_FIELDS],
        "summary_fields": [_summary_entry(f) for f in SUMMARY_FIELDS],
        "stat_aggregates": _texts(STAT_AGGREGATES),
        "sample_stats": [_stat_entry(s) for s in SAMPLE_STATS],
        "workspace_stats": [_stat_entry(s) for s in WORKSPACE_STATS],
        "headline_key_prefix": HEADLINE_KEY_PREFIX,
        "headline_scopes": _texts(_HEADLINE_SCOPES),
        "headline_poolings": [_term_entry(t) for t in HEADLINE_POOLING_TERMS],
        "headline_formats": [_term_entry(t) for t in HEADLINE_FORMAT_TERMS],
        "headline": [_headline_entry(h) for h in HEADLINE],
        "warning_codes": [_warning_entry(w) for w in WARNING_CODES],
        "warnings_cap": WARNINGS_CAP,
    }
