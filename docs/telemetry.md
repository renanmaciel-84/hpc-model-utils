# Job telemetry contract

A sampler runs on every node of a run's model and finalize jobs, and the
publish step exports what it recorded into three files below
`saidas/telemetry/` of the run's artifacts:

| File | Holds |
| --- | --- |
| `saidas/telemetry/samples.parquet` | one row per sample of every node, restart and exported job |
| `saidas/telemetry/workspace.parquet` | one row per workspace walk, taken on node 0 |
| `saidas/telemetry/summary.json` | the statistics of each node, the headline values and the warnings |

This document is the written contract of those three files (ADR-072). The
[reference](#reference) at its end is generated from
`src/hpc_model_utils/infra/telemetry/schema.py`, the single source of the
schema, and a contract test fails when the two differ. The sections before it
explain how to read the files.

Telemetry never changes a run's status, its exit code, the relayed job log or
the diagnosis. The one exception is the upload: a failed upload of a telemetry
file is a storage error, exit 4, as for any other artifact.

## The schema_version policy

`schema_version` is one integer. It is written in `summary.json`
`schema_version` and in the `hpcmu.telemetry.schema_version` key of the
key-value metadata of each Parquet file. The
[reference](#version-and-files) gives the current value.

- **Additions keep the version.** A new nullable column, header field,
  statistic name, `summary.json` field or warning code keeps it, so a reader
  should ignore the names it does not know.
- **Every other change bumps it.** Removing or renaming anything, or changing
  a type, a unit, a meaning, a sort order or a nullability, bumps the
  version.
- **A wording fix keeps it.** A meaning edit that does not change what is
  measured keeps the version. It goes through a reviewed entry that the
  contract tests check against the frozen snapshot
  `tests/goldens/telemetry/schema-v1.json`.

## Reading the files

The examples read `samples.parquet` in place with duckdb SQL, polars and
pandas. Replace `<bucket>` and `<artifacts-prefix>` with the run's artifacts
location. An `s3://` path needs S3 credentials, and duckdb needs its `httpfs`
extension for it. A downloaded copy reads the same way from a local path.

### How a statistic is computed

`summary.json` computes its statistics from the values as the Parquet files
hold them, so a reader can reproduce every one from the files (ADR-076). In
the rules below, `x` is the column and `dt_s` is the measured seconds of the
row.

- `min` and `max` are over the non-null values of `x`.
- `p95` and `p99` are the Hyndman-Fan type 7 quantile of the non-null values:
  the linear interpolation between the two values around position
  `(n - 1) * q`.
- `mean` is `sum(x * dt_s) / sum(dt_s)` and `total` is `sum(x * dt_s)`, both
  over the rows where `x` and `dt_s` are non-null. A mean is weighted by the
  measured time, and the total of a rate is its count over the measured time.

Compute in float64. Cast every `FLOAT32` column, `dt_s` included, with
`::DOUBLE` or its equivalent before any arithmetic: uncast float32
arithmetic drifts from `summary.json` by up to 7e-8 relative, while the cast
values agree within 1e-9. Restrict the denominator of a mean to the rows where
`x` is non-null, as `FILTER (WHERE x IS NOT NULL)` does below. A plain
`SUM(dt_s)` would also count the time of the null rows.

For the type 7 quantile, use duckdb `quantile_cont`, numpy
`numpy.quantile(x, q, method='linear')`, the pandas default, or polars with
`interpolation='linear'`. polars defaults to `'nearest'`, which gives another
value.

### Per-node statistics

A node entry of `summary.json`, `jobs[].nodes[]`, holds the statistics of one
staging file: one node of one attempt (`restart`) of one job. This query
reproduces three of them for every node:

```sql
SELECT
    phase, job_id, restart, node_index,
    SUM(cpu_busy_cores::DOUBLE * dt_s::DOUBLE)
        / SUM(dt_s::DOUBLE) FILTER (WHERE cpu_busy_cores IS NOT NULL)
        AS cpu_busy_cores_mean,
    quantile_cont(cpu_busy_cores::DOUBLE, 0.95) AS cpu_busy_cores_p95,
    SUM(nfs_read_bytes_per_s::DOUBLE * dt_s::DOUBLE) AS nfs_read_bytes_total
FROM 's3://<bucket>/<artifacts-prefix>/saidas/telemetry/samples.parquet'
GROUP BY phase, job_id, restart, node_index
ORDER BY phase, job_id, restart, node_index;
```

The two statistics of `cpu_busy_cores` in polars:

```python
import polars as pl

path = 's3://<bucket>/<artifacts-prefix>/saidas/telemetry/samples.parquet'
x = pl.col('cpu_busy_cores').cast(pl.Float64)
dt = pl.col('dt_s').cast(pl.Float64)
per_node = (
    pl.read_parquet(path)
    .group_by('phase', 'job_id', 'restart', 'node_index')
    .agg(
        cpu_busy_cores_mean=(x * dt).sum() / dt.filter(x.is_not_null()).sum(),
        cpu_busy_cores_p95=x.quantile(0.95, interpolation='linear'),
    )
    .sort('phase', 'job_id', 'restart', 'node_index')
)
```

And in pandas:

```python
import pandas as pd

path = 's3://<bucket>/<artifacts-prefix>/saidas/telemetry/samples.parquet'
samples = pd.read_parquet(path)
x = samples['cpu_busy_cores'].astype('float64')
dt = samples['dt_s'].astype('float64')
grouped = samples.assign(x=x, x_dt=x * dt, dt_x=dt.where(x.notna())).groupby(
    ['phase', 'job_id', 'restart', 'node_index']
)
per_node = pd.DataFrame({
    'cpu_busy_cores_mean': grouped['x_dt'].sum(min_count=1)
    / grouped['dt_x'].sum(min_count=1),
    'cpu_busy_cores_p95': grouped['x'].quantile(0.95),
}).reset_index()
```

Where `summary.json` leaves a statistic out because no value contributed,
these recipes give a null or a NaN instead.

### Headline values

A headline value pools the rows of every node and restart of one job, the
job that `headline.job_ids` names for its phase. Grouping by job instead of
by node gives the `node_rows` and `sum_nodes` values of every job, and the
rows of the reported jobs are the `headline.values`:

```sql
SELECT
    phase, job_id,
    MAX(job_mem_peak_bytes) AS job_mem_peak_bytes_max,
    SUM(cpu_busy_cores::DOUBLE * dt_s::DOUBLE)
        / SUM(dt_s::DOUBLE) FILTER (WHERE cpu_busy_cores IS NOT NULL)
        AS cpu_busy_cores_mean,
    quantile_cont(cpu_busy_cores::DOUBLE, 0.95) AS cpu_busy_cores_p95,
    SUM(nfs_read_bytes_per_s::DOUBLE * dt_s::DOUBLE) AS nfs_read_bytes_total
FROM 's3://<bucket>/<artifacts-prefix>/saidas/telemetry/samples.parquet'
GROUP BY phase, job_id
ORDER BY phase, job_id;
```

### NFS latency per operation

`nfs_read_rtt_ms` is the mean round-trip time of the READ operations of one
interval, and `nfs_read_ops_per_s * dt_s` is their count. The mean latency of
the operations weights each interval by that count:

```sql
SELECT
    phase, job_id, restart, node_index,
    SUM(nfs_read_rtt_ms::DOUBLE * nfs_read_ops_per_s::DOUBLE * dt_s::DOUBLE)
        / SUM(nfs_read_ops_per_s::DOUBLE * dt_s::DOUBLE)
        AS nfs_read_rtt_ms_per_op
FROM 's3://<bucket>/<artifacts-prefix>/saidas/telemetry/samples.parquet'
WHERE nfs_read_rtt_ms IS NOT NULL
GROUP BY phase, job_id, restart, node_index
ORDER BY phase, job_id, restart, node_index;
```

Writes work the same way with the `nfs_write_*` columns. This is why
`summary.json` publishes no `nfs_read_rtt_ms_mean` and no
`nfs_write_rtt_ms_mean`: a dt-weighted mean of per-interval average latencies
is not the mean latency of the operations. It would weigh a quiet interval
with one slow READ as much as a busy interval with thousands of fast ones.
The published `min`, `p95` and `max` of the latency columns describe the
intervals, and they answer whether a slow period happened.

## Reading guide

- **Poolings.** The `pooling` column of the headline table says how a value
  pools the rows of the reported job. A `node_rows` value is a per-node
  figure: on a 4-node job, `model.cpu_busy_cores_mean` is the mean busy cores
  of one node, not of the job, and `model.job_mem_peak_bytes_max` is the peak
  of the worst node. A `sum_nodes` value, such as
  `model.nfs_read_bytes_total`, is a whole-job sum across nodes and restarts.
  `workspace.allocated_bytes_max`, a `walks` value, is the walk maximum over
  the reported jobs.
- **The reported job.** The headline values of a phase come from one job, the
  last recorded job of that phase, which `headline.job_ids` names. An earlier
  job of the same phase is in `jobs` and in the Parquet files, but not in the
  headline.
- **Totals are lower bounds.** A total counts only the measured time. An
  interval whose rate or `dt_s` is null, a dropped line (a gap in `seq`) and
  a node without a valid staging file (a `missing_node` warning) all lose
  their share of the count.
- **OOM kills.** `oom_kills` and `job_oom_kills` are per-interval Δ counts
  with only a `max` statistic. `oom_kills_max > 0` means that at least one
  OOM kill happened. It is not the number of kills.
- **NFS bytes.** `nfs_read_bytes_*` and `nfs_write_bytes_*` are the bytes the
  application read and wrote on the workspace mount, page-cache hits
  included. The bytes on the wire are the `nfs_server_*` columns.
- **NFS operations on node 0.** On node 0, `nfs_ops_per_s` includes the
  workspace walk's own operations. Read the model's own operations on node 1
  and above.
- **Peak memory.** `job_mem_peak_bytes_max` is `memory.peak` of the job
  cgroup, which includes page cache and the sampler. It can be well above the
  memory that the job's processes use.
- **Workspace scopes.** `workspace.stats` in `summary.json` covers every kept
  walk of every exported job. The headline `workspace.allocated_bytes_max`
  covers only the walks of the reported jobs, so the two can differ.
- **All-null columns.** A column that is null on every row, such as
  `job_pids` when the job cgroup has no pids controller, has no statistic and
  no hook. Its Parquet column keeps its type and is all null. A column of
  real zeros still has statistics, so absent and 0 stay distinct.

## Staging, for operators

While a job runs, the sampler of each node appends to one staging file in the
workspace `<WS>`:

```text
<WS>/.hpcmu/telemetry/<phase>-<job_id>-r<restart>-<node_index>.ndjson
```

Each line is one ASCII JSON object of at most `STAGING_LINE_LIMIT` bytes
(65536), its newline included. There are four kinds of line:

| Line | `kind` | Content |
| --- | --- | --- |
| header | `hpcmu.telemetry.header` | the first line: the staging header fields of the reference |
| sample | no `kind` key | one sample: the `samples.parquet` columns |
| walk | `hpcmu.telemetry.walk` | node 0 only, one workspace walk: the `workspace.parquet` columns |
| error | `hpcmu.telemetry.error` | the first failure of a source and error class: `kind`, `ts`, `source`, `error`, `message` |

The sidecar's own output goes to the same directory:
`<phase>-<job_id>-r<restart>-<task>.log` holds the output of each node's
sampler task, and `<phase>-<job_id>-r<restart>-srun.log` the output of the
`srun` client. Neither reaches the relayed job log.

To follow node 0 of a running model job live:

```bash
tail -f <WS>/.hpcmu/telemetry/model-<job_id>-r0-0.ndjson | jq -c 'select(has("kind") | not) | {ts, cpu_busy_cores, mem_available_bytes}'
```

Staging is an internal format (ADR-069). It may change in any release without
a `schema_version` bump: only the three exported files are the contract.

## Reference

This section is generated by `render_markdown()` in
`src/hpc_model_utils/infra/telemetry/schema.py`, and
`tests/contract/test_telemetry_doc.py` checks it byte for byte. To change it,
change `schema.py` and run
`HPCMU_REGEN_GOLDENS=1 uv run pytest tests/contract/test_telemetry_doc.py -k block`
once.

<!-- BEGIN GENERATED: render_markdown() in src/hpc_model_utils/infra/telemetry/schema.py; do not edit -->
### Version and files

Schema version: `1` (`SCHEMA_VERSION`).

Phases: `model`, `finalize` (`PHASES`).

| file | kind | sort order |
| --- | --- | --- |
| `samples.parquet` | `hpcmu.telemetry.samples` (`SAMPLES_KIND`) | `phase`, `job_id`, `restart`, `node_index`, `ts` |
| `workspace.parquet` | `hpcmu.telemetry.workspace` (`WORKSPACE_KIND`) | `phase`, `job_id`, `restart`, `ts` |
| `summary.json` | `hpcmu.telemetry.summary` (`SUMMARY_KIND`) | - |

### Parquet key-value metadata

Both Parquet files carry these keys in their key-value metadata.

| key | meaning |
| --- | --- |
| `hpcmu.telemetry.schema_version` | SCHEMA_VERSION as a decimal string |
| `hpcmu.telemetry.kind` | SAMPLES_KIND or WORKSPACE_KIND |
| `hpcmu.tool_version` | the hpc-model-utils version that ran the export |
| `hpcmu.run_id` | the run_id of state.json |
| `hpcmu.model` | the plugin name of the run (state.json plugin) |
| `hpcmu.headers` | the JSON list of the valid staging headers, in sample sort order |

### Conventions

- Δ is the difference between this sample's read and the previous one.
- online CPUs is the number of cpuN lines in /proc/stat.
- Every rate is null on a counter decrease, on a missing baseline, or after a failed read of its source.
- The /proc/stat CPU columns follow the per-line ADR-093 rule written into their meanings.
- A source whose read fails leaves every column of that source null in that sample; its rates are also null in the next sample, which has no baseline.
- A source with nothing to read (no job cgroup, a workspace not on an NFS mount, no EFA device) leaves its columns null and writes no error line.
- A float that is not finite is written as null.
- Every cgroup column is null when the header's job_cgroup is null.
- A header field whose source cannot be read or parsed, a malformed Slurm variable included, is null.

### samples.parquet columns

| column | type | nullable | unit | source | meaning |
| --- | --- | --- | --- | --- | --- |
| `ts` | `TIMESTAMP_MS_UTC` | no | `-` | `identity` | scheduled sample instant, a multiple of interval_s on the wall-clock grid, except the off-grid final sample taken at SIGTERM or SIGINT; strictly increasing per staging file |
| `phase` | `STRING` | no | `-` | `identity` | job phase, one of PHASES |
| `job_id` | `STRING` | no | `-` | `identity` | Slurm job id (SLURM_JOB_ID) |
| `restart` | `INT16` | no | `-` | `identity` | SLURM_RESTART_COUNT of the attempt, 0 when unset |
| `node` | `STRING` | no | `-` | `identity` | Slurm node name (SLURMD_NODENAME), or the host name (uname nodename) when SLURMD_NODENAME is unset |
| `node_index` | `INT16` | no | `-` | `identity` | SLURM_NODEID |
| `seq` | `INT32` | no | `count` | `identity` | 0-based sample number within the staging file; a gap means a dropped line |
| `dt_s` | `FLOAT32` | yes | `s` | `sampler` | monotonic seconds since the previous counter read; for seq 0 the previous read is the unwritten start-up baseline |
| `sample_cost_ms` | `FLOAT32` | yes | `ms` | `sampler` | sampler CPU time (getrusage user + system) spent on this sample, the workspace walk excluded |
| `sampler_maxrss_bytes` | `INT64` | yes | `bytes` | `sampler` | sampler peak resident set size so far (ru_maxrss × 1024); ru_maxrss carries over execve, so it can only over-report |
| `cpu_busy_cores` | `FLOAT32` | yes | `cores` | `stat` | Δbusy / T × online CPUs from the /proc/stat cpu line, with Δbusy = Δ(user + nice + system + irq + softirq) and T = Δbusy + Δsteal + max(Δidle + Δiowait, 0); null when a busy field or steal decreases, when Δidle + Δiowait \< −1, or when T ≤ 0; steal is not busy |
| `cpu_util` | `FLOAT32` | yes | `fraction` | `stat` | cpu_busy_cores / online CPUs; null as cpu_busy_cores |
| `cpu_user_cores` | `FLOAT32` | yes | `cores` | `stat` | Δ(user + nice) / T × online CPUs; null as cpu_busy_cores |
| `cpu_system_cores` | `FLOAT32` | yes | `cores` | `stat` | Δ(system + irq + softirq) / T × online CPUs; null as cpu_busy_cores |
| `cpu_iowait_cores` | `FLOAT32` | yes | `cores` | `stat` | Δiowait / T × online CPUs; null as cpu_busy_cores, and also null when Δiowait \< 0 or Δidle \< 0 |
| `cpu_steal_cores` | `FLOAT32` | yes | `cores` | `stat` | Δsteal / T × online CPUs; null as cpu_busy_cores |
| `cpu_core_util_min` | `FLOAT32` | yes | `fraction` | `stat` | minimum over the cpuN lines of Δbusy / T of that CPU, under the same null rule per CPU; a CPU meeting the null rule is left out, and the value is null when every CPU is |
| `cpu_core_util_p50` | `FLOAT32` | yes | `fraction` | `stat` | median over the cpuN lines of Δbusy / T of that CPU, under the same null rule per CPU; a CPU meeting the null rule is left out, and the value is null when every CPU is |
| `cpu_core_util_max` | `FLOAT32` | yes | `fraction` | `stat` | maximum over the cpuN lines of Δbusy / T of that CPU, under the same null rule per CPU; a CPU meeting the null rule is left out, and the value is null when every CPU is |
| `cpu_active_cores` | `INT16` | yes | `count` | `stat` | number of cpuN lines with Δbusy / T >= 0.5, under the same null rule per CPU; null when every CPU is left out |
| `load1` | `FLOAT32` | yes | `-` | `loadavg` | first field of /proc/loadavg |
| `procs_running` | `INT32` | yes | `count` | `stat` | procs_running of /proc/stat at the sample |
| `procs_blocked` | `INT32` | yes | `count` | `stat` | procs_blocked of /proc/stat at the sample |
| `mem_used_bytes` | `INT64` | yes | `bytes` | `meminfo` | (MemTotal − MemAvailable) × 1024 |
| `mem_available_bytes` | `INT64` | yes | `bytes` | `meminfo` | MemAvailable × 1024 |
| `mem_anon_bytes` | `INT64` | yes | `bytes` | `meminfo` | AnonPages × 1024 |
| `mem_cached_bytes` | `INT64` | yes | `bytes` | `meminfo` | Cached × 1024 |
| `mem_shmem_bytes` | `INT64` | yes | `bytes` | `meminfo` | Shmem × 1024; includes MPI shared-memory segments in /dev/shm |
| `mem_dirty_bytes` | `INT64` | yes | `bytes` | `meminfo` | Dirty × 1024 |
| `mem_writeback_bytes` | `INT64` | yes | `bytes` | `meminfo` | Writeback × 1024 |
| `mem_slab_bytes` | `INT64` | yes | `bytes` | `meminfo` | Slab × 1024 |
| `swap_used_bytes` | `INT64` | yes | `bytes` | `meminfo` | (SwapTotal − SwapFree) × 1024 |
| `major_faults_per_s` | `FLOAT32` | yes | `1/s` | `vmstat` | Δpgmajfault / dt_s |
| `oom_kills` | `INT32` | yes | `count` | `vmstat` | Δoom_kill of /proc/vmstat in this interval |
| `job_cpu_cores` | `FLOAT32` | yes | `cores` | `cgroup` | (Δusage_usec of the job cgroup − Δusage_usec of the sampler's own step) / (1e6 × dt_s), clamped at 0; no subtraction when own_step_cgroup is null |
| `job_mem_bytes` | `INT64` | yes | `bytes` | `cgroup` | memory.current of the job cgroup − memory.current of the own step, clamped at 0; no subtraction when own_step_cgroup is null |
| `job_mem_anon_bytes` | `INT64` | yes | `bytes` | `cgroup` | memory.stat anon of the job cgroup − anon of the own step, clamped at 0; no subtraction when own_step_cgroup is null |
| `job_mem_file_bytes` | `INT64` | yes | `bytes` | `cgroup` | memory.stat file of the job cgroup − file of the own step, clamped at 0; no subtraction when own_step_cgroup is null |
| `job_mem_shmem_bytes` | `INT64` | yes | `bytes` | `cgroup` | memory.stat shmem of the job cgroup − shmem of the own step, clamped at 0; no subtraction when own_step_cgroup is null |
| `job_mem_peak_bytes` | `INT64` | yes | `bytes` | `cgroup` | memory.peak of the job cgroup: the kernel high-water mark since the cgroup was created, page cache and the sampler included; never subtracted |
| `job_pids` | `INT32` | yes | `count` | `cgroup` | pids.current of the job cgroup, the sampler step included; null when the job cgroup has no pids controller (pids.current absent) |
| `job_oom_kills` | `INT32` | yes | `count` | `cgroup` | Δoom_kill of the job cgroup's memory.events in this interval |
| `psi_cpu_some` | `FLOAT32` | yes | `fraction` | `pressure` | Δ some total (µs) of /proc/pressure/cpu / (1e6 × dt_s), clamped to [0, 1] |
| `psi_mem_some` | `FLOAT32` | yes | `fraction` | `pressure` | Δ some total of /proc/pressure/memory / (1e6 × dt_s), clamped to [0, 1] |
| `psi_mem_full` | `FLOAT32` | yes | `fraction` | `pressure` | Δ full total of /proc/pressure/memory / (1e6 × dt_s), clamped to [0, 1] |
| `psi_io_some` | `FLOAT32` | yes | `fraction` | `pressure` | Δ some total of /proc/pressure/io / (1e6 × dt_s), clamped to [0, 1] |
| `psi_io_full` | `FLOAT32` | yes | `fraction` | `pressure` | Δ full total of /proc/pressure/io / (1e6 × dt_s), clamped to [0, 1] |
| `disk_read_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `diskstats` | Σ Δsectors read × 512 / dt_s over whole local disks (a /sys/block/\<dev>/device entry, and a /sys/block/\<dev>/hidden file that is absent or holds 0) present in both reads |
| `disk_write_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `diskstats` | Σ Δsectors written × 512 / dt_s over the same disks |
| `disk_busy_max` | `FLOAT32` | yes | `fraction` | `diskstats` | maximum over the same disks of Δio_ticks (ms) / (1000 × dt_s), clamped to [0, 1] |
| `nfs_read_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `mountstats` | Δ(normalreadbytes + directreadbytes) / dt_s of the workspace mount (the mountstats entry whose mountpoint is the longest prefix of the workspace path, its symlinks resolved) |
| `nfs_write_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `mountstats` | Δ(normalwritebytes + directwritebytes) / dt_s of the workspace mount |
| `nfs_server_read_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `mountstats` | Δserverreadbytes / dt_s of the workspace mount |
| `nfs_server_write_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `mountstats` | Δserverwritebytes / dt_s of the workspace mount |
| `nfs_read_ops_per_s` | `FLOAT32` | yes | `1/s` | `mountstats` | ΔREAD ops / dt_s of the workspace mount |
| `nfs_write_ops_per_s` | `FLOAT32` | yes | `1/s` | `mountstats` | ΔWRITE ops / dt_s of the workspace mount |
| `nfs_ops_per_s` | `FLOAT32` | yes | `1/s` | `mountstats` | Δ(ops summed over every per-op line) / dt_s of the workspace mount; on node 0 it includes the walk's own operations |
| `nfs_read_rtt_ms` | `FLOAT32` | yes | `ms` | `mountstats` | ΔREAD rtt (ms) / ΔREAD ops; null when ΔREAD ops is 0 |
| `nfs_write_rtt_ms` | `FLOAT32` | yes | `ms` | `mountstats` | ΔWRITE rtt (ms) / ΔWRITE ops; null when ΔWRITE ops is 0 |
| `net_rx_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `netdev` | Σ Δreceive bytes / dt_s over interfaces with a /sys/class/net/\<if>/device entry, present in both reads |
| `net_tx_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `netdev` | Σ Δtransmit bytes / dt_s over the same interfaces |
| `net_errors_per_s` | `FLOAT32` | yes | `1/s` | `netdev` | Σ Δ(rx errs + rx drop + tx errs + tx drop) / dt_s over the same interfaces |
| `efa_rx_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `efa` | Σ Δhw_counters/rx_bytes / dt_s over /sys/class/infiniband/\*/ports/1; null without an EFA device; MPI traffic over EFA bypasses net_\* |
| `efa_tx_bytes_per_s` | `FLOAT64` | yes | `bytes/s` | `efa` | Σ Δhw_counters/tx_bytes / dt_s over the same ports; null without an EFA device |

### workspace.parquet columns

| column | type | nullable | unit | source | meaning |
| --- | --- | --- | --- | --- | --- |
| `ts` | `TIMESTAMP_MS_UTC` | no | `-` | `identity` | tick instant of the sample after which the walk ran |
| `phase` | `STRING` | no | `-` | `identity` | job phase, one of PHASES |
| `job_id` | `STRING` | no | `-` | `identity` | Slurm job id (SLURM_JOB_ID) |
| `restart` | `INT16` | no | `-` | `identity` | SLURM_RESTART_COUNT of the attempt, 0 when unset |
| `node` | `STRING` | no | `-` | `identity` | Slurm node name (SLURMD_NODENAME), or the host name (uname nodename) when SLURMD_NODENAME is unset |
| `apparent_bytes` | `INT64` | yes | `bytes` | `walk` | Σ lstat st_size over every entry below the workspace root, .hpcmu/telemetry excluded |
| `allocated_bytes` | `INT64` | yes | `bytes` | `walk` | Σ lstat st_blocks × 512 over the same entries |
| `files` | `INT64` | yes | `count` | `walk` | non-directory entries |
| `dirs` | `INT64` | yes | `count` | `walk` | directory entries below the root |
| `walk_s` | `FLOAT32` | yes | `s` | `walk` | wall seconds the walk took |
| `fs_used_bytes` | `INT64` | yes | `bytes` | `statvfs` | (f_blocks − f_bfree) × f_frsize of the workspace filesystem |
| `fs_avail_bytes` | `INT64` | yes | `bytes` | `statvfs` | f_bavail × f_frsize |

### Staging header fields

Each field is published verbatim in the `hpcmu.headers` metadata key and in `summary.json` `jobs[].nodes[].header`.

| field | JSON type | nullable | meaning |
| --- | --- | --- | --- |
| `kind` | `string` | no | line kind, hpcmu.telemetry.header |
| `schema_version` | `integer` | no | telemetry schema version of the writer (SCHEMA_VERSION) |
| `tool_version` | `string` | no | hpc-model-utils version, from the sampler's --tool-version argument |
| `phase` | `string` | no | job phase, one of PHASES |
| `job_id` | `string` | no | Slurm job id (SLURM_JOB_ID) |
| `restart` | `integer` | no | SLURM_RESTART_COUNT of the attempt, 0 when unset |
| `node` | `string` | no | Slurm node name (SLURMD_NODENAME), or the host name (uname nodename) when SLURMD_NODENAME is unset |
| `node_index` | `integer` | no | SLURM_NODEID |
| `nodes` | `integer` | yes | SLURM_JOB_NUM_NODES; null when it is unset or empty |
| `partition` | `string` | yes | SLURM_JOB_PARTITION, stripped; null when it is unset or blank |
| `cpus_on_node` | `integer` | yes | SLURM_JOB_CPUS_PER_NODE indexed by SLURM_NODEID, after expanding Slurm's compressed list (192(x2) is 192, 192); null when SLURM_JOB_CPUS_PER_NODE is unset or empty, or when SLURM_NODEID is outside the expanded list |
| `instance_type` | `string` | yes | DMI product_name |
| `cpu_model` | `string` | yes | value of the first model name line of /proc/cpuinfo |
| `ncpus` | `integer` | yes | online CPUs |
| `mem_total_bytes` | `integer` | yes | MemTotal × 1024 |
| `kernel` | `string` | yes | /proc/sys/kernel/osrelease |
| `job_cgroup` | `string` | yes | /sys/fs/cgroup path of the job cgroup, the parent of the step component of /proc/self/cgroup; null without a step component |
| `own_step_cgroup` | `string` | yes | /sys/fs/cgroup path of the sampler's own step, subtracted from the job values; null when the sampler's cgroup has no numbered step_N component (a step_batch, step_extern or step_interactive step, or no step), and then nothing is subtracted |
| `workspace_mount` | `object` | yes | {mountpoint, fstype} of the mountstats entry whose mountpoint is the longest prefix of the workspace path, its symlinks resolved; null when no entry matches |
| `interval_s` | `integer` | no | sample interval in seconds, the sampler's --interval argument |
| `workspace_interval_s` | `integer` | no | configured workspace walk interval in seconds, the sampler's --workspace-interval argument |
| `started_at` | `string` | no | sampler start time, ISO 8601 UTC with milliseconds and a Z |

### summary.json layout

A path joins the keys with `.`, and `[]` stands for the elements of an array. A `map` is an object whose keys are listed in another table. A required path is present in every object of its parent.

| path | JSON type | required | nullable | meaning |
| --- | --- | --- | --- | --- |
| `kind` | `string` | yes | no | the summary kind, hpcmu.telemetry.summary |
| `schema_version` | `integer` | yes | no | the schema version of the contract |
| `run` | `object` | yes | no | the run the export covers |
| `run.run_id` | `string` | yes | no | the run_id of state.json |
| `run.model` | `string` | yes | no | the plugin name of the run |
| `run.tool_version` | `string` | yes | no | the hpc-model-utils version that ran the export |
| `run.interval_s` | `integer` | yes | yes | interval_s of the first valid header; null when no header is valid |
| `run.workspace_interval_s` | `integer` | yes | yes | workspace_interval_s of the first valid header; null when no header is valid |
| `jobs` | `array` | yes | no | one entry per exported job, in the order of the recorded jobs |
| `jobs[].phase` | `string` | yes | no | a PHASES value |
| `jobs[].job_id` | `string` | yes | no | the Slurm job id |
| `jobs[].nodes` | `array` | yes | no | one entry per valid staging file of the job, in sample sort order |
| `jobs[].nodes[].header` | `map` | yes | no | the staging header, keys as in the staging header fields table |
| `jobs[].nodes[].first_ts` | `integer` | no | no | ts of the first kept sample, in ms since the epoch; absent without a sample |
| `jobs[].nodes[].last_ts` | `integer` | no | no | ts of the last kept sample; absent without a sample |
| `jobs[].nodes[].samples` | `integer` | yes | no | the number of kept samples |
| `jobs[].nodes[].gaps` | `integer` | yes | no | the number of samples whose dt_s exceeds 1.5 × interval_s |
| `jobs[].nodes[].max_gap_s` | `number` | no | no | the largest non-null dt_s; absent when every dt_s is null |
| `jobs[].nodes[].stats` | `map` | yes | no | the samples.parquet statistics of this file, names as in the statistics table |
| `workspace` | `object` | yes | no | the workspace walks of every exported job |
| `workspace.walks` | `integer` | yes | no | the number of kept walks |
| `workspace.stats` | `map` | yes | no | the workspace.parquet statistics over every kept walk |
| `headline` | `object` | yes | no | the headline values and the jobs they come from |
| `headline.job_ids` | `map` | yes | no | for each phase, the job id of its last recorded job, the one the headline values report |
| `headline.values` | `map` | yes | no | the headline values, keys as in the headline table |
| `warnings` | `array` | yes | no | at most WARNINGS_CAP warnings, in file order then the missing nodes |
| `warnings[].code` | `string` | yes | no | a warning code |
| `warnings[].phase` | `string` | yes | yes | the attempt's phase; null for warnings_capped |
| `warnings[].job_id` | `string` | yes | yes | the attempt's job id; null for warnings_capped |
| `warnings[].restart` | `integer` | yes | yes | the attempt's restart; null for warnings_capped |
| `warnings[].node_index` | `integer` | yes | yes | the attempt's node index; null for warnings_capped |
| `warnings[].count` | `integer` | yes | no | the occurrences this warning stands for |
| `warnings[].message` | `string` | yes | no | a human-readable detail |

### summary.json statistics

Every aggregate except `total` is published as `<column>_<aggregate>`, and the total under its listed name. The aggregates are `min`, `mean`, `p95`, `p99`, `max`, `total`, in this order. A statistic with no contributing value is absent, never 0.

| column | file | summary.json map | names |
| --- | --- | --- | --- |
| `sample_cost_ms` | `samples.parquet` | `jobs[].nodes[].stats` | `sample_cost_ms_mean`, `sample_cost_ms_p95`, `sample_cost_ms_p99`, `sample_cost_ms_max` |
| `sampler_maxrss_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `sampler_maxrss_bytes_max` |
| `cpu_busy_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_busy_cores_min`, `cpu_busy_cores_mean`, `cpu_busy_cores_p95`, `cpu_busy_cores_max` |
| `cpu_util` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_util_min`, `cpu_util_mean`, `cpu_util_p95`, `cpu_util_max` |
| `cpu_user_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_user_cores_min`, `cpu_user_cores_mean`, `cpu_user_cores_p95`, `cpu_user_cores_max` |
| `cpu_system_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_system_cores_min`, `cpu_system_cores_mean`, `cpu_system_cores_p95`, `cpu_system_cores_max` |
| `cpu_iowait_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_iowait_cores_min`, `cpu_iowait_cores_mean`, `cpu_iowait_cores_p95`, `cpu_iowait_cores_max` |
| `cpu_steal_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_steal_cores_min`, `cpu_steal_cores_mean`, `cpu_steal_cores_p95`, `cpu_steal_cores_max` |
| `cpu_core_util_min` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_core_util_min_min`, `cpu_core_util_min_mean`, `cpu_core_util_min_p95`, `cpu_core_util_min_max` |
| `cpu_core_util_p50` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_core_util_p50_min`, `cpu_core_util_p50_mean`, `cpu_core_util_p50_p95`, `cpu_core_util_p50_max` |
| `cpu_core_util_max` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_core_util_max_min`, `cpu_core_util_max_mean`, `cpu_core_util_max_p95`, `cpu_core_util_max_max` |
| `cpu_active_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `cpu_active_cores_min`, `cpu_active_cores_mean`, `cpu_active_cores_p95`, `cpu_active_cores_max` |
| `load1` | `samples.parquet` | `jobs[].nodes[].stats` | `load1_min`, `load1_mean`, `load1_p95`, `load1_max` |
| `procs_running` | `samples.parquet` | `jobs[].nodes[].stats` | `procs_running_min`, `procs_running_mean`, `procs_running_p95`, `procs_running_max` |
| `procs_blocked` | `samples.parquet` | `jobs[].nodes[].stats` | `procs_blocked_min`, `procs_blocked_mean`, `procs_blocked_p95`, `procs_blocked_max` |
| `mem_used_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_used_bytes_min`, `mem_used_bytes_mean`, `mem_used_bytes_p95`, `mem_used_bytes_max` |
| `mem_available_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_available_bytes_min`, `mem_available_bytes_mean`, `mem_available_bytes_p95`, `mem_available_bytes_max` |
| `mem_anon_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_anon_bytes_min`, `mem_anon_bytes_mean`, `mem_anon_bytes_p95`, `mem_anon_bytes_max` |
| `mem_cached_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_cached_bytes_min`, `mem_cached_bytes_mean`, `mem_cached_bytes_p95`, `mem_cached_bytes_max` |
| `mem_shmem_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_shmem_bytes_min`, `mem_shmem_bytes_mean`, `mem_shmem_bytes_p95`, `mem_shmem_bytes_max` |
| `mem_dirty_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_dirty_bytes_min`, `mem_dirty_bytes_mean`, `mem_dirty_bytes_p95`, `mem_dirty_bytes_max` |
| `mem_writeback_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_writeback_bytes_min`, `mem_writeback_bytes_mean`, `mem_writeback_bytes_p95`, `mem_writeback_bytes_max` |
| `mem_slab_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `mem_slab_bytes_min`, `mem_slab_bytes_mean`, `mem_slab_bytes_p95`, `mem_slab_bytes_max` |
| `swap_used_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `swap_used_bytes_min`, `swap_used_bytes_mean`, `swap_used_bytes_p95`, `swap_used_bytes_max` |
| `major_faults_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `major_faults_per_s_mean`, `major_faults_per_s_p95`, `major_faults_per_s_max`, `major_faults_total` |
| `oom_kills` | `samples.parquet` | `jobs[].nodes[].stats` | `oom_kills_max` |
| `job_cpu_cores` | `samples.parquet` | `jobs[].nodes[].stats` | `job_cpu_cores_min`, `job_cpu_cores_mean`, `job_cpu_cores_p95`, `job_cpu_cores_max` |
| `job_mem_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `job_mem_bytes_min`, `job_mem_bytes_mean`, `job_mem_bytes_p95`, `job_mem_bytes_max` |
| `job_mem_anon_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `job_mem_anon_bytes_min`, `job_mem_anon_bytes_mean`, `job_mem_anon_bytes_p95`, `job_mem_anon_bytes_max` |
| `job_mem_file_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `job_mem_file_bytes_min`, `job_mem_file_bytes_mean`, `job_mem_file_bytes_p95`, `job_mem_file_bytes_max` |
| `job_mem_shmem_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `job_mem_shmem_bytes_min`, `job_mem_shmem_bytes_mean`, `job_mem_shmem_bytes_p95`, `job_mem_shmem_bytes_max` |
| `job_mem_peak_bytes` | `samples.parquet` | `jobs[].nodes[].stats` | `job_mem_peak_bytes_max` |
| `job_pids` | `samples.parquet` | `jobs[].nodes[].stats` | `job_pids_min`, `job_pids_mean`, `job_pids_p95`, `job_pids_max` |
| `job_oom_kills` | `samples.parquet` | `jobs[].nodes[].stats` | `job_oom_kills_max` |
| `psi_cpu_some` | `samples.parquet` | `jobs[].nodes[].stats` | `psi_cpu_some_min`, `psi_cpu_some_mean`, `psi_cpu_some_p95`, `psi_cpu_some_max` |
| `psi_mem_some` | `samples.parquet` | `jobs[].nodes[].stats` | `psi_mem_some_min`, `psi_mem_some_mean`, `psi_mem_some_p95`, `psi_mem_some_max` |
| `psi_mem_full` | `samples.parquet` | `jobs[].nodes[].stats` | `psi_mem_full_min`, `psi_mem_full_mean`, `psi_mem_full_p95`, `psi_mem_full_max` |
| `psi_io_some` | `samples.parquet` | `jobs[].nodes[].stats` | `psi_io_some_min`, `psi_io_some_mean`, `psi_io_some_p95`, `psi_io_some_max` |
| `psi_io_full` | `samples.parquet` | `jobs[].nodes[].stats` | `psi_io_full_min`, `psi_io_full_mean`, `psi_io_full_p95`, `psi_io_full_max` |
| `disk_read_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `disk_read_bytes_per_s_mean`, `disk_read_bytes_per_s_p95`, `disk_read_bytes_per_s_max`, `disk_read_bytes_total` |
| `disk_write_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `disk_write_bytes_per_s_mean`, `disk_write_bytes_per_s_p95`, `disk_write_bytes_per_s_max`, `disk_write_bytes_total` |
| `disk_busy_max` | `samples.parquet` | `jobs[].nodes[].stats` | `disk_busy_max_min`, `disk_busy_max_mean`, `disk_busy_max_p95`, `disk_busy_max_max` |
| `nfs_read_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_read_bytes_per_s_mean`, `nfs_read_bytes_per_s_p95`, `nfs_read_bytes_per_s_max`, `nfs_read_bytes_total` |
| `nfs_write_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_write_bytes_per_s_mean`, `nfs_write_bytes_per_s_p95`, `nfs_write_bytes_per_s_max`, `nfs_write_bytes_total` |
| `nfs_server_read_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_server_read_bytes_per_s_mean`, `nfs_server_read_bytes_per_s_p95`, `nfs_server_read_bytes_per_s_max`, `nfs_server_read_bytes_total` |
| `nfs_server_write_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_server_write_bytes_per_s_mean`, `nfs_server_write_bytes_per_s_p95`, `nfs_server_write_bytes_per_s_max`, `nfs_server_write_bytes_total` |
| `nfs_read_ops_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_read_ops_per_s_mean`, `nfs_read_ops_per_s_p95`, `nfs_read_ops_per_s_max`, `nfs_read_ops_total` |
| `nfs_write_ops_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_write_ops_per_s_mean`, `nfs_write_ops_per_s_p95`, `nfs_write_ops_per_s_max`, `nfs_write_ops_total` |
| `nfs_ops_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_ops_per_s_mean`, `nfs_ops_per_s_p95`, `nfs_ops_per_s_max`, `nfs_ops_total` |
| `nfs_read_rtt_ms` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_read_rtt_ms_min`, `nfs_read_rtt_ms_p95`, `nfs_read_rtt_ms_max` |
| `nfs_write_rtt_ms` | `samples.parquet` | `jobs[].nodes[].stats` | `nfs_write_rtt_ms_min`, `nfs_write_rtt_ms_p95`, `nfs_write_rtt_ms_max` |
| `net_rx_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `net_rx_bytes_per_s_mean`, `net_rx_bytes_per_s_p95`, `net_rx_bytes_per_s_max`, `net_rx_bytes_total` |
| `net_tx_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `net_tx_bytes_per_s_mean`, `net_tx_bytes_per_s_p95`, `net_tx_bytes_per_s_max`, `net_tx_bytes_total` |
| `net_errors_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `net_errors_per_s_mean`, `net_errors_per_s_p95`, `net_errors_per_s_max`, `net_errors_total` |
| `efa_rx_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `efa_rx_bytes_per_s_mean`, `efa_rx_bytes_per_s_p95`, `efa_rx_bytes_per_s_max`, `efa_rx_bytes_total` |
| `efa_tx_bytes_per_s` | `samples.parquet` | `jobs[].nodes[].stats` | `efa_tx_bytes_per_s_mean`, `efa_tx_bytes_per_s_p95`, `efa_tx_bytes_per_s_max`, `efa_tx_bytes_total` |
| `apparent_bytes` | `workspace.parquet` | `workspace.stats` | `apparent_bytes_max` |
| `allocated_bytes` | `workspace.parquet` | `workspace.stats` | `allocated_bytes_max` |
| `files` | `workspace.parquet` | `workspace.stats` | `files_max` |
| `dirs` | `workspace.parquet` | `workspace.stats` | `dirs_max` |
| `walk_s` | `workspace.parquet` | `workspace.stats` | `walk_s_max` |
| `fs_used_bytes` | `workspace.parquet` | `workspace.stats` | `fs_used_bytes_max` |
| `fs_avail_bytes` | `workspace.parquet` | `workspace.stats` | `fs_avail_bytes_min` |

### Headline values and hooks

`headline.values` holds `<scope>.<statistic>`, and the ModelOps metadata hook of a value is `telemetry.<scope>.<statistic>` (`HEADLINE_KEY_PREFIX`). `<statistic>` is a published statistic name, and the scopes are `model`, `finalize`, `workspace`.

| pooling | meaning |
| --- | --- |
| `node_rows` | the statistic over the rows of every node and restart pooled together, so the value is a per-node figure: a mean is the mean of one node and a max is the worst node |
| `sum_nodes` | the sum over the rows of every node and restart, a whole-job figure |
| `walks` | the statistic over the walks of the reported jobs |

| format | meaning |
| --- | --- |
| `bytes` | an integer number of bytes, rounded half to even, written as plain digits |
| `cores` | a fixed-point number with 2 decimals, no sign and no exponent |
| `fraction` | a fixed-point number with 4 decimals, no sign and no exponent |

| hook key | scope | file | column | aggregate | pooling | format |
| --- | --- | --- | --- | --- | --- | --- |
| `telemetry.model.job_mem_peak_bytes_max` | `model` | `samples.parquet` | `job_mem_peak_bytes` | `max` | `node_rows` | `bytes` |
| `telemetry.model.mem_available_bytes_min` | `model` | `samples.parquet` | `mem_available_bytes` | `min` | `node_rows` | `bytes` |
| `telemetry.model.cpu_busy_cores_mean` | `model` | `samples.parquet` | `cpu_busy_cores` | `mean` | `node_rows` | `cores` |
| `telemetry.model.cpu_busy_cores_p95` | `model` | `samples.parquet` | `cpu_busy_cores` | `p95` | `node_rows` | `cores` |
| `telemetry.model.cpu_util_mean` | `model` | `samples.parquet` | `cpu_util` | `mean` | `node_rows` | `fraction` |
| `telemetry.model.nfs_read_bytes_total` | `model` | `samples.parquet` | `nfs_read_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.model.nfs_write_bytes_total` | `model` | `samples.parquet` | `nfs_write_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.model.disk_read_bytes_total` | `model` | `samples.parquet` | `disk_read_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.model.disk_write_bytes_total` | `model` | `samples.parquet` | `disk_write_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.finalize.job_mem_peak_bytes_max` | `finalize` | `samples.parquet` | `job_mem_peak_bytes` | `max` | `node_rows` | `bytes` |
| `telemetry.finalize.mem_available_bytes_min` | `finalize` | `samples.parquet` | `mem_available_bytes` | `min` | `node_rows` | `bytes` |
| `telemetry.finalize.cpu_busy_cores_mean` | `finalize` | `samples.parquet` | `cpu_busy_cores` | `mean` | `node_rows` | `cores` |
| `telemetry.finalize.cpu_busy_cores_p95` | `finalize` | `samples.parquet` | `cpu_busy_cores` | `p95` | `node_rows` | `cores` |
| `telemetry.finalize.cpu_util_mean` | `finalize` | `samples.parquet` | `cpu_util` | `mean` | `node_rows` | `fraction` |
| `telemetry.finalize.nfs_read_bytes_total` | `finalize` | `samples.parquet` | `nfs_read_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.finalize.nfs_write_bytes_total` | `finalize` | `samples.parquet` | `nfs_write_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.finalize.disk_read_bytes_total` | `finalize` | `samples.parquet` | `disk_read_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.finalize.disk_write_bytes_total` | `finalize` | `samples.parquet` | `disk_write_bytes_per_s` | `total` | `sum_nodes` | `bytes` |
| `telemetry.workspace.allocated_bytes_max` | `workspace` | `workspace.parquet` | `allocated_bytes` | `max` | `walks` | `bytes` |

### Warning codes

| code | meaning |
| --- | --- |
| `bad_header` | the file has no valid header line, and the file is skipped |
| `truncated_tail` | the last line has no newline and is dropped |
| `bad_line` | a line is not ASCII JSON within the line limit and is dropped |
| `bad_record` | a JSON line breaks the record rules of its kind and is dropped |
| `identity_mismatch` | a sample or walk names another attempt than its header and is dropped |
| `non_monotonic_ts` | a sample or walk ts is not after the previous kept one and is dropped |
| `sampler_error` | an error line written by the sampler |
| `missing_node` | a node of an attempt has no valid staging file |
| `warnings_capped` | the warning list reached WARNINGS_CAP, and the count holds the warning occurrences left out of the list |

The list holds at most `500` warnings (`WARNINGS_CAP`).
<!-- END GENERATED -->
