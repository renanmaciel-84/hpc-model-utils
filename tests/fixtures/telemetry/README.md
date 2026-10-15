# Telemetry Fixture Bundles

Each `*.json` file here is a sanitized capture of the `/proc`, `/sys` and
cgroup text the node sampler reads, so its parsers run against real kernel
output (R61, ADR-086). A bundle is data only: no file tree and no symlink is
ever committed. `tests/support/telemetry_tree.py` loads a bundle
(`load_bundle`) and writes one snapshot into a directory that stands in for
`/` (`materialize`). `tests/contract/test_telemetry_fixtures.py` scans every
file in this directory.

## Bundle format

One UTF-8 JSON object, keys sorted, indented by 1, ending in a newline:

| Key | Value |
| --- | --- |
| `format` | the string `hpcmu-telemetry-fixture` |
| `version` | the integer `1` |
| `source` | a non-empty string naming where the capture came from |
| `snapshots` | a list of exactly 2 objects, captured 5 seconds apart |

Each snapshot holds:

- `files`: a mapping from a POSIX path relative to `/` (for example
  `proc/stat`) to the file's text, byte for byte. A path never starts with
  `/`, never contains a backslash and never contains a `..` component.
- `dirs`: a list of relative paths that are directories, for the sysfs device
  links that a bundle cannot carry as symlinks.

`load_bundle("workstation")` resolves to `tests/fixtures/telemetry/workstation.json`.

## Provenance

| Bundle | Source | Captured | Kernel | How |
| --- | --- | --- | --- | --- |
| `workstation.json` | a developer workstation | 2026-10-14 | `7.0.0-31-generic` | `capture-fixtures.sh --local <out>`, then `bundle-fixtures.py --source workstation --hostname "$(hostname)" --mount-path <path> <out>/node-0 <bundle>` |
| `prd.json` | `ticket-004 capture job, node 0, c7a.48xlarge` | 2026-10-15 | `6.8.0-1024-aws` | `capture-fixtures.sh` in the ticket-004 capture job, then `bundle-fixtures.py --source prd --job-id <id> --job-id <id> --job-id <id> <capture>/node-0 <bundle>` |

The kernel is the bundle's own `proc/sys/kernel/osrelease`. The workstation
has no EFA device, so those files are absent from its bundle and the synthetic
bundle covers them. Its `proc/self/cgroup` has no `step_*` component, which is
the expected case of a process outside a Slurm step.

The prd snapshots are about 5.3 s apart, against the 5 s nominal gap of the
format. The job cgroup has no `pids.current` and no `io.stat` (T3), so
`job_pids` is null on prd. `proc/self/cgroup` ends in
`step_1/user/task_special`.

## Sanitization

The raw capture and the bundler that turns it into a bundle live under the
gitignored `plans/job-telemetry/`; raw captures never leave that directory.
Only the bundler's output is committed. It replaces identifying strings and
nothing else: counter values stay verbatim (A8.4). The rules run in this
order over every file text and every relative path:

1. EC2 host names `ip-<a>-<b>-<c>-<d>` become `ip-192-0-2-<n>`.
2. IPv4 addresses in the private ranges 10/8, 172.16/12 and 192.168/16 become
   `192.0.2.<n>`, with one stable `n` per distinct address, starting at 10.
   Rules 1 and 2 share the address map.
3. FSx ids `fs-<8 to 17 hex digits>` become `fs-placeholder`.
4. Any host name in the `amazonaws` DNS domain becomes `fsx.example.invalid`.
5. ParallelCluster node names `<queue>-dy|st-<cr>-<n>` become `node-<k>`,
   numbered from 1 by first appearance.
6. Every `--job-id <id>` given on the command line becomes `4711`. The option
   repeats once per id.
7. Every `--hostname <name>` given becomes `node-0`.
8. The user id, only in context: the digits in `user-<uid>.slice`,
   `user@<uid>.service` and `/run/user/<uid>` become `1000`. A bare digit
   match is never used, because it would rewrite counters.
9. Every `--mount-path <path>` given becomes `/mnt/data`. The option repeats
   once per path. The real path is passed on the command line and is never
   written into the bundler or into a committed file.
10. Random ids:
    - a UUID becomes the all-zero UUID;
    - a 64-hex id becomes a placeholder hex string of 64 digits, and a 12-hex
      id one of 12 digits, numbered from 1 by first appearance, so distinct
      ids stay distinct. A 12-hex id is matched only after
      `/run/docker/netns/` or when it holds at least one letter `a` to `f`,
      so a 12-digit counter is never touched;
    - `br-<12 hex>` becomes `br-placeholder`;
    - `veth<hex>` becomes `veth<k>`, numbered from 0 by first appearance.

The contract scan enforces the patterns of rules 1 to 5 only. Rules 6 to 10
depend on the bundler and on a by-eye review of every new bundle for: user
id, user names, `@` and corporate domains, project and mount names, MAC
addresses, serial numbers, machine ids, random container ids and internal DNS
names. Hardware model strings, such as the CPU model and the DMI product
name, are kept.

## synthetic.json

A hand-authored bundle, not a capture. The workstation bundle lacks the prd
structures the node sampler depends on, and ADR-086 supplies them here:

- an NFSv4 workspace mount in `mountstats`;
- a Slurm step cgroup tree with a numeric `step_<n>` component (R40, ADR-063);
- a job directory whose name does not contain the job id, which is the Slurm
  26.05 SLUID rename;
- an EFA device;
- a hidden NVMe multipath path and a partition, which the whole-disk filter
  must drop;
- virtual network interfaces, which the device-link filter must drop.

| Bundle | Source | Captured | Kernel | How |
| --- | --- | --- | --- | --- |
| `synthetic.json` | hand-authored, no host | not applicable | `6.8.0-1021-synthetic` | built by a throwaway script that is not committed; every counter is an integer and every delta is chosen so each derived column has a hand-computable value |

Nothing here came from a real host, so the sanitization rules above do not
apply. The bundle uses placeholders only, such as the RFC 5737 address
`192.0.2.20` and the host `fsx.example.invalid`.

### Design

- The two snapshots are 5 s apart by design, and every delta is
  non-negative.
- The aggregate `cpu` line of `proc/stat` is the exact sum of `cpu0` to `cpu3`
  in both snapshots. That holds only here, never on a real kernel.
- Decoys, which a correct filter ignores:
  - `proc/diskstats` also lists `nvme0n1p1` (a partition, with no
    `sys/block` directory), `nvme0c0n1` (a hidden path, `hidden` is `1`) and
    `dm-0` (no device link). Each moves 1000000 sectors in both directions.
  - `proc/net/dev` also lists `lo` and `docker0`, with no device link. Each
    moves 10000000 bytes.
  - `proc/self/mountstats` has an `nfs4` entry on `/shared`, a string prefix
    of `/shared-zfs` but not a path prefix. It must not be selected for the
    workspace.
  - The job directory `job_s7f3a9c2e` does not contain the job id `4711`, so
    the job cgroup is found from the `step_0` component of
    `proc/self/cgroup`.
- The `bytes:` and per-op counters of `mountstats` are chosen one by one for
  hand-computable values, so they are not consistent with each other. For
  example, `WRITE` ops stay at 200 while the server write bytes grow.
- Values that no expected value depends on are plausible filler: the
  snapshot 0 values of `proc/meminfo` and `proc/loadavg`, the extra keys of
  `memory.stat`, `cpu.stat` and `memory.events`, the step cgroup's
  `memory.peak` (24000000), `memory.events` (all zero) and `pids.current`
  (2), and the `intr`, `ctxt` and `softirq` lines. `job_pids` is the job
  cgroup's `pids.current` alone.

### Expected values

All values assume `dt_s = 5.0` and the workspace path `/shared-zfs/ws-test`.
Column meanings are in `src/hpc_model_utils/infra/telemetry/schema.py`
(`SAMPLE_COLUMNS`); the table gives only the value. All 54 collector columns
are listed.

| Group | Expected values |
| --- | --- |
| CPU, cores | `cpu_busy_cores` 1.55; `cpu_util` 0.3875; `cpu_user_cores` 1.2; `cpu_system_cores` 0.35; `cpu_iowait_cores` 0.1; `cpu_steal_cores` 0.05 |
| CPU, per core | `cpu_core_util_min` 0.0; `cpu_core_util_p50` 0.325; `cpu_core_util_max` 0.9; `cpu_active_cores` 2; `procs_running` 4; `procs_blocked` 0; `load1` 1.5 |
| Memory | `mem_used_bytes` 6144000000; `mem_available_bytes` 10240000000; `mem_anon_bytes` 4096000000; `mem_cached_bytes` 6144000000; `mem_shmem_bytes` 512000000; `mem_dirty_bytes` 1024000; `mem_writeback_bytes` 0; `mem_slab_bytes` 307200000; `swap_used_bytes` 102400000 |
| vmstat | `major_faults_per_s` 10.0; `oom_kills` 1 |
| Job cgroup | `job_cpu_cores` 1.99; `job_mem_bytes` 8080000000; `job_mem_anon_bytes` 5035000000; `job_mem_file_bytes` 2546000000; `job_mem_shmem_bytes` 400000000; `job_mem_peak_bytes` 9100000000; `job_pids` 42; `job_oom_kills` 1 |
| PSI | `psi_cpu_some` 0.1; `psi_mem_some` 0.05; `psi_mem_full` 0.02; `psi_io_some` 0.2; `psi_io_full` 0.1 |
| Disk | `disk_read_bytes_per_s` 1024000.0; `disk_write_bytes_per_s` 3072000.0; `disk_busy_max` 0.5 |
| NFS bytes | `nfs_read_bytes_per_s` 1000000.0; `nfs_write_bytes_per_s` 500000.0; `nfs_server_read_bytes_per_s` 1100000.0; `nfs_server_write_bytes_per_s` 600000.0 |
| NFS ops | `nfs_read_ops_per_s` 10.0; `nfs_write_ops_per_s` 0.0; `nfs_ops_per_s` 30.0; `nfs_read_rtt_ms` 5.0; `nfs_write_rtt_ms` null |
| Network, EFA | `net_rx_bytes_per_s` 1000000.0; `net_tx_bytes_per_s` 500000.0; `net_errors_per_s` 1.0; `efa_rx_bytes_per_s` 100000000.0; `efa_tx_bytes_per_s` 40000000.0 |
| Attribution | job cgroup `/sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e`; own step `/sys/fs/cgroup/system.slice/slurmstepd.scope/job_s7f3a9c2e/step_0`; workspace mount `{"mountpoint": "/shared-zfs", "fstype": "nfs4"}` |
