# Runbook: telemetry probe and fixture capture

**Status.** This runbook is the R68 gate (ADR-092) before the job-script
change (E3) is relied on in prd. In one operator session on the prd head node
it runs the amended telemetry probe, `probe-telemetry.sh`, and then the fixture
capture, `capture-fixtures.sh`. ticket-004 records the results in the private
probe record, `<private-plan-dir>/design/probe-facts.md`. Neither script is in
the repository: both live in the private plan directory, under
`<private-plan-dir>/spike/`, and the operator copies them to the head node.

The repository is public, so this file uses placeholders only (R89, R104,
R127). Every real value lives in the private environment file and in the
private cluster-facts record, and is referenced here by its key or by its fact
id (F1-F16). Keep every output of this runbook under
`<private-plan-dir>/design/capture/`, never in this file.

## Preconditions

| Placeholder | Where its value comes from |
| --- | --- |
| `<head-node>` | the prd head node, the host of the cluster-facts probe (F1-F16) |
| `<user>` | the ModelOps SSH user, the account the prd jobs run as: the user of the ModelOps SSH connection that env `sshConnectionId` names |
| `<partition>` | a model partition: env `queueNewave` or `queueDecomp` (the F6 partitions). Both scripts take the same one |
| `<nodes>` | `2`, the node count both commands pass. T4, T5 and T7 are only meaningful across nodes |
| `<mpich-path>` | env `mpichPath`, the MPICH that NEWAVE and DECOMP run with. It must ship Hydra's `bin/mpiexec` |
| `<root-path>` | env `rootPath`, the workspace parent on the shared filesystem (F5, F15) |
| `<capture-dir>` | a new directory under `<root-path>`, named by the operator. It must not exist yet |
| `<private-plan-dir>` | the gitignored `plans/job-telemetry/` directory of the operator's workstation checkout of this repository |

All of the following hold before the probe runs.

1. **The test jobs.** The probe submits two test jobs, one after the other:
   the main probe job, then the standalone T8b job (ADR-092). Each holds
   `<nodes>` exclusive nodes for at most 15 minutes. The capture then submits
   a third job of the same size and limit. They queue on `<partition>` like
   any other job.
2. **Where it runs.** Both scripts run on `<head-node>` as `<user>`, outside
   any allocation: not in an `salloc` shell and not inside a job. Each script
   checks the commands it needs, and the probe checks
   `<mpich-path>/bin/mpiexec`. A missing one exits 2 before any job is
   submitted.
3. **What they write.** The probe writes only a scratch directory
   `<root-path>/.probe-telemetry.<random>`, which it deletes when it exits.
   The capture writes `<capture-dir>`, a scratch directory
   `<root-path>/.capture-fixtures.<random>` that it deletes when it exits, and
   its job log, `slurm-<id>.out`, in the directory it runs from.
4. **One unbroken session.** Keep the head-node SSH session open until each
   script returns. A dropped session stops the script, and its exit trap then
   deletes the scratch directory that the running job still writes to or
   reads from.

## Run the probe

1. Copy `probe-telemetry.sh` to the head node, together with
   `capture-fixtures.sh`, which runs in the same session. Then open the
   head-node session.

   ```bash
   # on the workstation
   scp <private-plan-dir>/spike/probe-telemetry.sh \
     <private-plan-dir>/spike/capture-fixtures.sh <user>@<head-node>:
   ssh -o ServerAliveInterval=30 <user>@<head-node>
   ```

2. Run the probe from the directory the scripts were copied to.

   ```bash
   # on the head node
   bash probe-telemetry.sh <partition> 2 <mpich-path> <root-path> 2>&1 | tee probe-telemetry.out
   ```

   T1 and T2 print at once, then `job <id>` for the main job. Most of the
   rest is waiting, with no output:
   - the node boot, about 6 minutes, before the T9 results;
   - the main job itself, a few minutes, and then about 5 more minutes
     (MinJobAge, F2) before the `=== job <id> output ===` section, because
     the probe waits for the finished main job to leave `squeue` before it
     submits T8b;
   - the T8b job, after its own `job <id>` line, which boots its nodes again
     if Slurm has released them. It ends with the `T8b RESULT` line and the
     T8b job's files.

   Expect roughly 15 to 25 minutes in all.

3. Copy `probe-telemetry.out` back, from a second workstation terminal. Keep
   the head-node session open for the capture.

   ```bash
   # on the workstation
   mkdir -p <private-plan-dir>/design/capture
   scp <user>@<head-node>:probe-telemetry.out <private-plan-dir>/design/capture/
   ```

**If a probe job never reaches RUNNING.** For the main job, the probe prints
`job <id> did not reach RUNNING (last state: <state>); T9 skipped` after about
20 minutes. For the T8b job, nothing is printed: it is still `PENDING` in
`squeue -h -j <id> -o %T`, from a second session, 20 minutes after its
`job <id>` line. Then:

1. Stop the script with Ctrl-C if it is still running. Then cancel every job
   it submitted that is still in `squeue`: `scancel <id>` for each
   `job <id>` line it printed. Stop the script first, so that it cannot
   submit the T8b job once the cancelled main job leaves the queue.
2. Keep the first output: `mv probe-telemetry.out probe-telemetry.1.out`.
3. Rerun step 2 once.
4. If a job of the rerun does not reach RUNNING either, cancel it the same way
   and go on. In step 3, copy `probe-telemetry.1.out` back too. ticket-004
   reports the facts that job would have produced (T-facts) as `unknown`.

## Capture the fixtures

1. In the same head-node session and directory, after the probe has returned,
   run the capture. `<capture-dir>` must be on the shared filesystem under
   `<root-path>`, so that the head node sees what the job nodes write. It must
   not exist yet: the script refuses an existing one, so an earlier capture is
   never mixed in.

   ```bash
   # on the head node
   bash capture-fixtures.sh <partition> 2 <capture-dir> <root-path>
   ```

   It prints `job <id>, log <dir>/slurm-<id>.out`, then waits for the job:
   the node boot, if any, and about a minute of work. It then prints
   `job <id> ended: <state>`, `CAPTURE DONE <capture-dir>` and one `ls -l`
   line per `node-<i>.tar.gz`. With fewer than 2 tarballs it prints
   `expected 2 node tarballs, found <n>` and exits 1.

   If the capture job is still `PENDING` 20 minutes after its `job <id>` line,
   stop the script with Ctrl-C, run `scancel <id>` and
   `rm -rf -- <capture-dir>`, and rerun this step once.

2. Copy the `node-*.tar.gz` files back. After an
   `expected 2 node tarballs` line, also copy the job log `slurm-<id>.out`
   named on the `job <id>, log ...` line, for the hand back.

   ```bash
   # on the workstation
   scp '<user>@<head-node>:<capture-dir>/node-*.tar.gz' <private-plan-dir>/design/capture/
   # only after an "expected 2 node tarballs" line
   scp <user>@<head-node>:slurm-<id>.out <private-plan-dir>/design/capture/
   ```

   The tarballs are raw: they can hold host names, addresses, file-system DNS
   names and job ids. They stay under `<private-plan-dir>`, and only the
   sanitized bundle of ticket-030 is ever committed.

3. Delete `<capture-dir>` on the cluster, with the capture job log.

   ```bash
   # on the head node
   rm -rf -- <capture-dir>
   rm -f -- slurm-<id>.out
   ```

## Read the results

Each of T4, T4-pmix and T5 runs 3 baseline timings, with no overlap step
running (labels `T4-base`, `T4-pmix-base` and `T5-base`), and 3 overlap
timings next to the `--overlap` step (labels `T4`, `T4-pmix` and `T5`). Each
timing prints as `<label> run=<k> elapsed_ms=<n> rc=<rc>`. Its
`<label> RESULT` line compares the medians (ADR-092). PASS needs all 6 rc
values to be 0 and an overlap median at most max(1000 ms, 50 % of the baseline
median) above the baseline median: `delta_ms` is the overlap median minus the
baseline median, and `limit_ms` is that maximum. A missing value prints
`unknown` and counts as FAIL, and so does a RESULT line missing from the
output. The RESULT line is the verdict; the `run=` lines are its context and
are never judged by eye.

```bash
# on the workstation
grep -E 'RESULT|run=|CGROUP-STEP' <private-plan-dir>/design/capture/probe-telemetry.out
```

| Marker line | Pass rule | Consequence |
| --- | --- | --- |
| `T4 RESULT PASS\|FAIL base_median_ms=<n> overlap_median_ms=<n> delta_ms=<n> limit_ms=<n>` | PASS is required: all 6 rc values are 0 and `delta_ms` is at most `limit_ms` | A FAIL stops the rollout before E3: an ADR-060 remedy, chosen in ticket-004 |
| `T4-pmix RESULT PASS\|FAIL base_median_ms=<n> overlap_median_ms=<n> delta_ms=<n> limit_ms=<n>` | the same as T4, for `srun --mpi=pmix` | the same as T4 |
| `T5 RESULT PASS\|FAIL base_median_ms=<n> overlap_median_ms=<n> delta_ms=<n> limit_ms=<n>` | the same as T4, for Hydra's `mpiexec` | the same as T4 |
| `T8 RESULT PASS\|FAIL term_received=<n> nodes=<N>` | Recorded. FAIL is not a trigger (A1); a FAIL line ends in `(A1: not a fallback trigger)` | The `srun:` and `slurmstepd:` lines around it are copied verbatim into the probe record, apart from their host names, job, step and task ids and timestamps, which ticket-004 replaces with placeholders |
| `T8b RESULT PASS\|FAIL term_received=<n> nodes=<N> completing_s=<n> killwait_s=<n> log_lines=<n>` | PASS is required: every node received TERM (`term_received` equals `nodes`), COMPLETING lasted at most KillWait (`completing_s` is at most `killwait_s`), and the T8b job log holds no `srun:` or `slurmstepd:` line (`log_lines=0`) | A FAIL stops the rollout before E3: S1 or accepting the measured hold, chosen in ticket-004 |
| `<label> run=<k> elapsed_ms=<n> rc=<rc>` | Recorded | the inputs of the T4, T4-pmix and T5 RESULT lines |
| `CGROUP-STEP path=<cgroup> cpuset.cpus.effective=<cpus> memory.max=<value>` | Recorded | the overlap step's cpuset and memory limit, one line per node |
| `ENV batch`, `ENV task=<n>` | Recorded | the `SLURM_*` and `SRUN_*` variables of the batch step and of each overlap task |
| `=== T1 ...`, `=== T2 ...`, `=== T3 ...` sections, with `KillWait`, `WaitTime` and `UnkillableStepTimeout` among the T1 lines | Recorded | the Slurm version, plugins and stop timeouts, `cgroup.conf` and the job cgroup |
| `=== T6 ...`, `=== T7 ...`, `=== T9 ...` sections | Recorded | `sstat`, the per-node facts the sampler reads, and SSH from the head node |

A FAIL of a required line stops the rollout before E3, and the operator
decides in ticket-004:

- for a T4, T4-pmix or T5 FAIL, between the three ADR-060 remedies:
  1. a sidecar flag or start-order change under ADR-062;
  2. the design §12 node-0 batch-step fallback, which leaves the nodes past 0
     unsampled and needs its own stop block;
  3. `HPCMU_TELEMETRY` off by default until a redesign;
- for a T8b FAIL, between S1, an explicit stop in the job script (write
  `model.exit`, then `kill -TERM` and a bounded wait of at most 10 s), and
  accepting the measured hold, which is open only when `completing_s` is at
  most KillWait (ADR-061).

## Hand back

Paste `probe-telemetry.out` and the tarball list into the ticket-004 session.
After a rerun, paste `probe-telemetry.1.out` too, and after an
`expected 2 node tarballs` line, the capture job log. Never paste raw output
into a public issue or PR.

```bash
# on the workstation
ls -l <private-plan-dir>/design/capture/
```
