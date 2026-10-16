# Runbook: apply the cobre workflow definitions

Note (2026-10-16): from hpc-model-utils v2.3.1, cobre runs the novomodelo
fork's `novomodelo-mpi`. The `cobre-mpi` steps in this record describe the
v0.17.0 rollout. To upload a new version, follow
[Upload a novomodelo version](#upload-a-novomodelo-version).

This runbook is the ticket-073 apply (ADR-058, ADR-057, ADR-050, ADR-051,
ADR-054, R89 as amended 2026-10-05, R105, R131). It runs after the ticket-067c
sweep. When it ends, the ModelOps catalog holds the workflow `cobre` and its
two own Tasks, and the seven shared Tasks accept the model `cobre`.

What the apply changes:

- The seven model-agnostic shared Tasks `create-workdir`, `remove-workdir`,
  `fetch-executables`, `fetch-inputs`, `extract-sanitize`, `result-upload` and
  `cancel-run` widen two regular expressions in their scripts, `modelName` and
  `path`, from `newave|decomp` to `newave|decomp|cobre`. The behavior for
  NEWAVE and DECOMP does not change.
- Two Task documents are created: `ensure-utils`, which installs only
  `hpc-model-utils`, and `cobre-run`, which passes `--max-cores-per-node` and
  the cobre MPICH to the run command.
- The workflow `cobre` is created, pinned to the `hpc-model-utils` `v2.1.0`
  release. Its chain is `ensure-utils`, `create-workdir`, `fetch-executables`,
  `fetch-inputs`, `extract-sanitize`, `cobre-run`, `result-upload` and
  `remove-workdir`.
- No existing workflow changes, and nothing is deleted by the apply.

The repository is public, so this file uses placeholders only (R104, R127).
Every real value lives in the private environment files. Record evidence only
in the `## Cobre apply` section of the private rollout-evidence record, never
here. This runbook reuses P1 of `docs/runbooks/v2-pre-rollout.md` (the tunnel
and token) and the `active_executions` function of
`docs/runbooks/v2-switch.md`. `$EV` is the pre-rollout runbook's evidence
directory.

## Preconditions

| Placeholder | Where its value comes from |
| --- | --- |
| `<cobre-workflow-commit>` | the commit that holds the cobre definitions |
| `<pre-cobre-commit>` | the parent of the cobre commit: `git rev-parse HEAD^` on a clean checkout of it |
| `<env-file>` | the private environment file; it has `env.cobreMpichPath`, and no key yet for the Tasks `ensure-utils` and `cobre-run` or the workflow `cobre` |
| `<pre-cobre-env-file>` | its sibling, a byte copy of `<env-file>` taken before `cobreMpichPath` was added |
| `<utils-v2.1.0-sha12>` | the first 12 characters of the `utilsAppSha` default of the workflow `cobre` (the `hpc-model-utils` `v2.1.0` pin) |
| `<task-id>`, `<workflow-id>` | ids that ModelOps assigns |

All of the following hold before the Preview. The first comes first on
purpose: nothing below may run without it.

1. **The sweep has passed, and nothing is running.**
   - In the private rollout-evidence record, the `## Sweep` section reads
     `Status: PASS`, with the post-retirement
     `summary: 23 unchanged, 0 changed, 0 new, 0 failures` recorded.
   - No NEWAVE - PEM, DECOMP - PEM, Upload NEWAVE, Ranqueamento Prospectivo or
     Upload Versão execution is active. Run `active_executions` (the switch
     runbook defines it) now, and again immediately before the Apply. If any of
     them is executing, wait for it to end.

   If either is missing, stop. The repository change may already be committed;
   the prd apply may not run. The `hpc-model-utils` `v2.1.0` tag must exist
   too: the Preview proves it with its `PIN v2.1.0 … ok` line.
2. **The committed cobre tree.** Work from a clean checkout of
   `<cobre-workflow-commit>`: the apply refuses a dirty tree and stamps every
   write with `HEAD`. Record `<cobre-workflow-commit>` and `<pre-cobre-commit>`
   in the evidence now, because the rollback starts from the second.
3. **Tunnel and token** per P1 of the pre-rollout runbook, with the tunnel
   opened as `ssh -N -o ServerAliveInterval=30 -L …`, a token read without
   echo, and `MODELOPS_USER` exported.
4. **The environment files.** `<env-file>` has `env.cobreMpichPath`, and
   `<pre-cobre-env-file>` exists beside it. Do not add keys for the three new
   documents: they are `NEW` until the apply creates them.

## Preview

The Preview is a dry run with the environment file. It is read-only.

```bash
EV=<evidence-dir>
mkdir -p "$EV/cobre"
uv run python -m deploy.modelops.apply sync --env-file <env-file> \
  | tee "$EV/cobre/sync-preview-$(date -u +%Y%m%dT%H%M%SZ).txt"
```

Expected output:

- `CHANGED` for `create-workdir`, `remove-workdir`, `fetch-executables`,
  `fetch-inputs`, `extract-sanitize`, `result-upload` and `cancel-run`. Each
  `CHANGED` entry is followed by its diff against live: read them. Each shows
  two changed lines, the `modelName` check and the `path` check gaining
  `|cobre`, and nothing else.
- `NEW` for `ensure-utils`, `cobre-run` and `cobre`. The workflow is rendered
  with `<new:ensure-utils>` and `<new:cobre-run>` in place of the ids of the
  two Tasks that do not exist yet, and the dry run prints no diff for a `NEW`
  document.
- Every other line `UNCHANGED`, and the six own `v1-` Tasks `PROTECTED`:
  `v1-clone-simulprospec`, `v1-clone-evalprospec`, `v1-clone-ranking-utils`,
  `v1-ranking-fetch-inputs`, `v1-clone-upload-cli` and `v1-upload-version`.
- The ten `PIN` lines, in the order of the workflow slugs. `PIN v2.1.0
  <utils-v2.1.0-sha12> ok` comes first, the pin of the workflow `cobre`:

  ```text
  PIN v2.1.0 <utils-v2.1.0-sha12> ok
  PIN v2.0.2 13f5c0516496 ok
  PIN v3.0.1 b5f4bbaccc7f ok
  PIN v2.0.2 13f5c0516496 ok
  PIN v2.4.5 00305719ee16 ok
  PIN v1.0.1 f61dd5633c8c ok
  PIN v2.4.5 00305719ee16 ok
  PIN v2.0.2 13f5c0516496 ok
  PIN v2.4.5 00305719ee16 ok
  PIN upload-versao: no SHA pin (v1)
  ```

- No `IDENTIFIER-GAP` line.
- The summary line:

  ```text
  summary: 16 unchanged, 7 changed, 3 new, 0 failures
  ```

Any other result: stop before `--apply`. Nothing was written. The environment
file is wrong, the checkout is not the cobre commit, or a document drifted live
since the sweep.

## Apply

1. **Apply.** Re-check the active executions (`active_executions`,
   Preconditions item 1). Then run `sync --apply`, read the list of writes, and
   type `apply`:

   ```bash
   uv run python -m deploy.modelops.apply sync --env-file <env-file> --apply \
     | tee "$EV/cobre/sync-apply-$(date -u +%Y%m%dT%H%M%SZ).txt"
   ```

   Expect 7 `WRITTEN PUT` Task lines, then 3 `WRITTEN POST` lines (the two
   Tasks, then the workflow), then 3 `CREATED` lines, each with its
   `(add to env file: …)` hint. The Tasks are written in slug order, and the
   `CREATED` lines print after the last write, or when the apply aborts:

   ```text
   WRITTEN PUT task cancel-run <task-id>
   WRITTEN PUT task create-workdir <task-id>
   WRITTEN PUT task extract-sanitize <task-id>
   WRITTEN PUT task fetch-executables <task-id>
   WRITTEN PUT task fetch-inputs <task-id>
   WRITTEN PUT task remove-workdir <task-id>
   WRITTEN PUT task result-upload <task-id>
   WRITTEN POST task cobre-run
   WRITTEN POST task ensure-utils
   WRITTEN POST workflow cobre
   CREATED task cobre-run <task-id>  (add to env file: tasks.cobre-run)
   CREATED task ensure-utils <task-id>  (add to env file: tasks.ensure-utils)
   CREATED workflow cobre <workflow-id>  (add to env file: workflows.cobre)
   ```

2. **Record the three ids** from the `CREATED` lines in `<env-file>`, under
   `tasks.ensure-utils`, `tasks.cobre-run` and `workflows.cobre`.

3. **Run the post-apply dry run.**

   ```bash
   uv run python -m deploy.modelops.apply sync --env-file <env-file> \
     | tee "$EV/cobre/sync-post-apply-$(date -u +%Y%m%dT%H%M%SZ).txt"
   ```

   Expect every definition `UNCHANGED` (the six own `v1-` Tasks `PROTECTED`),
   the same ten `PIN` lines, and:

   ```text
   summary: 26 unchanged, 0 changed, 0 new, 0 failures
   ```

4. **Check the UI.** The workflow list has `cobre`, and its parameter list has
   the 19 parameters.

5. **Re-check the active executions** (`active_executions`). Record the first
   NEWAVE - PEM or DECOMP - PEM execution that starts after the apply
   (ExecutionId and stored status) in the evidence; it is not a gate.

**If a step deviates:**

- **A guard refuses the apply:** nothing was written. Read the one `apply:`
  line, fix the cause, and run the Preview again.
- **The apply aborts after a `CREATED` line:** record that id in `<env-file>`
  under its key, then run the Preview again. The created document shows
  `UNCHANGED`, and the documents not yet created are still `NEW`. Run
  `sync --apply` again for those.

## Rollback

Rollback means deleting the three new documents in the UI, then re-applying
the pre-cobre commit with its own environment file. Do not use `<env-file>`
for it.

1. **Delete in the UI, by id,** first the workflow `cobre`, then the Tasks
   `cobre-run` and `ensure-utils`. The ids are `workflows.cobre`,
   `tasks.cobre-run` and `tasks.ensure-utils` of `<env-file>`. The apply has no
   delete command, and its unmanaged-reference guard refuses to rewrite a
   shared Task while a live workflow outside the tree, the workflow `cobre`,
   still references it. Never delete by name.
2. **Switch the checkout** to the pre-cobre commit:

   ```bash
   git switch --detach <pre-cobre-commit>
   ```

3. **Run the dry run** with the pre-cobre environment file:

   ```bash
   uv run python -m deploy.modelops.apply sync --env-file <pre-cobre-env-file> \
     | tee "$EV/cobre/sync-rollback-preview-$(date -u +%Y%m%dT%H%M%SZ).txt"
   ```

   Expected output:

   - the same 7 `CHANGED` Tasks as in the Preview, their diffs now removing
     `|cobre`;
   - `UNCHANGED … PROTECTED` six times, for the six own `v1-` Tasks;
   - the nine `PIN` lines of the five existing workflows, without the
     `v2.1.0` line;
   - the summary line:

     ```text
     summary: 16 unchanged, 7 changed, 0 new, 0 failures
     ```

   Any other result: stop and report it. Nothing was written.
4. **Apply.** Run `sync --apply` with the same environment file and type
   `apply`. Expect the same 7 `WRITTEN PUT` Task lines.
5. **Run the dry run again** with the same environment file. Every definition
   is `UNCHANGED`:

   ```text
   summary: 23 unchanged, 0 changed, 0 new, 0 failures
   ```

6. Record the failure, the time and the affected executions in the evidence.
   Before any later apply, remove the three keys for `cobre-run`,
   `ensure-utils` and `cobre` from `<env-file>`: the documents they name are
   gone.

### Wrong-deletion recovery

A document deleted by mistake shows up in the dry run as `MISSING`, with a
failure in the summary. Recover it:

1. Remove its key from the environment file in use (`<env-file>` before the
   rollback, `<pre-cobre-env-file>` during it), under `tasks.<slug>` or
   `workflows.<slug>`, so the dry run shows `NEW` for it and `CHANGED` for
   every workflow that references it.
2. Run `sync --apply`. It creates the document again (`WRITTEN POST`) and
   writes the referencing workflows again. A workflow PUT that returns 422
   means that workflow is executing: wait for it to end and re-run.
3. Record the `CREATED` id under the same key. The dry run then reads
   `summary: 26 unchanged, 0 changed, 0 new, 0 failures` before the rollback,
   or `summary: 23 unchanged, 0 changed, 0 new, 0 failures` during it.

## Upload Versão cobre option

This section is the ticket-074 apply (ADR-058, R89 as amended 2026-10-05, R105,
R110). It runs after the Cobre apply above has passed. When it ends, Upload
Versão offers `cobre` among its `modelName` options and clones the
`upload-versoes-cli` `v1.1.0` release, and the cobre-mpi `v0.17.0` binary sits
under `versoes/cobre/v0.17.0/`, where the cobre cluster runs fetch it.

What the apply changes: two values of the workflow `upload-versao`, and
nothing else.

- The `modelName` options become `newave`, `decomp` and `cobre`. The default
  stays `newave`.
- The hidden `uploadCliVersion` default becomes `v1.1.0`. It is a tag-only pin:
  the CLI has no SHA pin (`PIN upload-versao: no SHA pin (v1)`).
- The Tasks `v1-clone-upload-cli` and `v1-upload-version` are not edited. The
  pin is the workflow parameter.

What the upload run delivers: the `cobre-mpi` binary, zipped alone. The
ModelOps run form accepts only `.zip`, `.csv`, `.json` and `.txt` for a File
parameter, so the release archive cannot be picked as it is. The operator zips
the `cobre-mpi` member of the release archive alone into
`cobre-mpi-0.17.0.zip`. The zip holds the single member `cobre-mpi`, and the
processor uploads it as `cobre-mpi`. The plain `cobre` command-line asset and
the archive's README and licence files are not delivered.

Security note (F12): ModelOps has no RBAC. Adding `cobre` widens which models
can be published, not who can publish. No compensating control is decided
here. This section records the binary's sha256 at each hop (the release
member, the extracted file, the zip member and the S3 read-back) and enforces
no checksum check; adding one needs an explicit security decision.

Record evidence only in the `## Upload Versão cobre` section of the private
rollout-evidence record, never here. `$EV` and `active_executions` are the ones
of the sections above.

| Placeholder | Where its value comes from |
| --- | --- |
| `<upload-versao-cobre-commit>` | the commit that holds the Upload Versão change |
| `<cobre-workflow-commit>` | the commit recorded in `## Cobre apply`; its Upload Versão still has the previous two values |
| `<env-file>` | the private environment file of the Cobre apply, now holding the three ids that apply created |
| `<versoes-bucket>` | env `versionsBucket` |
| `<root-path>` | env `rootPath`, without its trailing slash |
| `<inputs-uri>` | the S3 location the operator picks `executablesFile` from |
| `<head-scratch>` | a scratch directory on the machine that fetches the asset |
| `<execution-id>` | a ModelOps ExecutionId |

### Preconditions

All of the following hold before the Preview. The first comes first on
purpose: nothing below may run without it.

1. **The Cobre apply has passed, and nothing is running.**
   - In the private rollout-evidence record, the `## Cobre apply` section reads
     `Status: PASS`.
   - `git ls-remote https://github.com/rjmalves/upload-versoes-cli.git refs/tags/v1.1.0`
     prints one line.
   - No Upload Versão execution is active. Run `active_executions` now, and
     again immediately before the Apply. If one is executing, wait for it to
     end.

   If any is missing, stop. The repository change may already be committed; the
   prd apply may not run.
2. **The committed Upload Versão tree.** Work from a clean checkout of
   `<upload-versao-cobre-commit>`: the apply refuses a dirty tree and stamps
   the write with `HEAD`. Record `<upload-versao-cobre-commit>` in the
   evidence now, with the date of items 1 and 2.
3. **Tunnel, token and environment file** as in items 3 and 4 of the
   Preconditions of the Cobre apply. `<env-file>` needs no new key.

### Preview

The Preview is a dry run. It is read-only.

```bash
EV=<evidence-dir>
mkdir -p "$EV/cobre/upload"
uv run python -m deploy.modelops.apply sync --env-file <env-file> \
  | tee "$EV/cobre/upload/sync-preview-$(date -u +%Y%m%dT%H%M%SZ).txt"
```

Expected output:

- `CHANGED` for `upload-versao` only, followed by its diff against live (`-` is
  the rendered definition, `+` is live). Read it: it shows the options pair
  `"decomp",` and `"cobre"` against `"decomp"`, and the default
  `"v1.1.0"` against `"v1.0.0"`, and nothing else.
- Every other line `UNCHANGED`, and the six own `v1-` Tasks `PROTECTED`.
- The same ten `PIN` lines as in the Preview of the Cobre apply, the last one
  being `PIN upload-versao: no SHA pin (v1)`.
- No `IDENTIFIER-GAP` line, and no `NEW`.
- The summary line:

  ```text
  summary: 25 unchanged, 1 changed, 0 new, 0 failures
  ```

Any other result: stop before `--apply`. Nothing was written.

### Apply

1. **Apply.** Re-check the active executions. Then run `sync --apply`, read the
   list of writes, and type `apply`:

   ```bash
   uv run python -m deploy.modelops.apply sync --env-file <env-file> --apply \
     | tee "$EV/cobre/upload/sync-apply-$(date -u +%Y%m%dT%H%M%SZ).txt"
   ```

   Expect exactly one `WRITTEN PUT` line, and no `WRITTEN POST` or `CREATED`:

   ```text
   WRITTEN PUT workflow upload-versao <workflow-id>
   ```

2. **Run the post-apply dry run.**

   ```bash
   uv run python -m deploy.modelops.apply sync --env-file <env-file> \
     | tee "$EV/cobre/upload/sync-post-apply-$(date -u +%Y%m%dT%H%M%SZ).txt"
   ```

   Expect every definition `UNCHANGED` (the six own `v1-` Tasks `PROTECTED`),
   the same ten `PIN` lines, and:

   ```text
   summary: 26 unchanged, 0 changed, 0 new, 0 failures
   ```

3. **Check the UI.** The `modelName` list of Upload Versão offers `newave`,
   `decomp` and `cobre`, with `newave` selected.

**If a step deviates:**

- **A guard refuses the apply:** nothing was written. Read the one `apply:`
  line, fix the cause, and run the Preview again.
- **The workflow PUT returns 422:** an Upload Versão execution is active. Wait
  for it to end and run the Preview again.

### Build the cobre-mpi zip

On the head node or the workstation, in `<head-scratch>`.

1. **Download** `cobre-mpi-0.17.0-x86_64-unknown-linux-gnu.tar.gz` from
   `https://github.com/cobre-rs/cobre/releases/tag/v0.17.0`, confirming the
   name on that page. No other asset is needed.
2. **Check the archive** holds the binary. The command prints exactly
   `cobre-mpi`; `members.txt` is the member list, for the evidence:

   ```bash
   tar -tzf cobre-mpi-0.17.0-x86_64-unknown-linux-gnu.tar.gz \
     | tee members.txt | sed 's|^\./||' | grep -x cobre-mpi
   ```

3. **Record the archive hash and the binary hash,** and extract the binary:

   ```bash
   sha256sum cobre-mpi-0.17.0-x86_64-unknown-linux-gnu.tar.gz
   tar -xzOf cobre-mpi-0.17.0-x86_64-unknown-linux-gnu.tar.gz ./cobre-mpi | sha256sum
   tar -xzf cobre-mpi-0.17.0-x86_64-unknown-linux-gnu.tar.gz ./cobre-mpi
   sha256sum cobre-mpi
   ```

   The second and fourth commands print the same hash: the binary hash.
4. **Zip the binary alone** and check the zip:

   ```bash
   zip cobre-mpi-0.17.0.zip cobre-mpi
   python3 -c 'import sys, zipfile; print(*zipfile.ZipFile(sys.argv[1]).namelist(), sep="\n")' cobre-mpi-0.17.0.zip
   python3 -c 'import hashlib, sys, zipfile; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read("cobre-mpi")).hexdigest())' cobre-mpi-0.17.0.zip
   ```

   The listing prints exactly `cobre-mpi`, and the member hash equals the
   binary hash. Without `zip`, `python3 -m zipfile -c cobre-mpi-0.17.0.zip cobre-mpi`
   builds the same single-member zip. The zip does not keep the executable bit,
   and that does not matter: `fetch-executables` sets it on every fetched file.
5. **Copy the zip:**

   ```bash
   aws s3 cp cobre-mpi-0.17.0.zip <inputs-uri>/cobre-mpi-0.17.0.zip
   ```

### Upload run

Run Upload Versão with `modelName` `cobre`, `modelVersion` `v0.17.0` and
`executablesFile` `<inputs-uri>/cobre-mpi-0.17.0.zip`.

```bash
L="$EV/cobre/upload/execution-<execution-id>.log"
jq -r .executionStatus "$EV/cobre/upload/execution-<execution-id>.json"
grep -nF 'Created temporary dir <root-path>/cobre_' "$L"
grep -nF 'Using processor for model: cobre' "$L"
grep -nF 'Found 1 files in ' "$L"
grep -nF 'Will process: cobre-mpi -> cobre-mpi' "$L"
aws s3 ls "s3://<versoes-bucket>/versoes/cobre/v0.17.0/"
aws s3 cp "s3://<versoes-bucket>/versoes/cobre/v0.17.0/cobre-mpi" - | sha256sum
```

**Pass criterion:**

- status `Success`;
- the log has `Created temporary dir <root-path>/cobre_`,
  `Using processor for model: cobre`, `Found 1 files in ` (the extraction
  directory follows) and `Will process: cobre-mpi -> cobre-mpi`;
- `aws s3 ls s3://<versoes-bucket>/versoes/cobre/v0.17.0/`
  lists exactly one object, `cobre-mpi`;
- `aws s3 cp s3://<versoes-bucket>/versoes/cobre/v0.17.0/cobre-mpi - | sha256sum`
  equals the recorded binary hash.

**If the run fails:**

- **`Unsupported model`:** the clone is not `v1.1.0`. Check the tag and the
  applied `uploadCliVersion` default, and record the failure.
- **`S3 source URI must point to a .zip or .tar.gz file`:** the `executablesFile`
  URI lost its `.zip` suffix. Copy the zip again under its name, and record the
  deviation.
- **`Downloaded file is not a valid zip archive` or `No files to process`:**
  rebuild the zip from step 4 of the build, so that its only member is
  `cobre-mpi` at the top level, copy it again, and record the deviation.

If the run does not meet its pass criterion, run the Rollback below, record the
failure, and reopen ticket-074 with the evidence.

### Rollback

Rollback means re-applying `<cobre-workflow-commit>`. It reverts the two
values; it does not remove an uploaded binary.

1. **Switch the checkout:**

   ```bash
   git switch --detach <cobre-workflow-commit>
   ```

2. **Run the dry run** with the same `<env-file>`. Expect `CHANGED` for
   `upload-versao` only, and:

   ```text
   summary: 25 unchanged, 1 changed, 0 new, 0 failures
   ```

   Any other result: stop and report it. Nothing was written.
3. **Apply.** Run `sync --apply` with the same environment file and type
   `apply`. Expect one `WRITTEN PUT workflow upload-versao <workflow-id>` line.
4. **Run the dry run again.** Every definition is `UNCHANGED`:

   ```text
   summary: 26 unchanged, 0 changed, 0 new, 0 failures
   ```

5. Record the failure, the time and the affected executions in the evidence.
   Removing the `versoes/cobre/v0.17.0/` prefix is an operator choice, and the
   choice is recorded. List first, then delete:

   ```bash
   aws s3 rm --recursive --dryrun "s3://<versoes-bucket>/versoes/cobre/v0.17.0/"
   aws s3 rm --recursive "s3://<versoes-bucket>/versoes/cobre/v0.17.0/"
   ```

## Validation

This section is the ticket-075 validation (ADR-057, ADR-051, ADR-052, ADR-054,
ADR-058, R73, R105, R118, R135). It runs after the Upload Versão cobre option
above has passed. It proves that cobre `v0.17.0` runs from the ModelOps UI
through the workflow `cobre`, pinned to `hpc-model-utils` `v2.1.0`, on 1 node
and on 4 nodes × 100 threads. Both shapes launch `cobre-mpi` through
`srun --mpi=pmix`, with one rank per node. No plain `cobre` binary exists in
the delivery, and nothing here runs one.

Four rows, run in the order V1, V2, V2b, V3:

- **V1** single node, and **V2** multi-node (the R135 reference shape), on the
  validation deck;
- **V2b**, a supplementary EFA provider log, run by hand on the head node;
- **V3**, the fast-fail relay on a broken deck, on 1 node only. An erroring
  multi-node MPI job may hold its nodes until the time limit, so V3 never runs
  on more than one node.

Record evidence only in the private cobre validation record, in the section
of each row, never here. Each row's `Status:` takes the values of the private
rollout-evidence record: `PENDING`, `PASS`, `FAIL`, `BLOCKED` or `WAIVED`. A
waiver is recorded in the private amendments record with its residual
coverage. This section reuses P1 (the tunnel and token), P4 (the
`export_execution` function), P7 (the `s3_keys` function and the artifacts
layout) and the M14 `check_relay` steps of `docs/runbooks/v2-pre-rollout.md`.
`$EV` is the pre-rollout runbook's evidence directory.

Never mark V1 or V2 `PASS` on a `Success` status alone. The captured script,
the `Backend:` line, the rank and library evidence and `FI_PROVIDER` are part
of the criterion.

### Placeholders

| Placeholder | Where its value comes from |
| --- | --- |
| `<queue-decomp>` | env `queueDecomp`, the queue default of the workflow `cobre` |
| `<cobre-mpich-lib>` | the `lib` directory beside env `cobreMpichPath` (which names the cobre MPICH `bin`) |
| `<root-path>` | env `rootPath`, without its trailing slash |
| `<inputs-uri>` | the S3 location the operator picks `inputFile` from |
| `<versoes-bucket>` | env `versionsBucket` |
| `<artifacts-bucket>`, `<hash>` | env `outputsBucket`, and the execution's ExecutionHash (P7) |
| `<head-scratch>` | a private scratch directory of the operator on the head node, on the shared filesystem of `<root-path>`: V2b runs from it on compute nodes |
| `<cobre-checkout>` | a clone of `https://github.com/cobre-rs/cobre` with the tag `v0.17.0` |
| `<deck-dir>` | a scratch directory on the workstation for the two decks |
| `<execution-id>` | a ModelOps ExecutionId |
| `<model-id>`, `<finalize-id>` | the Slurm job ids from the `run` Task's `Submitted batch job` lines: the model job first, then the finalize job |
| `<suffix>` | the six characters after `cobre_` in the run's `Created temporary dir <root-path>/cobre_<suffix>` line |

### Preconditions

All of the following hold before V1. The first two come first on purpose:
nothing below may run without them.

1. **The rollout has passed.** In the private rollout-evidence record, the
   `## Cobre apply` and `## Upload Versão cobre` sections read `Status: PASS`.
2. **The workflow is visible.** The ModelOps UI lists the workflow `cobre`,
   and its `modelVersion` offers `v0.17.0`.
3. **Tunnel and token** per P1 of the pre-rollout runbook, for the P4 exports.
4. **The head node** has `jq`, `python3`, the AWS CLI and the `s3_keys`
   function (P7).

If item 1 or 2 is missing, stop. No validation run may start.

### Decks

The validation deck is cobre's public example `examples/4ree` at the tag
`v0.17.0` (Apache-2.0), zipped under the top folder `4ree/`. Both phases are
enabled: the training stops at `iteration_limit` 256, and the simulation
samples 100 scenarios. The fast-fail deck is the same zip with
`4ree/system/buses.json` replaced by the two bytes `{\n`. cobre rejects it at
load, after its MPI backend started: exit 1 or 2, a `DataError`.

On the workstation, build both decks with read-only git, then copy them to
`<inputs-uri>`:

```bash
git -C <cobre-checkout> archive --format=zip --prefix=4ree/ \
  -o <deck-dir>/cobre-4ree.zip v0.17.0:examples/4ree
python3 -c 'import sys, zipfile
src, dst = sys.argv[1:]
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
    for i in zi.infolist():
        zo.writestr(i, b"{\n" if i.filename == "4ree/system/buses.json" else zi.read(i))
' <deck-dir>/cobre-4ree.zip <deck-dir>/cobre-4ree-broken.zip
aws s3 cp <deck-dir>/cobre-4ree.zip <inputs-uri>/cobre-4ree.zip
aws s3 cp <deck-dir>/cobre-4ree-broken.zip <inputs-uri>/cobre-4ree-broken.zip
```

`git archive` stamps the members of a tree with the current time, so the zip
bytes differ between builds. The deck is identified by the tag and the path.

### Capture and download

V1, V2 and V3 collect their evidence the same way. `<row>` is `V1`, `V2` or
`V3`.

1. **Start the run** of the workflow `cobre` in the UI with the row's inputs.
   Record the ExecutionId, the parameters you set and the start time (UTC).
2. **Capture the model script.** While the Task
   `Executa e acompanha cobre no SLURM` is active, after its first
   `Submitted batch job <model-id>` line, copy the script on the head node.
   The run removes its workspace at the end, so the copy cannot be made later.

   ```bash
   mkdir -p <head-scratch>/cobre/<row>
   cp <root-path>/cobre_<suffix>/.hpcmu/jobs/model.sbatch <head-scratch>/cobre/<row>/
   ```

3. **Export the execution** (P4) after it reaches a terminal status:
   `export_execution <execution-id> "$EV/cobre/<row>"`. The stored
   `executionArtifactsPath` names `s3://<artifacts-bucket>/artifacts/<hash>/`.
4. **Download the artifacts** on the head node (P7). V3 publishes no
   `training/metadata.json`, so skip that copy there.

   ```bash
   A=s3://<artifacts-bucket>/artifacts/<hash>
   H=<head-scratch>/cobre/<row>
   s3_keys "$A/" saidas | tee "$H/saidas.txt"
   aws s3 cp "$A/saidas/run.json" "$H/"
   aws s3 cp "$A/saidas/metadata.modelops" "$H/"
   aws s3 cp "$A/saidas/training/metadata.json" "$H/training-metadata.json"
   aws s3 cp --recursive "$A/saidas/logs/" "$H/"
   ```

5. **Copy the files** to `$EV/cobre/<row>/` on the workstation, as in P7.

### V1 single node

**Inputs:** `inputFile` = `<inputs-uri>/cobre-4ree.zip`, `coreCount` 100,
`maxCoresPerNode` 100, the queue default, `jobTimeoutHours` 1.

**Steps.** Capture and download as above, then run the checks with `K=1`:

```bash
D="$EV/cobre/V1"; K=1
L="$D/execution-<execution-id>.log"
jq -r '.executionStatus, .executionAnnotation' "$D/execution-<execution-id>.json"
grep -nF 'cobre-mpi: regular ELF executable; comm and solver are verified in the model job' "$L"
for k in training.zip policy.zip simulation.zip training/metadata.json \
         simulation/metadata.json run.json metadata.modelops; do
  grep -qxF "saidas/$k" "$D/saidas.txt" || echo "missing saidas/$k"
done
grep '^saidas/logs/' "$D/saidas.txt"
grep -cxF 'saidas/cortes.zip' "$D/saidas.txt"
jq -r '.model_name, .study_name' "$D/metadata.modelops"
S="$D/model.sbatch"
grep -nxF "#SBATCH --nodes=$K" "$S"
grep -nxF '#SBATCH --ntasks-per-node=1' "$S"
grep -nxF '#SBATCH --cpus-per-task=100' "$S"
grep -nxF '#SBATCH --exclusive' "$S"
grep -nxF '#SBATCH --partition=<queue-decomp>' "$S"
grep -nxF 'export LD_LIBRARY_PATH=<cobre-mpich-lib>${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' "$S"
grep -nxF 'export FI_PROVIDER=efa' "$S"
grep -nE '^srun --mpi=pmix --ntasks-per-node=1 .*/assets/cobre-mpi run .* --threads 100 --comm-backend mpi$' "$S"
grep -cE '/assets/cobre[^-]' "$S"
M="$D/model-<model-id>.out"
grep -nF "HPCMU_SHAPE nodes=$K " "$M"
grep -nE '^ *(Backend|Solver): ' "$M"
jq '.distribution' "$D/training-metadata.json"
jq -e --argjson k "$K" '.distribution
  | .backend == "mpi" and .world_size == $k and .mpi_library == "MPICH 4.2.3"
    and ($k == 1 or ((.hosts | length) == $k and all(.hosts[]; (.ranks | length) == 1)))' \
  "$D/training-metadata.json"
```

**Evidence:** the ExecutionId; the fetch-executables static-check line; the
status and annotation; the `saidas/` listing; `metadata.modelops` verbatim;
the captured script and the outputs of its checks, with line numbers; the
model log's `HPCMU_SHAPE`, `Backend:` and `Solver:` lines; the `distribution`
object.

**Pass criterion:**

- status `Success`, with the annotation ending `[cobre.completed]`. This also
  means that `cobre.mpi_not_started` and `cobre.rank_count_mismatch` did not
  fire;
- the fetch-executables log has
  `cobre-mpi: regular ELF executable; comm and solver are verified in the model job`;
- `saidas/` holds `training.zip`, `policy.zip`, `simulation.zip`,
  `training/metadata.json`, `simulation/metadata.json`, `run.json`,
  `metadata.modelops` and the `logs/` files (the loop prints nothing, and
  `logs/` lists the model and finalize logs), and no `cortes.zip` (the count
  prints `0`);
- `metadata.modelops` has `model_name` `COBRE` and `study_name` `4ree`;
- the captured script holds `#SBATCH --nodes=1`, `#SBATCH --ntasks-per-node=1`
  and `#SBATCH --cpus-per-task=100`. It exports
  `LD_LIBRARY_PATH=<cobre-mpich-lib>${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}` and
  `export FI_PROVIDER=efa`, both at line numbers lower than that of the
  `srun --mpi=pmix --ntasks-per-node=1 … <root-path>/cobre_<suffix>/assets/cobre-mpi run … --threads 100 --comm-backend mpi`
  line. `grep -cE '/assets/cobre[^-]'` prints `0`: no plain binary;
- the model log has `HPCMU_SHAPE nodes=1`, cobre-mpi's own
  `  Backend:   MPI (MPICH 4.2.3, …)` line, and its `  Solver:    <name> <version>`
  line;
- the metadata has `distribution.backend == "mpi"`, `world_size == 1` and
  `mpi_library == "MPICH 4.2.3"` (`jq -e` prints `true`). On one node cobre
  leaves `hosts` empty. The NEWAVE and DECOMP MPICH is 4.3.0 (R135), so
  `MPICH 4.2.3` shows that the `libmpi.so.12` of the cobre install was the one
  loaded.

### V2 multi-node (R135)

**Inputs:** as V1, with `coreCount` 400 and `jobTimeoutHours` 2. The run is
4 nodes × 100 threads, one rank per node.

**Steps.**

1. **Record the capacity** on the head node, just before submitting. The
   queue's nodes are dynamic and boot on demand:

   ```bash
   { date -u +%Y-%m-%dT%H:%M:%SZ; sinfo -p <queue-decomp> -o '%P %a %D %t'; } \
     | tee <head-scratch>/cobre/V2-sinfo.txt
   ```

2. **Capture and download** as above, with `<row>` = `V2`.
3. **Run the checks** of V1 with `D="$EV/cobre/V2"; K=4`.

**Evidence:** the `sinfo` capacity line with its time, and every V1 evidence
field.

**Pass criterion:** every V1 criterion for K = 4, that is:

- status `Success`, with the annotation ending `[cobre.completed]`, the
  fetch-executables static-check line, the same `saidas/` listing and the same
  `metadata.modelops` values;
- the captured script holds `#SBATCH --nodes=4`, `#SBATCH --ntasks-per-node=1`,
  `#SBATCH --cpus-per-task=100`, `#SBATCH --exclusive` and
  `#SBATCH --partition=<queue-decomp>`, the cobre MPICH `lib` export and
  `export FI_PROVIDER=efa` before the same `srun --mpi=pmix … --comm-backend mpi`
  line, and `grep -cE '/assets/cobre[^-]'` prints `0`;
- the model log has `HPCMU_SHAPE nodes=4`, the `  Backend:   MPI (MPICH 4.2.3, …)`
  line and the `  Solver:` line;
- the metadata has `distribution.backend == "mpi"`, `world_size == 4`,
  `mpi_library == "MPICH 4.2.3"`, and `hosts` with 4 entries of exactly one
  rank each (`jq -e` prints `true`).

With `FI_PROVIDER=efa`, a `ch4:ofi` MPICH either initializes on EFA or fails
`MPI_Init`, with no TCP fallback. A `Success` with 4 ranks on 4 hosts is
therefore the EFA criterion. V2b shows the provider directly.

**If V2 stays `PENDING`:** the follower fails a job held on a never-clearing
reason fast (ADR-046). Record the row `BLOCKED` with the
`squeue -j <model-id> -o '%i %T %r'` reason and the `sinfo` line, and retry in
a later window.

### V2b EFA provider log

A supplementary 2-node run on the head node, outside ModelOps, that logs the
libfabric provider each rank selected.

**Steps.** On the head node:

1. **Copy the binary and the case** to `<head-scratch>`:

   ```bash
   aws s3 cp "s3://<versoes-bucket>/versoes/cobre/v0.17.0/cobre-mpi" <head-scratch>/cobre-mpi
   chmod u+x <head-scratch>/cobre-mpi
   aws s3 cp <inputs-uri>/cobre-4ree.zip <head-scratch>/cobre-4ree.zip
   python3 -m zipfile -e <head-scratch>/cobre-4ree.zip <head-scratch>
   ```

2. **Run** the two ranks:

   ```bash
   srun -p <queue-decomp> -N 2 --ntasks-per-node=1 --exclusive -t 10 --mpi=pmix \
     env LD_LIBRARY_PATH=<cobre-mpich-lib> FI_PROVIDER=efa FI_LOG_LEVEL=info \
     <head-scratch>/cobre-mpi run <head-scratch>/4ree --threads 4 --comm-backend mpi > <head-scratch>/v2b.log 2>&1
   echo "exit=$?"
   ```

3. **Count the provider lines:**

   ```bash
   grep -c ':efa:' <head-scratch>/v2b.log
   grep -cE ':(tcp|sockets):' <head-scratch>/v2b.log
   ```

**Evidence:** the time, the exit line, both counts, and the first `:efa:`
line verbatim. Copy `v2b.log` to `$EV/cobre/V2b/`.

**Pass criterion:** the first count is positive, and the second prints `0`.

**Waiver:** V2b may be `WAIVED`. Record the waiver in the private amendments
record with its residual: the provider is then shown only indirectly, by
`export FI_PROVIDER=efa` in both captured scripts and the `Success` 4-rank,
4-host V2 on a `ch4:ofi` MPICH built against the EFA libfabric.

### V3 fast-fail relay

**Inputs:** as V1, with `inputFile` = `<inputs-uri>/cobre-4ree-broken.zip`.
1 node only.

**Steps.**

1. **Capture and download** as above, with `<row>` = `V3`: the P4 export, and
   the P7 log download.
2. **Read the outcome**, and keep only the `run` Task's stored output:

   ```bash
   D="$EV/cobre/V3"
   L="$D/execution-<execution-id>.log"
   M="$D/model-<model-id>.out"
   jq -r '.executionStatus, .executionAnnotation' "$D/execution-<execution-id>.json"
   jq '.diagnosis | {status, rule_id, reason}' "$D/run.json"
   grep -nE '^ *Backend: +MPI' "$M"
   grep -nF 'error:' "$M" | head -n 1
   grep -F '[ExecutionOutput] Executa e acompanha cobre no SLURM: ' "$L" \
     > "$D/relayed-<execution-id>.txt"
   ```

3. **Check the relay** (M14), from the repository root:

   ```bash
   uv run python -m deploy.modelops.check_relay \
     --relayed "$D/relayed-<execution-id>.txt" \
     --job-log "$D/model-<model-id>.out" \
     --job-log "$D/finalize-<finalize-id>.out"
   ```

**Evidence:** the ExecutionId; both job ids; the status and annotation; the
`run.json` diagnosis; the `Backend:` line and the first `error:` line, with
their line numbers; the `check_relay` output verbatim.

**Pass criterion:**

- status `DataError`;
- the annotation ends `[cobre.validation_error]` or `[cobre.io_error]`, not
  `[cobre.mpi_not_started]`: the model log shows `Backend:   MPI` at a line
  number lower than that of the load error;
- `check_relay` prints `relay: PASS` (exit 0).

### If a row fails

Record `FAIL` with the annotation, the diagnosis `rule_id` from `run.json`
(`jq '.diagnosis | {status, rule_id, reason, evidence}' "$D/run.json"`) and
the logs. Do not re-run V2 before diagnosing. A failure here reopens the
ticket that owns the failing part; no definition changes in this section.

- **A fetch-executables `DataError` from the static check** points at the
  uploaded file under `versoes/cobre/v0.17.0/` (the Upload Versão cobre
  option).
- **`cobre.mpi_not_started`** points at the cobre MPICH path, PMIx or EFA, on
  that run's shape. On V1 it rules out the multi-node wiring.
- **`cobre.rank_count_mismatch`** points at the PMIx wiring. That is ADR-057's
  rework trigger: escalate to the operator, and never switch the launcher
  silently.

## Upload a novomodelo version

From hpc-model-utils v2.3.1, cobre runs the `novomodelo-mpi` binary of the
novomodelo fork (`ons-ccee-epe/novomodelo`), and the model keeps the name
`cobre`. Every `modelVersion` the cobre workflow offers needs its own
`novomodelo-mpi` upload under `versoes/cobre/v<version>/`. A fetched folder
that holds only the pre-fork binary fails at fetch with a message that asks
for this re-upload. `<version>` is the release number without the `v`, for
example `0.18.1`.

**Before offering the version:** a novomodelo release that changes the `run`
arguments, the exit codes, the summary labels or the stopping-rule names needs
an hpc-model-utils change first (`docs/architecture.md`, "cobre runtime
facts"). Read that section, and compare the release's source with it, before
the upload.

The placeholders are the ones of the `## Upload Versão cobre option` section.
Adding the version to `modelVersion` and to the bridge's tag rule is described
in [docs/architecture.md](../architecture.md), not here.

### Build the novomodelo-mpi zip

On the head node or the workstation, in `<head-scratch>`.

1. **Download** `novomodelo-mpi-<version>-x86_64-unknown-linux-gnu.tar.gz` from
   `https://github.com/ons-ccee-epe/novomodelo/releases/tag/v<version>`,
   confirming the name on that page. No other asset is needed. The archive
   also holds the licence and notice files, and they are not delivered.
2. **Check the archive** holds the binary. The member list shows
   `./novomodelo-mpi`, and the command prints exactly `novomodelo-mpi`;
   `members.txt` is the member list, for the evidence:

   ```bash
   tar -tzf novomodelo-mpi-<version>-x86_64-unknown-linux-gnu.tar.gz \
     | tee members.txt | sed 's|^\./||' | grep -x novomodelo-mpi
   ```

3. **Record the archive hash and the binary hash,** and extract the binary:

   ```bash
   sha256sum novomodelo-mpi-<version>-x86_64-unknown-linux-gnu.tar.gz
   tar -xzOf novomodelo-mpi-<version>-x86_64-unknown-linux-gnu.tar.gz ./novomodelo-mpi | sha256sum
   tar -xzf novomodelo-mpi-<version>-x86_64-unknown-linux-gnu.tar.gz ./novomodelo-mpi
   sha256sum novomodelo-mpi
   ```

   The second and fourth commands print the same hash: the binary hash.
4. **Zip the binary alone** and check the zip:

   ```bash
   zip novomodelo-mpi-<version>.zip novomodelo-mpi
   python3 -c 'import sys, zipfile; print(*zipfile.ZipFile(sys.argv[1]).namelist(), sep="\n")' novomodelo-mpi-<version>.zip
   python3 -c 'import hashlib, sys, zipfile; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read("novomodelo-mpi")).hexdigest())' novomodelo-mpi-<version>.zip
   ```

   The listing prints exactly `novomodelo-mpi`, and the member hash equals the
   binary hash. Without `zip`, `python3 -m zipfile -c novomodelo-mpi-<version>.zip novomodelo-mpi`
   builds the same single-member zip. The zip does not keep the executable bit,
   and that does not matter: `fetch-executables` sets it on every fetched file.
5. **Copy the zip:**

   ```bash
   aws s3 cp novomodelo-mpi-<version>.zip <inputs-uri>/novomodelo-mpi-<version>.zip
   ```

### Run the novomodelo upload

Run Upload Versão with `modelName` `cobre`, `modelVersion` `v<version>` and
`executablesFile` `<inputs-uri>/novomodelo-mpi-<version>.zip`.

```bash
L="$EV/cobre/upload/execution-<execution-id>.log"
jq -r .executionStatus "$EV/cobre/upload/execution-<execution-id>.json"
grep -nF 'Using processor for model: cobre' "$L"
grep -nF 'Found 1 files in ' "$L"
grep -nF 'Will process: novomodelo-mpi -> novomodelo-mpi' "$L"
aws s3 ls "s3://<versoes-bucket>/versoes/cobre/v<version>/"
aws s3 cp "s3://<versoes-bucket>/versoes/cobre/v<version>/novomodelo-mpi" - | sha256sum
```

**Pass criterion:**

- status `Success`;
- the log has `Using processor for model: cobre`, `Found 1 files in ` (the
  extraction directory follows) and
  `Will process: novomodelo-mpi -> novomodelo-mpi`;
- `aws s3 ls s3://<versoes-bucket>/versoes/cobre/v<version>/`
  lists exactly one object, `novomodelo-mpi`;
- `aws s3 cp s3://<versoes-bucket>/versoes/cobre/v<version>/novomodelo-mpi - | sha256sum`
  equals the recorded binary hash.

**If the run fails:** follow the failure list of the `### Upload run` section
above, and read its `cobre-mpi` as `novomodelo-mpi`. Record the failure and do
not offer the version.
