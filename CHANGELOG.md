# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## The 2.x series

2.0.0 is the drop-in v2 engine, with the status-semantics change described in [`docs/notices/v2-status-semantics.md`](docs/notices/v2-status-semantics.md). 2.0.1 and 2.0.2 carry the versioned workflow definitions and the relay fixes that the NEWAVE and DECOMP workflows run. 2.1.0 adds cobre. 2.2.0 tightens the DECOMP FC stage check ([`docs/notices/v2-fc-stage-mismatch.md`](docs/notices/v2-fc-stage-mismatch.md)), cleans the logs and adds a cobre results dashboard. 2.2.1 fixes the reporting of S3 upload and read failures and the cancellation of a terminated run. 2.2.2 fixes the Upload NEWAVE ingest of large offline runs. 2.2.3 supports cobre v0.18.0. 2.3.0 adds job telemetry.

## [2.3.0] - 2026-10-16

### Added

- Job telemetry: with telemetry on, the model and finalize job scripts start a sampler on every node of the job through one `srun --overlap` sidecar line. The sampler's own output goes to logs under `.hpcmu/telemetry/`, never to the relayed job log.
- `result_upload` exports what the sampler recorded into three files of the run's artifacts: `saidas/telemetry/samples.parquet` (one row per sample of every node, restart and exported job), `saidas/telemetry/workspace.parquet` (one row per workspace walk, taken on node 0) and `saidas/telemetry/summary.json` (the statistics of each node, the headline values and the warnings).
- `result_upload` reports the headline values of `summary.json` to ModelOps as `telemetry.<scope>.<statistic>` SetMetadata hooks, after the last upload; the scopes are `model`, `finalize` and `workspace`.
- [`docs/telemetry.md`](docs/telemetry.md): the contract of the three telemetry files, with their formats, the `schema_version` policy, how to read them with duckdb, polars and pandas, and a reference generated from the schema.
- Three settings, read from the environment: `HPCMU_TELEMETRY` (`on` or `off`, default `on`), `HPCMU_TELEMETRY_INTERVAL` (the sample interval, 5 to 30 seconds, default 5) and `HPCMU_TELEMETRY_WORKSPACE_INTERVAL` (the workspace walk interval, 60 to 3600 seconds, default 60). `run` applies them when it writes the job scripts, and a value outside its range is a usage error.
- pyarrow is a runtime dependency. It adds about 165 MB unpacked to each per-SHA install, before ZFS compression, from a Linux x86_64 wheel of about 54 MB.
- `deploy/modelops/scripts/ensure-tools.sh` imports `pyarrow.parquet` through the install's own interpreter when it installs hpc-model-utils, so a broken pyarrow install fails at install time.
- `deploy/modelops/compare_job_logs.py` compares the job log of a telemetry-off run with the job log of a telemetry-on run of the same job, after masking their volatile tokens, in a `strict` and a `prd` mode. It needs only the standard library.

### Changed

- With telemetry on, the model and finalize job scripts gain one sidecar line; with telemetry off, they are byte-identical to 2.2.3.
- `result_upload` exports the telemetry before any upload and uploads its three files after `sintese/*` and before the `saidas/status.modelops` and `saidas/metadata.modelops` markers.
- A failed upload of a telemetry file is a storage error, as for any other artifact: `result_upload` exits 4.

Telemetry never changes a run's status, exit code, relayed job log or diagnosis, except for the exit 4 of a failed telemetry upload. A failed or skipped export only leaves out the telemetry files and their hooks.

## [2.2.3] - 2026-10-14

### Fixed

- A successful cobre v0.18.0 run no longer reports `RUNTIME_ERROR` (`cobre.metadata_unreadable`, `cobre_version is missing`). v0.18.0 names its release `software_version` in `training/metadata.json`, and the diagnosis now reads that key, or `cobre_version` from earlier releases.
- A cobre v0.18.0 run stopped by SIGTERM or SIGINT writes its training outputs and policy checkpoint, then exits 5. The diagnosis now reads those outputs and reports `CANCELLED` (`cobre.shutdown_cancelled`), or `TIMEOUT` (`cobre.shutdown_timeout`) when Slurm reports a timeout, instead of `RUNTIME_ERROR` (`cobre.unexpected_exit`, `cobre exited 5`). The outputs are archived as before, and ModelOps marks the run as a runtime error, as it does every cancelled run.

### Changed

- The cobre stop reasons read `training stopped on request after <n> iterations` instead of `training stopped by graceful_shutdown after <n> iterations`; the rule ids are unchanged.

## [2.2.2] - 2026-10-09

### Fixed

- `ingest_offline_run` (the Upload NEWAVE workflow) encoding-sanitizes only the deck's input files, never the extracted outputs. Reading multi-GB output binaries whole exhausted the head node's memory and killed the ingest before it recorded the run as offline, so the next step ran NEWAVE on the uploaded deck, which then failed for lack of its licence file.

## [2.2.1] - 2026-10-06

### Fixed

- An S3 upload rejected inside boto3's transfer manager, and an S3 object read that fails mid-stream, are now storage errors like any other S3 failure: a failed upload of a partly published run writes the `RUNTIME_ERROR` status and reports the artifacts path, and the command exits 4 instead of 99.
- `run` stopped by SIGTERM or SIGHUP, or by a broken stdout pipe, now cancels its Slurm jobs and exits 128+n even when `.hpcmu/state.json` cannot be read or holds a malformed job id; before, it left the jobs running and exited 99.
- `deploy/modelops/apply.py snapshot`: a `git` that is missing or times out while checking `--out` now ends with the one-line `apply:` error and exit 2, and the git calls of that check no longer receive `MODELOPS_TOKEN`.

## [2.2.0] - 2026-10-05

### Behavior changes

- A chained DECOMP run whose dadger `FC NEWCUT` names a `cortes-NNN.dat` other than the stage its horizon end needs now fails in `preprocess` with `DATA_ERROR` (`SetDataError`) before any job is submitted; 2.0.0 to 2.1.0 only appended the mismatch to the annotation. See [`docs/notices/v2-fc-stage-mismatch.md`](docs/notices/v2-fc-stage-mismatch.md).
- The synthesis step's failure messages name the tool neutrally for every model: a failed step reads `synthesis failed: synthesis tool exited <n>` instead of `synthesis failed: sintetizador exited <n>`, and a missing binary reads `synthesis tool binary not found; tried: …`, with `; pass --synthesis-bin` when the option was not given; the `synthesis failed:` prefix, the statuses, the rule ids and `synthesis_status` are unchanged.
- A successful cobre run with a simulation phase now runs the synthesis step, so a direct `run cobre` outside ModelOps needs `--synthesis-bin` pointing at cobre-bridge; without it the run ends `RUNTIME_ERROR` (`core.synthesis_missing`), as NEWAVE and DECOMP already do without a sintetizador.

### Changed

- Relayed job-log lines and the sintetizador, nwlistcf and nwlistop output are written verbatim, without the CLI's timestamp and logger prefix, and `LOGLEVEL` no longer filters them.
- The finalize job log keeps only sintetizador's WARNING-or-above lines and a one-line summary; the full output is in `saidas/logs/synthesis.out`.
- A typed error (`DataError`, `UsageError`, …) on a command's fatal path is logged as one `<command> failed: <category>: <message>` line instead of a traceback; unexpected exceptions keep the traceback.
- `deploy/modelops/check_relay.py` reads the verbatim relay and still reads the prefixed relay of 2.0.x and 2.1.0 runs.
- `ModelPlugin.synthesis_args` takes the workspace as its first argument (`synthesis_args(ws, cpus)`); NEWAVE and DECOMP behavior is unchanged.
- `deploy/modelops/apply.py` verifies the cobre workflow's synthesis pin against the cobre-bridge repository.

### Added

- `saidas/logs/synthesis.out`: sintetizador's complete output.
- `docs/architecture.md` and a README rewritten for v2.
- `docs/notices/v2-fc-stage-mismatch.md`.
- `deploy/modelops/workflows/cobre.json`: the `cobre` workflow, with its own `cobre-run` and `ensure-utils` Tasks.
- `deploy/modelops/tasks/`: the seven shared Tasks (`create-workdir`, `remove-workdir`, `fetch-executables`, `fetch-inputs`, `extract-sanitize`, `result-upload` and `cancel-run`) accept the model `cobre`.
- `deploy/modelops/workflows/upload-versao.json`: `Upload Versão` offers the model `cobre` and runs upload-versoes-cli 1.1.0.
- `docs/runbooks/cobre-rollout.md`: the runbook that applies the cobre definitions to ModelOps and validates cobre on the cluster.
- cobre builds `sintese/dashboard.html` with `cobre-bridge dashboard` after a successful run with a simulation phase; a dashboard failure keeps `SUCCESS` and is recorded in the annotation and in `synthesis_status`.
- `deploy/modelops/tasks/ensure-utils.sh` installs cobre-bridge 0.17.0 beside hpc-model-utils, and `deploy/modelops/tasks/cobre-run.sh` passes it to `run` as `--synthesis-bin`.

### Removed

- `setup.sh` and `docker-compose.localstack.yml`; install with `uv sync`, and start LocalStack as the CI does.

## [2.1.0] - 2026-10-05

NEWAVE and DECOMP behavior is unchanged from 2.0.2.

### Added

- The cobre model plugin: `hpc-model-utils <command> cobre` runs native cobre case zips (with or without a top-level folder) with the MPI build `cobre-mpi` only, on one or several nodes, through `srun --mpi=pmix` with one rank per node and `--comm-backend mpi`; `check_and_fetch_executables` checks `cobre-mpi` statically (a regular, owner-executable ELF file) and never runs it.
- `run cobre` requires `--max-cores-per-node`: threads per node T = max_cores_per_node and nodes K = cores / T, and cores must be a multiple of T.
- The cobre outcome mapping: a run whose model log never shows cobre-mpi's `Backend:   MPI` line is `RUNTIME_ERROR`; otherwise exits 1 and 2 are `DATA_ERROR` and exits 3 and 4 are `RUNTIME_ERROR`; a run that exits 0 is `SUCCESS` only when every enabled phase wrote its metadata, training completed, and every simulation scenario completed; never `INFEASIBLE`.
- cobre artifacts: `saidas/training.zip`, `saidas/policy.zip`, `saidas/simulation.zip` and the raw `saidas/training/metadata.json` and `saidas/simulation/metadata.json`; no sintetizador step.
- The platform-identifier set covers the `maxCoresPerNode` workflow parameter.
- `deploy/modelops/workflows/ranqueamento.json` and `deploy/modelops/workflows/upload-versao.json`: the `Ranqueamento Prospectivo` and `Upload Versão` workflows are now managed under `deploy/modelops/` and use the v2 shared Tasks.
- `docs/runbooks/v2-switch.md`: the operator procedure that switches the original workflows to the v2 definitions, with its rollback.
- `docs/runbooks/v2-sweep.md`: the operator procedure that moves the shared Tasks onto the v2 set.

### Changed

- `deploy/modelops/workflows/`: `NEWAVE - PEM`, `DECOMP - PEM` and `Upload NEWAVE` now use the v2 Tasks, and the `[v2]` workflow copies are removed.
- `docs/notices/v2-status-semantics.md`: names the workflows that switch to v2 (v2.0.2 or later) and adds a checklist for encadeador developers.

## [2.0.2] - 2026-10-04

### Fixed

- The job-log relay no longer loses the output of a job that ends before its log file becomes visible on the login node (NFS negative-lookup caching); a log that never appears is reported with an explicit `[hpcmu] log never appeared` marker.

### Added

- `HPCMU_MISSING_LOG_GRACE`: how long the relay keeps looking for a job log it never saw, after the job leaves the queue (default 90 s).
- `deploy/modelops/workflows/*-v2.json`: the `NEWAVE - PEM [v2]`, `DECOMP - PEM [v2]` and `Upload NEWAVE [v2]` workflow copies, for validating v2 beside the original workflows.
- `docs/runbooks/v2-pre-rollout.md`: the validation runbook for the `[v2]` copies, run before the original workflows switch to v2.

## [2.0.1] - 2026-10-03

### Changed

- The stdout neutralizer also defangs the `HPCMU_TOOL` phrase, so relayed text cannot rewrite the tool-path parameters of the versioned workflow definitions.
- The platform-identifier set covers the four parameters the versioned workflow definitions add (`utilsAppSha`, `synthesisAppSha`, `utilsToolDir`, `synthesisToolDir`).

### Added

- `deploy/modelops/`: templated ModelOps Task and Workflow definitions, with a structural publication lint.
- `deploy/modelops/scripts/ensure-tools.sh`: immutable per-commit tool installs on an exact-patch shared interpreter.
- A read-only definitions snapshot command (`python -m deploy.modelops.apply snapshot`).
- `deploy/modelops/apply.py`: `python -m deploy.modelops.apply sync` prints a read-only dry-run diff of the definitions against ModelOps.
- `deploy/modelops/apply.py`: `python -m deploy.modelops.apply sync --apply` is the only write path, and it writes only from a clean working tree after a typed confirmation.
- `deploy/modelops/check_relay.py`: a relay-completeness checker that compares a run's relayed output with its job logs and reports the first lost, extra or changed line.

## [2.0.0] - 2026-10-03

### Behavior changes

- DECOMP max iterations (no convergence) now calls `SetRuntimeError` instead of `SetModelError`.
- DECOMP negative optimality gap now calls `SetRuntimeError` instead of `SetModelError`.
- A crashed DECOMP run, or one missing `relato`, now calls `SetRuntimeError` instead of `SetModelError`.
- A NEWAVE full run whose final simulation lacks the simulated-series cost table now calls `SetRuntimeError` instead of `SetSuccess`.
- A NEWAVE run with no `pmo.dat` now calls `SetRuntimeError`, where v1 called `SetDataError`
  (or `SetSuccess` for a consistency run).
- A job ended by timeout, node failure, licence failure or cancellation now calls `SetRuntimeError` (`TIMEOUT`/`INFRA_ERROR`/`LICENSE_ERROR`/`CANCELLED`) instead of depending on incidental output parsing.
- A successful run whose sintetizador or postprocess step failed still calls `SetSuccess`, but the reason is now prefixed `synthesis failed:`/`postprocess failed:` and `synthesis_status=failed` is recorded.
- A successful run with a missing sintetizador now calls `SetRuntimeError` (`core.synthesis_missing`) instead of `SetSuccess`.
- See [`docs/notices/v2-status-semantics.md`](docs/notices/v2-status-semantics.md) for the full comparison and the operator delivery checklist.

### Added

- The v2 `hpc_model_utils` engine with typed run state (`.hpcmu/state.json`).
- Kebab-case command aliases.
- `saidas/run.json`.
- Per-phase job logs under `saidas/logs/`.
- `saidas/relgnl.<ext>` for DECOMP.
- `run --synthesis-bin`.
- `HPCMU_*` engine tunables.
- Signal-driven cancellation of submitted jobs.

### Changed

- The `hpc-model-utils` console script now runs `hpc_model_utils.cli:main`.
- A missing sintetizador on a successful run is an annotated `RUNTIME_ERROR`.
- `botocore` and `cfinterface` are now declared direct dependencies.
- Production workflows adopt 2.0.0 only after a pre-rollout validation of the new workflow copies; until then they keep their current pins.

### Removed

- DESSEM and GEVAZP.
- The `output_compression_and_cleanup`, `download_executed_run` and `fetch_extract_raw_outputs` commands.
- `assets/jobs/*`.
- The `idessem` and `pytz` dependencies.

## [1.1.2] - 2026-07-22

### Fixed

- Offline-ingested NEWAVE runs no longer execute the full model. `run` now reads the `execution_source = OFFLINE` flag recorded by `ingest_offline_run` and submits only the post job (status → postprocess → compression), so an offline run driven by the external scheduler skips model execution without requiring an explicit `--skip` flag. The `--skip` flag remains available as an override.

## [1.1.1] - 2026-07-21

### Added

- Added insurance to avoid resource scarcity while creating new threads in `newave_post.job` and `decomp_post.job`

## [1.1.0] - 2026-07-17

### Added

- `ingest_offline_run` command (NEWAVE): ingests a run executed offline (outside the cluster) from three explicit S3 object keys — the inputs, outputs and Benders-cuts archives. The archives may arrive under arbitrary names, so ingestion does not depend on archive names: it downloads all three, extracts them together into the working directory, standardizes filenames and encoding, rebuilds the raw input-deck echo (`eco_deck.zip`) from the deck contents, points the process manager at the executables directory, and records study metadata. This lets an offline run flow through the same downstream steps (`generate_execution_status` → `postprocess` → `output_compression_and_cleanup` → `result_upload`) via `run --skip`, as if it had executed on the cluster.
- Offline runs are marked with an `execution_source = OFFLINE` metadata flag and a ModelOps annotation so they remain distinguishable from cluster executions.

## [1.0.6] - 2026-07-02

### Changed

- Added `qprevs-medio-usina.csv` to the list of NEWAVE files to be processed into `relatorios.zip`

## [1.0.5] - 2026-06-09

### Fixed

- Collect `deco_*.msg` files as DECOMP report outputs (were previously excluded from the uploaded outputs)

### Changed

- Bump dependencies to their latest releases: `boto3` 1.43, `click` 8.4, `idecomp` 1.10, `idessem` 1.2, `inewave` 1.13, `boto3-stubs` 1.43, `mypy` 2.1, `pytest-cov` 7.1, `pytest-timeout` 2.4, `requests` 2.34, `ruff` 0.15 (plus refreshed transitive dependencies in `uv.lock`)

## [1.0.4] - 2026-05-27

### Fixed

- Cap CPU count for `SYNTHESIS_APP` calls (at physical cores via `lscpu`) and `output_compression_and_cleanup` (at vCPUs via `nproc`) in `newave_post.job` and `decomp_post.job` to avoid resource over-subscription on HPC hosts
- Add missing `--processadores` flag to `decomp_post.job` synthesis call so it explicitly limits parallelism

## [1.0.3] - 2026-04-17

### Fixed

- Add support to DECOMP license filename `decomp_trial.cep`

## [1.0.2] - 2026-03-20

### Fixed

- Model version extraction in `check_and_fetch_executables` across all models (NEWAVE, DECOMP, DESSEM, GEVAZP) — was registering empty values due to path ending in '/'

## [1.0.1] - 2026-03-13

### Fixed

- Model version extraction in `check_and_fetch_executables` across all models (NEWAVE, DECOMP, DESSEM, GEVAZP) — was registering model name instead of version due to wrong split index
- Unit tests for NEWAVE and DECOMP repositories updated to match v1.0.0 API signatures (S3 path-based interface)
- Removed obsolete test references to deleted `generate_unique_input_id` method and renamed constants
- Fixed test mocks for `run` (mocking `submit_job`/`follow_submitted_job` directly instead of low-level terminal calls)
- Marked `test_uploads_empty_file` integration test as `xfail` due to LocalStack 3.0 bug with empty PutObject

## [1.0.0] - 2026-03-11

### Added

- Input validation layer with fail-fast semantics for all CLI commands
- Structured error hierarchy (`CLIError`, `ValidationError`, `SlurmError`, `S3Error`, `ModelError`) with distinct exit codes
- SLURM job monitoring redesign with `sacct` fallback for fast-finishing jobs
- Command timing decorator for observability
- Custom Click parameter types (`ModelNameType`, `S3PathType`, `PositiveIntType`)
- Per-command input validators
- Comprehensive unit test suite (validation, error handling, SLURM monitoring, Click types)
- S3 integration tests with LocalStack
- CI/CD pipeline with GitHub Actions (unit tests, type checking, linting, integration tests)
- Professional packaging with `pyproject.toml` and `hatchling` build backend
- Installation script (`setup.sh`) with automatic PATH symlink

### Models

- **NEWAVE**: Full workflow support — input parsing (inewave), status diagnosis from `pmo.dat`, postprocessing with `nwlistcf`/`nwlistop`, parallel output compression
- **DECOMP**: Full workflow support — input parsing (idecomp), status diagnosis from `relato`/`inviab_unic`, parent NEWAVE metadata handling
- **DESSEM**: Full workflow support — input parsing (idessem), status diagnosis from `DES_LOG_RELATO`, core count injection
