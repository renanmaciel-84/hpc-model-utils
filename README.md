# hpc-model-utils

CLI tool for running energy planning models (NEWAVE, DECOMP, cobre) in HPC clusters with SLURM job scheduling and AWS S3 integration.

An external scheduler (ModelOps) drives each execution as a sequence of discrete commands: fetch executables and inputs from S3, preprocess the deck, submit and monitor the SLURM job, diagnose the outcome, postprocess, and upload the results back to S3. Each command reports its result to ModelOps through hook lines on standard output. For the layers, contracts and engine internals, see [docs/architecture.md](docs/architecture.md).

## Models

| Model  | Deck format                          | Launcher                                  | Notes |
| ------ | ------------------------------------ | ----------------------------------------- | ----- |
| NEWAVE | NEWAVE deck zip (`caso.dat`, `arquivos.dat`) | `mpiexec` (Hydra)                | Runs the sintetizador step and the `nwlistcf`/`nwlistop` postprocessing. |
| DECOMP | DECOMP deck zip (`dadger`)           | `mpiexec` (Hydra)                         | Runs the sintetizador step. A chained deck whose `FC NEWCUT` names the wrong NEWAVE stage fails in `preprocess` with `DATA_ERROR` (see [the notice](docs/notices/v2-fc-stage-mismatch.md)). |
| cobre  | Native cobre case zip, with or without a top-level folder | `novomodelo-mpi` through `srun --mpi=pmix` | `novomodelo-mpi` is the only supported binary (the novomodelo fork's MPI build). `run cobre` requires `--max-cores-per-node`. After a successful run with a simulation phase it builds the cobre-bridge dashboard, published as `sintese/dashboard.html`; outside ModelOps, `run cobre` needs `--synthesis-bin <path to cobre-bridge>`, or such a run ends `RUNTIME_ERROR` (`core.synthesis_missing`). |

cobre is not yet validated on the production cluster.

## Commands

Every command works in the current directory. Pass the model name (`cobre`, `decomp` or `newave`) as the first argument. Examples use the snake_case canonical names; the kebab-case spelling (for example `check-and-fetch-inputs`) works too.

### Workflow commands

In workflow order:

| Command | Purpose | Example |
| ------- | ------- | ------- |
| `check_and_fetch_executables` | Download the model binaries from S3 and check them. | `hpc-model-utils check_and_fetch_executables newave s3://<bucket>/<executables-prefix>/` |
| `check_and_fetch_inputs` | Download the input deck from S3. Options: `--parent-path TEXT` (parent execution for a chained run), `--delete`. | `hpc-model-utils check_and_fetch_inputs decomp s3://<bucket>/<inputs-prefix>/` |
| `extract_sanitize_inputs` | Unzip the deck and sanitize its encoding. | `hpc-model-utils extract_sanitize_inputs newave` |
| `preprocess` | Model-specific deck preparation. Option: `--execution-name TEXT`. | `hpc-model-utils preprocess newave --execution-name <name>` |
| `run` | Submit the job to SLURM and monitor it until completion. Takes `QUEUE` and `CORES`. Options: `--max-cores-per-node`, `--max-job-time-hours`, `--mpich-path`, `--slurm-path`, `--skip`, `--synthesis-bin`. | `hpc-model-utils run cobre <queue> 128 --max-cores-per-node 32` |
| `ingest_offline_run` | NEWAVE only. Ingest a run executed outside the cluster from three S3 object keys (inputs, outputs, cuts). `run` then skips the model job. | `hpc-model-utils ingest_offline_run newave s3://<bucket>/<inputs>.zip s3://<bucket>/<outputs>.zip s3://<bucket>/<cuts>.zip` |
| `result_upload` | Upload the results to S3. | `hpc-model-utils result_upload newave s3://<bucket>/<results-prefix>/` |
| `cancel_run` | Cancel the SLURM job of the run. Options: `--job-id TEXT`, `--slurm-path`. | `hpc-model-utils cancel_run newave` |

### Toolbox commands

| Command | Purpose | Example |
| ------- | ------- | ------- |
| `generate_execution_status` | Diagnose the run outcome and report one status token. Option: `--job-id TEXT`. | `hpc-model-utils generate_execution_status newave` |
| `postprocess` | Run the model-specific postprocessing (for example `nwlistcf`/`nwlistop` for NEWAVE). | `hpc-model-utils postprocess newave` |

### Hidden command

`finalize` is a hidden job-side command: the generated SLURM job script calls it to diagnose, postprocess and record the outcome when the job ends. It does not appear in `--help`, and operators do not call it.

## Run status

Every diagnosis yields one of nine status tokens. The ModelOps hook method follows the token:

| Status | ModelOps hook |
| ------ | ------------- |
| `SUCCESS` | `SetSuccess` |
| `INFEASIBLE` | `SetModelError` |
| `DATA_ERROR` | `SetDataError` |
| `RUNTIME_ERROR` | `SetRuntimeError` |
| `TIMEOUT` | `SetRuntimeError` |
| `INFRA_ERROR` | `SetRuntimeError` |
| `LICENSE_ERROR` | `SetRuntimeError` |
| `CANCELLED` | `SetRuntimeError` |
| `UNKNOWN` | `SetRuntimeError` |

The full v1-to-v2 comparison, with the checklist for consumers of these statuses, is in [docs/notices/v2-status-semantics.md](docs/notices/v2-status-semantics.md). The DECOMP stage-mismatch data error is in [docs/notices/v2-fc-stage-mismatch.md](docs/notices/v2-fc-stage-mismatch.md).

## Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | Success. |
| `2` | Usage error: an invalid argument or option. |
| `3` | Scheduler error: a SLURM command failed. |
| `4` | Storage error: an S3 operation failed. |
| `5` | Data error: the deck or the outputs are invalid. |
| `99` | Internal error. |

A command that a signal ends exits with 128 plus the signal number. An expected failure prints one line to standard error.

## Logs and artifacts

- `saidas/logs/<phase>-<jobid>.out`: the log of each SLURM job.
- `saidas/logs/synthesis.out`: the output of the synthesis step (sintetizador for NEWAVE and DECOMP, cobre-bridge for cobre).
- `sintese/dashboard.html`: the cobre results dashboard, one HTML file that loads plotly.js from `cdn.plot.ly` when opened, so viewing it needs internet access.
- `saidas/run.json`: the run record, with the status, the reason and the rule that produced it.

`run` relays the lines of the child processes to ModelOps verbatim. cobre publishes `saidas/training.zip`, `saidas/policy.zip` and `saidas/simulation.zip`, plus the raw `saidas/training/metadata.json` and `saidas/simulation/metadata.json`.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

On a cluster, the workflows install the tool through `deploy/modelops/scripts/ensure-tools.sh`. It installs each tool once, into `<root>/<tool>/<sha40>/`, pinned by release tag plus the 40-hex commit SHA, with `uv sync --frozen --no-dev`. Installed directories are immutable.

For a manual install, check out a release tag and sync the locked environment:

```bash
git clone https://github.com/rjmalves/hpc-model-utils.git
cd hpc-model-utils
git checkout v<X.Y.Z>
uv sync --frozen --no-dev
uv run hpc-model-utils --version
```

## Workflows as code

The ModelOps Task and Workflow definitions live in [deploy/modelops/](deploy/modelops/README.md), with placeholders for every environment-specific value. Apply them with:

```bash
# Read the live definitions into a directory.
uv run python -m deploy.modelops.apply snapshot --out <dir>

# Dry run: print what would change.
uv run python -m deploy.modelops.apply sync --env-file <env-file>

# Write the changes after a typed confirmation.
uv run python -m deploy.modelops.apply sync --env-file <env-file> --apply
```

Operator runbooks, with placeholders, are in [docs/runbooks/](docs/runbooks/).

## Development

```bash
uv sync --frozen --dev
uv run ruff check src tests deploy
uv run mypy --strict src/hpc_model_utils
uv run mypy --strict deploy
uv run pytest tests/ -m "not integration" --cov=hpc_model_utils --cov-branch --cov-report=xml
```

Integration tests use [LocalStack](https://localstack.cloud/) for S3, so they need no AWS credentials:

```bash
docker run -d \
  --name localstack \
  -p 4566:4566 \
  -e SERVICES=s3 \
  -e DEBUG=0 \
  localstack/localstack:3.8

AWS_ENDPOINT_URL=http://localhost:4566 \
AWS_ACCESS_KEY_ID=test \
AWS_SECRET_ACCESS_KEY=test \
AWS_DEFAULT_REGION=us-east-1 \
uv run pytest tests/integration/ -v -m integration --cov=hpc_model_utils --cov-branch --cov-report=xml --cov-append

docker rm -f localstack
```

## License

[MIT](LICENSE)
