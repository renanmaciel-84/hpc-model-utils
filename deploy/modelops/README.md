# deploy/modelops

Versioned ModelOps Task and Workflow definitions (JSON plus one script file
per Task) and the operator tooling that applies them. This repository is
public: every environment-specific value is a placeholder in the committed
files, and the real values live only in an uncommitted environment file
(ADR-031, ADR-049).

## Layout

| Path | Content |
| --- | --- |
| `workflows/<slug>.json` | One Workflow document: `newave-pem`, `decomp-pem`, `upload-newave`, `ranqueamento`, `upload-versao`, `cobre` |
| `tasks/<slug>.json` | One Task document without its `script` |
| `tasks/<slug>.sh` | The Task's script, verbatim, LF line endings |
| `env/prd.example.json` | The environment file shape, dummy values only |
| `scripts/` | Shell helpers, referenced from Task scripts by `@@script:FILE@@` |
| `variants/telemetry-off/<slug>.sh` | The telemetry off lever (ADR-066): `run.sh` and `cobre-run.sh`, each its `tasks/` script plus `export HPCMU_TELEMETRY=off`; never rendered, copied over `tasks/` by an operator commit (`docs/runbooks/v2-final-repin.md`, `## Telemetry off lever`) |
| `apply.py`, `modelops_api.py` | The operator CLI and its API client |

Audit fields (`_id`, `createdBy`, `createdDate`, `lastChangeBy`,
`lastChangeDate`, `scheduleStatus`) are never committed. The apply script sets
`createdBy` and `lastChangeBy` when it writes.

`cobre-run` and `ensure-utils` are cobre's own Tasks, and the seven shared
Tasks accept `cobre` (ticket-072b).

## Placeholder grammar

A placeholder is a whole JSON string value (or a token inside a `.sh` file):

| Token | NAME / SLUG / FILE must fullmatch | Replaced by |
| --- | --- | --- |
| `@@env:NAME@@` | `[A-Za-z][A-Za-z0-9]*` | `env.NAME` of the environment file |
| `@@task:SLUG@@` | `[a-z0-9][a-z0-9-]*` | the live `_id` of Task `SLUG` (`tasks.SLUG`) |
| `@@script:FILE@@` | `[a-z0-9][a-z0-9-]*\.sh`, a file in `scripts/` | the file's content; allowed only in `tasks/*.sh` |

ModelOps' own `{{...}}` and `${...}` stay untouched. `workflowTaskId` values
are client-chosen and internal to a document, so they stay literal.

## Environment file

The apply script takes the real file through its required `--env-file` option.
`env/prd.example.json` has the same shape with dummy values only:

```json
{
  "env": {"<NAME>": "<value>"},
  "workflows": {"<workflow-slug>": "<live workflow _id>"},
  "tasks": {"<task-slug>": "<live task _id>"},
  "protectedTaskIds": ["<live task _id>"]
}
```

Every prd Workflow is now a managed definition. `protectedTaskIds` lists the
live Task `_id`s that no definition may write: the apply refuses a change to a
listed Task. They are the legacy `v1-` documents of Ranqueamento Prospectivo
and Upload Versão, except `v1-ranking-run`. The seven ids the ticket-067
switch overwrote in place (`cancel-run`, `extract-sanitize`, `fetch-inputs`,
`ingest-offline`, `preprocess`, `result-upload`, `run`) left the list, because
those documents now hold the v2 content under their original names (R89 as
amended 2026-10-05).

Real files never enter this tree. `.gitignore` excludes
`deploy/modelops/env/*.json` except `*.example.json`, as defense in depth
should a real file be copied here. The structural lint
`tests/deploy/test_definitions_structure.py` reads no real value and gates
publication (ADR-049), together with the pre-push hook in `.githooks` and the
CI secret scan.

## The `v1-` baseline

The `v1-` prefix marks a legacy Task exported verbatim, exempt from the idiom
lint. Seven remain, the own Tasks of Ranqueamento Prospectivo
(`v1-clone-simulprospec`, `v1-clone-evalprospec`, `v1-clone-ranking-utils`,
`v1-ranking-fetch-inputs`, `v1-ranking-run`) and of Upload Versão
(`v1-clone-upload-cli`, `v1-upload-version`). Six are protected: the apply
refuses to write them.

`v1-ranking-run` is the exception. It is not protected, and it carries the
ADR-050 capture block for `utilsToolDir` and `synthesisToolDir`, followed by
the exports `HPCMU_UTILS_APP` and `HPCMU_SYNTHESIS_APP`, ahead of its three
legacy lines. Those lines stay byte for byte, with no `set -e`, so
`result_upload` still runs after a failed `run`.

Both workflows use the v2 `create-workdir`, `remove-workdir` and
`fetch-executables`, and Ranqueamento Prospectivo uses `ensure-tools` instead
of its two clone Tasks.

## Operator entry point

```sh
uv run python -m deploy.modelops.apply --help
```
