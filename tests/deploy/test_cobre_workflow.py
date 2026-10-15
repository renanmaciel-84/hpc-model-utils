"""ticket-073 contract for the cobre workflow and its two own Tasks (ADR-056).

The ``check_*`` functions take the ``deploy/modelops`` directory so a mutated
copy in ``tmp_path`` is checked the same way as the real tree. The behavior
tests render ``cobre-run`` and ``ensure-utils`` the way ModelOps does and run
them under real bash against a stub CLI, reusing the idiom suite's sandboxes.
Ticket-074 adds the Upload Versao check: the ``cobre`` option and the CLI pin.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from deploy.modelops.render import (
    EnvFile,
    load_env,
    render_task,
    render_workflow,
)
from tests.deploy import test_task_script_idiom as idiom
from tests.deploy.test_task_script_idiom import (
    _BASH_STUB,
    CAPTURED,
    ENSURE_PYTHON,
    ENSURE_SYNTHESIS_SHA,
    ENSURE_SYNTHESIS_TAG,
    ENSURE_UTILS_SHA,
    ENSURE_UTILS_TAG,
    Call,
    EnsureSandbox,
    Layout,
    _run_ensure,
)
from tests.deploy.test_v2_workflows import (
    SHA,
    TAG,
    UUID4,
    _load,
    _params,
    _task_slug,
    _walk,
    require,
)
from tests.support.script_harness import (
    EXECUTION_ID,
    EXECUTION_ID_PARAMETER,
    prepare_command,
)

# pytest registers a module attribute as a fixture under that attribute's name
layout = idiom.layout
ensure_sandbox = idiom.ensure_sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"
WORKFLOW = "workflows/cobre.json"
RUN_SCRIPT = "tasks/cobre-run.sh"
ENSURE_SCRIPT = "tasks/ensure-utils.sh"

CHAIN = (
    "ensure-utils",
    "create-workdir",
    "fetch-executables",
    "fetch-inputs",
    "extract-sanitize",
    "cobre-run",
    "result-upload",
    "remove-workdir",
)
CANCEL_CHAIN = ("cancel-run", "remove-workdir")
PARAMETER_NAMES = (
    "awsRegion",
    "outputsBucket",
    "mpichPath",
    "slurmPath",
    "rootPath",
    "inputFile",
    "parentPath",
    "coreCount",
    "maxCoresPerNode",
    "queue",
    "modelName",
    "modelVersion",
    "path",
    "jobId",
    "utilsToolDir",
    "synthesisToolDir",
    "utilsAppVersion",
    "utilsAppSha",
    "synthesisAppVersion",
    "synthesisAppSha",
    "versionsBucket",
    "jobTimeoutHours",
)
UTILS_TAG = "v2.2.2"
BRIDGE_TAG = "v0.17.0"
BRIDGE_SHA = "1b059ac505a8d57b6549c0a60411c41a9d7167cb"
UPLOAD_WORKFLOW = "workflows/upload-versao.json"
UPLOAD_MODEL_OPTIONS = ["newave", "decomp", "cobre"]
UPLOAD_CLI_TAG = "v1.1.0"

# name -> (type, defaultValue, visible, regexPattern, format, options)
SHAPES: dict[str, tuple[str, str, bool, str, str, list[str]]] = {
    "awsRegion": ("String", "@@env:awsRegion@@", False, "", "", []),
    "outputsBucket": ("String", "@@env:outputsBucket@@", False, "", "", []),
    "mpichPath": ("String", "@@env:cobreMpichPath@@", False, "", "", []),
    "slurmPath": ("String", "@@env:slurmPath@@", False, "", "", []),
    "rootPath": ("String", "@@env:rootPath@@", False, "", "", []),
    "inputFile": ("File", "", True, "", "", []),
    "parentPath": ("String", "''", False, "", "", []),
    "coreCount": ("Int", "400", True, "", "", []),
    "maxCoresPerNode": ("Int", "100", True, "", "", []),
    "queue": (
        "List",
        "@@env:queueCobre@@",
        True,
        "",
        "",
        ["@@env:queueNewave@@", "@@env:queueDecomp@@", "@@env:queueCobre@@"],
    ),
    "modelName": ("String", "cobre", False, "", "", []),
    "modelVersion": ("List", "v0.17.0", True, "", "", ["v0.17.0"]),
    "path": (
        "String",
        "",
        False,
        r"(?<=Created temporary dir\s)(/[^ ]+)",
        "{0}",
        [],
    ),
    "jobId": ("String", "", False, r"Submitted batch job (\d+)", "{1}", []),
    "utilsToolDir": (
        "String",
        "",
        False,
        r"HPCMU_TOOL hpc-model-utils (/\S+)",
        "{1}",
        [],
    ),
    "synthesisToolDir": (
        "String",
        "",
        False,
        r"HPCMU_TOOL cobre-bridge (/\S+)",
        "{1}",
        [],
    ),
    "versionsBucket": ("String", "@@env:versionsBucket@@", False, "", "", []),
    "jobTimeoutHours": ("Int", "8", True, "", "", []),
}
PIN_SHAPE: tuple[str, bool, str, str, list[str]] = (
    "String",
    False,
    "",
    "",
    [],
)

RUN_EXEC = (
    'exec "$UTILS_DIR/.venv/bin/hpc-model-utils" run "$MODEL" "$QUEUE"'
    ' "$CORES" --max-cores-per-node "$MAX_CORES"'
    ' --max-job-time-hours "$JOB_HOURS" --mpich-path "$MPICH_PATH"'
    ' --slurm-path "$SLURM_PATH"'
    ' --synthesis-bin "$SYNTHESIS_DIR/.venv/bin/cobre-bridge"'
)
ENSURE_EXEC = (
    "exec bash -s -- --root '@@env:toolsRoot@@' --uv '@@env:uvBin@@'"
    " --python '3.12.13' --tool hpc-model-utils"
    " https://github.com/rjmalves/hpc-model-utils.git"
    ' "$UTILS_TAG" "$UTILS_SHA" hpc-model-utils'
    " --tool cobre-bridge https://github.com/cobre-rs/cobre-bridge.git"
    ' "$SYNTHESIS_TAG" "$SYNTHESIS_SHA" cobre-bridge'
    " <<'HPCMU_ENSURE_TOOLS_EOF'"
)


def _all_nodes(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return list(doc["workflowTasks"]) + list(doc["cancelationTasks"])


def _metadata_ids(doc: dict[str, Any], field: str) -> list[str]:
    return sorted(m["workflowTaskId"] for m in doc[field])


def check_chain(root: Path) -> list[str]:
    errors: list[str] = []
    doc = _load(root / WORKFLOW)
    nodes = {n["workflowTaskId"]: n for n in doc["workflowTasks"]}
    cancel_nodes = {n["workflowTaskId"]: n for n in doc["cancelationTasks"]}
    if len(nodes) != len(doc["workflowTasks"]):
        errors.append(f"{WORKFLOW}: duplicate workflowTaskId")
    if set(nodes) & set(cancel_nodes):
        errors.append(f"{WORKFLOW}: a workflowTaskId is shared with the cancel")
    for node in _all_nodes(doc):
        task = _task_slug(node)
        if task is None:
            errors.append(f"{WORKFLOW}: taskId is not @@task:@@")
        elif task.startswith("v1-"):
            errors.append(f"{WORKFLOW}: taskId names the v1 Task {task}")
        if not UUID4.fullmatch(node["workflowTaskId"]):
            errors.append(
                f"{WORKFLOW}: workflowTaskId is not a lowercase uuid4"
            )
        if node.get("alias") is not None or node.get("executionCriteria") != "":
            errors.append(
                f"{WORKFLOW}: alias or executionCriteria is not empty"
            )
    errors += [
        f"{WORKFLOW}: onFailure is not 1"
        for node in doc["workflowTasks"]
        if node.get("onFailure") != 1
    ]
    edges = (
        doc["taskExecutionCriterias"]
        + doc["cancellationTaskExecutionCriterias"]
    )
    errors += [
        f"{WORKFLOW}: criteria edge expression is not true"
        for edge in edges
        if edge["criteriaExpression"] != "true"
    ]
    walked, ended = _walk(doc)
    sequence = tuple(
        _task_slug(nodes[i]) if i in nodes else None for i in walked
    )
    if sequence != CHAIN or not ended:
        errors.append(
            f"{WORKFLOW}: the criteria chain from start differs from the"
            " expected sequence"
        )
    if sorted(walked) != sorted(nodes):
        errors.append(f"{WORKFLOW}: workflowTasks holds nodes off the chain")
    cancel_walked, cancel_ended = _walk(
        {"taskExecutionCriterias": doc["cancellationTaskExecutionCriterias"]}
    )
    cancel_sequence = tuple(
        _task_slug(cancel_nodes[i]) if i in cancel_nodes else None
        for i in cancel_walked
    )
    if cancel_sequence != CANCEL_CHAIN or not cancel_ended:
        errors.append(
            f"{WORKFLOW}: the cancel chain from start differs from the"
            " expected sequence"
        )
    if tuple(_task_slug(n) for n in doc["cancelationTasks"]) != CANCEL_CHAIN:
        errors.append(f"{WORKFLOW}: cancelationTasks differ from the pair")
    if _metadata_ids(doc, "workflowTaskMetadata") != sorted(
        [*nodes, "startNode", "endNode"]
    ):
        errors.append(
            f"{WORKFLOW}: workflowTaskMetadata differs from the nodes"
        )
    if _metadata_ids(doc, "cancellationTaskMetadata") != sorted(
        [*cancel_nodes, "startNode", "endNode"]
    ):
        errors.append(
            f"{WORKFLOW}: cancellationTaskMetadata differs from the cancel nodes"
        )
    return errors


def check_parameters(root: Path) -> list[str]:
    errors: list[str] = []
    text = (root / WORKFLOW).read_text(encoding="utf-8")
    doc = _load(root / WORKFLOW)
    parameters = doc["parameters"]
    if tuple(p["name"] for p in parameters) != PARAMETER_NAMES:
        errors.append(f"{WORKFLOW}: parameter names differ")
    if [p["order"] for p in parameters] != list(range(len(parameters))):
        errors.append(f"{WORKFLOW}: parameter order is not contiguous")
    params = _params(doc)
    for name, expected in SHAPES.items():
        param = params.get(name, {})
        shape = (
            param.get("type"),
            param.get("defaultValue"),
            param.get("visible"),
            param.get("regexPattern"),
            param.get("format"),
            param.get("options"),
        )
        if shape != expected:
            errors.append(f"{WORKFLOW}: parameter {name} has the wrong shape")
    if "@@env:mpichPath@@" in text:
        errors.append(f"{WORKFLOW}: references the NEWAVE/DECOMP mpichPath")
    return errors


def check_pins(root: Path) -> list[str]:
    errors: list[str] = []
    params = _params(_load(root / WORKFLOW))
    version = str(params.get("utilsAppVersion", {}).get("defaultValue", ""))
    sha = str(params.get("utilsAppSha", {}).get("defaultValue", ""))
    if version != UTILS_TAG or not TAG.fullmatch(version):
        errors.append(f"{WORKFLOW}: utilsAppVersion is not {UTILS_TAG}")
    if not SHA.fullmatch(sha):
        errors.append(f"{WORKFLOW}: utilsAppSha is not 40 hex")
    if params.get("synthesisAppVersion", {}).get("defaultValue") != BRIDGE_TAG:
        errors.append(f"{WORKFLOW}: synthesisAppVersion is not {BRIDGE_TAG}")
    if params.get("synthesisAppSha", {}).get("defaultValue") != BRIDGE_SHA:
        errors.append(f"{WORKFLOW}: synthesisAppSha is not {BRIDGE_SHA}")
    for name in (
        "utilsAppVersion",
        "utilsAppSha",
        "synthesisAppVersion",
        "synthesisAppSha",
    ):
        param = params.get(name, {})
        shape = (
            param.get("type"),
            param.get("visible"),
            param.get("regexPattern"),
            param.get("format"),
            param.get("options"),
        )
        if shape != PIN_SHAPE:
            errors.append(f"{WORKFLOW}: parameter {name} has the wrong shape")
    return errors


def check_cobre_run(root: Path) -> list[str]:
    errors: list[str] = []
    lines = (root / RUN_SCRIPT).read_text(encoding="utf-8").splitlines()
    if lines[-1] != RUN_EXEC:
        errors.append(f"{RUN_SCRIPT}: the final line is not the cobre command")
    text = "\n".join(lines)
    if "sintetizador" in text:
        errors.append(f"{RUN_SCRIPT}: the script references a sintetizador")
    if "@@env:mpichPath@@" in text:
        errors.append(f"{RUN_SCRIPT}: references the NEWAVE/DECOMP mpichPath")
    if "2>" in text:
        errors.append(f"{RUN_SCRIPT}: redirects stderr")
    if "[[ \"$MODEL\" =~ ^cobre$ ]] || fail 'invalid modelName'" not in lines:
        errors.append(f"{RUN_SCRIPT}: modelName is not restricted to cobre")
    return errors


def check_ensure_utils(root: Path) -> list[str]:
    errors: list[str] = []
    lines = (root / ENSURE_SCRIPT).read_text(encoding="utf-8").splitlines()
    if lines[-3:] != [
        ENSURE_EXEC,
        "@@script:ensure-tools.sh@@",
        "HPCMU_ENSURE_TOOLS_EOF",
    ]:
        errors.append(f"{ENSURE_SCRIPT}: the final three lines differ")
    text = "\n".join(lines)
    if "{{modelName}}" in text:
        errors.append(f"{ENSURE_SCRIPT}: the script is not model-agnostic")
    if "sintetizador" in text:
        errors.append(f"{ENSURE_SCRIPT}: the script installs a sintetizador")
    return errors


def check_upload_versao(root: Path) -> list[str]:
    errors: list[str] = []
    params = _params(_load(root / UPLOAD_WORKFLOW))
    model = params.get("modelName", {})
    if model.get("options") != UPLOAD_MODEL_OPTIONS:
        errors.append(
            f"{UPLOAD_WORKFLOW}: modelName options are not"
            f" {UPLOAD_MODEL_OPTIONS}"
        )
    if model.get("defaultValue") != "newave":
        errors.append(f"{UPLOAD_WORKFLOW}: modelName default is not newave")
    cli = params.get("uploadCliVersion", {})
    if cli.get("defaultValue") != UPLOAD_CLI_TAG:
        errors.append(
            f"{UPLOAD_WORKFLOW}: uploadCliVersion is not {UPLOAD_CLI_TAG}"
        )
    return errors


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("workflows", "tasks"):
        shutil.copytree(MODELOPS / name, tmp_path / name)
    return tmp_path


def _rewrite(
    root: Path,
    change: Callable[[dict[str, Any]], None],
    workflow: str = WORKFLOW,
) -> None:
    path = root / workflow
    doc = _load(path)
    change(doc)
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _node_id(doc: dict[str, Any], task: str) -> str:
    (found,) = (
        n["workflowTaskId"]
        for n in doc["workflowTasks"]
        if n["taskId"] == f"@@task:{task}@@"
    )
    return str(found)


def _parameter(doc: dict[str, Any], name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [
        p for p in doc["parameters"] if p["name"] == name
    ]
    (found,) = matches
    return found


def _drop_node(doc: dict[str, Any]) -> None:
    gone = _node_id(doc, "fetch-inputs")
    doc["workflowTasks"] = [
        n for n in doc["workflowTasks"] if n["workflowTaskId"] != gone
    ]


def _use_v1_task(doc: dict[str, Any]) -> None:
    doc["workflowTasks"][1]["taskId"] = "@@task:v1-list-files@@"


def _extra_edge(doc: dict[str, Any]) -> None:
    doc["taskExecutionCriterias"].append(
        {
            "criteriaExpression": "true",
            "sourceWorkflowTaskId": _node_id(doc, "cobre-run"),
            "targetWorkflowTaskId": None,
        }
    )


def _null_on_failure(doc: dict[str, Any]) -> None:
    doc["workflowTasks"][0]["onFailure"] = None


def _non_uuid_id(doc: dict[str, Any]) -> None:
    old = doc["workflowTasks"][0]["workflowTaskId"]
    doc["workflowTasks"][0]["workflowTaskId"] = old.upper()
    for edge in doc["taskExecutionCriterias"]:
        for key in ("sourceWorkflowTaskId", "targetWorkflowTaskId"):
            if edge[key] == old:
                edge[key] = old.upper()


def _swap_cancel_pair(doc: dict[str, Any]) -> None:
    doc["cancelationTasks"].reverse()


def _cut_cancel_chain(doc: dict[str, Any]) -> None:
    doc["cancellationTaskExecutionCriterias"][1]["targetWorkflowTaskId"] = None


def _drop_metadata(doc: dict[str, Any]) -> None:
    doc["workflowTaskMetadata"].pop(1)


def _share_cancel_id(doc: dict[str, Any]) -> None:
    doc["cancelationTasks"][0]["workflowTaskId"] = doc["workflowTasks"][0][
        "workflowTaskId"
    ]


def _max_cores_as_string(doc: dict[str, Any]) -> None:
    _parameter(doc, "maxCoresPerNode")["type"] = "String"


def _newave_mpich_default(doc: dict[str, Any]) -> None:
    _parameter(doc, "mpichPath")["defaultValue"] = "@@env:mpichPath@@"


def _drop_synthesis_tool_dir(doc: dict[str, Any]) -> None:
    doc["parameters"] = [
        p for p in doc["parameters"] if p["name"] != "synthesisToolDir"
    ]


def _gap_in_order(doc: dict[str, Any]) -> None:
    doc["parameters"][-1]["order"] += 1


def _wrong_model_version_options(doc: dict[str, Any]) -> None:
    _parameter(doc, "modelVersion")["options"] = ["v0.17.0", "v0.16.0"]


def _wrong_path_regex(doc: dict[str, Any]) -> None:
    _parameter(doc, "path")["regexPattern"] = ""


def _branch_tag(doc: dict[str, Any]) -> None:
    _parameter(doc, "utilsAppVersion")["defaultValue"] = "main"


def _short_sha(doc: dict[str, Any]) -> None:
    _parameter(doc, "utilsAppSha")["defaultValue"] = "abc"


def _visible_sha(doc: dict[str, Any]) -> None:
    _parameter(doc, "utilsAppSha")["visible"] = True


def _branch_bridge_tag(doc: dict[str, Any]) -> None:
    _parameter(doc, "synthesisAppVersion")["defaultValue"] = "main"


def _short_bridge_sha(doc: dict[str, Any]) -> None:
    _parameter(doc, "synthesisAppSha")["defaultValue"] = BRIDGE_SHA[:39]


def test_check_chain_real_tree_reports_nothing() -> None:
    require(check_chain(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_drop_node, r"chain from start differs"),
        (_use_v1_task, r"names the v1 Task v1-list-files"),
        (_use_v1_task, r"chain from start differs"),
        (_extra_edge, r"chain from start differs"),
        (_null_on_failure, r"onFailure is not 1"),
        (_non_uuid_id, r"workflowTaskId is not a lowercase uuid4"),
        (_swap_cancel_pair, r"cancelationTasks differ from the pair"),
        (_cut_cancel_chain, r"cancel chain from start differs"),
        (_drop_metadata, r"workflowTaskMetadata differs from the nodes"),
        (_share_cancel_id, r"a workflowTaskId is shared with the cancel"),
    ],
)
def test_check_chain_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _rewrite(tree, mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_chain(tree))


def test_cobre_workflow_chain_renders_with_the_example_env_ids() -> None:
    env = load_env(MODELOPS / "env" / "prd.example.json")

    rendered = render_workflow("cobre", MODELOPS, env, {})

    assert rendered["workflowName"] == "cobre"
    assert sorted(n["taskId"] for n in rendered["workflowTasks"]) == sorted(
        env.tasks[task] for task in CHAIN
    )
    assert [n["taskId"] for n in rendered["cancelationTasks"]] == [
        env.tasks[task] for task in CANCEL_CHAIN
    ]


def test_check_parameters_real_tree_reports_nothing() -> None:
    require(check_parameters(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            _max_cores_as_string,
            r"parameter maxCoresPerNode has the wrong shape",
        ),
        (_newave_mpich_default, r"parameter mpichPath has the wrong shape"),
        (_newave_mpich_default, r"references the NEWAVE/DECOMP mpichPath"),
        (_drop_synthesis_tool_dir, r"parameter names differ"),
        (_gap_in_order, r"parameter order is not contiguous"),
        (
            _wrong_model_version_options,
            r"parameter modelVersion has the wrong shape",
        ),
        (_wrong_path_regex, r"parameter path has the wrong shape"),
    ],
)
def test_check_parameters_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _rewrite(tree, mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_parameters(tree))


def test_check_pins_real_tree_reports_nothing() -> None:
    require(check_pins(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_branch_tag, r"utilsAppVersion is not v"),
        (_short_sha, r"utilsAppSha is not 40 hex"),
        (_visible_sha, r"parameter utilsAppSha has the wrong shape"),
        (_branch_bridge_tag, r"synthesisAppVersion is not v0\.17\.0"),
        (_short_bridge_sha, rf"synthesisAppSha is not {BRIDGE_SHA}"),
    ],
)
def test_check_pins_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _rewrite(tree, mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_pins(tree))


def test_check_cobre_run_real_tree_reports_nothing() -> None:
    require(check_cobre_run(MODELOPS))


def _edit_script(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            '/.venv/bin/cobre-bridge"',
            '/.venv/bin/sintetizador-cobre"',
            r"the final line is not the cobre command",
        ),
        (
            'cd -- "$WORKDIR"',
            '# sintetizador\ncd -- "$WORKDIR"',
            r"references a sintetizador",
        ),
        (
            "@@env:cobreMpichPath@@",
            "@@env:mpichPath@@",
            r"references the NEWAVE/DECOMP mpichPath",
        ),
        (
            '--slurm-path "$SLURM_PATH"',
            '--slurm-path "$SLURM_PATH" 2>&1',
            r"redirects stderr",
        ),
        (
            "=~ ^cobre$ ]]",
            "=~ ^(newave|cobre)$ ]]",
            r"modelName is not restricted to cobre",
        ),
    ],
)
def test_check_cobre_run_mutated_copy_is_reported(
    tree: Path, old: str, new: str, message: str
) -> None:
    _edit_script(tree, RUN_SCRIPT, old, new)

    with pytest.raises(AssertionError, match=message):
        require(check_cobre_run(tree))


def test_check_ensure_utils_real_tree_reports_nothing() -> None:
    require(check_ensure_utils(MODELOPS))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "cobre-bridge <<",
            "cobre-bridge --tool sintetizador-newave <<",
            r"the final three lines differ",
        ),
        (
            "IFS= read -r -d '' UTILS_TAG",
            "{{modelName}}\nIFS= read -r -d '' UTILS_TAG",
            r"the script is not model-agnostic",
        ),
        (
            "@@script:ensure-tools.sh@@",
            "@@script:ensure-tools.sh@@\n# sintetizador",
            r"the script installs a sintetizador",
        ),
    ],
)
def test_check_ensure_utils_mutated_copy_is_reported(
    tree: Path, old: str, new: str, message: str
) -> None:
    _edit_script(tree, ENSURE_SCRIPT, old, new)

    with pytest.raises(AssertionError, match=message):
        require(check_ensure_utils(tree))


def _drop_cobre_option(doc: dict[str, Any]) -> None:
    _parameter(doc, "modelName")["options"].remove("cobre")


def _cobre_default(doc: dict[str, Any]) -> None:
    _parameter(doc, "modelName")["defaultValue"] = "cobre"


def _old_upload_cli(doc: dict[str, Any]) -> None:
    _parameter(doc, "uploadCliVersion")["defaultValue"] = "v1.0.0"


def test_upload_versao_offers_cobre_with_cli_v1_1_0() -> None:
    require(check_upload_versao(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_drop_cobre_option, r"modelName options are not"),
        (_cobre_default, r"modelName default is not newave"),
        (_old_upload_cli, r"uploadCliVersion is not v1\.1\.0"),
    ],
)
def test_check_upload_versao_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _rewrite(tree, mutation, UPLOAD_WORKFLOW)

    with pytest.raises(AssertionError, match=message):
        require(check_upload_versao(tree))


def _cobre_values(sandbox: Layout) -> dict[str, str]:
    return {
        EXECUTION_ID_PARAMETER: EXECUTION_ID,
        "modelName": "cobre",
        "rootPath": sandbox.env["rootPath"],
        "path": str(sandbox.workdir("cobre")),
        "queue": sandbox.env["queueDecomp"],
        "coreCount": "400",
        "maxCoresPerNode": "100",
        "jobTimeoutHours": "8",
        "mpichPath": sandbox.env["cobreMpichPath"],
        "slurmPath": sandbox.env["slurmPath"],
        "utilsToolDir": str(sandbox.tool_dir("hpc-model-utils")),
        "synthesisToolDir": str(sandbox.tool_dir("cobre-bridge")),
    }


def _run_cobre_run(
    sandbox: Layout, overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    sandbox.workdir("cobre").mkdir(exist_ok=True)
    env = EnvFile(
        env=sandbox.env,
        workflows={},
        tasks={},
        protected_task_ids=frozenset(),
    )
    script = prepare_command(
        render_task("cobre-run", MODELOPS, env)["script"],
        {**_cobre_values(sandbox), **(overrides or {})},
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=sandbox.base,
        env={"PATH": os.environ["PATH"], "LC_ALL": "C"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _assert_cobre_run_rejected(
    sandbox: Layout,
    result: subprocess.CompletedProcess[str],
    parameter: str,
) -> None:
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"hpcmu-task: invalid {parameter}\n"
    assert sandbox.calls == []
    assert not sandbox.sentinel.exists()


def test_cobre_run_valid_values_exec_the_stub_once(layout: Layout) -> None:
    result = _run_cobre_run(layout)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert layout.calls == [
        Call(
            str(layout.workdir("cobre")),
            "unset",
            [
                "run",
                "cobre",
                layout.env["queueDecomp"],
                "400",
                "--max-cores-per-node",
                "100",
                "--max-job-time-hours",
                "8",
                "--mpich-path",
                layout.env["cobreMpichPath"],
                "--slurm-path",
                layout.env["slurmPath"],
                "--synthesis-bin",
                f"{layout.tool_dir('cobre-bridge')}/.venv/bin/cobre-bridge",
            ],
        )
    ]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("modelName", "newave"),
        ("maxCoresPerNode", "0"),
        ("maxCoresPerNode", "10000"),
        ("maxCoresPerNode", "4;id"),
    ],
)
def test_cobre_run_invalid_value_is_rejected(
    layout: Layout, parameter: str, value: str
) -> None:
    result = _run_cobre_run(layout, {parameter: value})

    _assert_cobre_run_rejected(layout, result, parameter)


_SYNTHESIS_DIR_CASES: dict[str, Callable[[Layout], str]] = {
    "other-model": lambda sb: str(sb.tool_dir("sintetizador-decomp")),
    "outside-tools-root": lambda sb: str(sb.tool_dir("cobre-bridge")).replace(
        "/tools/", "/x/"
    ),
    "short-sha": lambda sb: str(sb.tool_dir("cobre-bridge"))[:-1],
}


@pytest.mark.parametrize("case", _SYNTHESIS_DIR_CASES)
def test_cobre_run_synthesis_tool_dir_outside_the_contract_is_rejected(
    layout: Layout, case: str
) -> None:
    value = _SYNTHESIS_DIR_CASES[case](layout)

    result = _run_cobre_run(layout, {"synthesisToolDir": value})

    _assert_cobre_run_rejected(layout, result, "synthesisToolDir")


def test_cobre_run_newave_decomp_mpich_path_is_rejected(layout: Layout) -> None:
    assert layout.env["mpichPath"] != layout.env["cobreMpichPath"]

    result = _run_cobre_run(layout, {"mpichPath": layout.env["mpichPath"]})

    _assert_cobre_run_rejected(layout, result, "mpichPath")


@pytest.mark.parametrize("parameter", CAPTURED["cobre-run"])
def test_cobre_run_every_captured_value_rejects_an_injected_line(
    layout: Layout, parameter: str
) -> None:
    result = _run_cobre_run(layout, {parameter: f"x\ntouch {layout.sentinel}"})

    _assert_cobre_run_rejected(layout, result, parameter)


def test_cobre_run_max_cores_edge_values_pass_through(layout: Layout) -> None:
    result = _run_cobre_run(layout, {"maxCoresPerNode": "9999"})

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert layout.calls[0].argv[5] == "9999"


def _stub_bash(sandbox: EnsureSandbox) -> None:
    stub = sandbox.shims / "bash"
    stub.write_text(
        _BASH_STUB.format(bash=sandbox.bash, base=sandbox.base),
        encoding="utf-8",
    )
    stub.chmod(0o755)


def test_ensure_utils_hands_bash_the_utils_arguments_and_the_script(
    ensure_sandbox: EnsureSandbox,
) -> None:
    _stub_bash(ensure_sandbox)

    result = _run_ensure(ensure_sandbox, slug="ensure-utils")

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    argv = (ensure_sandbox.base / "bash-argv").read_bytes().split(b"\0")[:-1]
    assert [item.decode("utf-8") for item in argv] == [
        "-s",
        "--",
        "--root",
        str(ensure_sandbox.root),
        "--uv",
        str(ensure_sandbox.base / "uv"),
        "--python",
        ENSURE_PYTHON,
        "--tool",
        "hpc-model-utils",
        "https://github.com/rjmalves/hpc-model-utils.git",
        ENSURE_UTILS_TAG,
        ENSURE_UTILS_SHA,
        "hpc-model-utils",
        "--tool",
        "cobre-bridge",
        "https://github.com/cobre-rs/cobre-bridge.git",
        ENSURE_SYNTHESIS_TAG,
        ENSURE_SYNTHESIS_SHA,
        "cobre-bridge",
    ]
    assert not any("sintetizador" in item.decode() for item in argv)
    script = (MODELOPS / "scripts" / "ensure-tools.sh").read_text("utf-8")
    stdin = (ensure_sandbox.base / "bash-stdin").read_text("utf-8")
    assert stdin == script + "\n"


def test_ensure_utils_cache_hit_prints_two_tool_lines_and_runs_nothing(
    ensure_sandbox: EnsureSandbox,
) -> None:
    ensure_sandbox.install("hpc-model-utils", ENSURE_UTILS_SHA)
    ensure_sandbox.install("cobre-bridge", ENSURE_SYNTHESIS_SHA)

    result = _run_ensure(ensure_sandbox, slug="ensure-utils")

    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        ensure_sandbox.tool_line("hpc-model-utils", ENSURE_UTILS_SHA)
        + ensure_sandbox.tool_line("cobre-bridge", ENSURE_SYNTHESIS_SHA),
        "",
    )
    assert result.stdout.count("HPCMU_TOOL ") == 2
    assert ensure_sandbox.recorded() == []


def test_ensure_utils_ignores_the_model_name(
    ensure_sandbox: EnsureSandbox,
) -> None:
    ensure_sandbox.install("hpc-model-utils", ENSURE_UTILS_SHA)
    ensure_sandbox.install("cobre-bridge", ENSURE_SYNTHESIS_SHA)

    result = _run_ensure(ensure_sandbox, model="cobre", slug="ensure-utils")

    assert result.returncode == 0
    assert result.stdout == ensure_sandbox.tool_line(
        "hpc-model-utils", ENSURE_UTILS_SHA
    ) + ensure_sandbox.tool_line("cobre-bridge", ENSURE_SYNTHESIS_SHA)


@pytest.mark.parametrize("parameter", CAPTURED["ensure-utils"])
def test_ensure_utils_forged_terminator_is_rejected(
    ensure_sandbox: EnsureSandbox, parameter: str
) -> None:
    forged = (
        f"x\nHPCMU_{{{{{EXECUTION_ID_PARAMETER}}}}}"
        f"\ntouch {ensure_sandbox.sentinel}"
    )

    result = _run_ensure(
        ensure_sandbox, {parameter: forged}, slug="ensure-utils"
    )

    assert (result.returncode, result.stdout, result.stderr) == (
        2,
        "",
        f"hpcmu-task: invalid {parameter}\n",
    )
    assert ensure_sandbox.recorded() == []
    assert not ensure_sandbox.sentinel.exists()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("utilsAppVersion", "main"),
        ("utilsAppVersion", "v2.1"),
        ("utilsAppSha", "A" * 40),
        ("utilsAppSha", "a" * 39),
        ("synthesisAppVersion", "main"),
        ("synthesisAppSha", "a" * 39),
    ],
)
def test_ensure_utils_invalid_value_is_rejected(
    ensure_sandbox: EnsureSandbox, parameter: str, value: str
) -> None:
    result = _run_ensure(
        ensure_sandbox, {parameter: value}, slug="ensure-utils"
    )

    assert (result.returncode, result.stdout, result.stderr) == (
        2,
        "",
        f"hpcmu-task: invalid {parameter}\n",
    )
    assert ensure_sandbox.recorded() == []
