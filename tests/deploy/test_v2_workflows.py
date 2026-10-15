"""ticket-064/067 structure contract for the v2 original workflows (ADR-047).

The ticket-067 switch gave the three originals the content of their [v2]
copies, so the checks below run on ``workflows/<slug>.json``. Each ``check_*``
function takes the ``deploy/modelops`` directory so a mutated copy in
``tmp_path`` is checked the same way as the real tree. The expected values are
constants here; with the copies gone there is nothing to compare them with.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from deploy.modelops.render import load_env, render_workflow

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"

SLUGS = ("newave-pem", "decomp-pem", "upload-newave")
MODEL = {
    "newave-pem": "newave",
    "decomp-pem": "decomp",
    "upload-newave": "newave",
}

_PEM_CHAIN = (
    "ensure-tools",
    "create-workdir",
    "fetch-executables",
    "fetch-inputs",
    "extract-sanitize",
    "preprocess",
    "run",
    "result-upload",
    "remove-workdir",
)
CHAINS = {
    "newave-pem": _PEM_CHAIN,
    "decomp-pem": _PEM_CHAIN,
    "upload-newave": (
        "ensure-tools",
        "create-workdir",
        "fetch-executables",
        "ingest-offline",
        "run",
        "result-upload",
        "remove-workdir",
    ),
}
CANCEL_CHAIN = ("cancel-run", "remove-workdir")

_PEM_PARAMETERS = (
    "awsRegion",
    "outputsBucket",
    "mpichPath",
    "slurmPath",
    "rootPath",
    "inputFile",
    "parentPath",
    "coreCount",
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
PARAMETER_NAMES = {
    "newave-pem": _PEM_PARAMETERS,
    "decomp-pem": _PEM_PARAMETERS,
    "upload-newave": (
        "awsRegion",
        "outputsBucket",
        "mpichPath",
        "slurmPath",
        "rootPath",
        "inputFile",
        "outputFile",
        "cutFile",
        "coreCount",
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
    ),
}

UTILS_TAG = "v2.2.2"
TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
SHA = re.compile(r"[0-9a-f]{40}")
TASK_TOKEN = re.compile(r"@@task:([a-z0-9][a-z0-9-]*)@@")
UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

WORKFLOW_NAMES = {
    "newave-pem": "NEWAVE - PEM",
    "decomp-pem": "DECOMP - PEM",
    "upload-newave": "Upload NEWAVE",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(slug: str) -> str:
    return f"workflows/{slug}.json"


def _params(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in doc["parameters"]}


def _default(params: dict[str, dict[str, Any]], name: str) -> str:
    return str(params.get(name, {}).get("defaultValue", ""))


def _task_slug(node: dict[str, Any]) -> str | None:
    match = TASK_TOKEN.fullmatch(node.get("taskId") or "")
    return match.group(1) if match else None


def require(errors: list[str]) -> None:
    if errors:
        raise AssertionError("\n".join(errors))


def _walk(doc: dict[str, Any]) -> tuple[list[str], bool]:
    """Node ids reached from the start edge, and whether the end is reached."""
    edges = doc["taskExecutionCriterias"]
    successors: dict[str | None, list[str | None]] = {}
    for edge in edges:
        successors.setdefault(edge["sourceWorkflowTaskId"], []).append(
            edge["targetWorkflowTaskId"]
        )
    walked: list[str] = []
    current: str | None = None
    for _ in range(len(edges)):
        following = successors.get(current, [])
        if len(following) != 1:
            break
        current = following[0]
        if current is None:
            return walked, True
        walked.append(current)
    return walked, False


def check_chain(root: Path) -> list[str]:
    errors: list[str] = []
    for slug in SLUGS:
        rel = _rel(slug)
        doc = _load(root / rel)
        nodes = {n["workflowTaskId"]: n for n in doc["workflowTasks"]}
        if len(nodes) != len(doc["workflowTasks"]):
            errors.append(f"{rel}: duplicate workflowTaskId")
        for node in doc["workflowTasks"] + doc["cancelationTasks"]:
            task = _task_slug(node)
            if task is None:
                errors.append(f"{rel}: taskId is not @@task:@@")
            elif task.startswith("v1-"):
                errors.append(f"{rel}: taskId names the v1 Task {task}")
        errors += [
            f"{rel}: onFailure is not 1"
            for node in doc["workflowTasks"]
            if node.get("onFailure") != 1
        ]
        errors += [
            f"{rel}: criteria edge expression is not true"
            for edge in doc["taskExecutionCriterias"]
            if edge["criteriaExpression"] != "true"
        ]
        walked, ended = _walk(doc)
        sequence = tuple(
            _task_slug(nodes[i]) if i in nodes else None for i in walked
        )
        if sequence != CHAINS[slug] or not ended:
            errors.append(
                f"{rel}: the criteria chain from start differs from the"
                " expected sequence"
            )
        if sorted(walked) != sorted(nodes):
            errors.append(f"{rel}: workflowTasks holds nodes off the chain")
        ensure = [
            i for i, n in nodes.items() if _task_slug(n) == "ensure-tools"
        ]
        errors += [
            f"{rel}: ensure-tools workflowTaskId is not a lowercase uuid4"
            for i in ensure
            if not UUID4.fullmatch(i)
        ]
        cancel = tuple(_task_slug(n) for n in doc["cancelationTasks"])
        if cancel != CANCEL_CHAIN:
            errors.append(f"{rel}: cancelationTasks differ from the v2 pair")
        metadata = sorted(
            m["workflowTaskId"] for m in doc["workflowTaskMetadata"]
        )
        if metadata != sorted([*nodes, "startNode", "endNode"]):
            errors.append(f"{rel}: workflowTaskMetadata differs from the nodes")
    return errors


def check_parameter_names(root: Path) -> list[str]:
    errors: list[str] = []
    for slug in SLUGS:
        parameters = _load(root / _rel(slug))["parameters"]
        if tuple(p["name"] for p in parameters) != PARAMETER_NAMES[slug]:
            errors.append(f"{_rel(slug)}: parameter names differ")
        if [p["order"] for p in parameters] != list(range(len(parameters))):
            errors.append(f"{_rel(slug)}: parameter order is not contiguous")
    return errors


def check_pins(root: Path) -> list[str]:
    errors: list[str] = []
    utils_shas: set[str] = set()
    for slug in SLUGS:
        rel = _rel(slug)
        params = _params(_load(root / rel))
        if _default(params, "utilsAppVersion") != UTILS_TAG:
            errors.append(f"{rel}: utilsAppVersion is not {UTILS_TAG}")
        if not SHA.fullmatch(_default(params, "utilsAppSha")):
            errors.append(f"{rel}: utilsAppSha is not 40 hex")
        utils_shas.add(_default(params, "utilsAppSha"))
        if not TAG.fullmatch(_default(params, "synthesisAppVersion")):
            errors.append(f"{rel}: synthesisAppVersion is not a vX.Y.Z tag")
        if not SHA.fullmatch(_default(params, "synthesisAppSha")):
            errors.append(f"{rel}: synthesisAppSha is not 40 hex")
        for name in ("utilsAppSha", "synthesisAppSha"):
            param = params.get(name, {})
            shape = (
                param.get("type"),
                param.get("visible"),
                param.get("regexPattern"),
                param.get("format"),
                param.get("options"),
            )
            if shape != ("String", False, "", "", []):
                errors.append(f"{rel}: parameter {name} has the wrong shape")
    if len(utils_shas) > 1:
        errors.append("utilsAppSha differs across the workflows")
    return errors


def check_dynamic_parameters(root: Path) -> list[str]:
    errors: list[str] = []
    for slug in SLUGS:
        rel = _rel(slug)
        params = _params(_load(root / rel))
        expected = {
            "utilsToolDir": r"HPCMU_TOOL hpc-model-utils (/\S+)",
            "synthesisToolDir": (
                rf"HPCMU_TOOL sintetizador-{MODEL[slug]} (/\S+)"
            ),
        }
        for name, pattern in expected.items():
            param = params.get(name, {})
            shape = (
                param.get("type"),
                param.get("defaultValue"),
                param.get("visible"),
                param.get("regexPattern"),
                param.get("format"),
            )
            if shape != ("String", "", False, pattern, "{1}"):
                errors.append(f"{rel}: parameter {name} has the wrong shape")
    return errors


def check_names(root: Path) -> list[str]:
    return [
        f"{_rel(slug)}: workflowName is {name!r}, expected"
        f" {WORKFLOW_NAMES[slug]!r}"
        for slug in SLUGS
        if (name := _load(root / _rel(slug))["workflowName"])
        != WORKFLOW_NAMES[slug]
    ]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    shutil.copytree(MODELOPS / "workflows", tmp_path / "workflows")
    return tmp_path


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    doc = _load(path)
    change(doc)
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _mutate(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    _rewrite(root / _rel("newave-pem"), change)


def _set_workflow_name(name: str) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        doc["workflowName"] = name

    return change


def _set_default(name: str, value: str) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        _params(doc)[name]["defaultValue"] = value

    return change


_LIST_FILES_NODE = {
    "workflowTaskId": "7aee66bf-a099-4aae-abc7-98e062d70d4a",
    "alias": None,
    "taskId": "@@task:v1-list-files@@",
    "executionCriteria": "",
    "onFailure": 1,
}


def _node_id(doc: dict[str, Any], task: str) -> str:
    (found,) = (
        n["workflowTaskId"]
        for n in doc["workflowTasks"]
        if n["taskId"] == f"@@task:{task}@@"
    )
    return str(found)


def _edge_from(doc: dict[str, Any], source: str | None) -> dict[str, Any]:
    edges: list[dict[str, Any]] = [
        e
        for e in doc["taskExecutionCriterias"]
        if e["sourceWorkflowTaskId"] == source
    ]
    (found,) = edges
    return found


def _add_orphan_list_files(doc: dict[str, Any]) -> None:
    doc["workflowTasks"].append(dict(_LIST_FILES_NODE))


def _splice_list_files_after_run(doc: dict[str, Any]) -> None:
    doc["workflowTasks"].append(dict(_LIST_FILES_NODE))
    edge = _edge_from(doc, _node_id(doc, "run"))
    after = edge["targetWorkflowTaskId"]
    edge["targetWorkflowTaskId"] = _LIST_FILES_NODE["workflowTaskId"]
    doc["taskExecutionCriterias"].append(
        {
            "criteriaExpression": "true",
            "sourceWorkflowTaskId": _LIST_FILES_NODE["workflowTaskId"],
            "targetWorkflowTaskId": after,
        }
    )


def _bypass_ensure_tools(doc: dict[str, Any]) -> None:
    ensure = _node_id(doc, "ensure-tools")
    after = _edge_from(doc, ensure)["targetWorkflowTaskId"]
    doc["taskExecutionCriterias"] = [
        e
        for e in doc["taskExecutionCriterias"]
        if e["sourceWorkflowTaskId"] != ensure
    ]
    _edge_from(doc, None)["targetWorkflowTaskId"] = after


def _null_on_failure(doc: dict[str, Any]) -> None:
    doc["workflowTasks"][0]["onFailure"] = None


def _restore_aws_key(doc: dict[str, Any]) -> None:
    doc["parameters"].append(
        {
            "description": "AWS KEY ID",
            "name": "awsKeyId",
            "type": "String",
            "defaultValue": "",
            "visible": False,
            "regexPattern": "",
            "format": "",
            "order": len(doc["parameters"]),
            "options": [],
        }
    )


def _swap_synthesis_model(doc: dict[str, Any]) -> None:
    _params(doc)["synthesisToolDir"]["regexPattern"] = (
        r"HPCMU_TOOL sintetizador-decomp (/\S+)"
    )


def test_check_chain_real_tree_reports_nothing() -> None:
    require(check_chain(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_add_orphan_list_files, r"names the v1 Task v1-list-files"),
        (_add_orphan_list_files, r"holds nodes off the chain"),
        (_splice_list_files_after_run, r"names the v1 Task v1-list-files"),
        (_splice_list_files_after_run, r"chain from start differs"),
        (_bypass_ensure_tools, r"chain from start differs"),
        (_null_on_failure, r"onFailure is not 1"),
    ],
)
def test_check_chain_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _mutate(tree, mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_chain(tree))


def test_check_parameter_names_real_tree_reports_nothing() -> None:
    require(check_parameter_names(MODELOPS))


def test_check_parameter_names_restored_aws_key_is_reported(
    tree: Path,
) -> None:
    _mutate(tree, _restore_aws_key)

    with pytest.raises(AssertionError, match=r"parameter names differ"):
        require(check_parameter_names(tree))


def test_check_pins_real_tree_reports_nothing() -> None:
    require(check_pins(MODELOPS))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_set_default("utilsAppVersion", "v2.0.0"), r"utilsAppVersion is not"),
        (_set_default("utilsAppSha", "b" * 40), r"utilsAppSha differs across"),
        (_set_default("synthesisAppSha", "abc"), r"synthesisAppSha is not 40"),
        (
            _set_default("synthesisAppVersion", "main"),
            r"synthesisAppVersion is not a vX.Y.Z",
        ),
    ],
)
def test_check_pins_mutated_copy_is_reported(
    tree: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    _mutate(tree, mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_pins(tree))


def test_check_dynamic_parameters_real_tree_reports_nothing() -> None:
    require(check_dynamic_parameters(MODELOPS))


def test_check_dynamic_parameters_wrong_model_is_reported(tree: Path) -> None:
    _mutate(tree, _swap_synthesis_model)

    with pytest.raises(
        AssertionError, match=r"parameter synthesisToolDir has the wrong shape"
    ):
        require(check_dynamic_parameters(tree))


def test_check_names_real_tree_reports_nothing() -> None:
    require(check_names(MODELOPS))


def test_check_names_restored_v2_suffix_is_reported(tree: Path) -> None:
    _mutate(tree, _set_workflow_name("NEWAVE - PEM [v2]"))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "workflows/newave-pem.json: workflowName is 'NEWAVE - PEM [v2]',"
            " expected 'NEWAVE - PEM'"
        ),
    ):
        require(check_names(tree))


@pytest.mark.parametrize("slug", SLUGS)
def test_v2_workflow_renders_with_the_example_env_ids(slug: str) -> None:
    env = load_env(MODELOPS / "env" / "prd.example.json")

    rendered = render_workflow(slug, MODELOPS, env, {})

    assert sorted(n["taskId"] for n in rendered["workflowTasks"]) == sorted(
        env.tasks[task] for task in CHAINS[slug]
    )
    assert [n["taskId"] for n in rendered["cancelationTasks"]] == [
        env.tasks[task] for task in CANCEL_CHAIN
    ]
