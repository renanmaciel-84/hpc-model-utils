"""ticket-067c contract for the swept Ranqueamento and Upload Versao (ADR-044).

The sweep repoints both managed workflows to the shared v2 Tasks. The
``check_*`` functions take the ``deploy/modelops`` directory so a mutated copy
in ``tmp_path`` is checked the same way as the real tree. The ranking run Task
keeps its legacy lines; the behavior tests render it with the example env,
prepare it like ModelOps does and run it against a stub CLI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from deploy.modelops.render import load_env, render_task, render_workflow
from tests.support.script_harness import (
    EXECUTION_HASH,
    EXECUTION_ID,
    EXECUTION_ID_PARAMETER,
    SHA,
    prepare_command,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"
EXAMPLE_ENV = MODELOPS / "env" / "prd.example.json"

SLUGS = ("ranqueamento", "upload-versao")
CHAINS = {
    "ranqueamento": (
        "ensure-tools",
        "create-workdir",
        "v1-clone-simulprospec",
        "v1-clone-evalprospec",
        "v1-clone-ranking-utils",
        "fetch-executables",
        "v1-ranking-fetch-inputs",
        "v1-ranking-run",
        "remove-workdir",
    ),
    "upload-versao": (
        "create-workdir",
        "v1-clone-upload-cli",
        "v1-upload-version",
        "remove-workdir",
    ),
}
CANCEL_CHAIN = ("remove-workdir",)
RETIRED_TASKS = frozenset(
    {
        "v1-create-workdir",
        "v1-remove-workdir",
        "v1-fetch-executables",
        "v1-clone-hpc-model-utils",
        "v1-clone-sintetizador-newave",
    }
)
KEPT_NODE_IDS = {
    "ranqueamento": {
        "create-workdir": "bb1fe491-c0cc-421c-a745-fb06d22c22f7",
        "v1-clone-simulprospec": "abfdcd47-72c9-4dfc-bef4-42284853c4f3",
        "v1-clone-evalprospec": "f911e86d-2f21-4e18-a761-1e34897d0789",
        "v1-clone-ranking-utils": "5e3662eb-13e4-4953-8bf8-f3376ad539ae",
        "fetch-executables": "50abf2db-d264-4862-8402-029834b371b3",
        "v1-ranking-fetch-inputs": "0c32ee5d-8b64-4894-a690-bcdc9c92e1dc",
        "v1-ranking-run": "44055993-904f-44bc-a47e-7fb45f92c7f1",
        "remove-workdir": "bdf750d6-447a-4cd4-92fb-49d16ff5783c",
    },
    "upload-versao": {
        "create-workdir": "3af8b15c-bccf-4031-a866-232417da90ba",
        "v1-clone-upload-cli": "eb99d164-4aea-4aba-b16c-e84646d03628",
        "v1-upload-version": "5c45bb67-c7c9-4827-9f4e-d7cf0358dafb",
        "remove-workdir": "ef6153e2-7c9e-46b3-bb20-5cd2d664859f",
    },
}
KEPT_CANCEL_IDS = {
    "ranqueamento": "71f60a91-1f00-4534-af4b-d44b46139ad4",
    "upload-versao": "544e60b4-f9b3-44c0-837d-a1dc7d498f81",
}

PARAMETER_NAMES = {
    "ranqueamento": (
        "modelName",
        "utilsAppVersion",
        "synthesisAppVersion",
        "awsKeyId",
        "awsSecretKey",
        "awsRegion",
        "outputsBucket",
        "mpichPath",
        "slurmPath",
        "rootPath",
        "inputFile",
        "parentPath",
        "coreCount",
        "queue",
        "modelVersion",
        "path",
        "jobId",
        "versionsBucket",
        "jobTimeoutHours",
        "simulprospecAppVersion",
        "evalprospecAppVersion",
        "rankingAppVersion",
        "utilsAppSha",
        "synthesisAppSha",
        "utilsToolDir",
        "synthesisToolDir",
    ),
    "upload-versao": (
        "modelName",
        "modelVersion",
        "executablesFile",
        "uploadCliVersion",
        "path",
        "rootPath",
        "versionsBucket",
    ),
}
NEW_PARAMETERS = (
    "utilsAppSha",
    "synthesisAppSha",
    "utilsToolDir",
    "synthesisToolDir",
)
SHAPE_KEYS = ("description", "type", "visible", "regexPattern", "format")

UTILS_PIN = ("v1.0.1", "f61dd5633c8c2fd339bf51300cf7ca692407ac8c")
SYNTHESIS_PIN = ("v2.4.6", "3df54dff1d510529a6ef159564c124a1624acb27")
RANKING_TAG = "v1.1.1"

TASK_TOKEN = re.compile(r"@@task:([a-z0-9][a-z0-9-]*)@@")
UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


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


def _walk(edges: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """Node ids reached from the start edge, and whether the end is reached."""
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
            elif task in RETIRED_TASKS:
                errors.append(f"{rel}: taskId names the retired v1 Task {task}")
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
        walked, ended = _walk(doc["taskExecutionCriterias"])
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
        errors += [
            f"{rel}: ensure-tools workflowTaskId is not a lowercase uuid4"
            for i, n in nodes.items()
            if _task_slug(n) == "ensure-tools" and not UUID4.fullmatch(i)
        ]
        by_task = {_task_slug(n): i for i, n in nodes.items()}
        errors += [
            f"{rel}: the workflowTaskId of {task} changed"
            for task, node_id in KEPT_NODE_IDS[slug].items()
            if by_task.get(task) != node_id
        ]
        cancel = doc["cancelationTasks"]
        if tuple(_task_slug(n) for n in cancel) != CANCEL_CHAIN:
            errors.append(f"{rel}: cancelationTasks differ from the v2 remove")
        if [n["workflowTaskId"] for n in cancel] != [KEPT_CANCEL_IDS[slug]]:
            errors.append(f"{rel}: the cancel workflowTaskId changed")
        cancel_walked, cancel_ended = _walk(
            doc["cancellationTaskExecutionCriterias"]
        )
        if cancel_walked != [KEPT_CANCEL_IDS[slug]] or not cancel_ended:
            errors.append(f"{rel}: the cancel criteria chain differs")
        for key, ids in (
            ("workflowTaskMetadata", nodes),
            (
                "cancellationTaskMetadata",
                {n["workflowTaskId"]: n for n in cancel},
            ),
        ):
            metadata = sorted(m["workflowTaskId"] for m in doc[key])
            if metadata != sorted([*ids, "startNode", "endNode"]):
                errors.append(f"{rel}: {key} differs from the nodes")
    return errors


def check_parameters(root: Path) -> list[str]:
    errors: list[str] = []
    for slug in SLUGS:
        parameters = _load(root / _rel(slug))["parameters"]
        if tuple(p["name"] for p in parameters) != PARAMETER_NAMES[slug]:
            errors.append(f"{_rel(slug)}: parameter names differ")
        if [p["order"] for p in parameters] != list(range(len(parameters))):
            errors.append(f"{_rel(slug)}: parameter order is not contiguous")
    ranking = _params(_load(root / _rel("ranqueamento")))
    reference = _params(_load(root / _rel("newave-pem")))
    for name in NEW_PARAMETERS:
        shape = {k: ranking.get(name, {}).get(k) for k in SHAPE_KEYS}
        if shape != {k: reference[name][k] for k in SHAPE_KEYS}:
            errors.append(
                f"{_rel('ranqueamento')}: parameter {name} has the wrong shape"
            )
        if ranking.get(name, {}).get("options") != []:
            errors.append(
                f"{_rel('ranqueamento')}: parameter {name} has the wrong shape"
            )
    for name in ("utilsToolDir", "synthesisToolDir"):
        if _default(ranking, name) != "":
            errors.append(
                f"{_rel('ranqueamento')}: parameter {name} has the wrong shape"
            )
    return errors


def check_pins(root: Path) -> list[str]:
    errors: list[str] = []
    rel = _rel("ranqueamento")
    ranking = _params(_load(root / rel))
    for (tag_name, sha_name), (tag, sha) in (
        (("utilsAppVersion", "utilsAppSha"), UTILS_PIN),
        (("synthesisAppVersion", "synthesisAppSha"), SYNTHESIS_PIN),
    ):
        if _default(ranking, tag_name) != tag:
            errors.append(f"{rel}: {tag_name} is not {tag}")
        if _default(ranking, sha_name) != sha:
            errors.append(f"{rel}: {sha_name} is not the {tag} commit")
    if _default(ranking, "rankingAppVersion") != RANKING_TAG:
        errors.append(f"{rel}: rankingAppVersion is not {RANKING_TAG}")
    upload = _params(_load(root / _rel("upload-versao")))
    errors += [
        f"{_rel('upload-versao')}: declares {name}, a v1 workflow has no pin"
        for name in ("utilsAppSha", "synthesisAppSha")
        if name in upload
    ]
    return errors


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


_RETIRED_NODE = {
    "workflowTaskId": "2672778b-c593-4391-8319-ff0d6ae351cd",
    "alias": None,
    "taskId": "@@task:v1-clone-hpc-model-utils@@",
    "executionCriteria": None,
    "onFailure": 1,
    "hidden": False,
    "taskName": None,
}


def _restore_retired_clone(doc: dict[str, Any]) -> None:
    doc["workflowTasks"].append(dict(_RETIRED_NODE))
    edge = _edge_from(doc, _node_id(doc, "create-workdir"))
    after = edge["targetWorkflowTaskId"]
    edge["targetWorkflowTaskId"] = _RETIRED_NODE["workflowTaskId"]
    doc["taskExecutionCriterias"].append(
        {
            "criteriaExpression": "true",
            "sourceWorkflowTaskId": _RETIRED_NODE["workflowTaskId"],
            "targetWorkflowTaskId": after,
        }
    )


def _use_retired_create(doc: dict[str, Any]) -> None:
    for node in doc["workflowTasks"]:
        if node["taskId"] == "@@task:create-workdir@@":
            node["taskId"] = "@@task:v1-create-workdir@@"


def _use_retired_cancel(doc: dict[str, Any]) -> None:
    doc["cancelationTasks"][0]["taskId"] = "@@task:v1-remove-workdir@@"


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


def _regenerate_create_node_id(doc: dict[str, Any]) -> None:
    for node in doc["workflowTasks"]:
        if node["taskId"] == "@@task:create-workdir@@":
            node["workflowTaskId"] = "11111111-1111-4111-8111-111111111111"


def _uppercase_ensure_node_id(doc: dict[str, Any]) -> None:
    for node in doc["workflowTasks"]:
        if node["taskId"] == "@@task:ensure-tools@@":
            node["workflowTaskId"] = node["workflowTaskId"].upper()


def test_check_chain_real_tree_reports_nothing() -> None:
    require(check_chain(MODELOPS))


@pytest.mark.parametrize(
    ("slug", "mutation", "message"),
    [
        (
            "ranqueamento",
            _restore_retired_clone,
            r"names the retired v1 Task v1-clone-hpc-model-utils",
        ),
        ("ranqueamento", _restore_retired_clone, r"chain from start differs"),
        (
            "ranqueamento",
            _use_retired_create,
            r"names the retired v1 Task v1-create-workdir",
        ),
        (
            "upload-versao",
            _use_retired_create,
            r"names the retired v1 Task v1-create-workdir",
        ),
        (
            "ranqueamento",
            _use_retired_cancel,
            r"names the retired v1 Task v1-remove-workdir",
        ),
        (
            "upload-versao",
            _use_retired_cancel,
            r"cancelationTasks differ from the v2 remove",
        ),
        ("ranqueamento", _bypass_ensure_tools, r"chain from start differs"),
        ("ranqueamento", _null_on_failure, r"onFailure is not 1"),
        ("upload-versao", _null_on_failure, r"onFailure is not 1"),
        (
            "ranqueamento",
            _regenerate_create_node_id,
            r"the workflowTaskId of create-workdir changed",
        ),
        (
            "ranqueamento",
            _uppercase_ensure_node_id,
            r"ensure-tools workflowTaskId is not a lowercase uuid4",
        ),
    ],
)
def test_check_chain_mutated_copy_is_reported(
    tree: Path,
    slug: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _rewrite(tree / _rel(slug), mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_chain(tree))


def _drop_parameter(name: str) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        doc["parameters"] = [p for p in doc["parameters"] if p["name"] != name]

    return change


def _gap_in_order(doc: dict[str, Any]) -> None:
    doc["parameters"][-1]["order"] += 1


def _set_field(
    name: str, field: str, value: object
) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        _params(doc)[name][field] = value

    return change


def test_check_parameters_real_tree_reports_nothing() -> None:
    require(check_parameters(MODELOPS))


def test_check_parameters_ranqueamento_has_the_26_names_in_order() -> None:
    parameters = _load(MODELOPS / _rel("ranqueamento"))["parameters"]

    assert len(parameters) == 26
    assert [p["name"] for p in parameters[22:]] == list(NEW_PARAMETERS)
    assert [p["order"] for p in parameters] == list(range(26))


@pytest.mark.parametrize(
    ("slug", "mutation", "message"),
    [
        (
            "ranqueamento",
            _drop_parameter("synthesisToolDir"),
            r"parameter names differ",
        ),
        (
            "ranqueamento",
            _drop_parameter("utilsAppSha"),
            r"parameter names differ",
        ),
        (
            "ranqueamento",
            _gap_in_order,
            r"parameter order is not contiguous",
        ),
        (
            "upload-versao",
            _drop_parameter("uploadCliVersion"),
            r"parameter names differ",
        ),
        (
            "ranqueamento",
            _set_field(
                "synthesisToolDir",
                "regexPattern",
                r"HPCMU_TOOL sintetizador-decomp (/\S+)",
            ),
            r"parameter synthesisToolDir has the wrong shape",
        ),
        (
            "ranqueamento",
            _set_field("utilsToolDir", "visible", True),
            r"parameter utilsToolDir has the wrong shape",
        ),
        (
            "ranqueamento",
            _set_field("utilsToolDir", "defaultValue", "/x"),
            r"parameter utilsToolDir has the wrong shape",
        ),
        (
            "ranqueamento",
            _set_field("synthesisAppSha", "regexPattern", "x"),
            r"parameter synthesisAppSha has the wrong shape",
        ),
        (
            "ranqueamento",
            _set_field("utilsAppSha", "options", ["x"]),
            r"parameter utilsAppSha has the wrong shape",
        ),
    ],
)
def test_check_parameters_mutated_copy_is_reported(
    tree: Path,
    slug: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _rewrite(tree / _rel(slug), mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_parameters(tree))


def test_check_pins_real_tree_reports_nothing() -> None:
    require(check_pins(MODELOPS))


def _declare_sha(doc: dict[str, Any]) -> None:
    doc["parameters"].append(
        dict(_params(_load(MODELOPS / _rel("newave-pem")))["utilsAppSha"])
    )


@pytest.mark.parametrize(
    ("slug", "mutation", "message"),
    [
        (
            "ranqueamento",
            _set_field("utilsAppVersion", "defaultValue", "v1.0.0"),
            rf"utilsAppVersion is not {re.escape(UTILS_PIN[0])}",
        ),
        (
            "ranqueamento",
            _set_field("utilsAppSha", "defaultValue", "b" * 40),
            rf"utilsAppSha is not the {re.escape(UTILS_PIN[0])} commit",
        ),
        (
            "ranqueamento",
            _set_field("synthesisAppVersion", "defaultValue", "v2.3.0"),
            rf"synthesisAppVersion is not {re.escape(SYNTHESIS_PIN[0])}",
        ),
        (
            "ranqueamento",
            _set_field("synthesisAppSha", "defaultValue", "abc"),
            rf"synthesisAppSha is not the {re.escape(SYNTHESIS_PIN[0])} commit",
        ),
        (
            "ranqueamento",
            _set_field("rankingAppVersion", "defaultValue", "main"),
            rf"rankingAppVersion is not {re.escape(RANKING_TAG)}",
        ),
        (
            "ranqueamento",
            _set_field("rankingAppVersion", "defaultValue", "v1.1.2"),
            rf"rankingAppVersion is not {re.escape(RANKING_TAG)}",
        ),
        ("upload-versao", _declare_sha, r"declares utilsAppSha"),
    ],
)
def test_check_pins_mutated_copy_is_reported(
    tree: Path,
    slug: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _rewrite(tree / _rel(slug), mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_pins(tree))


@pytest.mark.parametrize("slug", SLUGS)
def test_swept_workflow_renders_with_the_example_env_ids(slug: str) -> None:
    env = load_env(EXAMPLE_ENV)

    rendered = render_workflow(slug, MODELOPS, env, {})

    assert sorted(n["taskId"] for n in rendered["workflowTasks"]) == sorted(
        env.tasks[task] for task in CHAINS[slug]
    )
    assert [n["taskId"] for n in rendered["cancelationTasks"]] == [
        env.tasks[task] for task in CANCEL_CHAIN
    ]


UTILS_DIR = f"/example/tools/hpc-model-utils/{SHA}"
SYNTHESIS_DIR = f"/example/tools/sintetizador-newave/{SHA}"
LEGACY_TAIL = (
    "cd {{path}}\n"
    "ranqueamento-prospectivo-utils/venv/bin/ranqueamento-prospectivo-utils"
    " run {{queue}} {{coreCount}} --max-job-time-hours {{jobTimeoutHours}}"
    " --mpich-path {{mpichPath}} --slurm-path {{slurmPath}}\n"
    "ranqueamento-prospectivo-utils/venv/bin/ranqueamento-prospectivo-utils"
    ' result_upload "s3://{{outputsBucket}}/artifacts/'
    '{{CurrentExecution.ExecutionHash}}"'
)
_STUB = """#!{bash}
set -eu
printf '%s|%s|%s\\n' "${{HPCMU_UTILS_APP-unset}}" \\
  "${{HPCMU_SYNTHESIS_APP-unset}}" "$*" >> "{log}"
if [ "$1" = run ] && [ -e "{fail_marker}" ]; then exit 3; fi
"""


@dataclass(frozen=True)
class Call:
    utils_app: str
    synthesis_app: str
    argv: str


@dataclass(frozen=True)
class Sandbox:
    base: Path

    @property
    def workdir(self) -> Path:
        return self.base / "workdir"

    @property
    def log(self) -> Path:
        return self.base / "calls.log"

    @property
    def fail_marker(self) -> Path:
        return self.base / "fail-run"

    @property
    def sentinel(self) -> Path:
        return self.base / "sentinel"

    @property
    def calls(self) -> list[Call]:
        if not self.log.exists():
            return []
        return [
            Call(*line.split("|", 2))
            for line in self.log.read_text("utf-8").splitlines()
        ]


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    bash = shutil.which("bash")
    assert bash is not None
    assert re.fullmatch(r"/[A-Za-z0-9._@/-]+", str(tmp_path)), (
        "TMPDIR is outside the legacy lines' unquoted path"
    )
    box = Sandbox(tmp_path)
    stub = (
        box.workdir
        / "ranqueamento-prospectivo-utils"
        / "venv"
        / "bin"
        / "ranqueamento-prospectivo-utils"
    )
    stub.parent.mkdir(parents=True)
    stub.write_text(
        _STUB.format(bash=bash, log=box.log, fail_marker=box.fail_marker),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return box


def _values(
    box: Sandbox, overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = load_env(EXAMPLE_ENV).env
    return {
        EXECUTION_ID_PARAMETER: EXECUTION_ID,
        "utilsToolDir": UTILS_DIR,
        "synthesisToolDir": SYNTHESIS_DIR,
        "path": str(box.workdir),
        "queue": env["queueNewave"],
        "coreCount": "16",
        "jobTimeoutHours": "2",
        "mpichPath": env["mpichPath"],
        "slurmPath": env["slurmPath"],
        "outputsBucket": env["outputsBucket"],
        "CurrentExecution.ExecutionHash": EXECUTION_HASH,
        **(overrides or {}),
    }


def _run_ranking(
    box: Sandbox, overrides: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    script = render_task("v1-ranking-run", MODELOPS, load_env(EXAMPLE_ENV))[
        "script"
    ]
    return subprocess.run(
        ["bash", "-c", prepare_command(script, _values(box, overrides))],
        cwd=box.base,
        env={"PATH": os.environ["PATH"], "LC_ALL": "C"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _expected_calls() -> list[Call]:
    env = load_env(EXAMPLE_ENV).env
    apps = (
        f"{UTILS_DIR}/.venv/bin/hpc-model-utils",
        f"{SYNTHESIS_DIR}/.venv/bin/sintetizador-newave",
    )
    return [
        Call(
            *apps,
            f"run {env['queueNewave']} 16 --max-job-time-hours 2"
            f" --mpich-path {env['mpichPath']}"
            f" --slurm-path {env['slurmPath']}",
        ),
        Call(
            *apps,
            f"result_upload s3://{env['outputsBucket']}/artifacts/"
            f"{EXECUTION_HASH}",
        ),
    ]


def test_ranking_run_exports_the_tool_paths_and_runs_both_legacy_calls(
    sandbox: Sandbox,
) -> None:
    result = _run_ranking(sandbox)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert sandbox.calls == _expected_calls()


def test_ranking_run_failed_run_still_uploads_and_reports_the_upload_status(
    sandbox: Sandbox,
) -> None:
    sandbox.fail_marker.touch()

    result = _run_ranking(sandbox)

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert sandbox.calls == _expected_calls()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("utilsToolDir", f"/other/tools/hpc-model-utils/{SHA}"),
        ("utilsToolDir", f"{UTILS_DIR};id"),
        ("utilsToolDir", f"{UTILS_DIR}\nid"),
        ("utilsToolDir", f"/example/tools/sintetizador-newave/{SHA}"),
        ("utilsToolDir", ""),
        ("synthesisToolDir", f"/other/tools/sintetizador-newave/{SHA}"),
        ("synthesisToolDir", f"{SYNTHESIS_DIR};id"),
        ("synthesisToolDir", f"{SYNTHESIS_DIR}\nid"),
        ("synthesisToolDir", f"/example/tools/sintetizador-decomp/{SHA}"),
        ("synthesisToolDir", ""),
    ],
)
def test_ranking_run_rejects_a_tool_dir_outside_the_root_or_injected(
    sandbox: Sandbox, parameter: str, value: str
) -> None:
    result = _run_ranking(sandbox, {parameter: value})

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"hpcmu-task: invalid {parameter}\n"
    assert sandbox.calls == []


def test_ranking_run_forged_tool_line_is_data_not_shell(
    sandbox: Sandbox,
) -> None:
    forged = (
        f"{UTILS_DIR}$(touch {sandbox.sentinel}) `touch {sandbox.sentinel}`"
    )

    result = _run_ranking(sandbox, {"utilsToolDir": forged})

    assert result.returncode == 2
    assert result.stderr == "hpcmu-task: invalid utilsToolDir\n"
    assert not sandbox.sentinel.exists()
    assert sandbox.calls == []


def test_ranking_run_invalid_execution_id_is_rejected(
    sandbox: Sandbox,
) -> None:
    result = _run_ranking(sandbox, {EXECUTION_ID_PARAMETER: "bad id"})

    assert result.returncode == 2
    assert result.stderr == "hpcmu-task: invalid CurrentExecution.ExecutionId\n"
    assert sandbox.calls == []


def test_ranking_run_script_ends_with_the_legacy_lines_byte_for_byte() -> None:
    script = (MODELOPS / "tasks" / "v1-ranking-run.sh").read_bytes()

    assert script.endswith(LEGACY_TAIL.encode("utf-8"))
    assert b"set -e" not in script


@pytest.mark.parametrize("parameter", ["utilsToolDir", "synthesisToolDir"])
def test_ranking_run_script_reads_each_tool_dir_only_inside_its_heredoc(
    parameter: str,
) -> None:
    script = (MODELOPS / "tasks" / "v1-ranking-run.sh").read_text("utf-8")
    body = script.split("\n")
    at = [i for i, line in enumerate(body) if "{{" + parameter in line]

    assert len(at) == 1
    assert body[at[0] - 1].startswith("IFS= read -r -d '' ")
    assert body[at[0] + 1] == "HPCMU_{{CurrentExecution.ExecutionId}}"
