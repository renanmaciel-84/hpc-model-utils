"""Tests for the ticket-063 dry run: ``deploy.modelops.render`` and
``apply sync`` without ``--apply``.

Acceptance criteria, selected with ``-k``: ``render``,
``classify or canonical``, ``identifier or pin``, ``readonly or redact``.
Every test runs against the in-memory ``FakeModelOpsServer`` and, for the
tag pins, a local bare git repository; nothing reaches GitHub or prd.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from deploy.modelops import apply as apply_module
from deploy.modelops.apply import (
    _PIN_REPOS,
    _SYNTHESIS_TOOLS,
    _TAG_TOOLS,
    _scrub,
    _sync,
    _verify_pin,
    cli,
)
from deploy.modelops.modelops_api import ModelOpsClient
from deploy.modelops.render import (
    EnvFile,
    EnvFileError,
    RenderError,
    canonical,
    comparison_keys,
    diff,
    load_env,
    render_task,
    render_workflow,
)
from hpc_model_utils.platform.encoding import PLATFORM_IDENTIFIERS
from tests.support.fake_modelops import FakeModelOpsServer

TOKEN = "test-token-abc123"
AKIA = "AKIAABCDEFGHIJKLMNOP"  # gitleaks:allow (fake key for redaction tests)
STAMP = "\n[deploy/modelops " + "a" * 40 + "]"
ANNOTATED_TAG = "v1.0.0"
LIGHTWEIGHT_TAG = "v1.0.1"
BRIDGE = "novomodelo-bridge"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"
_GIT_TIMEOUT = 60

HELPER_SH = "helper() { echo helper; }\nhelper\n"
ALPHA_SH = (
    "#!/usr/bin/env bash\n"
    "ROOT='@@env:rootPath@@'\n"
    "EXT='@@task:external@@'\n"
    "@@script:helper.sh@@\n"
    "echo '{{modelName}}'\n"
)
BETA_SH = "echo beta\n"
TASK_KEYS = {
    "description": "a task",
    "scriptType": "BASH",
    "tags": [],
    "parameters": [],
    "version": "1.0.0",
    "hidden": False,
}


@dataclass
class Defs:
    root: Path
    env_path: Path
    env_data: dict[str, Any]

    def env(self) -> EnvFile:
        self.save_env()
        return load_env(self.env_path)

    def save_env(self) -> None:
        text = json.dumps(self.env_data)
        if not self.env_path.exists() or self.env_path.read_text() != text:
            self.env_path.write_text(text, encoding="utf-8")

    def edit(self, rel: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.root / rel
        doc = json.loads(path.read_text(encoding="utf-8"))
        change(doc)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def set_params(self, **defaults: str) -> None:
        def change(doc: dict[str, Any]) -> None:
            by_name = {p["name"]: p for p in doc["parameters"]}
            for name, value in defaults.items():
                by_name.setdefault(
                    name, {"name": name, "type": "String", "options": []}
                )["defaultValue"] = value
            doc["parameters"] = list(by_name.values())

        self.edit("workflows/flow.json", change)

    def write(self, rel: str, content: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


@pytest.fixture
def defs(tmp_path: Path) -> Defs:
    root = tmp_path / "defs"
    built = Defs(
        root=root,
        env_path=tmp_path / "env.json",
        env_data={
            "env": {
                "rootPath": "/r/root",
                "conn": "conn-guid",
                "note": "alpha note",
            },
            "workflows": {"flow": "wf-1"},
            "tasks": {"alpha": "task-a", "external": "task-ext"},
            "protectedTaskIds": ["task-ext"],
        },
    )
    built.write("scripts/helper.sh", HELPER_SH)
    built.write("tasks/alpha.sh", ALPHA_SH)
    built.write("tasks/beta.sh", BETA_SH)
    for slug, name, observation in (
        ("alpha", "Alpha", "@@env:note@@"),
        ("beta", "Beta", ""),
    ):
        task = {
            "taskName": name,
            **TASK_KEYS,
            "observation": observation,
        }
        built.write(f"tasks/{slug}.json", json.dumps(task, indent=2))
    flow = {
        "workflowName": "Flow",
        "workflowDescription": "a workflow",
        "version": "1.0.0",
        "cancelationCriteria": "",
        "sshConnectionId": "@@env:conn@@",
        "timeout": 0,
        "parameters": [
            {
                "name": "modelName",
                "type": "String",
                "defaultValue": "newave",
                "options": [],
            },
            {
                "name": "utilsAppVersion",
                "type": "String",
                "defaultValue": "v1.0.0",
                "options": [],
            },
        ],
        "workflowTasks": [
            {
                "workflowTaskId": "t1",
                "taskId": "@@task:alpha@@",
                "onFailure": 1,
            },
            {
                "workflowTaskId": "t2",
                "taskId": "@@task:external@@",
                "onFailure": 1,
            },
        ],
        "cancelationTasks": [],
        "tags": [],
        "observation": "",
        "isSchedulesActive": False,
        "schedules": [],
    }
    built.write("workflows/flow.json", json.dumps(flow, indent=2))
    built.save_env()
    return built


@pytest.fixture
def fake() -> Iterator[FakeModelOpsServer]:
    with FakeModelOpsServer() as server:
        yield server


def _live(rendered: dict[str, Any], doc_id: str) -> dict[str, Any]:
    live = {k: v for k, v in rendered.items() if k != "timeout"}
    live.update(
        {
            "_id": doc_id,
            "createdBy": "alice",
            "createdDate": "2026-01-01T00:00:00Z",
            "lastChangeBy": "bob",
            "lastChangeDate": "2026-02-02T00:00:00Z",
            "scheduleStatus": "None",
            "observation": rendered["observation"] + STAMP,
        }
    )
    return live


def _seed(fake: FakeModelOpsServer, defs: Defs) -> None:
    """Seed live copies of every managed document that has an env id."""
    env = defs.env()
    fake.tasks = [
        _live(render_task(slug, defs.root, env), env.tasks[slug])
        for slug in sorted(p.stem for p in (defs.root / "tasks").glob("*.json"))
        if slug in env.tasks
    ]
    fake.workflows = [
        _live(render_workflow(slug, defs.root, env, {}), env.workflows[slug])
        for slug in sorted(
            p.stem for p in (defs.root / "workflows").glob("*.json")
        )
        if slug in env.workflows
    ]


def _live_task(fake: FakeModelOpsServer, doc_id: str) -> dict[str, Any]:
    (found,) = [t for t in fake.tasks if t["_id"] == doc_id]
    return found


def _run(
    fake: FakeModelOpsServer,
    defs: Defs,
    capsys: pytest.CaptureFixture[str],
    **kwargs: Any,
) -> tuple[int, list[str], str]:
    client = ModelOpsClient(base_url=fake.url, token=TOKEN)
    code = _sync(client, defs.env(), defs.root, **kwargs)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


# ------------------------------------------------------------------ git remote


@dataclass(frozen=True)
class Remote:
    url: str
    commit: str
    tag_object: str


def _git(cwd: Path, home: Path, *args: str) -> str:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    ).stdout.strip()


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Remote:
    assert shutil.which("git") is not None, "git not found on PATH"
    base = tmp_path_factory.mktemp("remote").resolve()
    home = base / "home"
    home.mkdir()
    work = base / "work"
    work.mkdir()
    _git(work, home, "-c", "init.defaultBranch=main", "init", "-q")
    (work / "file.txt").write_text("content\n")
    _git(work, home, "add", "file.txt")
    _git(work, home, "commit", "-q", "-m", "release")
    _git(work, home, "tag", "-a", ANNOTATED_TAG, "-m", "annotated release")
    _git(work, home, "tag", LIGHTWEIGHT_TAG)
    bare = base / "remote.git"
    _git(base, home, "clone", "-q", "--bare", str(work), str(bare))
    commit = _git(work, home, "rev-parse", "HEAD")
    tag_object = _git(work, home, "rev-parse", ANNOTATED_TAG)
    assert tag_object != commit
    return Remote(url=str(bare), commit=commit, tag_object=tag_object)


# ------------------------------------------------------------------ AC: render


def test_render_task_resolves_env_task_and_script_placeholders(
    defs: Defs,
) -> None:
    rendered = render_task("alpha", defs.root, defs.env())

    assert rendered["observation"] == "alpha note"
    assert rendered["script"] == (
        "#!/usr/bin/env bash\n"
        "ROOT='/r/root'\n"
        "EXT='task-ext'\n" + HELPER_SH + "\necho '{{modelName}}'\n"
    )
    assert "@@" not in json.dumps(rendered)


def test_render_script_inlines_the_file_byte_for_byte(defs: Defs) -> None:
    helper = "line one\r\n\tindented\n\n\xe9ç trailing   \n\nend"
    defs.write("scripts/helper.sh", helper)

    rendered = render_task("alpha", defs.root, defs.env())

    assert helper in rendered["script"]


def test_render_real_ensure_tools_script_inlines_byte_for_byte(
    defs: Defs,
) -> None:
    source = (MODELOPS / "scripts" / "ensure-tools.sh").read_bytes()
    defs.write("scripts/ensure-tools.sh", source.decode("utf-8"))
    defs.write("tasks/beta.sh", "@@script:ensure-tools.sh@@")

    rendered = render_task("beta", defs.root, defs.env())

    assert rendered["script"].encode("utf-8") == source


def test_render_workflow_resolves_ids_and_task_ids_override_env(
    defs: Defs,
) -> None:
    env = defs.env()

    plain = render_workflow("flow", defs.root, env, {})
    overridden = render_workflow("flow", defs.root, env, {"alpha": "new-id"})

    assert plain["sshConnectionId"] == "conn-guid"
    assert [t["taskId"] for t in plain["workflowTasks"]] == [
        "task-a",
        "task-ext",
    ]
    assert [t["taskId"] for t in overridden["workflowTasks"]] == [
        "new-id",
        "task-ext",
    ]


def test_render_unknown_env_token_raises_naming_file_and_token(
    defs: Defs,
) -> None:
    defs.write("tasks/alpha.sh", "echo '@@env:nope@@'\n")

    with pytest.raises(
        RenderError, match=r"tasks/alpha\.sh: unresolved @@env:nope@@"
    ):
        render_task("alpha", defs.root, defs.env())


def test_render_lists_every_unresolved_token_with_its_file(defs: Defs) -> None:
    defs.write("tasks/alpha.sh", "@@env:one@@\n@@task:two@@\n")
    defs.edit(
        "tasks/alpha.json", lambda d: d.update(description="@@env:three@@")
    )

    with pytest.raises(RenderError) as excinfo:
        render_task("alpha", defs.root, defs.env())

    assert excinfo.value.problems == (
        "tasks/alpha.json: unresolved @@env:three@@",
        "tasks/alpha.sh: unresolved @@env:one@@",
        "tasks/alpha.sh: unresolved @@task:two@@",
    )


def test_render_unpaired_marker_raises(defs: Defs) -> None:
    defs.write("tasks/beta.sh", "echo a@@b\n")

    with pytest.raises(RenderError, match=r"tasks/beta\.sh: unpaired @@"):
        render_task("beta", defs.root, defs.env())


def test_render_script_token_in_json_value_raises(defs: Defs) -> None:
    defs.edit(
        "tasks/alpha.json",
        lambda d: d.update(description="@@script:helper.sh@@"),
    )

    with pytest.raises(
        RenderError,
        match=r"tasks/alpha\.json: @@script:helper\.sh@@ is allowed only in a task script",
    ):
        render_task("alpha", defs.root, defs.env())


@pytest.mark.parametrize("marker", ["{{", "@@"])
def test_render_inlined_file_with_forbidden_text_raises_without_echo(
    defs: Defs, marker: str
) -> None:
    defs.write("scripts/helper.sh", f"echo secret-line {marker} x\n")

    with pytest.raises(
        RenderError, match=r"scripts/helper\.sh contains @@ or \{\{"
    ) as excinfo:
        render_task("alpha", defs.root, defs.env())

    assert "secret-line" not in str(excinfo.value)


def test_render_missing_script_file_raises_naming_it(defs: Defs) -> None:
    (defs.root / "scripts" / "helper.sh").unlink()

    with pytest.raises(RenderError, match=r"scripts/helper\.sh: cannot read"):
        render_task("alpha", defs.root, defs.env())


def test_render_env_value_with_marker_raises_without_echoing_the_value(
    defs: Defs,
) -> None:
    defs.env_data["env"]["note"] = "leaky-value@@x"

    with pytest.raises(
        RenderError, match=r"@@env:note@@ resolves to"
    ) as excinfo:
        render_task("alpha", defs.root, defs.env())

    assert "leaky-value" not in str(excinfo.value)


def test_render_reference_only_task_must_be_protected(defs: Defs) -> None:
    defs.env_data["protectedTaskIds"] = []

    with pytest.raises(
        RenderError,
        match=r"workflows/flow\.json: @@task:external@@ has no task file",
    ):
        render_workflow("flow", defs.root, defs.env(), {})


def test_render_workflow_task_without_id_is_unresolved(defs: Defs) -> None:
    del defs.env_data["tasks"]["alpha"]

    with pytest.raises(
        RenderError, match=r"workflows/flow\.json: unresolved @@task:alpha@@"
    ):
        render_workflow("flow", defs.root, defs.env(), {})


def test_render_rejects_invalid_json_and_bad_slug(defs: Defs) -> None:
    defs.write("workflows/flow.json", "{not json")

    with pytest.raises(
        RenderError, match=r"workflows/flow\.json: invalid JSON"
    ):
        render_workflow("flow", defs.root, defs.env(), {})
    with pytest.raises(RenderError, match="slug is not"):
        render_task("../alpha", defs.root, defs.env())


def test_render_real_tree_with_example_env_raises_nothing() -> None:
    env = load_env(MODELOPS / "env" / "prd.example.json")
    tasks = sorted(p.stem for p in (MODELOPS / "tasks").glob("*.json"))
    workflows = sorted(p.stem for p in (MODELOPS / "workflows").glob("*.json"))
    assert tasks and workflows

    for slug in tasks:
        rendered = render_task(slug, MODELOPS, env)
        assert "@@" not in json.dumps(rendered)
        assert rendered["script"]
    for slug in workflows:
        assert "@@" not in json.dumps(render_workflow(slug, MODELOPS, env, {}))


# ---------------------------------------------------------------- env file


def test_load_env_example_file_has_the_four_sections() -> None:
    env = load_env(MODELOPS / "env" / "prd.example.json")

    assert env.env["rootPath"] == "/example/root"
    assert "newave-pem" in env.workflows
    assert "run" in env.tasks
    assert env.protected_task_ids


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("tasks"), "tasks is not an object of strings"),
        (lambda d: d.update(env={"a": 1}), "env is not an object of strings"),
        (
            lambda d: d.update(protectedTaskIds="task-ext"),
            "protectedTaskIds is not a list of strings",
        ),
        (
            lambda d: d["tasks"].update(alpha="has space"),
            r"tasks\.alpha is not a valid id",
        ),
        (
            lambda d: d["workflows"].update({"Bad_Slug": "wf-2"}),
            "workflows has a key that is not a slug",
        ),
        (
            lambda d: d.update(protectedTaskIds=["x" * 65]),
            "protectedTaskIds has an entry that is not a valid id",
        ),
    ],
)
def test_load_env_rejects_wrong_shapes_without_echoing_values(
    defs: Defs, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    defs.env_data["env"]["note"] = "do-not-echo"
    change(defs.env_data)

    with pytest.raises(EnvFileError, match=message) as excinfo:
        defs.env()

    assert "has space" not in str(excinfo.value)
    assert "do-not-echo" not in str(excinfo.value)


def test_load_env_rejects_malformed_json_and_missing_file(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")

    with pytest.raises(EnvFileError, match="cannot read"):
        load_env(bad)
    with pytest.raises(EnvFileError, match="cannot read"):
        load_env(tmp_path / "absent.json")


def test_load_env_allows_extra_slugs_and_env_names(defs: Defs) -> None:
    defs.env_data["tasks"]["spare-v1-task"] = (
        "00000000-0000-0000-0000-000000000001"
    )
    defs.env_data["env"]["unused"] = "value"

    assert "spare-v1-task" in defs.env().tasks


# ------------------------------------------------------------ AC: canonical


def test_canonical_projects_onto_keys_with_sorted_keys_and_indent() -> None:
    doc = {"b": 1, "a": [3, 1, 2], "extra": "dropped"}

    assert canonical(doc, {"a", "b", "absent"}) == (
        '{\n  "a": [\n    3,\n    1,\n    2\n  ],\n  "b": 1\n}'
    )


@pytest.mark.parametrize(
    "observation",
    ["text" + STAMP, STAMP, "text\n[deploy/modelops " + "a" * 40 + "]"],
)
def test_canonical_strips_a_trailing_stamp(observation: str) -> None:
    stripped = canonical({"observation": observation}, {"observation"})

    assert "deploy/modelops" not in stripped


@pytest.mark.parametrize(
    "observation",
    [
        "[deploy/modelops " + "a" * 39 + "]",
        "[deploy/modelops " + "A" * 40 + "]",
        "[deploy/modelops " + "a" * 40 + "] tail",
        "[deploy/modelops " + "a" * 40 + "]\n",
    ],
)
def test_canonical_keeps_text_that_is_not_a_trailing_stamp(
    observation: str,
) -> None:
    kept = canonical({"observation": observation}, {"observation"})

    assert "deploy/modelops" in kept


def test_canonical_keeps_unicode_and_array_order() -> None:
    kept = canonical({"x": ["z", "é", "a"]}, {"x"})

    assert "é" in kept
    assert kept.index("z") < kept.index("é") < kept.index('"a"')


def test_canonical_comparison_keys_are_the_document_keys_without_timeout() -> (
    None
):
    rendered = {"a": 1, "timeout": 0, "b": 2}

    assert comparison_keys(rendered) == {"a", "b"}
    assert diff(rendered, {"a": 1, "b": 2}) == ""
    assert diff(rendered, {"a": 1, "b": 2, "timeout": 5, "_id": "x"}) == ""


def test_canonical_diff_is_unified_and_empty_only_when_equal() -> None:
    rendered = {"script": "echo a", "tags": []}

    text = diff(rendered, {"script": "echo b", "tags": []})

    assert text.splitlines()[:2] == ["--- rendered", "+++ live"]
    assert '-  "script": "echo a",' in text
    assert '+  "script": "echo b",' in text
    assert diff(rendered, rendered) == ""


# ------------------------------------------------------------ AC: classify


def test_classify_identical_documents_report_unchanged(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)

    code, lines, err = _run(fake, defs, capsys)

    assert lines == [
        "UNCHANGED task alpha",
        "NEW task beta",
        "UNCHANGED workflow flow",
        "PIN flow: no SHA pin (v1)",
        "summary: 2 unchanged, 0 changed, 1 new, 0 failures",
    ]
    assert (code, err) == (0, "")


def test_classify_one_character_script_change_reports_changed_with_diff(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)
    live = _live_task(fake, "task-a")
    live["script"] = live["script"].replace("ROOT=", "R00T=")

    code, lines, _ = _run(fake, defs, capsys)

    assert "CHANGED task alpha" in lines
    assert any(
        line.startswith("-") and "ROOT=" in line and "R00T=" not in line
        for line in lines
    )
    assert any(
        line.startswith("+") and "R00T=" in line and "ROOT=" not in line
        for line in lines
    )
    assert "summary: 1 unchanged, 1 changed, 1 new, 0 failures" in lines
    assert code == 0


def test_classify_slug_without_id_reports_new(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)

    code, lines, _ = _run(fake, defs, capsys)

    assert "NEW task beta" in lines
    assert code == 0


def test_classify_id_absent_from_live_reports_missing_and_exits_1(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)
    fake.tasks = [t for t in fake.tasks if t["_id"] != "task-a"]

    code, lines, _ = _run(fake, defs, capsys)

    assert "MISSING task alpha" in lines
    assert lines[-1] == "summary: 1 unchanged, 0 changed, 1 new, 1 failures"
    assert code == 1


def test_classify_protected_task_carries_protected_label(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    defs.env_data["protectedTaskIds"].append("task-a")
    _seed(fake, defs)
    _live_task(fake, "task-a")["script"] += "x"

    _, lines, _ = _run(fake, defs, capsys)

    assert "CHANGED task alpha PROTECTED" in lines
    assert "NEW task beta" in lines
    assert "UNCHANGED workflow flow" in lines


def test_classify_live_only_fields_and_timeout_do_not_change_status(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)
    fake.workflows[0]["lastChangeBy"] = "someone-else"
    fake.workflows[0]["unrelatedLiveField"] = [1, 2, 3]

    _, lines, _ = _run(fake, defs, capsys)

    assert "UNCHANGED workflow flow" in lines


# ------------------------------------------------------ AC: identifier / pin


def test_identifier_gap_in_unmanaged_live_workflow_is_printed_and_fails(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(fake, defs)
    fake.workflows.append(
        {
            "_id": "other",
            "workflowName": "Unmanaged One",
            "parameters": [
                {"name": "notInSet"},
                {"name": "modelName"},
                {"name": "notInSet"},
            ],
        }
    )

    code, lines, _ = _run(fake, defs, capsys)

    assert [line for line in lines if line.startswith("IDENTIFIER-GAP")] == [
        "IDENTIFIER-GAP Unmanaged One: notInSet"
    ]
    assert lines[-1] == "summary: 2 unchanged, 0 changed, 1 new, 1 failures"
    assert code == 1


def test_identifier_check_passes_when_every_name_is_in_the_set(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    assert {"modelName", "utilsAppVersion"} <= PLATFORM_IDENTIFIERS
    _seed(fake, defs)

    code, lines, _ = _run(fake, defs, capsys)

    assert not [line for line in lines if line.startswith("IDENTIFIER-GAP")]
    assert code == 0


def _pin_flow(defs: Defs, tag: str, sha: str) -> None:
    defs.set_params(utilsAppVersion=tag, utilsAppSha=sha)


def test_pin_equal_to_peeled_commit_prints_ok(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, ANNOTATED_TAG, remote.commit)
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert f"PIN {ANNOTATED_TAG} {remote.commit[:12]} ok" in lines
    assert "PIN flow: no SHA pin (v1)" not in lines
    assert code == 0


def test_pin_equal_to_tag_object_sha_prints_mismatch_and_exits_1(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, ANNOTATED_TAG, remote.tag_object)
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert f"PIN {ANNOTATED_TAG} {remote.tag_object[:12]} MISMATCH" in lines
    assert lines[-1] == "summary: 2 unchanged, 0 changed, 1 new, 1 failures"
    assert code == 1


def test_pin_lightweight_tag_compares_the_plain_sha(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, LIGHTWEIGHT_TAG, remote.commit)
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert f"PIN {LIGHTWEIGHT_TAG} {remote.commit[:12]} ok" in lines
    assert code == 0


def test_pin_unknown_tag_prints_missing_and_exits_1(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, "v9.9.9", remote.commit)
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert f"PIN v9.9.9 {remote.commit[:12]} MISSING (tag not found)" in lines
    assert code == 1


def test_pin_unreachable_repository_prints_missing_with_reason(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, ANNOTATED_TAG, remote.commit)
    _seed(fake, defs)
    absent = str(tmp_path / "absent.git")

    code, lines, _ = _run(fake, defs, capsys, repos={"hpc-model-utils": absent})

    (line,) = [x for x in lines if x.startswith(f"PIN {ANNOTATED_TAG}")]
    assert line.startswith(
        f"PIN {ANNOTATED_TAG} {remote.commit[:12]} MISSING ("
    )
    assert "exit" in line
    assert code == 1


def test_pin_repository_without_a_mapping_entry_prints_missing(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin_flow(defs, ANNOTATED_TAG, remote.commit)
    _seed(fake, defs)

    code, lines, _ = _run(fake, defs, capsys, repos={})

    assert (
        f"PIN {ANNOTATED_TAG} {remote.commit[:12]} MISSING"
        " (no repository for hpc-model-utils)"
    ) in lines
    assert code == 1


@pytest.mark.parametrize(
    ("model", "tool"),
    [("decomp", "sintetizador-decomp"), ("newave", "sintetizador-newave")],
)
def test_pin_synthesis_repository_follows_the_model_name_default(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    tool: str,
) -> None:
    defs.set_params(
        utilsAppVersion=ANNOTATED_TAG,
        utilsAppSha=remote.commit,
        synthesisAppVersion=ANNOTATED_TAG,
        synthesisAppSha=remote.commit,
        modelName=model,
    )
    _seed(fake, defs)
    unused = str(tmp_path / "must-not-be-used.git")
    repos = {name: unused for name in _SYNTHESIS_TOOLS.values()}
    repos["hpc-model-utils"] = remote.url
    repos[tool] = remote.url

    code, lines, _ = _run(fake, defs, capsys, repos=repos)

    pins = [x for x in lines if x.startswith("PIN ")]
    assert pins == [f"PIN {ANNOTATED_TAG} {remote.commit[:12]} ok"] * 2
    assert code == 0


def test_pin_synthesis_model_without_a_tool_prints_missing(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    defs.set_params(
        utilsAppVersion=ANNOTATED_TAG,
        utilsAppSha=remote.commit,
        synthesisAppVersion=ANNOTATED_TAG,
        synthesisAppSha=remote.commit,
        modelName="dessem",
    )
    _seed(fake, defs)
    repos = {"hpc-model-utils": remote.url, "sintetizador-dessem": remote.url}

    code, lines, _ = _run(fake, defs, capsys, repos=repos)

    assert (
        f"PIN {ANNOTATED_TAG} {remote.commit[:12]} MISSING"
        " (no synthesis tool for model 'dessem')"
    ) in lines
    assert f"PIN {ANNOTATED_TAG} {remote.commit[:12]} ok" in lines
    assert lines[-1].endswith("1 failures")
    assert code == 1


def test_pin_v1_workflow_prints_the_no_sha_skip_once(
    fake: FakeModelOpsServer, defs: Defs, capsys: pytest.CaptureFixture[str]
) -> None:
    defs.set_params(synthesisAppVersion="v2.4.5")
    _seed(fake, defs)

    code, lines, _ = _run(fake, defs, capsys)

    assert [x for x in lines if x.startswith("PIN")] == [
        "PIN flow: no SHA pin (v1)"
    ]
    assert code == 0


@pytest.mark.parametrize(
    "params",
    [
        {"utilsAppSha": "abc"},
        {"utilsAppVersion": "main", "utilsAppSha": "a" * 40},
        {"utilsAppSha": "A" * 40},
        {"synthesisAppSha": "a" * 40},
    ],
)
def test_pin_incomplete_or_malformed_pair_fails_instead_of_skipping(
    fake: FakeModelOpsServer,
    defs: Defs,
    capsys: pytest.CaptureFixture[str],
    params: dict[str, str],
) -> None:
    defs.set_params(**params)
    _seed(fake, defs)

    code, lines, _ = _run(fake, defs, capsys)

    assert any("is not a complete pin pair" in line for line in lines)
    assert code == 1


def test_pin_verify_runs_ls_remote_as_argv_without_prompt_or_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit, tag_object = "c" * 40, "d" * 40
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(
        argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        out = (
            f"{tag_object}\trefs/tags/v1.2.3\n{commit}\trefs/tags/v1.2.3^{{}}\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setenv("MODELOPS_TOKEN", TOKEN)

    status = _verify_pin(
        "https://example.test/r.git", "v1.2.3", commit, runner=runner
    )

    ((argv, kwargs),) = calls
    assert argv == [
        "git",
        "ls-remote",
        "https://example.test/r.git",
        "refs/tags/v1.2.3",
        "refs/tags/v1.2.3^{}",
    ]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert "MODELOPS_TOKEN" not in kwargs["env"]
    assert "shell" not in kwargs
    assert kwargs["timeout"] > 0
    assert status == "ok"


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("git"), subprocess.TimeoutExpired(["git"], 30)],
)
def test_pin_verify_git_failure_is_missing_not_skipped(
    error: Exception,
) -> None:
    def runner(
        argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        raise error

    status = _verify_pin(
        "https://example.test/r.git", "v1.2.3", "c" * 40, runner=runner
    )

    assert status.startswith("MISSING (")


def _tag_tool_flow(defs: Defs, commit: str, options: list[str]) -> None:
    defs.set_params(
        utilsAppVersion=ANNOTATED_TAG,
        utilsAppSha=commit,
        modelName="cobre",
        modelVersion=options[0] if options else "",
    )

    def change(doc: dict[str, Any]) -> None:
        (param,) = [p for p in doc["parameters"] if p["name"] == "modelVersion"]
        param["type"] = "List"
        param["options"] = options

    defs.edit("workflows/flow.json", change)


def test_pin_tag_tool_resolves_every_model_version_option_to_its_commit(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tag_tool_flow(defs, remote.commit, [LIGHTWEIGHT_TAG, ANNOTATED_TAG])
    _seed(fake, defs)

    code, lines, _ = _run(
        fake,
        defs,
        capsys,
        repos={"hpc-model-utils": remote.url, BRIDGE: remote.url},
    )

    assert [x for x in lines if x.startswith("PIN ")] == [
        f"PIN {ANNOTATED_TAG} {remote.commit[:12]} ok",
        f"PIN {BRIDGE} {LIGHTWEIGHT_TAG} {remote.commit[:12]} resolved",
        f"PIN {BRIDGE} {ANNOTATED_TAG} {remote.commit[:12]} resolved",
    ]
    assert code == 0


def test_pin_tag_tool_unknown_tag_prints_missing_and_exits_1(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tag_tool_flow(defs, remote.commit, [ANNOTATED_TAG, "v9.9.9"])
    _seed(fake, defs)

    code, lines, _ = _run(
        fake,
        defs,
        capsys,
        repos={"hpc-model-utils": remote.url, BRIDGE: remote.url},
    )

    assert f"PIN {BRIDGE} v9.9.9 MISSING (tag not found)" in lines
    assert (
        f"PIN {BRIDGE} {ANNOTATED_TAG} {remote.commit[:12]} resolved" in lines
    )
    assert lines[-1].endswith(" 1 failures")
    assert code == 1


def test_pin_tag_tool_unreachable_repository_prints_missing_with_reason(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tag_tool_flow(defs, remote.commit, [ANNOTATED_TAG])
    _seed(fake, defs)
    absent = str(tmp_path / "absent.git")

    code, lines, _ = _run(
        fake,
        defs,
        capsys,
        repos={"hpc-model-utils": remote.url, BRIDGE: absent},
    )

    (line,) = [x for x in lines if x.startswith(f"PIN {BRIDGE}")]
    assert line.startswith(
        f"PIN {BRIDGE} {ANNOTATED_TAG} MISSING (git ls-remote exit "
    )
    assert code == 1


def test_pin_tag_tool_without_a_repository_entry_prints_missing(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tag_tool_flow(defs, remote.commit, [ANNOTATED_TAG])
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert (
        f"PIN {BRIDGE} {ANNOTATED_TAG} MISSING (no repository for {BRIDGE})"
    ) in lines
    assert code == 1


@pytest.mark.parametrize(
    ("options", "line"),
    [
        (
            [ANNOTATED_TAG, "main"],
            "PIN flow: modelVersion option 'main' is not a vX.Y.Z tag",
        ),
        ([], "PIN flow: modelVersion offers no tag"),
    ],
)
def test_pin_tag_tool_option_that_is_not_a_tag_fails_without_git(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    line: str,
) -> None:
    _tag_tool_flow(defs, remote.commit, options)
    _seed(fake, defs)

    code, lines, _ = _run(
        fake,
        defs,
        capsys,
        repos={"hpc-model-utils": remote.url, BRIDGE: remote.url},
    )

    assert line in lines
    assert lines[-1].endswith(" 1 failures")
    assert code == 1


def test_pin_tag_tool_runs_only_for_its_model(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tag_tool_flow(defs, remote.commit, ["main"])
    defs.set_params(modelName="decomp")
    _seed(fake, defs)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert [x for x in lines if x.startswith("PIN ")] == [
        f"PIN {ANNOTATED_TAG} {remote.commit[:12]} ok"
    ]
    assert code == 0


def test_pin_tag_tools_map_cobre_to_the_bridge_and_its_model_version() -> None:
    assert dict(_TAG_TOOLS) == {"cobre": (BRIDGE, "modelVersion")}


def test_pin_production_repositories_are_the_four_github_projects() -> None:
    assert dict(_PIN_REPOS) == {
        "hpc-model-utils": "https://github.com/rjmalves/hpc-model-utils.git",
        "sintetizador-newave": "https://github.com/rjmalves/sintetizador-newave.git",
        "sintetizador-decomp": "https://github.com/rjmalves/sintetizador-decomp.git",
        "novomodelo-bridge": "https://github.com/ons-ccee-epe/novomodelo-bridge.git",
    }


def test_pin_synthesis_tools_map_each_model() -> None:
    assert dict(_SYNTHESIS_TOOLS) == {
        "newave": "sintetizador-newave",
        "decomp": "sintetizador-decomp",
    }


# ------------------------------------------------------ AC: readonly / redact


def _every_scenario(
    fake: FakeModelOpsServer, defs: Defs, remote: Remote
) -> None:
    _pin_flow(defs, ANNOTATED_TAG, remote.commit)
    _seed(fake, defs)
    _live_task(fake, "task-a")["script"] += "changed"
    fake.workflows.append(
        {
            "_id": "other",
            "workflowName": "Other",
            "parameters": [{"name": "notInSet"}],
        }
    )
    fake.workflows = [w for w in fake.workflows if w["_id"] != "wf-1"]


def test_readonly_dry_run_sends_only_get_requests(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _every_scenario(fake, defs, remote)

    code, lines, _ = _run(
        fake, defs, capsys, repos={"hpc-model-utils": remote.url}
    )

    assert "CHANGED task alpha" in lines
    assert "NEW task beta" in lines
    assert "MISSING workflow flow" in lines
    assert "IDENTIFIER-GAP Other: notInSet" in lines
    assert code == 1
    assert sorted((r.method, r.path) for r in fake.requests) == [
        ("GET", "/api/Task/all"),
        ("GET", "/api/Workflow/all"),
    ]


def test_readonly_fake_records_a_write_attempt_so_the_check_is_not_vacuous(
    fake: FakeModelOpsServer,
) -> None:
    request = urllib.request.Request(
        f"{fake.url}/api/Task/task-a", data=b"{}", method="PUT"
    )

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)  # noqa: S310
    excinfo.value.close()

    assert [(r.method, r.path) for r in fake.requests] == [
        ("PUT", "/api/Task/task-a")
    ]


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def test_readonly_run_leaves_git_status_and_tree_unchanged(
    fake: FakeModelOpsServer,
    defs: Defs,
    remote: Remote,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(defs.root, repo / "defs")
    shutil.copy2(defs.env_path, repo / "env.json")
    home = tmp_path / "home"
    home.mkdir()
    _git(repo, home, "-c", "init.defaultBranch=main", "init", "-q")
    moved = Defs(repo / "defs", repo / "env.json", defs.env_data)
    _every_scenario(fake, moved, remote)
    before_status = _git(
        repo, home, "status", "--porcelain", "--untracked-files=all"
    )
    before_tree = _tree_state(repo)

    _run(fake, moved, capsys, repos={"hpc-model-utils": remote.url})

    assert (
        _git(repo, home, "status", "--porcelain", "--untracked-files=all")
        == before_status
    )
    assert _tree_state(repo) == before_tree


def _cli(
    fake: FakeModelOpsServer,
    defs: Defs,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str | None = TOKEN,
) -> Any:
    monkeypatch.setattr(apply_module, "_MODELOPS_ROOT", defs.root)
    defs.save_env()
    return CliRunner().invoke(
        cli,
        ["sync", "--env-file", str(defs.env_path)],
        env={"MODELOPS_URL": fake.url, "MODELOPS_TOKEN": token},
    )


def test_redact_token_and_aws_key_never_reach_the_output(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(fake, defs)
    _live_task(fake, "task-a")["script"] += (
        f"\nexport KEY={AKIA}\nAWS_SECRET_ACCESS_KEY=hunter2hunter2\n"
        f"curl -H 'Authorization: Bearer {TOKEN}' http://x\nT={TOKEN}\n"
    )
    fake.workflows.append(
        {
            "_id": "other",
            "workflowName": f"Name {TOKEN}",
            "parameters": [{"name": f"p{AKIA}{TOKEN}"}],
        }
    )

    result = _cli(fake, defs, monkeypatch)

    output = result.stdout + result.stderr
    assert "CHANGED task alpha" in result.stdout
    assert "IDENTIFIER-GAP" in result.stdout
    assert "<redacted>" in result.stdout
    for secret in (TOKEN, AKIA, "hunter2hunter2"):
        assert secret not in output
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"k={AKIA} z", "k=<redacted> z"),
        ("Aws_Secret_Access_Key=abc/def+ghi next", "<redacted> next"),
        ("AKIA123", "AKIA123"),
        (f"Bearer {TOKEN} and {TOKEN}", "<redacted> and <redacted>"),
    ],
)
def test_redact_scrub_masks_aws_credentials_and_the_token(
    text: str, expected: str
) -> None:
    assert _scrub(text, TOKEN) == expected


def test_redact_api_error_keeps_the_token_out_of_stderr(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.script(
        "/api/Task/all",
        status=500,
        body=f"boom Authorization: Bearer {TOKEN} {TOKEN}".encode(),
    )

    result = _cli(fake, defs, monkeypatch)

    assert result.exit_code == 1
    assert TOKEN not in result.stdout + result.stderr
    assert result.stderr.startswith("apply: GET /api/Task/all: HTTP 500")
    assert len(result.stderr.splitlines()) == 1


SECRET_VALUE = "zzz-secret-value"
AWS_ERROR_BODY = (
    f"boom {AKIA} aws_secret_access_key={SECRET_VALUE} Bearer {TOKEN}".encode()
)


@pytest.mark.parametrize("path", ["/api/Task/all", "/api/Workflow/all"])
def test_redact_api_error_body_masks_aws_credentials_on_stderr(
    fake: FakeModelOpsServer,
    defs: Defs,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    fake.script(path, status=500, body=AWS_ERROR_BODY)

    result = _cli(fake, defs, monkeypatch)

    output = result.stdout + result.stderr
    assert result.exit_code == 1
    for secret in (AKIA, SECRET_VALUE, TOKEN):
        assert secret not in output
    assert result.stderr.startswith(f"apply: GET {path}: HTTP 500: boom ")
    assert "<redacted>" in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_redact_snapshot_error_shares_the_same_masking(
    fake: FakeModelOpsServer, tmp_path: Path
) -> None:
    fake.script("/api/Workflow/all", status=500, body=AWS_ERROR_BODY)

    result = CliRunner().invoke(
        cli,
        ["snapshot", "--out", str(tmp_path / "snapshot")],
        env={"MODELOPS_URL": fake.url, "MODELOPS_TOKEN": TOKEN},
    )

    assert result.exit_code == 1
    for secret in (AKIA, SECRET_VALUE, TOKEN):
        assert secret not in result.stdout + result.stderr
    assert "<redacted>" in result.stderr


# ------------------------------------------------------- the click command


def test_apply_sync_clean_dry_run_exits_0_through_the_cli(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(fake, defs)

    result = _cli(fake, defs, monkeypatch)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[-1] == (
        "summary: 2 unchanged, 0 changed, 1 new, 0 failures"
    )
    assert result.stderr == ""


def test_apply_sync_without_env_file_exits_2(fake: FakeModelOpsServer) -> None:
    result = CliRunner().invoke(
        cli,
        ["sync"],
        env={"MODELOPS_URL": fake.url, "MODELOPS_TOKEN": TOKEN},
    )

    assert result.exit_code == 2
    assert "--env-file" in result.stderr
    assert fake.requests == []


def test_apply_sync_absent_env_file_exits_2(
    fake: FakeModelOpsServer, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        cli,
        ["sync", "--env-file", str(tmp_path / "absent.json")],
        env={"MODELOPS_URL": fake.url, "MODELOPS_TOKEN": TOKEN},
    )

    assert result.exit_code == 2
    assert fake.requests == []


def test_apply_sync_malformed_env_file_exits_2_with_one_line_and_no_values(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    defs.env_data["tasks"]["alpha"] = "SECRET VALUE!"
    monkeypatch.setattr(apply_module, "_MODELOPS_ROOT", defs.root)
    defs.env_path.write_text(json.dumps(defs.env_data), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["sync", "--env-file", str(defs.env_path)],
        env={"MODELOPS_URL": fake.url, "MODELOPS_TOKEN": TOKEN},
    )

    assert result.exit_code == 2
    assert result.stderr == "apply: env file: tasks.alpha is not a valid id\n"
    assert fake.requests == []


def test_apply_sync_missing_token_exits_2_before_any_request(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli(fake, defs, monkeypatch, token=None)

    assert result.exit_code == 2
    assert result.stderr.startswith("apply: ")
    assert fake.requests == []


def test_apply_sync_unresolved_tokens_exit_1_listing_all_before_any_request(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    defs.write("tasks/alpha.sh", "@@env:one@@\n")
    defs.write("tasks/beta.sh", "@@env:two@@\n")

    result = _cli(fake, defs, monkeypatch)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "apply: tasks/alpha.sh: unresolved @@env:one@@",
        "apply: tasks/beta.sh: unresolved @@env:two@@",
    ]
    assert fake.requests == []


def test_apply_sync_unexpected_list_payload_exits_1_with_one_line(
    fake: FakeModelOpsServer, defs: Defs, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.script("/api/Task/all", status=200, body=b'{"not": "a list"}')

    result = _cli(fake, defs, monkeypatch)

    assert result.exit_code == 1
    assert (
        result.stderr == "apply: GET /api/Task/all: not a list of documents\n"
    )
