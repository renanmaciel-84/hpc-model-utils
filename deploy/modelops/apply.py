"""deploy.modelops.apply: the ModelOps operator CLI (ADR-033, R98).

Run as ``uv run python -m deploy.modelops.apply <command>``. ``snapshot``
is read-only and ``sync`` is a read-only dry run; ``sync --apply`` is the
only write path (ADR-033): ``plan()`` computes the dry run, ``execute()``
checks every precondition, asks for the typed confirmation and then writes.

Every command-level failure -- a bad ``--out``, a ``ConfigError`` from
the environment, an ``EnvFileError`` from ``--env-file``, an
``ApplyRefused`` from a write guard, or a ``ModelOpsApiError`` from the
API -- is caught
in ``_ApplyGroup.invoke()`` rather than left to Click's own
standalone-mode formatting. ``CliRunner`` (used by the tests) always
runs a command's ``main()`` in standalone mode and never sees a
separate wrapper `main()` defined by this module, so catching these
exceptions here is what makes the single ``apply: <redacted reason>``
stderr line and exit code happen the same way under the tests and
under the real ``python -m`` entry point.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import click

from deploy.modelops.modelops_api import (
    ConfigError,
    ModelOpsApiError,
    ModelOpsClient,
    redact,
)
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

_MODELOPS_ROOT = Path(__file__).resolve().parent
_GIT_TIMEOUT_S = 30
_PIN_REPOS: Mapping[str, str] = MappingProxyType(
    {
        "hpc-model-utils": "https://github.com/rjmalves/hpc-model-utils.git",
        "sintetizador-newave": (
            "https://github.com/rjmalves/sintetizador-newave.git"
        ),
        "sintetizador-decomp": (
            "https://github.com/rjmalves/sintetizador-decomp.git"
        ),
        "novomodelo-bridge": (
            "https://github.com/ons-ccee-epe/novomodelo-bridge.git"
        ),
    }
)
_SYNTHESIS_TOOLS: Mapping[str, str] = MappingProxyType(
    {
        "newave": "sintetizador-newave",
        "decomp": "sintetizador-decomp",
    }
)
_TAG_TOOLS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {"cobre": ("novomodelo-bridge", "modelVersion")}
)
_PINS = (
    ("utilsAppVersion", "utilsAppSha"),
    ("synthesisAppVersion", "synthesisAppSha"),
)
_TAG_RE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
# A model version is a release tag, or a patched build of one with a semver
# pre-release suffix (v0.19.1-fixinflows). A tag tool follows the release, so
# a patched build runs the tool of its base release.
_MODEL_VERSION_RE = re.compile(
    rf"(?P<release>{_TAG_RE.pattern})(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_USER_RE = re.compile(r"[A-Za-z0-9._@-]{1,64}")
_API_KINDS = MappingProxyType({"task": "Task", "workflow": "Workflow"})
_NAME_KEYS = MappingProxyType({"task": "taskName", "workflow": "workflowName"})


class OutGuardError(Exception):
    pass


class ApplyRefused(Exception):
    def __init__(self, reason: str, code: int = 1) -> None:
        super().__init__(reason)
        self.code = code


def _git(
    cwd: Path, *args: str, error: Callable[[str], Exception]
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "MODELOPS_TOKEN"}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise error(f"git {args[0]} failed: {exc}") from exc


def _write_snapshot_files(out: Path, files: Mapping[str, Any]) -> None:
    """Write every file's ``.part`` first, then rename all of them, so a
    snapshot is all-or-nothing: a failure at any point removes every
    ``.part`` AND every final file this call already renamed, instead of
    leaving a partial snapshot behind (``--out`` is required to be empty
    or absent, so clearing these specific names on failure is safe).
    """
    parts = {name: out / f"{name}.part" for name in files}
    try:
        for name, payload in files.items():
            parts[name].write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        for name in files:
            os.replace(parts[name], out / name)
    except BaseException:
        for name in files:
            parts[name].unlink(missing_ok=True)
            (out / name).unlink(missing_ok=True)
        raise


def _git_toplevel(cwd: Path) -> Path | None:
    result = _git(cwd, "rev-parse", "--show-toplevel", error=OutGuardError)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def _git_check_ignore(cwd: Path, target: Path) -> bool:
    result = _git(cwd, "check-ignore", "-q", str(target), error=OutGuardError)
    return result.returncode == 0


def _first_existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    raise OutGuardError(f"no existing ancestor for --out: {path}")


def _check_out_guard(out: Path) -> None:
    if out.exists():
        if not out.is_dir():
            raise OutGuardError(f"--out exists and is not a directory: {out}")
        if any(out.iterdir()):
            raise OutGuardError(f"--out is not an empty directory: {out}")
        anchor = out
    else:
        anchor = _first_existing_ancestor(out)
    toplevel = _git_toplevel(anchor)
    if toplevel is None:
        # Not inside any git work tree at all, so nothing can accidentally
        # end up tracked there; the ADR-033 guard below does not apply.
        return
    if not _git_check_ignore(toplevel, out):
        raise OutGuardError(f"--out is not git-ignored: {out}")


def _scrub(text: str, token: str) -> str:
    return redact(text, token)


def _emit(token: str, line: str, *, err: bool = False) -> None:
    click.echo(_scrub(line, token), err=err)


def _fail(reason: str, code: int) -> NoReturn:
    token = os.environ.get("MODELOPS_TOKEN", "")
    click.echo(f"apply: {_scrub(reason, token)}", err=True)
    raise click.exceptions.Exit(code)


class _ApplyGroup(click.Group):
    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (ConfigError, OutGuardError, EnvFileError) as exc:
            _fail(str(exc), 2)
        except ApplyRefused as exc:
            _fail(str(exc), exc.code)
        except (ModelOpsApiError, OSError) as exc:
            _fail(str(exc), 1)


@click.group(cls=_ApplyGroup)
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(path_type=Path),
)
def snapshot(out: Path) -> None:
    client = ModelOpsClient.from_env(os.environ)
    resolved = out.resolve()
    _check_out_guard(resolved)
    workflows = client.get_json("/api/Workflow/all")
    tasks = client.get_json("/api/Task/all")
    resolved.mkdir(parents=True, exist_ok=True)
    meta = {
        "takenAt": datetime.now(tz=UTC).isoformat(),
        "workflowCount": len(workflows),
        "taskCount": len(tasks),
    }
    _write_snapshot_files(
        resolved,
        {"workflows.json": workflows, "tasks.json": tasks, "meta.json": meta},
    )
    click.echo(
        f"snapshot: {len(workflows)} workflows, {len(tasks)} tasks "
        f"-> {resolved}"
    )


def _live_documents(payload: Any, path: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, list) or not all(
        isinstance(doc, dict) and isinstance(doc.get("_id"), str)
        for doc in payload
    ):
        raise ModelOpsApiError(f"GET {path}: not a list of documents")
    return {doc["_id"]: doc for doc in payload}


def _render_all(
    root: Path, env: EnvFile
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    tasks: dict[str, dict[str, Any]] = {}
    workflows: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    task_slugs = sorted(p.stem for p in (root / "tasks").glob("*.json"))
    provisional = {
        slug: f"<new:{slug}>" for slug in task_slugs if slug not in env.tasks
    }
    for slug in task_slugs:
        try:
            tasks[slug] = render_task(slug, root, env)
        except RenderError as exc:
            problems.extend(exc.problems)
    for slug in sorted(p.stem for p in (root / "workflows").glob("*.json")):
        try:
            workflows[slug] = render_workflow(slug, root, env, provisional)
        except RenderError as exc:
            problems.extend(exc.problems)
    return tasks, workflows, problems


@dataclass(frozen=True)
class Entry:
    kind: str
    slug: str
    status: str
    doc_id: str | None
    rendered: Mapping[str, Any]
    live_canonical: str | None
    protected: bool


@dataclass(frozen=True)
class Plan:
    root: Path
    env: EnvFile
    entries: tuple[Entry, ...]
    live_tasks: Mapping[str, Mapping[str, Any]]
    live_workflows: Mapping[str, Mapping[str, Any]]
    lines: tuple[str, ...]
    problems: tuple[str, ...]
    failures: int

    @property
    def failed(self) -> bool:
        return bool(self.problems or self.failures)


def _classify(
    kind: str,
    rendered: Mapping[str, Mapping[str, Any]],
    ids: Mapping[str, str],
    live: Mapping[str, Mapping[str, Any]],
    protected: frozenset[str],
    counts: Counter[str],
    emit: Callable[[str], None],
) -> list[Entry]:
    entries: list[Entry] = []
    for slug in sorted(rendered):
        doc_id = ids.get(slug)
        delta = ""
        live_canonical = None
        if doc_id is None:
            status = "NEW"
        elif doc_id not in live:
            status = "MISSING"
        else:
            delta = diff(rendered[slug], live[doc_id])
            live_canonical = canonical(
                live[doc_id], comparison_keys(rendered[slug])
            )
            status = "CHANGED" if delta else "UNCHANGED"
        counts["failures" if status == "MISSING" else status.lower()] += 1
        is_protected = kind == "task" and doc_id in protected
        emit(f"{status} {kind} {slug}{' PROTECTED' if is_protected else ''}")
        for line in delta.splitlines():
            emit(line)
        entries.append(
            Entry(
                kind,
                slug,
                status,
                doc_id,
                rendered[slug],
                live_canonical,
                is_protected,
            )
        )
    return entries


def _identifier_gaps(live: Mapping[str, Mapping[str, Any]]) -> list[str]:
    gaps: list[str] = []
    for workflow in live.values():
        name = workflow.get("workflowName", "?")
        reported: set[str] = set()
        for parameter in workflow.get("parameters") or []:
            param_name = parameter.get("name")
            if (
                param_name not in PLATFORM_IDENTIFIERS
                and param_name not in reported
            ):
                reported.add(param_name)
                gaps.append(f"IDENTIFIER-GAP {name}: {param_name}")
    return gaps


class _TagUnresolvedError(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def _resolve_tag(
    repo: str,
    tag: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    peeled_ref = f"refs/tags/{tag}^{{}}"
    env = {k: v for k, v in os.environ.items() if k != "MODELOPS_TOKEN"}
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = runner(
            ["git", "ls-remote", repo, f"refs/tags/{tag}", peeled_ref],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _TagUnresolvedError(f"MISSING ({exc})") from exc
    if result.returncode != 0:
        reason = " ".join(result.stderr.split())[:200]
        raise _TagUnresolvedError(
            f"MISSING (git ls-remote exit {result.returncode}: {reason})"
        )
    shas: dict[str, str] = {}
    for line in result.stdout.splitlines():
        found, _, ref = line.partition("\t")
        shas[ref] = found
    actual = shas.get(peeled_ref) or shas.get(f"refs/tags/{tag}")
    if actual is None:
        raise _TagUnresolvedError("MISSING (tag not found)")
    return actual


def _verify_pin(
    repo: str,
    tag: str,
    sha: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    try:
        actual = _resolve_tag(repo, tag, runner=runner)
    except _TagUnresolvedError as exc:
        return exc.status
    return "ok" if actual == sha else "MISMATCH"


def _release_tag(version: object) -> str | None:
    """The release tag of a model version: v0.19.1 for v0.19.1-fixinflows."""
    if not isinstance(version, str):
        return None
    match = _MODEL_VERSION_RE.fullmatch(version)
    return None if match is None else match["release"]


def _tag_lines(
    slug: str,
    workflow: Mapping[str, Any],
    tool: tuple[str, str],
    repos: Mapping[str, str],
) -> tuple[list[str], int]:
    name, param = tool
    options = next(
        (
            p.get("options")
            for p in workflow["parameters"]
            if p["name"] == param
        ),
        None,
    )
    if not (isinstance(options, list) and options):
        return [f"PIN {slug}: {param} offers no tag"], 1
    lines: list[str] = []
    failures = 0
    for option in options:
        tag = _release_tag(option)
        if tag is None:
            lines.append(
                f"PIN {slug}: {param} option {option!r} is not a"
                " vX.Y.Z[-suffix] version"
            )
            failures += 1
            continue
        label = tag if tag == option else f"{tag} for {option}"
        if name not in repos:
            lines.append(
                f"PIN {name} {label} MISSING (no repository for {name})"
            )
            failures += 1
            continue
        try:
            sha = _resolve_tag(repos[name], tag)
        except _TagUnresolvedError as exc:
            lines.append(f"PIN {name} {label} {exc.status}")
            failures += 1
        else:
            lines.append(f"PIN {name} {label} {sha[:12]} resolved")
    return lines, failures


def _pin_lines(
    slug: str, workflow: Mapping[str, Any], repos: Mapping[str, str]
) -> tuple[list[str], int]:
    params = {p["name"]: p["defaultValue"] for p in workflow["parameters"]}
    if not any(sha_param in params for _, sha_param in _PINS):
        return [f"PIN {slug}: no SHA pin (v1)"], 0
    lines: list[str] = []
    failures = 0
    for tag_param, sha_param in _PINS:
        if tag_param not in params and sha_param not in params:
            continue
        tag = params.get(tag_param, "")
        sha = params.get(sha_param, "")
        if not (_TAG_RE.fullmatch(tag) and _SHA_RE.fullmatch(sha)):
            lines.append(
                f"PIN {slug}: {tag_param}/{sha_param} is not a complete pin pair"
            )
            failures += 1
            continue
        model = params.get("modelName", "")
        name: str | None
        if tag_param == "utilsAppVersion":
            name = "hpc-model-utils"
        else:
            name = _SYNTHESIS_TOOLS.get(model)
        if name is None:
            status = f"MISSING (no synthesis tool for model {model!r})"
        elif name not in repos:
            status = f"MISSING (no repository for {name})"
        else:
            status = _verify_pin(repos[name], tag, sha)
        lines.append(f"PIN {tag} {sha[:12]} {status}")
        failures += status != "ok"
    tag_tool = _TAG_TOOLS.get(params.get("modelName", ""))
    if tag_tool is not None:
        tag_lines, tag_failures = _tag_lines(slug, workflow, tag_tool, repos)
        lines.extend(tag_lines)
        failures += tag_failures
    return lines, failures


def plan(
    root: Path,
    env: EnvFile,
    client: ModelOpsClient,
    *,
    repos: Mapping[str, str] = _PIN_REPOS,
) -> Plan:
    tasks, workflows, problems = _render_all(root, env)
    if problems:
        return Plan(root, env, (), {}, {}, (), tuple(problems), 0)
    live_tasks = _live_documents(
        client.get_json("/api/Task/all"), "/api/Task/all"
    )
    live_workflows = _live_documents(
        client.get_json("/api/Workflow/all"), "/api/Workflow/all"
    )
    counts: Counter[str] = Counter()
    lines: list[str] = []
    entries = [
        *_classify(
            "task",
            tasks,
            env.tasks,
            live_tasks,
            env.protected_task_ids,
            counts,
            lines.append,
        ),
        *_classify(
            "workflow",
            workflows,
            env.workflows,
            live_workflows,
            env.protected_task_ids,
            counts,
            lines.append,
        ),
    ]
    for gap in _identifier_gaps(live_workflows):
        lines.append(gap)
        counts["failures"] += 1
    for slug in sorted(workflows):
        pin_lines, failed = _pin_lines(slug, workflows[slug], repos)
        lines.extend(pin_lines)
        counts["failures"] += failed
    lines.append(
        f"summary: {counts['unchanged']} unchanged, {counts['changed']} "
        f"changed, {counts['new']} new, {counts['failures']} failures"
    )
    return Plan(
        root,
        env,
        tuple(entries),
        live_tasks,
        live_workflows,
        tuple(lines),
        (),
        counts["failures"],
    )


def _print_report(report: Plan, token: str) -> None:
    for problem in report.problems:
        _emit(token, f"apply: {problem}", err=True)
    for line in report.lines:
        _emit(token, line)


def _sync(
    client: ModelOpsClient,
    env: EnvFile,
    root: Path,
    *,
    repos: Mapping[str, str] = _PIN_REPOS,
) -> int:
    report = plan(root, env, client, repos=repos)
    _print_report(report, client.token)
    return 1 if report.failed else 0


def _run_git(cwd: Path, *args: str) -> str:
    result = _git(cwd, *args, error=ApplyRefused)
    if result.returncode != 0:
        reason = " ".join(result.stderr.split())[:200]
        raise ApplyRefused(f"git {args[0]} exit {result.returncode}: {reason}")
    return result.stdout


def _git_head(root: Path) -> str:
    head = _run_git(root, "rev-parse", "HEAD").strip()
    if not _SHA_RE.fullmatch(head):
        raise ApplyRefused("git rev-parse HEAD is not a 40-hex commit id")
    return head


def _require_clean_tree(root: Path) -> None:
    status = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
    if status.strip():
        raise ApplyRefused(
            f"the working tree is not clean ({len(status.splitlines())} "
            "entries in git status --porcelain); commit or remove them"
        )


def _tty_confirm() -> bool:
    if not sys.stdin.isatty():
        raise ApplyRefused("stdin is not a terminal; cannot ask to confirm")
    try:
        return input("type 'apply' to write the changes above: ") == "apply"
    except EOFError:
        return False


def _protected_writes(plan: Plan) -> list[str]:
    return [
        f"task {entry.slug} is in protectedTaskIds"
        for entry in plan.entries
        if entry.protected and entry.status == "CHANGED"
    ]


def _unmanaged_references(plan: Plan) -> list[str]:
    managed = {
        entry.doc_id for entry in plan.entries if entry.kind == "workflow"
    }
    referenced_by: dict[str | None, set[str]] = {}
    for workflow_id, workflow in plan.live_workflows.items():
        if workflow_id in managed:
            continue
        for field in ("workflowTasks", "cancelationTasks"):
            for item in workflow.get(field) or []:
                referenced_by.setdefault(item.get("taskId"), set()).add(
                    workflow.get("workflowName", "?")
                )
    return [
        f"task {entry.slug} is referenced by unmanaged workflow "
        + ", ".join(sorted(referenced_by[entry.doc_id]))
        for entry in plan.entries
        if entry.kind == "task"
        and entry.status == "CHANGED"
        and entry.doc_id in referenced_by
    ]


def _name_collisions(plan: Plan) -> list[str]:
    reasons: list[str] = []
    for kind, live in (
        ("task", plan.live_tasks),
        ("workflow", plan.live_workflows),
    ):
        key = _NAME_KEYS[kind]
        live_names = {doc.get(key) for doc in live.values()}
        new = [e for e in plan.entries if e.kind == kind and e.status == "NEW"]
        new_names = Counter(entry.rendered.get(key) for entry in new)
        for entry in new:
            name = entry.rendered.get(key)
            if name in live_names:
                reasons.append(
                    f"{kind} {entry.slug}: {key} already exists live; "
                    f"record its id under {kind}s.{entry.slug} in the env file"
                )
            elif new_names[name] > 1:
                reasons.append(
                    f"{kind} {entry.slug}: {key} is shared with another "
                    f"new {kind}"
                )
    return reasons


def _ordered_writes(plan: Plan) -> list[tuple[str, Entry]]:
    return [
        (method, entry)
        for kind in ("task", "workflow")
        for method, status in (("PUT", "CHANGED"), ("POST", "NEW"))
        for entry in plan.entries
        if entry.kind == kind and entry.status == status
    ]


def _write_line(method: str, entry: Entry) -> str:
    target = f" {entry.doc_id}" if method == "PUT" else ""
    return f"{method} {entry.kind} {entry.slug}{target}"


def _created_id(client: ModelOpsClient, entry: Entry, name: str) -> str:
    path = f"/api/{_API_KINDS[entry.kind]}/all"
    key = _NAME_KEYS[entry.kind]
    found = [
        doc_id
        for doc_id, doc in _live_documents(client.get_json(path), path).items()
        if doc.get(key) == name
    ]
    if len(found) != 1:
        raise ApplyRefused(
            f"{entry.kind} {entry.slug}: expected exactly one live {key} "
            f"match after the POST, found {len(found)}"
        )
    return found[0]


def _write_all(
    plan: Plan,
    writes: list[tuple[str, Entry]],
    *,
    client: ModelOpsClient,
    user: str,
    head: str,
) -> None:
    created: list[str] = []
    new_task_ids: dict[str, str] = {}
    try:
        for method, entry in writes:
            path = f"/api/{_API_KINDS[entry.kind]}"
            doc = (
                entry.rendered
                if entry.kind == "task"
                else render_workflow(
                    entry.slug, plan.root, plan.env, new_task_ids
                )
            )
            body = {
                **doc,
                "observation": (
                    f"{doc.get('observation', '')}\n[deploy/modelops {head}]"
                ),
                "lastChangeBy": user,
            }
            if method == "PUT":
                target = f"{path}/{entry.doc_id}"
                fresh = client.get_json(target)
                if (
                    canonical(fresh, comparison_keys(entry.rendered))
                    != entry.live_canonical
                ):
                    raise ApplyRefused(
                        f"{entry.kind} {entry.slug} changed on the server "
                        "since the plan was computed"
                    )
                client.put_json(
                    target, {**body, "createdBy": fresh.get("createdBy")}
                )
            else:
                client.post_json(path, {**body, "createdBy": user})
            _emit(client.token, f"WRITTEN {_write_line(method, entry)}")
            if method == "POST":
                name = doc[_NAME_KEYS[entry.kind]]
                doc_id = _created_id(client, entry, name)
                if entry.kind == "task":
                    new_task_ids[entry.slug] = doc_id
                created.append(
                    f"CREATED {entry.kind} {entry.slug} {doc_id}  "
                    f"(add to env file: {entry.kind}s.{entry.slug})"
                )
    finally:
        for line in created:
            _emit(client.token, line)


def execute(
    plan: Plan,
    *,
    client: ModelOpsClient,
    user: str,
    head: str,
    confirm: Callable[[], bool],
) -> int:
    emit = partial(_emit, client.token)
    try:
        _require_clean_tree(plan.root)
        _print_report(plan, client.token)
        if plan.failed:
            raise ApplyRefused("the dry run reported failures; nothing written")
        if not _USER_RE.fullmatch(user):
            raise ApplyRefused(
                "MODELOPS_USER must be set to [A-Za-z0-9._@-]{1,64}", 2
            )
        refusals = [
            *_protected_writes(plan),
            *_unmanaged_references(plan),
            *_name_collisions(plan),
        ]
        if refusals:
            raise ApplyRefused("; ".join(refusals))
        writes = _ordered_writes(plan)
        if not writes:
            emit("nothing to write")
            return 0
        for method, entry in writes:
            emit(_write_line(method, entry))
        if not confirm():
            raise ApplyRefused("confirmation not given; nothing written")
        _require_clean_tree(plan.root)
        if _git_head(plan.root) != head:
            raise ApplyRefused(
                "HEAD changed while waiting for the confirmation; "
                "nothing written"
            )
        _write_all(plan, writes, client=client, user=user, head=head)
    except ApplyRefused as exc:
        emit(f"apply: {exc}", err=True)
        return exc.code
    except ModelOpsApiError as exc:
        emit(f"apply: {exc}", err=True)
        return 1
    except RenderError as exc:
        emit(f"apply: {'; '.join(exc.problems)}", err=True)
        return 1
    return 0


@cli.command()
@click.option(
    "--env-file",
    "env_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--apply",
    "apply_",
    is_flag=True,
    help="Write the changes to ModelOps after a typed confirmation.",
)
def sync(env_file: Path, apply_: bool) -> None:
    client = ModelOpsClient.from_env(os.environ)
    env = load_env(env_file)
    if apply_:
        head = _git_head(_MODELOPS_ROOT)
        code = execute(
            plan(_MODELOPS_ROOT, env, client),
            client=client,
            user=os.environ.get("MODELOPS_USER", ""),
            head=head,
            confirm=_tty_confirm,
        )
    else:
        code = _sync(client, env, _MODELOPS_ROOT)
    if code:
        raise click.exceptions.Exit(code)


if __name__ == "__main__":
    cli()
