"""ADR-050 lint and behavior tests for the v2 Task scripts (ticket-062).

The lint (``check_scripts``) reads each ``tasks/*.sh`` whose slug does not
start with ``v1-`` and enforces the quoted-heredoc parameter idiom as a
closed set of line forms. Its messages name the file, the line and the rule,
never the offending text. The behavioral tests render each script the way
ModelOps does (``@@env:`` at apply time, then ``PrepareCommand``) and run it
under real bash against a stub CLI that records its argv.

``ensure-tools`` is the one script that does not end in the hpc-model-utils
call: its last three lines are the ``exec bash -s`` line, the
``@@script:ensure-tools.sh@@`` line and the heredoc terminator (ticket-063b).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from deploy.modelops.render import load_env, render_task
from tests.support.script_harness import (
    EXECUTION_HASH,
    EXECUTION_ID,
    EXECUTION_ID_PARAMETER,
    SHA,
    prepare_command,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"
TASKS = MODELOPS / "tasks"

SLUGS = (
    "create-workdir",
    "fetch-executables",
    "fetch-inputs",
    "extract-sanitize",
    "preprocess",
    "run",
    "result-upload",
    "remove-workdir",
    "cancel-run",
    "ingest-offline",
)
ENSURE_SLUG = "ensure-tools"
ENSURE_SLUGS = (ENSURE_SLUG, "ensure-utils")
COBRE_SLUGS = ("cobre-run",)
CLI_SLUGS = tuple(
    slug for slug in SLUGS if slug not in ("create-workdir", "remove-workdir")
)
AWS_SLUGS = frozenset(
    {"fetch-executables", "fetch-inputs", "result-upload", "ingest-offline"}
)

PARAMETER_VARIABLES = {
    "modelName": "MODEL",
    "rootPath": "ROOT_PATH",
    "path": "WORKDIR",
    "jobId": "JOB_ID",
    "inputFile": "INPUT_FILE",
    "outputFile": "OUTPUT_FILE",
    "cutFile": "CUT_FILE",
    "parentPath": "PARENT_PATH",
    "coreCount": "CORES",
    "queue": "QUEUE",
    "modelVersion": "MODEL_VERSION",
    "jobTimeoutHours": "JOB_HOURS",
    "maxCoresPerNode": "MAX_CORES",
    "mpichPath": "MPICH_PATH",
    "slurmPath": "SLURM_PATH",
    "outputsBucket": "OUTPUTS_BUCKET",
    "versionsBucket": "VERSIONS_BUCKET",
    "awsRegion": "AWS_REGION",
    "utilsToolDir": "UTILS_DIR",
    "synthesisToolDir": "SYNTHESIS_DIR",
    "utilsAppVersion": "UTILS_TAG",
    "utilsAppSha": "UTILS_SHA",
    "synthesisAppVersion": "SYNTHESIS_TAG",
    "synthesisAppSha": "SYNTHESIS_SHA",
    "CurrentExecution.ExecutionName": "EXECUTION_NAME",
    "CurrentExecution.ExecutionHash": "EXECUTION_HASH",
}
BRIDGE_TAG_LINE = "BRIDGE_TAG=${MODEL_VERSION%%-*}"
# slug -> (variable, line): the one line of that Task that derives the
# variable from a validated capture. No other Task may use the variable.
DERIVED: dict[str, tuple[str, str]] = {
    "ensure-utils": ("BRIDGE_TAG", BRIDGE_TAG_LINE),
}
_BASE = ("modelName", "rootPath", "path")
CAPTURED: dict[str, tuple[str, ...]] = {
    "create-workdir": ("modelName", "rootPath"),
    "fetch-executables": (
        *_BASE,
        "versionsBucket",
        "modelVersion",
        "awsRegion",
        "utilsToolDir",
    ),
    "fetch-inputs": (
        *_BASE,
        "inputFile",
        "parentPath",
        "awsRegion",
        "utilsToolDir",
    ),
    "extract-sanitize": (*_BASE, "utilsToolDir"),
    "preprocess": (
        *_BASE,
        "CurrentExecution.ExecutionName",
        "utilsToolDir",
    ),
    "run": (
        *_BASE,
        "queue",
        "coreCount",
        "jobTimeoutHours",
        "mpichPath",
        "slurmPath",
        "utilsToolDir",
        "synthesisToolDir",
    ),
    "result-upload": (
        *_BASE,
        "outputsBucket",
        "awsRegion",
        "CurrentExecution.ExecutionHash",
        "utilsToolDir",
    ),
    "remove-workdir": _BASE,
    "cancel-run": (*_BASE, "jobId", "slurmPath", "utilsToolDir"),
    "ingest-offline": (
        *_BASE,
        "inputFile",
        "outputFile",
        "cutFile",
        "awsRegion",
        "utilsToolDir",
    ),
    ENSURE_SLUG: (
        "modelName",
        "utilsAppVersion",
        "synthesisAppVersion",
        "utilsAppSha",
        "synthesisAppSha",
    ),
    "ensure-utils": (
        "utilsAppVersion",
        "utilsAppSha",
        "modelVersion",
    ),
    "cobre-run": (
        *_BASE,
        "queue",
        "coreCount",
        "maxCoresPerNode",
        "jobTimeoutHours",
        "mpichPath",
        "slurmPath",
        "utilsToolDir",
        "synthesisToolDir",
    ),
}

FIXED_LINES = (
    "readonly HPCMU_EXECUTION_ID='{{CurrentExecution.ExecutionId}}'",
    "set -euo pipefail",
    "fail() { printf 'hpcmu-task: %s\\n' \"$1\" >&2; exit 2; }",
    '[[ "$HPCMU_EXECUTION_ID" =~ ^[A-Za-z0-9-]{1,64}$ ]]'
    " || fail 'invalid CurrentExecution.ExecutionId'",
)
DELIMITER = "HPCMU_{{CurrentExecution.ExecutionId}}"
EXPORT_REGION = 'export AWS_DEFAULT_REGION="$AWS_REGION"'
CHANGE_DIRECTORY = 'cd -- "$WORKDIR"'
TELEMETRY_OFF = "export HPCMU_TELEMETRY=off"
TELEMETRY_OFF_SLUGS = frozenset({"run", "cobre-run"})
MKTEMP = 'WORKDIR=$(mktemp -d -p "$ROOT_PATH" -t "${MODEL}_XXXXXX")'
ENSURE_TERMINATOR = "HPCMU_ENSURE_TOOLS_EOF"
ENSURE_SCRIPT_LINE = "@@script:ensure-tools.sh@@"
FINAL_LINES = {
    "create-workdir": "printf 'Created temporary dir %s\\n' \"$WORKDIR\"",
    "remove-workdir": 'rm -r -- "$WORKDIR"',
    ENSURE_SLUG: ENSURE_TERMINATOR,
    "ensure-utils": ENSURE_TERMINATOR,
}

_NAME = r"[A-Z][A-Z_]*"
_VARIABLES = "|".join(
    sorted(
        {
            *PARAMETER_VARIABLES.values(),
            *(variable for variable, _ in DERIVED.values()),
        }
    )
)
_OPENER = re.compile(
    rf"IFS= read -r -d '' ({_NAME}) <<'{re.escape(DELIMITER)}' \|\| :"
)
_BODY = re.compile(r"\{\{([A-Za-z][A-Za-z0-9.]*)\}\}")
_VARIABLE_USE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
_VALIDATION = re.compile(r"\[\[ (.+) \]\] \|\| fail '([^'$]*)'")
_UNSAFE_VALIDATION = re.compile(r"\$\(|`|;|[<>]\(| \]\] ")
_REFERENCE = rf"\$(?:(?:{_VARIABLES})(?![A-Za-z0-9_])|\{{(?:{_VARIABLES})\}})"
_QUOTED = rf'"(?:[^"\\`$]|{_REFERENCE})*"'
_LITERAL = r"[A-Za-z0-9_./:=-]+"
_SINGLE_QUOTED = r"'[^'$`\\]*'"
_EXEC_LINE = re.compile(
    rf'exec "\$UTILS_DIR/\.venv/bin/hpc-model-utils"'
    rf"(?: (?:{_QUOTED}|{_LITERAL}))+"
)
_ENSURE_EXEC_LINE = re.compile(
    rf"exec bash -s --(?: (?:{_QUOTED}|{_SINGLE_QUOTED}|{_LITERAL}))+"
    rf" <<'{ENSURE_TERMINATOR}'"
)
_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stderr redirection", re.compile(r"2>|&>|\|&")),
    ("backtick", re.compile(r"`")),
    (
        "command word",
        re.compile(r"\b(?:eval|source|echo|ls|curl|wget|pip)\b|(?<!\.)\bgit\b"),
    ),
    ("clone layout", re.compile(r"hpc-model-utils/venv/")),
    (
        "AWS key",
        re.compile(
            r"awsKeyId|awsSecretKey|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY"
        ),
    ),
)


def _validation_message(line: str) -> str | None:
    match = _VALIDATION.fullmatch(line)
    if match is None or _UNSAFE_VALIDATION.search(match.group(1)):
        return None
    return match.group(2)


def _is_validation(line: str) -> bool:
    return _validation_message(line) is not None


def _refers_to(line: str, variable: str) -> bool:
    return re.search(rf"\$\{{?{variable}(?![A-Za-z0-9_])", line) is not None


@dataclass(frozen=True)
class _Assignment:
    variable: str
    parameter: str
    end: int


def _find_captures(
    where: str, lines: list[str], errors: list[str]
) -> tuple[list[_Assignment], set[int]]:
    assignments: list[_Assignment] = []
    inside: set[int] = set()
    index = len(FIXED_LINES)
    while index < len(lines):
        opener = _OPENER.fullmatch(lines[index])
        if opener is None:
            index += 1
            continue
        variable = opener.group(1)
        block = lines[index : index + 4]
        body = _BODY.fullmatch(block[1]) if len(block) == 4 else None
        strip = f"{variable}=${{{variable}%$'\\n'}}"
        if body is None or block[2] != DELIMITER or block[3] != strip:
            errors.append(
                f"{where}:{index + 1}: placement: malformed capture block"
            )
            index += 1
            continue
        parameter = body.group(1)
        if PARAMETER_VARIABLES.get(parameter) != variable:
            errors.append(
                f"{where}:{index + 1}: capture: parameter is not captured"
                " into its variable"
            )
        assignments.append(_Assignment(variable, parameter, index + 3))
        inside.update(range(index, index + 4))
        index += 4
    return assignments, inside


def _check_validation_order(
    where: str,
    lines: list[str],
    inside: set[int],
    assignments: list[_Assignment],
    errors: list[str],
) -> None:
    ordinary = [
        (index, line)
        for index, line in enumerate(lines)
        if index >= len(FIXED_LINES) and index not in inside
    ]
    for item in assignments:
        message = f"invalid {item.parameter}"
        own = [
            i for i, line in ordinary if _validation_message(line) == message
        ]
        uses = [
            i
            for i, line in ordinary
            if _refers_to(line, item.variable) and not _is_validation(line)
        ]
        early = [
            i
            for i, line in ordinary
            if i < item.end and _refers_to(line, item.variable)
        ]
        if early:
            errors.append(
                f"{where}:{early[0] + 1}: validation-order: {item.variable}"
                " is referenced before its capture"
            )
        if not own:
            errors.append(
                f"{where}: validation-order: {item.parameter} has no"
                " validation line"
            )
        elif uses and max(own) > min(uses):
            errors.append(
                f"{where}:{max(own) + 1}: validation-order: validation line"
                f" for {item.parameter} comes after the first use of"
                f" {item.variable}"
            )


def lint_script(slug: str, text: str) -> list[str]:
    where = f"tasks/{slug}.sh"
    if not text.endswith("\n"):
        return [f"{where}: fixed-line: no final newline"]
    lines = text[:-1].split("\n")
    errors: list[str] = []

    for number, expected in enumerate(FIXED_LINES, start=1):
        if lines[number - 1 : number] != [expected]:
            errors.append(f"{where}:{number}: fixed-line: line differs")

    first = text.find("{{")
    if first == -1 or "\n" in text[:first]:
        errors.append(
            f"{where}: placement: first template reference is not on line 1"
        )

    assignments, inside = _find_captures(where, lines, errors)
    if slug == "create-workdir":
        mktemp = [i for i, line in enumerate(lines) if line == MKTEMP]
        assignments += [_Assignment("WORKDIR", "path", i) for i in mktemp]
    _check_validation_order(where, lines, inside, assignments, errors)

    seen: set[str] = set()
    for item in assignments:
        if item.variable in seen:
            errors.append(f"{where}: capture: {item.variable} captured twice")
        seen.add(item.variable)

    defined = {item.variable for item in assignments} | {"HPCMU_EXECUTION_ID"}
    allowed_extra = {MKTEMP} if slug == "create-workdir" else set()
    if slug in TELEMETRY_OFF_SLUGS:
        allowed_extra.add(TELEMETRY_OFF)
    if slug in DERIVED:
        variable, derivation = DERIVED[slug]
        allowed_extra.add(derivation)
        if derivation in lines:
            defined.add(variable)
    last = len(lines) - 1
    for index, line in enumerate(lines):
        number = index + 1
        for rule, pattern in _FORBIDDEN:
            if pattern.search(line):
                errors.append(f"{where}:{number}: forbidden: {rule}")
        if ">&2" in line and index != 2:
            errors.append(f"{where}:{number}: forbidden: stderr redirection")
        if index < len(FIXED_LINES) or index in inside:
            continue
        if not line.startswith("# "):
            errors += [
                f"{where}:{number}: uncaptured: {name} is referenced but"
                " never captured"
                for name in dict.fromkeys(_VARIABLE_USE.findall(line))
                if name not in defined
            ]
        if "{{" in line:
            errors.append(
                f"{where}:{number}: placement: template reference outside"
                " a capture"
            )
        elif slug in ENSURE_SLUGS and index == last - 2:
            if _ENSURE_EXEC_LINE.fullmatch(line) is None:
                errors.append(
                    f"{where}:{number}: closed-set: line is not the"
                    " exec bash -s line"
                )
        elif slug in ENSURE_SLUGS and index == last - 1:
            if line != ENSURE_SCRIPT_LINE:
                errors.append(
                    f"{where}:{number}: closed-set: line is not the"
                    " @@script: line"
                )
        elif index == last:
            final = FINAL_LINES.get(slug)
            if final is not None:
                ok = line == final
            else:
                ok = _EXEC_LINE.fullmatch(line) is not None
            if not ok:
                errors.append(
                    f"{where}:{number}: closed-set: last line is not the"
                    " Task's final command"
                )
        elif not (
            line == ""
            or line.startswith("# ")
            or _is_validation(line)
            or line in (EXPORT_REGION, CHANGE_DIRECTORY)
            or line in allowed_extra
        ):
            errors.append(
                f"{where}:{number}: closed-set: line is not in the allowed set"
            )
    return errors


def check_scripts(tasks: Path) -> list[str]:
    scripts = [
        path
        for path in sorted(tasks.glob("*.sh"))
        if not path.stem.startswith("v1-")
    ]
    if not scripts:
        raise AssertionError("no v2 tasks/*.sh in the tree")
    return [
        error
        for path in scripts
        for error in lint_script(path.stem, path.read_text(encoding="utf-8"))
    ]


def lint_ensure_script(text: str) -> list[str]:
    return [
        f"scripts/ensure-tools.sh:{number}: terminator: line is the heredoc"
        " terminator"
        for number, line in enumerate(text.split("\n"), start=1)
        if line == ENSURE_TERMINATOR
    ]


def require(errors: list[str]) -> None:
    if errors:
        raise AssertionError("\n".join(errors))


@pytest.fixture
def tasks(tmp_path: Path) -> Path:
    shutil.copytree(TASKS, tmp_path / "tasks")
    return tmp_path / "tasks"


def _edit(path: Path, change: Callable[[list[str]], None]) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    change(lines)
    path.write_text("\n".join(lines), encoding="utf-8")


def _index(lines: list[str], predicate: Callable[[str], bool]) -> int:
    return next(i for i, line in enumerate(lines) if predicate(line))


def test_lint_real_tree_reports_nothing() -> None:
    require(check_scripts(TASKS))


def test_lint_covers_exactly_the_v2_scripts() -> None:
    v2 = {
        path.stem
        for path in TASKS.glob("*.sh")
        if not path.stem.startswith("v1-")
    }

    assert v2 == {*SLUGS, *ENSURE_SLUGS, *COBRE_SLUGS}


def test_lint_v1_baseline_is_not_checked(tasks: Path) -> None:
    for path in tasks.glob("*.sh"):
        if path.stem not in (
            "extract-sanitize",
            "v1-clone-simulprospec",
            "v1-upload-version",
        ):
            path.unlink()

    require(check_scripts(tasks))
    assert lint_script(
        "v1-clone-simulprospec",
        (tasks / "v1-clone-simulprospec.sh").read_text("utf-8"),
    )


def test_lint_without_a_v2_script_is_an_error(tasks: Path) -> None:
    for path in tasks.glob("[!v]*.sh"):
        path.unlink()

    with pytest.raises(AssertionError, match=r"no v2 tasks/\*\.sh"):
        check_scripts(tasks)


def _append_redirection(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += " 2>&1"


def _use_input_file_outside_heredoc(lines: list[str]) -> None:
    lines.insert(
        _index(lines, lambda s: s.startswith("exec ")), "touch {{inputFile}}"
    )


def _comment_with_reference(lines: list[str]) -> None:
    lines.insert(len(FIXED_LINES), "# {{path}}")


def _validate_after_use(lines: list[str]) -> None:
    moved = lines.pop(
        _index(lines, lambda s: s.endswith("fail 'invalid path'"))
    )
    lines.insert(_index(lines, lambda s: s.startswith("cd -- ")) + 1, moved)


def _echo_model(lines: list[str]) -> None:
    lines.insert(
        _index(lines, lambda s: s.startswith("exec ")), 'echo "$MODEL"'
    )


def _drop_first_line(lines: list[str]) -> None:
    del lines[0]


def _derive_before_validation(lines: list[str]) -> None:
    moved = lines.pop(_index(lines, lambda s: s == BRIDGE_TAG_LINE))
    lines.insert(
        _index(lines, lambda s: s.endswith("fail 'invalid modelVersion'")),
        moved,
    )


@pytest.mark.parametrize(
    ("script", "mutation", "message"),
    [
        (
            "extract-sanitize",
            _append_redirection,
            r"extract-sanitize\.sh:\d+: forbidden: stderr redirection",
        ),
        (
            "ensure-utils",
            _derive_before_validation,
            r"ensure-utils\.sh:\d+: validation-order: validation line"
            r" for modelVersion comes after the first use of MODEL_VERSION",
        ),
        (
            "fetch-inputs",
            _use_input_file_outside_heredoc,
            r"fetch-inputs\.sh:\d+: placement: template reference outside",
        ),
        (
            "extract-sanitize",
            _comment_with_reference,
            r"extract-sanitize\.sh:5: placement: template reference outside",
        ),
        (
            "extract-sanitize",
            _validate_after_use,
            r"extract-sanitize\.sh:\d+: validation-order: validation line"
            r" for path comes after the first use of WORKDIR",
        ),
        (
            "extract-sanitize",
            _echo_model,
            r"extract-sanitize\.sh:\d+: forbidden: command word",
        ),
        (
            "extract-sanitize",
            _drop_first_line,
            r"extract-sanitize\.sh:1: fixed-line: line differs",
        ),
    ],
)
def test_lint_acceptance_negative_is_reported(
    tasks: Path,
    script: str,
    mutation: Callable[[list[str]], None],
    message: str,
) -> None:
    _edit(tasks / f"{script}.sh", mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_scripts(tasks))


@pytest.mark.parametrize(
    "line",
    [
        "# 2>&1",
        "# 1>&2",
        "# &> out",
        "# |& tee",
        "# eval",
        "# source",
        "# echo",
        "# ls",
        "# curl",
        "# wget",
        "# git",
        "# pip",
        "# `id`",
        "# hpc-model-utils/venv/bin",
        "# awsKeyId",
        "# awsSecretKey",
        "# AWS_ACCESS_KEY_ID",
        "# AWS_SECRET_ACCESS_KEY",
    ],
)
def test_lint_forbidden_content_is_reported(tasks: Path, line: str) -> None:
    _edit(
        tasks / "extract-sanitize.sh",
        lambda lines: lines.insert(len(FIXED_LINES), line),
    )

    with pytest.raises(AssertionError, match=r"5: forbidden: "):
        require(check_scripts(tasks))


def _unquoted_argument(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += " $MODEL"


def _command_substitution_argument(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += ' "$(id)"'


def _escape_in_argument(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += ' "a\\b"'


def _unknown_variable_argument(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += ' "$HOME"'


def _chained_command(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s.startswith("exec "))] += "; id"


def _extra_stage(lines: list[str]) -> None:
    lines.insert(_index(lines, lambda s: s.startswith("exec ")), "id")


def _blank_last_line(lines: list[str]) -> None:
    lines.insert(len(lines) - 1, "")


def _unvalidated_command_substitution(lines: list[str]) -> None:
    lines.insert(
        _index(lines, lambda s: s.startswith("exec ")),
        "[[ $(id) ]] || fail 'invalid x'",
    )


def _two_tests_in_one_line(lines: list[str]) -> None:
    lines.insert(
        _index(lines, lambda s: s.startswith("exec ")),
        "[[ a ]] || id || fail 'invalid x'",
    )


def _derivation_of_another_task(lines: list[str]) -> None:
    lines.insert(
        _index(lines, lambda s: s.startswith("exec ")), BRIDGE_TAG_LINE
    )


@pytest.mark.parametrize(
    "mutation",
    [
        _unquoted_argument,
        _command_substitution_argument,
        _escape_in_argument,
        _unknown_variable_argument,
        _chained_command,
        _extra_stage,
        _blank_last_line,
        _unvalidated_command_substitution,
        _two_tests_in_one_line,
        _derivation_of_another_task,
    ],
)
def test_lint_closed_set_violation_is_reported(
    tasks: Path, mutation: Callable[[list[str]], None]
) -> None:
    _edit(tasks / "extract-sanitize.sh", mutation)

    with pytest.raises(AssertionError, match=r"closed-set: "):
        require(check_scripts(tasks))


def _drop_validation(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s.endswith("fail 'invalid modelName'"))]


def _drop_terminator(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s == DELIMITER)]


def _capture_into_wrong_variable(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s == "{{modelName}}")] = "{{queue}}"


def _use_before_capture(lines: list[str]) -> None:
    lines.insert(len(FIXED_LINES), '[[ "$MODEL" == "$MODEL" ]] || fail \'x\'')


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_drop_validation, r"validation-order: modelName has no validation"),
        (_drop_terminator, r"placement: malformed capture block"),
        (_capture_into_wrong_variable, r"capture: parameter is not captured"),
        (_use_before_capture, r"validation-order: MODEL is referenced before"),
    ],
)
def test_lint_capture_violation_is_reported(
    tasks: Path, mutation: Callable[[list[str]], None], message: str
) -> None:
    _edit(tasks / "extract-sanitize.sh", mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_scripts(tasks))


def _append_to_exec(argument: str) -> Callable[[list[str]], None]:
    def mutate(lines: list[str]) -> None:
        lines[_index(lines, lambda s: s.startswith("exec "))] += f" {argument}"

    return mutate


def _insert_before_exec(line: str) -> Callable[[list[str]], None]:
    def mutate(lines: list[str]) -> None:
        lines.insert(_index(lines, lambda s: s.startswith("exec ")), line)

    return mutate


def _drop_mktemp(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s == MKTEMP)]


def _drop_synthesis_sha_capture(lines: list[str]) -> None:
    start = _index(lines, lambda s: " SYNTHESIS_SHA <<" in s)
    del lines[start : start + 5]


def _drop_bridge_tag(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s == BRIDGE_TAG_LINE)]


@pytest.mark.parametrize(
    ("script", "mutation", "variable"),
    [
        ("extract-sanitize", _append_to_exec('"$UTILS_SHA"'), "UTILS_SHA"),
        ("run", _append_to_exec('"$SYNTHESIS_TAG"'), "SYNTHESIS_TAG"),
        ("extract-sanitize", _append_to_exec('"$CORES"'), "CORES"),
        ("fetch-inputs", _append_to_exec('"x${QUEUE}y"'), "QUEUE"),
        (
            "extract-sanitize",
            _insert_before_exec(EXPORT_REGION),
            "AWS_REGION",
        ),
        ("create-workdir", _drop_mktemp, "WORKDIR"),
        (ENSURE_SLUG, _drop_synthesis_sha_capture, "SYNTHESIS_SHA"),
        ("ensure-utils", _drop_bridge_tag, "BRIDGE_TAG"),
    ],
)
def test_lint_uncaptured_variable_is_reported(
    tasks: Path,
    script: str,
    mutation: Callable[[list[str]], None],
    variable: str,
) -> None:
    _edit(tasks / f"{script}.sh", mutation)

    with pytest.raises(
        AssertionError,
        match=rf"{script}\.sh:\d+: uncaptured: {variable} is referenced",
    ):
        require(check_scripts(tasks))


def test_lint_create_workdir_use_before_validation_is_reported(
    tasks: Path,
) -> None:
    def use_before_validation(lines: list[str]) -> None:
        lines.insert(_index(lines, lambda s: s == MKTEMP) + 1, CHANGE_DIRECTORY)

    _edit(tasks / "create-workdir.sh", use_before_validation)

    with pytest.raises(
        AssertionError,
        match=r"create-workdir\.sh:\d+: validation-order: validation line",
    ):
        require(check_scripts(tasks))


def test_ensure_lint_accepts_the_ensure_task() -> None:
    script = (TASKS / f"{ENSURE_SLUG}.sh").read_text(encoding="utf-8")

    require(lint_script(ENSURE_SLUG, script))


def _drop_script_line(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s == ENSURE_SCRIPT_LINE)]


def _continue_exec_line(lines: list[str]) -> None:
    index = _index(lines, lambda s: s.startswith("exec "))
    head, tail = lines[index].split(" --tool ", 1)
    lines[index : index + 1] = [f"{head} \\", f"  --tool {tail}"]


def _unquoted_tag(lines: list[str]) -> None:
    index = _index(lines, lambda s: s.startswith("exec "))
    lines[index] = lines[index].replace('"$UTILS_TAG"', "$UTILS_TAG")


def _variable_in_single_quotes(lines: list[str]) -> None:
    index = _index(lines, lambda s: s.startswith("exec "))
    lines[index] = lines[index].replace("'@@env:uvBin@@'", "'$HOME'")


def _rename_terminator(lines: list[str]) -> None:
    lines[_index(lines, lambda s: s == ENSURE_TERMINATOR)] += "_X"


def _drop_sha_validation(lines: list[str]) -> None:
    del lines[_index(lines, lambda s: s.endswith("fail 'invalid utilsAppSha'"))]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            _append_redirection,
            r"ensure-tools\.sh:\d+: forbidden: stderr redirection",
        ),
        (
            _append_redirection,
            r"ensure-tools\.sh:\d+: closed-set: line is not the exec bash -s",
        ),
        (
            _drop_script_line,
            r"ensure-tools\.sh:\d+: closed-set: line is not the @@script:",
        ),
        (
            _continue_exec_line,
            r"ensure-tools\.sh:\d+: closed-set: line is not the exec bash -s",
        ),
        (
            _continue_exec_line,
            r"ensure-tools\.sh:\d+: closed-set: line is not in the allowed",
        ),
        (
            _unquoted_tag,
            r"ensure-tools\.sh:\d+: closed-set: line is not the exec bash -s",
        ),
        (
            _variable_in_single_quotes,
            r"ensure-tools\.sh:\d+: closed-set: line is not the exec bash -s",
        ),
        (
            _rename_terminator,
            r"ensure-tools\.sh:\d+: closed-set: last line is not",
        ),
        (
            _drop_sha_validation,
            r"ensure-tools\.sh: validation-order: utilsAppSha has no validation",
        ),
    ],
)
def test_ensure_lint_violation_is_reported(
    tasks: Path, mutation: Callable[[list[str]], None], message: str
) -> None:
    _edit(tasks / f"{ENSURE_SLUG}.sh", mutation)

    with pytest.raises(AssertionError, match=message):
        require(check_scripts(tasks))


def test_ensure_lint_script_holds_no_terminator_line() -> None:
    script = (MODELOPS / "scripts" / f"{ENSURE_SLUG}.sh").read_text(
        encoding="utf-8"
    )

    require(lint_ensure_script(script))
    with pytest.raises(AssertionError, match=r"ensure-tools\.sh:2: terminator"):
        require(lint_ensure_script(f"#!x\n{ENSURE_TERMINATOR}\n{script}"))


@pytest.mark.parametrize("slug", [*SLUGS, *ENSURE_SLUGS, *COBRE_SLUGS])
def test_scripts_capture_the_requirement_1_parameters(slug: str) -> None:
    text = (TASKS / f"{slug}.sh").read_text(encoding="utf-8")
    names = set(re.findall(r"\{\{(.*?)\}\}", text)) - {EXECUTION_ID_PARAMETER}

    assert names == set(CAPTURED[slug])


@pytest.mark.parametrize("slug", [*SLUGS, *ENSURE_SLUGS, *COBRE_SLUGS])
def test_scripts_pass_bash_syntax_check(slug: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(TASKS / f"{slug}.sh")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert (result.returncode, result.stderr) == (0, "")


TASK_DOCUMENTS: dict[str, dict[str, Any]] = {
    "create-workdir": {
        "taskName": "Cria diretorio temporario para execucao",
        "description": "Cria diretorio temporario para execucao de modelo",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Cria diretório temporário para execução de modelo",
    },
    "fetch-executables": {
        "taskName": "Obtem executaveis dos modelos do S3",
        "description": "Obtem executaveis dos modelos do S3",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Obtem executaveis dos modelos do S3",
    },
    "fetch-inputs": {
        "taskName": "Obtem dados de entrada do S3",
        "description": "Copia os dados de entrada do S3 para o diretorio de execucao",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Copia os dados de entrada do S3 para o diretorio de execucao",
    },
    "extract-sanitize": {
        "taskName": "Extrai e trata encoding dos dados de entrada do modelo",
        "description": "Extrai e trata encoding dos dados de entrada do modelo",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Extrai e trata encoding dos dados de entrada do modelo",
    },
    "preprocess": {
        "taskName": "Preprocessamento especifico do modelo",
        "description": "Preprocessamento especifico do modelo",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Preprocessamento especifico do modelo",
    },
    "run": {
        "taskName": "Executa e acompanha modelo no SLURM",
        "description": "Executa o modelo através da submissão de um job ao SLURM e acompanha a",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Executa o modelo através da submissão de um job ao SLURM e acompanha a execução",
    },
    "result-upload": {
        "taskName": "Upload das saidas do modelo para o S3",
        "description": "Upload das saidas do modelo para o S3",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Upload das saidas do modelo para o S3",
    },
    "remove-workdir": {
        "taskName": "Remove diretorio temporario da execucao",
        "description": "Remove diretorio temporario da execucao",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Remove diretorio temporario da execucao",
    },
    "cancel-run": {
        "taskName": "Cancela job submetido na fila do SLURM",
        "description": "Cancela um job que foi submetido à fila do SLURM para uma rodada",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Cancela um job que foi submetido à fila do SLURM para uma rodada",
    },
    "ingest-offline": {
        "taskName": "Obtem dados de rodada para upload do S3",
        "description": "Obtem dados de rodada para upload do S3",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Obtem dados de rodada para upload do S3",
    },
    "ensure-tools": {
        "taskName": "Garante ferramentas versionadas",
        "description": "Instala ou reutiliza instalacoes imutaveis de ferramentas, uma por commit",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.0.0",
        "hidden": False,
        "observation": "Instala ou reutiliza instalacoes imutaveis de ferramentas, uma por commit",
    },
    "ensure-utils": {
        "taskName": "Garante hpc-model-utils e novomodelo-bridge versionados",
        "description": "Instala ou reutiliza as instalacoes imutaveis do hpc-model-utils e do novomodelo-bridge, uma por commit",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.1.0",
        "hidden": False,
        "observation": "Instala ou reutiliza as instalacoes imutaveis do hpc-model-utils e do novomodelo-bridge, uma por commit",
    },
    "cobre-run": {
        "taskName": "Executa e acompanha cobre no SLURM",
        "description": "Executa o cobre através da submissão de jobs ao SLURM e acompanha a execução",
        "scriptType": "BASH",
        "tags": [],
        "parameters": [],
        "version": "2.1.0",
        "hidden": False,
        "observation": "Executa o cobre através da submissão de jobs ao SLURM e acompanha a execução",
    },
}


@pytest.mark.parametrize("slug", [*SLUGS, *ENSURE_SLUGS, *COBRE_SLUGS])
def test_task_documents_carry_the_switch_names(slug: str) -> None:
    document = json.loads((TASKS / f"{slug}.json").read_text(encoding="utf-8"))

    assert document == TASK_DOCUMENTS[slug]


_ENV_TOKEN = re.compile(r"@@env:([A-Za-z][A-Za-z0-9]*)@@")


def _render_env(text: str, env: Mapping[str, str]) -> str:
    return _ENV_TOKEN.sub(lambda match: env[match.group(1)], text)


def test_prepare_command_replaces_each_name_everywhere() -> None:
    script = "{{a}} {{b}} {{a}}"

    assert prepare_command(script, {"a": "1", "b": "2"}) == "1 2 1"


def test_prepare_command_unknown_name_becomes_empty() -> None:
    assert prepare_command("x{{unknown}}y", {}) == "xy"


def test_prepare_command_value_names_expand_only_for_later_names() -> None:
    values = {"a": "{{b}}", "b": "B"}

    assert prepare_command("{{a}}\n{{b}}", values) == "B\nB"
    assert prepare_command("{{b}}\n{{a}}", values) == "B\n{{b}}"


def test_prepare_command_injected_execution_id_stays_literal() -> None:
    token = "{{" + EXECUTION_ID_PARAMETER + "}}"
    script = f"{token}\n{{{{x}}}}\n{token}"
    values = {EXECUTION_ID_PARAMETER: "GUID", "x": f"v\n{token}"}

    assert prepare_command(script, values) == f"GUID\nv\n{token}\nGUID"


def test_render_env_replaces_every_token() -> None:
    assert _render_env("a @@env:k@@ b @@env:k@@", {"k": "v"}) == "a v b v"


def test_render_env_unknown_name_is_an_error() -> None:
    with pytest.raises(KeyError, match="missing"):
        _render_env("@@env:missing@@", {"k": "v"})


EXECUTION_NAME = "PMO Água 'x' \"y\" $z $(id) ${HOME}"
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._/-]+")
_STUB = """#!{bash}
set -eu
{{ printf '%s\\0' "$PWD" "${{AWS_DEFAULT_REGION-unset}}"; printf '%s\\0' "$@"; }} \\
  > "{calls}/$$"
printf '%s' "${{HPCMU_TELEMETRY-unset}}" > "{telemetry}/$$"
"""


@dataclass(frozen=True)
class Call:
    cwd: str
    region: str
    argv: list[str]


@dataclass(frozen=True)
class Layout:
    base: Path
    env: dict[str, str]

    @property
    def root(self) -> Path:
        return self.base / "root"

    @property
    def tools(self) -> Path:
        return self.base / "tools"

    @property
    def sentinel(self) -> Path:
        return self.base / "sentinel"

    @property
    def calls(self) -> list[Call]:
        records = [
            path.read_bytes().decode("utf-8").split("\0")[:-1]
            for path in sorted((self.base / "calls").iterdir())
        ]
        return [Call(r[0], r[1], r[2:]) for r in records]

    @property
    def telemetry(self) -> list[str]:
        return [
            path.read_text(encoding="utf-8")
            for path in sorted((self.base / "telemetry").iterdir())
        ]

    def workdir(self, model: str = "newave") -> Path:
        return self.root / f"{model}_AbC123"

    def tool_dir(self, name: str) -> Path:
        return self.tools / name / SHA

    def values(self, model: str = "newave") -> dict[str, str]:
        return {
            EXECUTION_ID_PARAMETER: EXECUTION_ID,
            "modelName": model,
            "rootPath": self.env["rootPath"],
            "path": str(self.workdir(model)),
            "jobId": "4242",
            "inputFile": "s3://bkt-in/inputs/deck.zip",
            "outputFile": "s3://bkt-out/outputs/result.zip",
            "cutFile": "s3://bkt-cuts/cuts/cuts.zip",
            "parentPath": "",
            "coreCount": "16",
            "queue": self.env["queueNewave"],
            "modelVersion": "1.2.3_example",
            "jobTimeoutHours": "2",
            "mpichPath": self.env["mpichPath"],
            "slurmPath": self.env["slurmPath"],
            "outputsBucket": self.env["outputsBucket"],
            "versionsBucket": self.env["versionsBucket"],
            "awsRegion": self.env["awsRegion"],
            "utilsToolDir": str(self.tool_dir("hpc-model-utils")),
            "synthesisToolDir": str(self.tool_dir(f"sintetizador-{model}")),
            "CurrentExecution.ExecutionName": "execution name",
            "CurrentExecution.ExecutionHash": EXECUTION_HASH,
        }


@pytest.fixture
def layout() -> Iterator[Layout]:
    example = json.loads(
        (MODELOPS / "env" / "prd.example.json").read_text(encoding="utf-8")
    )
    bash = shutil.which("bash")
    assert bash is not None
    with tempfile.TemporaryDirectory(prefix="hpcmu-idiom-") as name:
        base = Path(name)
        assert _SAFE_PATH.fullmatch(str(base)), "TMPDIR is outside the regexes"
        sandbox = Layout(
            base,
            {
                **example["env"],
                "rootPath": str(base / "root"),
                "toolsRoot": str(base / "tools"),
            },
        )
        (base / "calls").mkdir()
        (base / "telemetry").mkdir()
        sandbox.root.mkdir()
        for model in ("newave", "decomp"):
            sandbox.workdir(model).mkdir()
        stub = sandbox.tool_dir("hpc-model-utils") / ".venv" / "bin"
        stub.mkdir(parents=True)
        (stub / "hpc-model-utils").write_text(
            _STUB.format(
                bash=bash, calls=base / "calls", telemetry=base / "telemetry"
            ),
            encoding="utf-8",
        )
        (stub / "hpc-model-utils").chmod(0o755)
        for model in ("newave", "decomp"):
            sandbox.tool_dir(f"sintetizador-{model}").mkdir(parents=True)
        yield sandbox


def _run(
    sandbox: Layout,
    slug: str,
    overrides: Mapping[str, str] | None = None,
    *,
    model: str = "newave",
    locale: str = "C",
    wrapped: bool = False,
    text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    values = {**sandbox.values(model), **(overrides or {})}
    if text is None:
        text = (TASKS / f"{slug}.sh").read_text("utf-8")
    script = prepare_command(_render_env(text, sandbox.env), values)
    if wrapped:
        script = "exec bash -c '" + script.replace("'", "'\"'\"'") + "'"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=sandbox.base,
        env={"PATH": os.environ["PATH"], "LC_ALL": locale},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _expected_argv(slug: str, v: Mapping[str, str]) -> list[str]:
    model = v["modelName"]
    version_uri = f"s3://{v['versionsBucket']}/versoes/{model}"
    upload_uri = f"s3://{v['outputsBucket']}/artifacts/{v['CurrentExecution.ExecutionHash']}"
    synthesis = f"{v['synthesisToolDir']}/.venv/bin/sintetizador-{model}"
    return {
        "fetch-executables": [
            "check_and_fetch_executables",
            model,
            f"{version_uri}/{v['modelVersion']}/",
        ],
        "fetch-inputs": [
            "check_and_fetch_inputs",
            model,
            v["inputFile"],
            "--parent-path",
            v["parentPath"],
            "--delete",
        ],
        "extract-sanitize": ["extract_sanitize_inputs", model],
        "preprocess": [
            "preprocess",
            model,
            "--execution-name",
            v["CurrentExecution.ExecutionName"],
        ],
        "run": [
            "run",
            model,
            v["queue"],
            v["coreCount"],
            "--max-job-time-hours",
            v["jobTimeoutHours"],
            "--mpich-path",
            v["mpichPath"],
            "--slurm-path",
            v["slurmPath"],
            "--synthesis-bin",
            synthesis,
        ],
        "result-upload": ["result_upload", model, upload_uri],
        "cancel-run": [
            "cancel_run",
            model,
            "--job-id",
            v["jobId"],
            "--slurm-path",
            v["slurmPath"],
        ],
        "ingest-offline": [
            "ingest_offline_run",
            model,
            v["inputFile"],
            v["outputFile"],
            v["cutFile"],
        ],
    }[slug]


def _assert_called_once_with(
    sandbox: Layout,
    slug: str,
    result: subprocess.CompletedProcess[str],
    overrides: Mapping[str, str] | None = None,
    model: str = "newave",
) -> None:
    values = {**sandbox.values(model), **(overrides or {})}
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert sandbox.calls == [
        Call(
            str(sandbox.workdir(model)),
            values["awsRegion"] if slug in AWS_SLUGS else "unset",
            _expected_argv(slug, values),
        )
    ]


def _assert_rejected(
    sandbox: Layout, result: subprocess.CompletedProcess[str], parameter: str
) -> None:
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"hpcmu-task: invalid {parameter}\n"
    assert sandbox.calls == []
    assert not sandbox.sentinel.exists()


@pytest.mark.parametrize("model", ["newave", "decomp"])
@pytest.mark.parametrize("slug", CLI_SLUGS)
def test_behavior_cli_argv_equals_the_requirement_1_command(
    layout: Layout, slug: str, model: str
) -> None:
    result = _run(layout, slug, model=model)

    _assert_called_once_with(layout, slug, result, model=model)


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_behavior_execution_name_arrives_as_one_verbatim_argument(
    layout: Layout, wrapped: bool, locale: str
) -> None:
    overrides = {"CurrentExecution.ExecutionName": EXECUTION_NAME}

    result = _run(
        layout, "preprocess", overrides, wrapped=wrapped, locale=locale
    )

    _assert_called_once_with(layout, "preprocess", result, overrides)
    assert layout.calls[0].argv[-1] == EXECUTION_NAME


@pytest.mark.parametrize("parent", ["''", '""', "s3://bkt-par/p/", ""])
def test_behavior_parent_path_arrives_verbatim(
    layout: Layout, parent: str
) -> None:
    result = _run(layout, "fetch-inputs", {"parentPath": parent})

    _assert_called_once_with(
        layout, "fetch-inputs", result, {"parentPath": parent}
    )
    assert layout.calls[0].argv[3:5] == ["--parent-path", parent]


def _hostile_tail(sandbox: Layout) -> str:
    return (
        f"k 'x' \"y\" $z $(touch {sandbox.sentinel}) ${{HOME}}"
        f" `touch {sandbox.sentinel}` \\n \\"
    )


@pytest.mark.parametrize(
    ("slug", "parameter"),
    [
        ("fetch-inputs", "inputFile"),
        ("fetch-inputs", "parentPath"),
        ("ingest-offline", "inputFile"),
        ("ingest-offline", "outputFile"),
        ("ingest-offline", "cutFile"),
    ],
)
def test_behavior_hostile_uri_text_is_data_not_shell(
    layout: Layout, slug: str, parameter: str
) -> None:
    overrides = {parameter: f"s3://bkt-x/{_hostile_tail(layout)}"}

    result = _run(layout, slug, overrides)

    _assert_called_once_with(layout, slug, result, overrides)
    assert not layout.sentinel.exists()


@pytest.mark.parametrize("bucket", ["bkt", "b"])
def test_behavior_forged_terminator_in_input_file_is_rejected(
    layout: Layout, bucket: str
) -> None:
    token = "{{" + EXECUTION_ID_PARAMETER + "}}"
    forged = f"s3://{bucket}/k\nHPCMU_{token}\ntouch {layout.sentinel}"

    result = _run(layout, "fetch-inputs", {"inputFile": forged})

    _assert_rejected(layout, result, "inputFile")


def test_behavior_outputs_bucket_differing_from_its_literal_is_rejected(
    layout: Layout,
) -> None:
    result = _run(layout, "result-upload", {"outputsBucket": "other-bucket"})

    _assert_rejected(layout, result, "outputsBucket")


@pytest.mark.parametrize(
    ("slug", "parameter"),
    [
        ("extract-sanitize", "rootPath"),
        ("run", "mpichPath"),
        ("run", "slurmPath"),
        ("fetch-executables", "versionsBucket"),
        ("ingest-offline", "awsRegion"),
    ],
)
def test_behavior_environment_constant_mismatch_is_rejected(
    layout: Layout, slug: str, parameter: str
) -> None:
    result = _run(layout, slug, {parameter: layout.env[parameter] + "x"})

    _assert_rejected(layout, result, parameter)


@pytest.mark.parametrize(
    "path",
    [
        "{base}/elsewhere/newave_AbC123",
        "{root}/nested/newave_AbC123",
        "{root}/../root/newave_AbC123",
        "{root}/decomp_AbC123",
        "{root}/newave_AbC12",
        "{root}/newave_AbC123/",
        "{root}/newave_AbC123\n",
        "{root}",
        "",
    ],
)
def test_behavior_remove_workdir_outside_root_deletes_nothing(
    layout: Layout, path: str
) -> None:
    (layout.base / "elsewhere" / "newave_AbC123").mkdir(parents=True)
    (layout.root / "nested" / "newave_AbC123").mkdir(parents=True)
    value = path.format(base=layout.base, root=layout.root)

    result = _run(layout, "remove-workdir", {"path": value})

    assert result.returncode == 2
    assert result.stderr == "hpcmu-task: invalid path\n"
    assert layout.workdir().is_dir()
    assert (layout.base / "elsewhere" / "newave_AbC123").is_dir()
    assert (layout.root / "decomp_AbC123").is_dir()


def test_behavior_remove_workdir_removes_the_validated_directory(
    layout: Layout,
) -> None:
    (layout.workdir() / "inner").mkdir()

    result = _run(layout, "remove-workdir")

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert not layout.workdir().exists()
    assert layout.root.is_dir()


@pytest.mark.parametrize("suffix", ["", "/"])
def test_behavior_create_workdir_prints_the_path_feeding_remove_workdir(
    layout: Layout, suffix: str
) -> None:
    layout.env["rootPath"] += suffix

    created = _run(layout, "create-workdir")

    printed = re.fullmatch(
        r"Created temporary dir (/\S+/newave_[A-Za-z0-9]{6})\n",
        created.stdout,
    )
    assert printed is not None
    assert created.returncode == 0 and created.stderr == ""
    workdir = Path(printed.group(1))
    assert workdir.parent == layout.root and workdir.is_dir()
    removed = _run(layout, "remove-workdir", {"path": str(workdir)})
    assert removed.returncode == 0
    assert not workdir.exists()


def test_behavior_create_workdir_missing_root_fails_on_stderr(
    layout: Layout,
) -> None:
    shutil.rmtree(layout.root)

    result = _run(layout, "create-workdir")

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr != ""


@pytest.mark.parametrize(
    ("slug", "parameter", "value"),
    [
        ("extract-sanitize", "modelName", "NEWAVE"),
        ("extract-sanitize", "modelName", "newave\n"),
        ("extract-sanitize", "modelName", "newave;id"),
        ("extract-sanitize", "modelName", ""),
        ("cancel-run", "jobId", "12a"),
        ("cancel-run", "jobId", "-1"),
        ("cancel-run", "jobId", "4242\n"),
        ("run", "coreCount", "0"),
        ("run", "coreCount", "016"),
        ("run", "coreCount", "1000000"),
        ("run", "coreCount", "6 4"),
        ("run", "coreCount", "64\n"),
        ("run", "queue", "a b"),
        ("run", "queue", ""),
        ("run", "queue", "a;b"),
        ("run", "queue", "q" * 65),
        ("fetch-executables", "modelVersion", ""),
        ("fetch-executables", "modelVersion", "1.0 beta"),
        ("fetch-executables", "modelVersion", "$x"),
        ("fetch-executables", "modelVersion", "v" * 65),
        ("run", "jobTimeoutHours", "0"),
        ("run", "jobTimeoutHours", "10000"),
        ("run", "jobTimeoutHours", "2h"),
        ("preprocess", "CurrentExecution.ExecutionName", ""),
        ("preprocess", "CurrentExecution.ExecutionName", "a\tb"),
        ("preprocess", "CurrentExecution.ExecutionName", "a\nb"),
        ("preprocess", "CurrentExecution.ExecutionName", "a\x07b"),
        ("result-upload", "CurrentExecution.ExecutionHash", "A" * 64),
        ("result-upload", "CurrentExecution.ExecutionHash", "a" * 63),
        ("result-upload", "CurrentExecution.ExecutionHash", "g" * 64),
        ("result-upload", "CurrentExecution.ExecutionHash", "a" * 64 + "\n"),
        ("fetch-inputs", "inputFile", "http://bkt/k"),
        ("fetch-inputs", "inputFile", "s3://BKT/k"),
        ("fetch-inputs", "inputFile", "s3://bkt"),
        ("fetch-inputs", "inputFile", "s3://bkt/"),
        ("fetch-inputs", "inputFile", "s3://bkt/k\x07"),
        ("fetch-inputs", "inputFile", ""),
        ("fetch-inputs", "parentPath", "'"),
        ("fetch-inputs", "parentPath", "''x"),
        ("fetch-inputs", "parentPath", "relative/path"),
        ("fetch-inputs", "parentPath", "s3://bkt-par/p\n"),
        ("ingest-offline", "outputFile", "s3://bkt/k\n"),
        ("ingest-offline", "cutFile", "bkt/k"),
    ],
)
def test_behavior_invalid_value_is_rejected(
    layout: Layout, slug: str, parameter: str, value: str
) -> None:
    result = _run(layout, slug, {parameter: value})

    _assert_rejected(layout, result, parameter)


_TOOL_CASES: dict[str, tuple[str, str, Callable[[Layout], str]]] = {
    "utils-outside-tools-root": (
        "extract-sanitize",
        "utilsToolDir",
        lambda sb: str(sb.tool_dir("hpc-model-utils")).replace(
            "/tools/", "/x/"
        ),
    ),
    "utils-short-sha": (
        "extract-sanitize",
        "utilsToolDir",
        lambda sb: str(sb.tool_dir("hpc-model-utils"))[:-1],
    ),
    "utils-uppercase-sha": (
        "extract-sanitize",
        "utilsToolDir",
        lambda sb: str(sb.tool_dir("hpc-model-utils"))[:-40] + "A" * 40,
    ),
    "utils-trailing-slash": (
        "extract-sanitize",
        "utilsToolDir",
        lambda sb: str(sb.tool_dir("hpc-model-utils")) + "/",
    ),
    "utils-dot-dot-root": (
        "extract-sanitize",
        "utilsToolDir",
        lambda sb: f"{sb.tools}/../tools/hpc-model-utils/{SHA}",
    ),
    "synthesis-other-model": (
        "run",
        "synthesisToolDir",
        lambda sb: str(sb.tool_dir("sintetizador-decomp")),
    ),
    "synthesis-outside-tools-root": (
        "run",
        "synthesisToolDir",
        lambda sb: str(sb.tool_dir("sintetizador-newave")).replace(
            "/tools/", "/x/"
        ),
    ),
    "synthesis-short-sha": (
        "run",
        "synthesisToolDir",
        lambda sb: str(sb.tool_dir("sintetizador-newave"))[:-1],
    ),
}


@pytest.mark.parametrize("case", _TOOL_CASES)
def test_behavior_tool_dir_outside_the_contract_is_rejected(
    layout: Layout, case: str
) -> None:
    slug, parameter, make = _TOOL_CASES[case]

    result = _run(layout, slug, {parameter: make(layout)})

    _assert_rejected(layout, result, parameter)


@pytest.mark.parametrize(
    ("slug", "overrides"),
    [
        ("cancel-run", {"jobId": ""}),
        ("fetch-executables", {"modelVersion": "32.15.1_coin-X.y"}),
        ("run", {"queue": "Queue_name-01", "coreCount": "999999"}),
        ("run", {"jobTimeoutHours": "9999"}),
        ("preprocess", {"CurrentExecution.ExecutionName": "x"}),
        ("preprocess", {"CurrentExecution.ExecutionName": "  spaced  "}),
        ("fetch-inputs", {"inputFile": "s3://a.b-c/k/with spaces/ç.zip"}),
        ("result-upload", {"CurrentExecution.ExecutionHash": "f" * 64}),
    ],
)
def test_behavior_valid_edge_values_pass_through(
    layout: Layout, slug: str, overrides: dict[str, str]
) -> None:
    result = _run(layout, slug, overrides)

    _assert_called_once_with(layout, slug, result, overrides)


@pytest.mark.parametrize(
    ("slug", "parameter"),
    [(slug, parameter) for slug in SLUGS for parameter in CAPTURED[slug]],
)
def test_behavior_every_captured_parameter_rejects_an_injected_line(
    layout: Layout, slug: str, parameter: str
) -> None:
    hostile = f"x\ntouch {layout.sentinel}"

    result = _run(layout, slug, {parameter: hostile})

    _assert_rejected(layout, result, parameter)


def test_behavior_execution_id_is_validated_before_any_capture(
    layout: Layout,
) -> None:
    result = _run(layout, "extract-sanitize", {EXECUTION_ID_PARAMETER: "a b"})

    assert result.returncode == 2
    assert result.stderr == "hpcmu-task: invalid CurrentExecution.ExecutionId\n"
    assert layout.calls == []


WIDENED_SLUGS = (
    "create-workdir",
    "remove-workdir",
    "fetch-executables",
    "fetch-inputs",
    "extract-sanitize",
    "result-upload",
    "cancel-run",
)


def test_widened_scripts_accept_cobre_in_both_closed_sets() -> None:
    for slug in WIDENED_SLUGS:
        text = (TASKS / f"{slug}.sh").read_text(encoding="utf-8")

        assert "^(newave|decomp|cobre)$" in text, slug
        assert "/(newave|decomp|cobre)_[A-Za-z0-9]{6}$" in text, slug
        assert "^(newave|decomp)$" not in text, slug


def test_behavior_cobre_create_workdir_prints_a_cobre_path(
    layout: Layout,
) -> None:
    result = _run(layout, "create-workdir", model="cobre")

    printed = re.fullmatch(
        rf"Created temporary dir ({re.escape(str(layout.root))}"
        r"/cobre_[A-Za-z0-9]{6})\n",
        result.stdout,
    )
    assert printed is not None
    assert result.returncode == 0 and result.stderr == ""
    assert Path(printed.group(1)).is_dir()


def test_behavior_cobre_remove_workdir_removes_the_directory(
    layout: Layout,
) -> None:
    (layout.workdir("cobre") / "inner").mkdir(parents=True)

    result = _run(layout, "remove-workdir", model="cobre")

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert not layout.workdir("cobre").exists()
    assert layout.root.is_dir()


@pytest.mark.parametrize(
    "slug", [slug for slug in CLI_SLUGS if slug in WIDENED_SLUGS]
)
def test_behavior_cobre_cli_argv_equals_the_command(
    layout: Layout, slug: str
) -> None:
    layout.workdir("cobre").mkdir()

    result = _run(layout, slug, model="cobre")

    _assert_called_once_with(layout, slug, result, model="cobre")


@pytest.mark.parametrize("slug", ["run", "preprocess", "ingest-offline"])
def test_behavior_cobre_is_rejected_by_the_closed_tasks(
    layout: Layout, slug: str
) -> None:
    result = _run(layout, slug, model="cobre")

    _assert_rejected(layout, result, "modelName")


ENSURE_PYTHON = "3.12.13"
ENSURE_UTILS_TAG = "v2.0.2"
ENSURE_UTILS_SHA = "a" * 40
ENSURE_SYNTHESIS_TAG = "v2.4.5"
ENSURE_SYNTHESIS_SHA = "b" * 40
ENSURE_MODEL_VERSION = "v0.18.0"
_RECORDING_STUB = """#!{bash}
printf '%s\\n' "${{0##*/}} $*" >> "{calls}"
exit 99
"""
_BASH_STUB = """#!{bash}
printf '%s\\0' "$@" > "{base}/bash-argv"
cat > "{base}/bash-stdin"
"""


@dataclass(frozen=True)
class EnsureSandbox:
    base: Path
    bash: str
    env_file: Path

    @property
    def root(self) -> Path:
        return self.base / "tools"

    @property
    def shims(self) -> Path:
        return self.base / "shims"

    @property
    def sentinel(self) -> Path:
        return self.base / "sentinel"

    def recorded(self) -> list[str]:
        return (self.base / "calls.log").read_text("utf-8").splitlines()

    def values(self, model: str = "newave") -> dict[str, str]:
        return {
            EXECUTION_ID_PARAMETER: EXECUTION_ID,
            "modelName": model,
            "utilsAppVersion": ENSURE_UTILS_TAG,
            "utilsAppSha": ENSURE_UTILS_SHA,
            "synthesisAppVersion": ENSURE_SYNTHESIS_TAG,
            "synthesisAppSha": ENSURE_SYNTHESIS_SHA,
            "modelVersion": ENSURE_MODEL_VERSION,
        }

    def tool_line(self, name: str, sha: str) -> str:
        return f"HPCMU_TOOL {name} {self.root / name / sha}\n"

    def record_tag(self, name: str, repo: str, tag: str, sha: str) -> None:
        record = self.root / ".tags" / name / tag
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            f"tool={name}\nrepo={repo}\ntag={tag}\nsha={sha}\n",
            encoding="utf-8",
        )

    def install(self, name: str, sha: str) -> None:
        interpreter = (
            self.root
            / ".python"
            / f"cpython-{ENSURE_PYTHON}-linux-x86_64-gnu"
            / "bin"
        )
        interpreter.mkdir(parents=True, exist_ok=True)
        (interpreter / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
        (interpreter / "python3").chmod(0o755)
        install = self.root / name / sha
        (install / ".venv" / "bin").mkdir(parents=True)
        (install / ".venv" / "bin" / "python").symlink_to(
            interpreter / "python3"
        )
        (install / ".venv" / "pyvenv.cfg").write_text(
            f"home = {interpreter}\nversion_info = {ENSURE_PYTHON}\n",
            encoding="utf-8",
        )
        (install / ".ready").write_text(
            f"tool={name}\nsha={sha}\npython={ENSURE_PYTHON}\n",
            encoding="utf-8",
        )


@pytest.fixture
def ensure_sandbox(tmp_path: Path) -> Iterator[EnsureSandbox]:
    example = json.loads(
        (MODELOPS / "env" / "prd.example.json").read_text(encoding="utf-8")
    )
    bash = shutil.which("bash")
    assert bash is not None
    with tempfile.TemporaryDirectory(prefix="hpcmu-ensure-") as name:
        base = Path(name).resolve()
        assert _SAFE_PATH.fullmatch(str(base)), "TMPDIR is outside the regexes"
        (base / "shims").mkdir()
        (base / "calls.log").touch()
        for shim in (base / "shims" / "git", base / "uv"):
            shim.write_text(
                _RECORDING_STUB.format(bash=bash, calls=base / "calls.log"),
                encoding="utf-8",
            )
            shim.chmod(0o755)
        example["env"].update(
            toolsRoot=str(base / "tools"), uvBin=str(base / "uv")
        )
        env_file = tmp_path / "env.json"
        env_file.write_text(json.dumps(example), encoding="utf-8")
        yield EnsureSandbox(base, bash, env_file)


def _run_ensure(
    sandbox: EnsureSandbox,
    overrides: Mapping[str, str] | None = None,
    *,
    model: str = "newave",
    slug: str = ENSURE_SLUG,
) -> subprocess.CompletedProcess[str]:
    task = render_task(slug, MODELOPS, load_env(sandbox.env_file))
    script = prepare_command(
        task["script"], {**sandbox.values(model), **(overrides or {})}
    )
    return subprocess.run(
        [sandbox.bash, "-c", script],
        cwd=sandbox.base,
        env={
            "PATH": f"{sandbox.shims}{os.pathsep}{os.environ['PATH']}",
            "LC_ALL": "C",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("model", ["newave", "decomp"])
def test_ensure_behavior_cache_hit_prints_the_tool_lines_and_runs_nothing(
    ensure_sandbox: EnsureSandbox, model: str
) -> None:
    synthesis = f"sintetizador-{model}"
    ensure_sandbox.install("hpc-model-utils", ENSURE_UTILS_SHA)
    ensure_sandbox.install(synthesis, ENSURE_SYNTHESIS_SHA)

    result = _run_ensure(ensure_sandbox, model=model)

    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        ensure_sandbox.tool_line("hpc-model-utils", ENSURE_UTILS_SHA)
        + ensure_sandbox.tool_line(synthesis, ENSURE_SYNTHESIS_SHA),
        "",
    )
    assert ensure_sandbox.recorded() == []


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock")
@pytest.mark.parametrize("model", ["newave", "decomp"])
def test_ensure_behavior_cache_miss_reaches_git_with_the_derived_repository(
    ensure_sandbox: EnsureSandbox, model: str
) -> None:
    ensure_sandbox.install("hpc-model-utils", ENSURE_UTILS_SHA)

    result = _run_ensure(ensure_sandbox, model=model)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        f"ensure-tools: sintetizador-{model}@{ENSURE_SYNTHESIS_SHA}:"
        " git ls-remote failed"
    )
    assert ensure_sandbox.recorded() == [
        f"git ls-remote https://github.com/rjmalves/sintetizador-{model}.git"
        f" refs/tags/{ENSURE_SYNTHESIS_TAG}"
        f" refs/tags/{ENSURE_SYNTHESIS_TAG}^{{}}"
    ]


@pytest.mark.parametrize("model", ["newave", "decomp"])
def test_ensure_behavior_hands_bash_the_derived_arguments_and_the_script(
    ensure_sandbox: EnsureSandbox, model: str
) -> None:
    stub = ensure_sandbox.shims / "bash"
    stub.write_text(
        _BASH_STUB.format(bash=ensure_sandbox.bash, base=ensure_sandbox.base),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    synthesis = f"sintetizador-{model}"

    result = _run_ensure(ensure_sandbox, model=model)

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
        synthesis,
        f"https://github.com/rjmalves/{synthesis}.git",
        ENSURE_SYNTHESIS_TAG,
        ENSURE_SYNTHESIS_SHA,
        synthesis,
    ]
    script = (MODELOPS / "scripts" / f"{ENSURE_SLUG}.sh").read_text("utf-8")
    stdin = (ensure_sandbox.base / "bash-stdin").read_text("utf-8")
    assert stdin == script + "\n"


def _assert_ensure_rejected(
    sandbox: EnsureSandbox,
    result: subprocess.CompletedProcess[str],
    parameter: str,
) -> None:
    assert (result.returncode, result.stdout, result.stderr) == (
        2,
        "",
        f"hpcmu-task: invalid {parameter}\n",
    )
    assert sandbox.recorded() == []
    assert not sandbox.sentinel.exists()


@pytest.mark.parametrize("parameter", CAPTURED[ENSURE_SLUG])
def test_ensure_injection_forged_terminator_is_rejected(
    ensure_sandbox: EnsureSandbox, parameter: str
) -> None:
    forged = (
        f"x\nHPCMU_{{{{{EXECUTION_ID_PARAMETER}}}}}"
        f"\ntouch {ensure_sandbox.sentinel}"
    )

    result = _run_ensure(ensure_sandbox, {parameter: forged})

    _assert_ensure_rejected(ensure_sandbox, result, parameter)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("modelName", "NEWAVE"),
        ("modelName", "newave\n"),
        ("utilsAppVersion", "v2.0"),
        ("utilsAppVersion", "2.0.1"),
        ("synthesisAppVersion", "main"),
        ("synthesisAppVersion", "v2.4.5-rc1"),
        ("utilsAppSha", "A" * 40),
        ("utilsAppSha", "a" * 39),
        ("synthesisAppSha", "a" * 41),
        ("synthesisAppSha", ""),
    ],
)
def test_ensure_behavior_invalid_value_is_rejected(
    ensure_sandbox: EnsureSandbox, parameter: str, value: str
) -> None:
    result = _run_ensure(ensure_sandbox, {parameter: value})

    _assert_ensure_rejected(ensure_sandbox, result, parameter)


def test_ensure_behavior_cobre_model_is_rejected(
    ensure_sandbox: EnsureSandbox,
) -> None:
    result = _run_ensure(ensure_sandbox, model="cobre")

    _assert_ensure_rejected(ensure_sandbox, result, "modelName")
