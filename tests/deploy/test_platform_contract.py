"""AM-001b/ADR-007 contracts between deploy/modelops and the platform encoder.

Three contracts: every parameter name a definition declares is a
``PLATFORM_IDENTIFIERS`` member; every ``{{x}}`` in a Task script resolves;
every RegexPattern parameter is covered by the ``TRIGGER_PATTERNS``
neutralizer. Each ``check_*`` function takes the ``deploy/modelops`` directory
so a mutated ``tmp_path`` copy is checked the same way as the real tree.
Messages name files and parameters, never values.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from hpc_model_utils.platform.encoding import (
    PLATFORM_IDENTIFIERS,
    TRIGGER_PATTERNS,
    neutralize,
)
from hpc_model_utils.platform.modelops import HookMethod

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELOPS = REPO_ROOT / "deploy" / "modelops"

_IDENTIFIER_SYNTAX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED = frozenset(
    {"currentexecution", *(method.value.casefold() for method in HookMethod)}
)
_REFERENCE = re.compile(r"\{\{(.*?)\}\}")
_EXECUTION_REFERENCE = re.compile(
    r"CurrentExecution\.(ExecutionId|ExecutionName|ExecutionHash)"
)

_SAMPLES: dict[str, tuple[str, ...]] = {
    "path": ("Created temporary dir /x/newave_AbC123",),
    "jobId": ("Submitted batch job 4242",),
    "utilsToolDir": (
        "HPCMU_TOOL hpc-model-utils /x/hpc-model-utils/" + "a" * 40,
    ),
    "synthesisToolDir": (
        "HPCMU_TOOL sintetizador-newave /x/sintetizador-newave/" + "a" * 40,
        "HPCMU_TOOL sintetizador-decomp /x/sintetizador-decomp/" + "a" * 40,
        "HPCMU_TOOL novomodelo-bridge /x/novomodelo-bridge/" + "a" * 40,
    ),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(errors: list[str]) -> None:
    if errors:
        raise AssertionError("\n".join(errors))


def _parameters(root: Path) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    declaring = 0
    for path in sorted(root.rglob("*.json")):
        doc = _load(path)
        if isinstance(doc, dict) and isinstance(doc.get("parameters"), list):
            declaring += 1
            rel = path.relative_to(root).as_posix()
            found += [(rel, param) for param in doc["parameters"]]
    if not declaring:
        raise AssertionError("no definition with a parameters list in the tree")
    return found


def check_identifiers(root: Path) -> list[str]:
    errors: list[str] = []
    for where, param in _parameters(root):
        name = param["name"]
        if name not in PLATFORM_IDENTIFIERS:
            errors.append(
                f"{where}: parameter {name} is not in PLATFORM_IDENTIFIERS"
            )
        if not _IDENTIFIER_SYNTAX.fullmatch(name):
            errors.append(
                f"{where}: parameter {name} is not a valid identifier"
            )
        if name.casefold() in _RESERVED:
            errors.append(
                f"{where}: parameter {name} collides with a reserved name"
            )
    return errors


def check_references(root: Path) -> list[str]:
    scripts = sorted((root / "tasks").glob("*.sh"))
    if not scripts:
        raise AssertionError("no tasks/*.sh in the tree")
    errors: list[str] = []
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        for match in _REFERENCE.finditer(text):
            reference = match.group(1)
            if (
                reference in PLATFORM_IDENTIFIERS
                or _EXECUTION_REFERENCE.fullmatch(reference)
            ):
                continue
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"tasks/{script.name}:{line}: {match.group(0)} is not a"
                " platform identifier"
            )
    return errors


def check_trigger_coverage(
    root: Path, samples: Mapping[str, Sequence[str]] = _SAMPLES
) -> list[str]:
    errors: list[str] = []
    for where, param in _parameters(root):
        pattern = param.get("regexPattern")
        if not pattern:
            continue
        name = param["name"]
        if name not in samples:
            errors.append(
                f"{where}: parameter {name} has a regexPattern but no"
                " _SAMPLES entry"
            )
            continue
        regex = re.compile(pattern, re.I)
        matching = [line for line in samples[name] if regex.search(line)]
        if not matching:
            errors.append(
                f"{where}: no _SAMPLES line for {name} matches its regexPattern"
            )
        errors += [
            f"{where}: a _SAMPLES line for {name} still matches its"
            " regexPattern after neutralize()"
            for line in matching
            if regex.search(neutralize(line))
        ]
    return errors


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("workflows", "tasks"):
        shutil.copytree(MODELOPS / name, tmp_path / name)
    return tmp_path


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    doc = _load(path)
    change(doc)
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _declare(root: Path, glob: str, param: dict[str, Any]) -> None:
    target = sorted(root.glob(glob))[0]
    _rewrite(target, lambda doc: doc["parameters"].append(param))


def test_check_identifiers_real_tree_reports_nothing() -> None:
    require(check_identifiers(MODELOPS))


@pytest.mark.parametrize("glob", ["workflows/*.json", "tasks/*.json"])
@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Path2x", r"parameter Path2x is not in PLATFORM_IDENTIFIERS"),
        ("bad-name", r"parameter bad-name is not a valid identifier"),
        ("CurrentExecution", r"CurrentExecution collides with a reserved"),
        ("SetSuccess", r"SetSuccess collides with a reserved"),
    ],
)
def test_check_identifiers_bad_parameter_name_is_reported(
    tree: Path, glob: str, name: str, message: str
) -> None:
    _declare(tree, glob, {"name": name})

    with pytest.raises(AssertionError, match=message):
        require(check_identifiers(tree))


def test_check_identifiers_covers_the_managed_ranking_workflow(
    tree: Path,
) -> None:
    _declare(tree, "workflows/ranqueamento.json", {"name": "Path2x"})

    with pytest.raises(
        AssertionError, match=r"workflows/ranqueamento.json: parameter Path2x"
    ):
        require(check_identifiers(tree))


def test_check_references_real_tree_reports_nothing() -> None:
    require(check_references(MODELOPS))


@pytest.mark.parametrize(
    "typo", ["{{inputFle}}", "{{Path}}", "{{CurrentExecution.ExecutionIdx}}"]
)
def test_check_references_typo_is_reported(tree: Path, typo: str) -> None:
    script = sorted((tree / "tasks").glob("*.sh"))[0]
    script.write_text(
        script.read_text(encoding="utf-8") + f"\necho {typo}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=re.escape(typo)):
        require(check_references(tree))


def test_check_trigger_coverage_real_tree_reports_nothing() -> None:
    require(check_trigger_coverage(MODELOPS))


def test_check_trigger_coverage_pattern_without_sample_is_reported(
    tree: Path,
) -> None:
    _declare(
        tree,
        "workflows/*.json",
        {"name": "newCapture", "regexPattern": r"new capture (\d+)"},
    )

    with pytest.raises(
        AssertionError, match=r"parameter newCapture has a regexPattern but no"
    ):
        require(check_trigger_coverage(tree))


def test_check_trigger_coverage_unmatched_sample_is_reported() -> None:
    samples = {**_SAMPLES, "path": ("nothing to capture",)}

    with pytest.raises(
        AssertionError, match=r"no _SAMPLES line for path matches"
    ):
        require(check_trigger_coverage(MODELOPS, samples))


def test_check_trigger_coverage_uncovered_pattern_is_reported(
    tree: Path,
) -> None:
    _declare(
        tree,
        "workflows/*.json",
        {"name": "plainCapture", "regexPattern": r"plain (\d+)"},
    )
    samples = {**_SAMPLES, "plainCapture": ("plain 7",)}

    with pytest.raises(
        AssertionError,
        match=r"_SAMPLES line for plainCapture still matches .* neutralize",
    ):
        require(check_trigger_coverage(tree, samples))


@pytest.mark.parametrize(
    "line", [line for lines in _SAMPLES.values() for line in lines]
)
def test_samples_are_trigger_lines_that_neutralize_defangs(line: str) -> None:
    assert any(pattern.search(line) for pattern in TRIGGER_PATTERNS)
    assert not any(
        pattern.search(neutralize(line)) for pattern in TRIGGER_PATTERNS
    )
