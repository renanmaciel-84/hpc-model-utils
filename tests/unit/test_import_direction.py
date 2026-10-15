"""ADR-002: enforce the layered import direction of src/hpc_model_utils/."""

from __future__ import annotations

import ast
import re
import shutil
import sys
import tomllib
from collections.abc import Iterable
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

PACKAGE_NAME = "hpc_model_utils"
PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
_DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

ALLOWED_EDGES: dict[str, frozenset[str]] = {
    "infra.telemetry": frozenset(),
    "infra": frozenset(),
    "core": frozenset({"infra", "infra.telemetry"}),
    "platform": frozenset({"core"}),
    "models": frozenset({"core", "infra"}),
    "cli": frozenset(
        {"core", "platform", "infra", "infra.telemetry", "models"}
    ),
}

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / PACKAGE_NAME
TESTS_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_LITERAL = "${CurrentExecution"
ALLOWED_LITERAL_MODULE = Path("platform") / "modelops.py"

RESTRICTED_IMPORTS: dict[str, frozenset[str]] = {
    "inewave": frozenset({"models/"}),
    "idecomp": frozenset({"models/"}),
    "pyarrow": frozenset({"infra/telemetry/export.py"}),
    f"{PACKAGE_NAME}.infra.telemetry.export": frozenset(),
}

TELEMETRY_TEST_ALLOWED_TOP_LEVEL = frozenset(sys.stdlib_module_names) | {
    "pytest",
    "hypothesis",
    "pyarrow",
}
TELEMETRY_TEST_ALLOWED_PREFIXES = (
    f"{PACKAGE_NAME}.infra.telemetry",
    "tests.support.telemetry_tree",
    "tests.unit.infra.telemetry",
    "tests.support.telemetry_staging",
)


def _layer_of(module_parts: tuple[str, ...]) -> str | None:
    matches = [
        layer
        for layer in ALLOWED_EDGES
        if module_parts[: layer.count(".") + 1] == tuple(layer.split("."))
    ]
    return max(matches, key=len, default=None)


def _package_parts(relative_path: Path) -> tuple[str, ...]:
    parts = relative_path.parts
    if relative_path.name == "__init__.py":
        return (PACKAGE_NAME, *parts[:-1])
    return (PACKAGE_NAME, *parts[:-1], relative_path.stem)


def _resolve_relative_base(
    own_package: tuple[str, ...], level: int
) -> tuple[str, ...]:
    return own_package[: len(own_package) - (level - 1)]


def _iter_import_targets(
    node: ast.Import | ast.ImportFrom,
    package_parts: tuple[str, ...],
    is_package: bool,
) -> list[tuple[str, ...]]:
    if isinstance(node, ast.Import):
        return [tuple(alias.name.split(".")) for alias in node.names]
    if node.level == 0:
        base = tuple(node.module.split(".")) if node.module else ()
    else:
        own_package = package_parts if is_package else package_parts[:-1]
        base = _resolve_relative_base(own_package, node.level)
        if node.module:
            base = (*base, *node.module.split("."))
    return [(*base, alias.name) for alias in node.names]


def _iter_python_files(src_root: Path) -> list[Path]:
    return sorted(src_root.rglob("*.py"))


def _parse(py_file: Path) -> ast.Module:
    return ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))


def _import_nodes(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def find_violations(src_root: Path) -> list[str]:
    violations: list[str] = []
    for py_file in _iter_python_files(src_root):
        relative = py_file.relative_to(src_root)
        package_parts = _package_parts(relative)
        importer_layer = _layer_of(package_parts[1:])
        if importer_layer is None:
            continue
        is_package = py_file.name == "__init__.py"
        for node in _import_nodes(_parse(py_file)):
            for target in _iter_import_targets(node, package_parts, is_package):
                if target[0] != PACKAGE_NAME:
                    continue
                target_layer = _layer_of(target[1:])
                if target_layer is None:
                    if importer_layer == "infra.telemetry":
                        violations.append(
                            f"{relative.as_posix()}: {importer_layer} -> "
                            f"{PACKAGE_NAME}"
                        )
                    continue
                if target_layer == importer_layer:
                    continue
                if target_layer not in ALLOWED_EDGES[importer_layer]:
                    module = ".".join(package_parts)
                    violations.append(
                        f"{module}: {importer_layer} -> {target_layer}"
                    )
    return violations


def find_forbidden_literal_usages(src_root: Path) -> list[str]:
    violations: list[str] = []
    for py_file in _iter_python_files(src_root):
        relative = py_file.relative_to(src_root)
        if relative == ALLOWED_LITERAL_MODULE:
            continue
        if FORBIDDEN_LITERAL in py_file.read_text(encoding="utf-8"):
            violations.append(str(relative))
    return violations


def find_restricted_imports(src_root: Path) -> list[str]:
    violations: list[str] = []
    for py_file in _iter_python_files(src_root):
        relative = py_file.relative_to(src_root)
        location = relative.as_posix()
        package_parts = _package_parts(relative)
        is_package = py_file.name == "__init__.py"
        for node in _import_nodes(_parse(py_file)):
            for target in _iter_import_targets(node, package_parts, is_package):
                name = ".".join(target)
                for prefix, allowed in RESTRICTED_IMPORTS.items():
                    if name != prefix and not name.startswith(f"{prefix}."):
                        continue
                    if not any(
                        location.startswith(entry)
                        if entry.endswith("/")
                        else location == entry
                        for entry in allowed
                    ):
                        violations.append(f"{location}: {prefix}")
    return violations


def find_telemetry_test_import_violations(paths: Iterable[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        for node in _import_nodes(_parse(path)):
            if isinstance(node, ast.ImportFrom) and node.level:
                names = ["." * node.level + (node.module or "")]
            else:
                names = [
                    ".".join(target)
                    for target in _iter_import_targets(node, (), False)
                ]
            for name in names:
                if name.split(".")[0] in TELEMETRY_TEST_ALLOWED_TOP_LEVEL:
                    continue
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in TELEMETRY_TEST_ALLOWED_PREFIXES
                ):
                    continue
                violations.append(f"{path}: {name}")
    return violations


def _tree_with_probe(tmp_path: Path, relative: str, body: str) -> Path:
    tree_root = tmp_path / PACKAGE_NAME
    shutil.copytree(SRC_ROOT, tree_root)
    (tree_root / relative).write_text(
        f"from __future__ import annotations\n\n{body}\n", encoding="utf-8"
    )
    return tree_root


def test_find_violations_real_tree_reports_none() -> None:
    assert find_violations(SRC_ROOT) == []


def test_find_violations_infra_imports_core_reports_edge(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "infra/_probe.py",
        "from hpc_model_utils.core import errors",
    )
    violations = find_violations(tree_root)
    assert any("infra -> core" in v for v in violations)


def test_find_violations_relative_import_core_to_platform_reports_edge(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path, "core/_probe.py", "from ..platform import x"
    )
    violations = find_violations(tree_root)
    assert any("core -> platform" in v for v in violations)


def test_find_violations_telemetry_imports_core_reports_edge(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "infra/telemetry/_probe.py",
        "from hpc_model_utils.core import errors",
    )
    violations = find_violations(tree_root)
    assert any("infra.telemetry -> core" in v for v in violations)


def test_find_violations_telemetry_imports_infra_reports_edge(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "infra/telemetry/_probe.py",
        "from hpc_model_utils.infra import errors",
    )
    violations = find_violations(tree_root)
    assert any("infra.telemetry -> infra" in v for v in violations)


def test_find_violations_core_imports_telemetry_reports_none(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "core/_probe.py",
        "from hpc_model_utils.infra.telemetry import schema",
    )
    assert find_violations(tree_root) == []


def test_find_violations_telemetry_imports_itself_reports_none(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "infra/telemetry/_probe.py",
        "from hpc_model_utils.infra.telemetry import schema",
    )
    assert find_violations(tree_root) == []


@pytest.mark.parametrize(
    "body",
    [
        "from hpc_model_utils import __version__",
        "import hpc_model_utils",
        "from ... import __version__",
    ],
)
def test_find_violations_telemetry_imports_package_root_reports_root(
    tmp_path: Path, body: str
) -> None:
    tree_root = _tree_with_probe(tmp_path, "infra/telemetry/_probe.py", body)
    violations = find_violations(tree_root)
    assert len(violations) == 1
    assert "infra.telemetry -> hpc_model_utils" in violations[0]


def test_find_forbidden_literal_usages_real_tree_reports_none() -> None:
    assert find_forbidden_literal_usages(SRC_ROOT) == []


def test_find_forbidden_literal_usages_outside_modelops_reports_file(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path, "core/_probe.py", "VALUE = '${CurrentExecution}/out'"
    )
    violations = find_forbidden_literal_usages(tree_root)
    assert str(Path("core") / "_probe.py") in violations


def test_find_restricted_imports_real_tree_reports_none() -> None:
    assert find_restricted_imports(SRC_ROOT) == []


def test_find_restricted_imports_outside_models_reports_file(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(tmp_path, "cli/_probe.py", "import inewave")
    assert "cli/_probe.py: inewave" in find_restricted_imports(tree_root)


def test_find_restricted_imports_pyarrow_outside_export_reports_file(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path, "cli/_probe.py", "import pyarrow.parquet"
    )
    assert "cli/_probe.py: pyarrow" in find_restricted_imports(tree_root)


def test_find_restricted_imports_pyarrow_in_export_reports_none(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path, "infra/telemetry/export.py", "import pyarrow.parquet"
    )
    assert find_restricted_imports(tree_root) == []


def test_find_restricted_imports_export_module_has_no_importer(
    tmp_path: Path,
) -> None:
    tree_root = _tree_with_probe(
        tmp_path,
        "core/_probe.py",
        "import hpc_model_utils.infra.telemetry.export",
    )
    assert find_restricted_imports(tree_root) == [
        "core/_probe.py: hpc_model_utils.infra.telemetry.export"
    ]


def test_find_telemetry_test_import_violations_real_tree_reports_none() -> None:
    paths = [
        *sorted((TESTS_ROOT / "unit" / "infra" / "telemetry").rglob("*.py")),
        *sorted((TESTS_ROOT / "support").glob("telemetry*.py")),
    ]
    assert find_telemetry_test_import_violations(paths) == []


def test_find_telemetry_test_import_violations_core_import_reports_module(
    tmp_path: Path,
) -> None:
    test_x = tmp_path / "test_x.py"
    test_x.write_text(
        "import pytest\nimport hpc_model_utils.core.workspace\n",
        encoding="utf-8",
    )
    assert find_telemetry_test_import_violations([test_x]) == [
        f"{test_x}: hpc_model_utils.core.workspace"
    ]


def test_find_telemetry_test_import_violations_relative_import_reports_it(
    tmp_path: Path,
) -> None:
    test_x = tmp_path / "test_x.py"
    test_x.write_text("from .. import helper\n", encoding="utf-8")
    assert find_telemetry_test_import_violations([test_x]) == [f"{test_x}: .."]


def test_find_telemetry_test_import_violations_allowed_imports_report_none(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "telemetry_helper.py"
    helper.write_text(
        "from __future__ import annotations\n"
        "import pytest\n"
        "import hpc_model_utils.infra.telemetry.schema\n"
        "from hpc_model_utils.infra import telemetry\n"
        "from tests.support import telemetry_tree\n",
        encoding="utf-8",
    )
    assert find_telemetry_test_import_violations([helper]) == []


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _type_checking_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If) and _is_type_checking_guard(node.test)
        ):
            continue
        for child in ast.walk(node):
            lineno = getattr(child, "lineno", None)
            if lineno is not None:
                lines.add(lineno)
    return lines


def find_runtime_third_party_top_level_names(src_root: Path) -> set[str]:
    names: set[str] = set()
    for py_file in _iter_python_files(src_root):
        tree = _parse(py_file)
        excluded = _type_checking_lines(tree)
        for node in _import_nodes(tree):
            if node.lineno in excluded:
                continue
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {PACKAGE_NAME}


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_dependency_names(pyproject_path: Path) -> set[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    names = set[str]()
    for dependency in data["project"]["dependencies"]:
        match = _DEPENDENCY_NAME_RE.match(dependency)
        assert match, f"could not parse dependency name from {dependency!r}"
        names.add(_normalize_distribution_name(match.group(1)))
    return names


def test_runtime_third_party_imports_equal_declared_dependencies() -> None:
    top_level_names = find_runtime_third_party_top_level_names(SRC_ROOT)
    distributions = packages_distributions()
    imported = {
        _normalize_distribution_name(distributions[name][0])
        for name in top_level_names
    }
    declared = declared_dependency_names(PYPROJECT_PATH)

    undeclared = imported - declared
    unused = declared - imported
    assert imported == declared, (
        f"undeclared runtime imports (add to [project].dependencies): "
        f"{sorted(undeclared)}; declared but unused dependencies (remove "
        f"from [project].dependencies): {sorted(unused)}"
    )
