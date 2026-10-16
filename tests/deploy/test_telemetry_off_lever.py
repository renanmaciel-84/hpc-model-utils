"""ticket-061 contract for the telemetry off lever (ADR-066).

Each ``variants/telemetry-off/<slug>.sh`` is its live ``tasks/`` script plus
one ``export HPCMU_TELEMETRY=off`` line directly before ``cd -- "$WORKDIR"``.
The variants are never rendered; an operator commit copies them over
``tasks/``. These tests hold on the normal tree and on such a lever commit
alike, and fail as soon as a variant drifts from its live script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.deploy import test_task_script_idiom as idiom
from tests.deploy.test_cobre_workflow import _cobre_values
from tests.deploy.test_task_script_idiom import (
    CHANGE_DIRECTORY,
    TASKS,
    TELEMETRY_OFF,
    Layout,
    _run,
    lint_script,
)

# pytest registers a module attribute as a fixture under that attribute's name
layout = idiom.layout

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = REPO_ROOT / "deploy" / "modelops" / "variants" / "telemetry-off"
SLUGS = ("run", "cobre-run")
CLOSED_SET = "closed-set: line is not in the allowed set"


def remove_off(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if line != TELEMETRY_OFF)


def _variant(slug: str) -> str:
    return (VARIANTS / f"{slug}.sh").read_text(encoding="utf-8")


def _live(slug: str) -> str:
    return (TASKS / f"{slug}.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize("slug", SLUGS)
def test_variant_is_the_live_script_plus_the_off_line(slug: str) -> None:
    variant = _variant(slug)
    lines = variant.split("\n")
    off = [index for index, line in enumerate(lines) if line == TELEMETRY_OFF]
    change = [
        index for index, line in enumerate(lines) if line == CHANGE_DIRECTORY
    ]

    assert remove_off(variant) == remove_off(_live(slug))
    assert len(off) == 1
    assert len(change) == 1
    assert off[0] == change[0] - 1


@pytest.mark.parametrize("slug", SLUGS)
def test_variant_passes_the_task_script_lint(slug: str) -> None:
    assert lint_script(slug, _variant(slug)) == []


@pytest.mark.parametrize(
    ("slug", "model"),
    [
        pytest.param("run", "newave", id="run-newave"),
        pytest.param("run", "decomp", id="run-decomp"),
        pytest.param("cobre-run", "cobre", id="cobre-run"),
    ],
)
def test_variant_runs_the_live_argv_with_telemetry_off(
    layout: Layout, slug: str, model: str
) -> None:
    overrides: dict[str, str] | None = None
    if slug == "cobre-run":
        layout.workdir("cobre").mkdir(exist_ok=True)
        overrides = _cobre_values(layout)

    live = _run(layout, slug, overrides, model=model)
    live_argv = [call.argv for call in layout.calls]
    for name in ("calls", "telemetry"):
        for path in (layout.base / name).iterdir():
            path.unlink()
    result = _run(layout, slug, overrides, model=model, text=_variant(slug))

    assert (live.returncode, result.returncode) == (0, 0)
    assert [call.argv for call in layout.calls] == live_argv
    assert layout.telemetry == ["off"]


def test_off_line_is_rejected_for_other_tasks() -> None:
    lines = _live("fetch-inputs").split("\n")
    index = lines.index(CHANGE_DIRECTORY)
    lines.insert(index, TELEMETRY_OFF)

    errors = lint_script("fetch-inputs", "\n".join(lines))

    assert errors == [f"tasks/fetch-inputs.sh:{index + 1}: {CLOSED_SET}"]


def test_variants_directory_holds_exactly_the_two_scripts() -> None:
    assert sorted(p.name for p in VARIANTS.iterdir()) == [
        "cobre-run.sh",
        "run.sh",
    ]
