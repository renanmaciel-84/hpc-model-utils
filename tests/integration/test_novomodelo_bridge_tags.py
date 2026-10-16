"""The cobre bridge follows the run's ``modelVersion``: every option has a tag.

These tests need network access to GitHub, not LocalStack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.modelops.apply import (
    _PIN_REPOS,
    _SHA_RE,
    _TAG_TOOLS,
    _resolve_tag,
    _TagUnresolvedError,
)

pytestmark = pytest.mark.integration

_TOOL, _PARAM = _TAG_TOOLS["cobre"]
_COBRE = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "modelops"
    / "workflows"
    / "cobre.json"
)
_OPTIONS = next(
    p["options"]
    for p in json.loads(_COBRE.read_text(encoding="utf-8"))["parameters"]
    if p["name"] == _PARAM
)


@pytest.mark.parametrize("tag", _OPTIONS)
def test_bridge_tag_exists_for_every_cobre_model_version(tag: str) -> None:
    try:
        sha = _resolve_tag(_PIN_REPOS[_TOOL], tag)
    except _TagUnresolvedError as exc:
        pytest.fail(f"{_TOOL} {tag}: {exc.status}")
    assert _SHA_RE.fullmatch(sha)
