"""The cobre bridge follows the release of the run's ``modelVersion``: the
release of every option has a bridge tag.

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
    _release_tag,
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


@pytest.mark.parametrize("version", _OPTIONS)
def test_bridge_tag_exists_for_every_cobre_model_version(version: str) -> None:
    tag = _release_tag(version)
    assert tag is not None, f"{version} is not a vX.Y.Z[-suffix] version"
    try:
        sha = _resolve_tag(_PIN_REPOS[_TOOL], tag)
    except _TagUnresolvedError as exc:
        pytest.fail(f"{_TOOL} {tag}: {exc.status}")
    assert _SHA_RE.fullmatch(sha)
