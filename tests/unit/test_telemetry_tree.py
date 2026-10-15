"""ADR-086/R61: tests/support/telemetry_tree.py builder coverage."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support import telemetry_tree
from tests.support.telemetry_tree import (
    FIXTURES_DIR,
    Bundle,
    load_bundle,
    materialize,
)

_SNAPSHOT_0: dict[str, Any] = {
    "files": {
        "proc/stat": "cpu  1 2\r\n",
        "proc/loadavg": "0.1 0.2 0.3 1/2 3\n",
        "sys/block/sda/stat": "1 2 3\n",
    },
    "dirs": ["sys/block/nvme0n1/device"],
}
_SNAPSHOT_1: dict[str, Any] = {
    "files": {
        "proc/stat": "cpu  5 6\r\n",
        "proc/loadavg": "0.4 0.5 0.6 1/2 3\n",
    },
    "dirs": ["sys/class/net/eth0"],
}
_DOCUMENT: dict[str, Any] = {
    "format": "hpcmu-telemetry-fixture",
    "version": 1,
    "source": "workstation",
    "snapshots": [_SNAPSHOT_0, _SNAPSHOT_1],
}


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _document(**changes: Any) -> dict[str, Any]:
    document = copy.deepcopy(_DOCUMENT)
    document.update(changes)
    return document


def _bundle(tmp_path: Path) -> Bundle:
    return load_bundle(_write(tmp_path / "bundle.json", _DOCUMENT))


def test_fixtures_dir_default_points_at_tests_fixtures_telemetry() -> None:
    expected = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry"

    assert FIXTURES_DIR == expected


def test_load_bundle_bare_name_resolves_under_fixtures_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "workstation.json", _DOCUMENT)
    monkeypatch.setattr(telemetry_tree, "FIXTURES_DIR", tmp_path)

    bundle = load_bundle("workstation")

    assert bundle.source == "workstation"


def test_load_bundle_valid_document_returns_frozen_records(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    assert len(bundle.snapshots) == 2
    assert bundle.snapshots[0].dirs == ("sys/block/nvme0n1/device",)
    assert bundle.snapshots[1].files["proc/stat"] == "cpu  5 6\r\n"
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.source = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        bundle.snapshots[0].files["x"] = "y"  # type: ignore[index]


@pytest.mark.parametrize(
    ("document", "key"),
    [
        pytest.param(_document(version=2), "version", id="version-2"),
        pytest.param(_document(version=True), "version", id="version-bool"),
        pytest.param(_document(format="other"), "format", id="format"),
        pytest.param(_document(source=""), "source", id="source-empty"),
        pytest.param(_document(source=3), "source", id="source-not-string"),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0]), "snapshots", id="one-snapshot"
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0] * 3),
            "snapshots",
            id="three-snapshots",
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0, "x"]),
            r"snapshots\[1\]",
            id="snapshot-not-object",
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0, {"files": {"a": 1}, "dirs": []}]),
            r"snapshots\[1\]\.files",
            id="file-value-not-string",
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0, {"files": [], "dirs": []}]),
            r"snapshots\[1\]\.files",
            id="files-not-mapping",
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0, {"files": {}, "dirs": "a"}]),
            r"snapshots\[1\]\.dirs",
            id="dirs-not-list",
        ),
        pytest.param(
            _document(snapshots=[_SNAPSHOT_0, {"files": {}, "dirs": [1]}]),
            r"snapshots\[1\]\.dirs",
            id="dirs-entry-not-string",
        ),
        pytest.param(["not", "an", "object"], "<document>", id="not-object"),
    ],
)
def test_load_bundle_invalid_document_raises_value_error_naming_key(
    tmp_path: Path, document: object, key: str
) -> None:
    path = _write(tmp_path / "bad.json", document)

    with pytest.raises(ValueError, match=rf"bad\.json.*{key}"):
        load_bundle(path)


@pytest.mark.parametrize(
    "relpath",
    ["../etc/passwd", "/proc/stat", "", "proc\\stat", "proc/../../x"],
)
@pytest.mark.parametrize("section", ["files", "dirs"])
def test_load_bundle_invalid_relpath_raises_value_error(
    tmp_path: Path, relpath: str, section: str
) -> None:
    snapshot = {"files": {"a": "x"}, "dirs": ["d"]}
    snapshot[section] = {relpath: "x"} if section == "files" else [relpath]
    path = _write(
        tmp_path / "bad.json",
        _document(snapshots=[_SNAPSHOT_0, snapshot]),
    )

    with pytest.raises(ValueError, match=r"bad\.json.*invalid relative path"):
        load_bundle(path)


def test_load_bundle_malformed_json_raises_value_error_naming_bundle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.json"):
        load_bundle(path)


def test_load_bundle_missing_file_propagates_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "absent.json")


def test_materialize_steps_from_snapshot_0_to_1_replaces_and_removes(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    root = tmp_path / "root"

    assert materialize(bundle, 0, root) == root
    assert (root / "sys/block/sda/stat").is_file()
    assert (root / "sys/block/nvme0n1/device").is_dir()

    assert materialize(bundle, 1, root) == root
    for relpath, text in bundle.snapshots[1].files.items():
        assert (root / relpath).read_bytes() == text.encode("utf-8")
    assert not (root / "sys/block/sda/stat").exists()
    assert (root / "sys/class/net/eth0").is_dir()


def test_materialize_text_keeps_exact_bytes_without_newline_translation(
    tmp_path: Path,
) -> None:
    root = materialize(_bundle(tmp_path), 0, tmp_path / "root")

    assert (root / "proc/stat").read_bytes() == b"cpu  1 2\r\n"


def test_materialize_overrides_replace_add_and_delete_files(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    root = materialize(
        bundle,
        0,
        tmp_path / "root",
        {"proc/stat": "x\n", "proc/loadavg": None, "proc/extra": "e\n"},
    )

    assert (root / "proc/stat").read_text(encoding="utf-8") == "x\n"
    assert not (root / "proc/loadavg").exists()
    assert (root / "proc/extra").read_text(encoding="utf-8") == "e\n"
    assert (root / "sys/block/sda/stat").is_file()


def test_materialize_override_does_not_outlive_its_call(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    root = tmp_path / "root"
    materialize(bundle, 0, root, {"proc/extra": "e\n", "proc/loadavg": None})

    materialize(bundle, 0, root)

    assert not (root / "proc/extra").exists()
    assert (root / "proc/loadavg").is_file()


def test_materialize_override_none_removes_file_not_written_by_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "proc").mkdir(parents=True)
    (root / "proc/extra").write_text("e\n", encoding="utf-8")

    materialize(_bundle(tmp_path), 0, root, {"proc/extra": None})

    assert not (root / "proc/extra").exists()


@pytest.mark.parametrize(
    "relpath", ["../escape", "/abs", "", "a\\b", "a/../../escape"]
)
def test_materialize_invalid_override_relpath_raises_before_writing(
    tmp_path: Path, relpath: str
) -> None:
    root = tmp_path / "root"

    with pytest.raises(ValueError, match="invalid relative path"):
        materialize(_bundle(tmp_path), 0, root, {relpath: "x"})

    assert not root.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("index", [-1, 2, 5])
def test_materialize_snapshot_index_out_of_range_raises_value_error(
    tmp_path: Path, index: int
) -> None:
    with pytest.raises(ValueError, match="snapshot index"):
        materialize(_bundle(tmp_path), index, tmp_path / "root")


def test_materialize_tampered_manifest_raises_without_deleting_outside_root(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / ".telemetry-tree-manifest").write_text(
        json.dumps(["../victim"]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid relative path"):
        materialize(_bundle(tmp_path), 0, root)

    assert victim.read_text(encoding="utf-8") == "keep\n"
