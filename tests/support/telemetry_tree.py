"""ADR-086/R61: load telemetry fixture bundles and materialize snapshots.

A bundle is a JSON document holding two snapshots of the relevant `/proc`,
`/sys` and cgroup files, as text keyed by their path relative to `/`, plus
the directory markers that sysfs device links represent. `materialize`
writes one snapshot under a caller-owned root, so a fixture tree can stand
in for `/` in the sampler's injectable root. No file tree or symlink is ever
committed, and the builder knows nothing about which files a collector reads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry"

_FORMAT = "hpcmu-telemetry-fixture"
_VERSION = 1
_SNAPSHOT_COUNT = 2
_MANIFEST = ".telemetry-tree-manifest"


@dataclass(frozen=True, slots=True)
class Snapshot:
    files: Mapping[str, str]
    dirs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Bundle:
    source: str
    snapshots: tuple[Snapshot, ...]


def _check_relpath(relpath: str, where: str) -> None:
    if (
        not relpath
        or relpath.startswith("/")
        or "\\" in relpath
        or ".." in relpath.split("/")
    ):
        raise ValueError(f"{where}: invalid relative path {relpath!r}")


def _invalid(path: Path, key: str, expected: str) -> ValueError:
    return ValueError(f"{path}: '{key}' must be {expected}")


def _parse_snapshot(path: Path, index: int, raw: object) -> Snapshot:
    key = f"snapshots[{index}]"
    if not isinstance(raw, dict):
        raise _invalid(path, key, "an object")
    files = raw.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(text, str) for text in files.values()
    ):
        raise _invalid(path, f"{key}.files", "a mapping of strings to strings")
    dirs = raw.get("dirs")
    if not isinstance(dirs, list) or not all(
        isinstance(entry, str) for entry in dirs
    ):
        raise _invalid(path, f"{key}.dirs", "a list of strings")
    for relpath in (*files, *dirs):
        _check_relpath(relpath, f"{path}: {key}")
    return Snapshot(MappingProxyType(dict(files)), tuple(dirs))


def load_bundle(name_or_path: str | Path) -> Bundle:
    """Load and validate a fixture bundle.

    A `str` is a bare bundle name resolved to `FIXTURES_DIR/<name>.json`; a
    `Path` is used as is. Raises `ValueError` naming the bundle and the
    offending key on any violation; an `OSError` from reading propagates.
    """
    path = (
        name_or_path
        if isinstance(name_or_path, Path)
        else FIXTURES_DIR / f"{name_or_path}.json"
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _invalid(path, "<document>", "a JSON object")
    if data.get("format") != _FORMAT:
        raise _invalid(path, "format", repr(_FORMAT))
    version = data.get("version")
    if type(version) is not int or version != _VERSION:
        raise _invalid(path, "version", str(_VERSION))
    source = data.get("source")
    if not isinstance(source, str) or not source:
        raise _invalid(path, "source", "a non-empty string")
    raw = data.get("snapshots")
    if not isinstance(raw, list) or len(raw) != _SNAPSHOT_COUNT:
        raise _invalid(
            path, "snapshots", f"a list of exactly {_SNAPSHOT_COUNT}"
        )
    return Bundle(
        source,
        tuple(_parse_snapshot(path, i, item) for i, item in enumerate(raw)),
    )


def _read_manifest(manifest: Path) -> set[str]:
    try:
        text = manifest.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    relpaths: list[str] = json.loads(text)
    for relpath in relpaths:
        _check_relpath(relpath, str(manifest))
    return set(relpaths)


def materialize(
    bundle: Bundle,
    snapshot: int,
    root: Path,
    overrides: Mapping[str, str | None] | None = None,
) -> Path:
    """Write snapshot `snapshot` of `bundle` under `root` and return `root`.

    An override string replaces or adds a file and `None` deletes it. Files
    written by an earlier call on the same root and absent from this one are
    removed, tracked through the hidden `.telemetry-tree-manifest` at `root`.
    Raises `ValueError` for an index outside the bundle or an invalid relpath;
    an `OSError` from the filesystem propagates.
    """
    if not 0 <= snapshot < len(bundle.snapshots):
        raise ValueError(
            f"snapshot index {snapshot} outside 0..{len(bundle.snapshots) - 1}"
        )
    chosen = bundle.snapshots[snapshot]
    overrides = overrides or {}
    for relpath in (*chosen.files, *chosen.dirs):
        _check_relpath(relpath, f"snapshot {snapshot}")
    for relpath in overrides:
        _check_relpath(relpath, "override")
    merged = {**chosen.files, **overrides}
    wanted = {rel: text for rel, text in merged.items() if text is not None}

    manifest = root / _MANIFEST
    deleted = merged.keys() - wanted.keys()
    stale = (_read_manifest(manifest) - wanted.keys()) | deleted
    root.mkdir(parents=True, exist_ok=True)
    for relpath in stale:
        (root / relpath).unlink(missing_ok=True)
    manifest.write_text(json.dumps(sorted(wanted)), encoding="utf-8")
    for relpath in chosen.dirs:
        (root / relpath).mkdir(parents=True, exist_ok=True)
    for relpath, text in wanted.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")
    return root
