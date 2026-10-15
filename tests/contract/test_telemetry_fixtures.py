"""Contract tests for sanitized telemetry fixture bundles (ADR-086, R61, R24)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from tests.support.telemetry_tree import load_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "telemetry"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-ipv4",
        re.compile(
            r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)"
            r"\.\d{1,3}\.\d{1,3}\b"
        ),
    ),
    ("ec2-hostname", re.compile(r"\bip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b")),
    ("fsx-id", re.compile(r"\bfs-[0-9a-f]{8,17}\b")),
    ("aws-dns", re.compile(r"\.amazonaws\.com\b")),
    (
        "pcluster-node",
        re.compile(
            r"\b[a-z0-9]+(?:-[a-z0-9]+)*-(?:dy|st)-[a-z0-9]+"
            r"(?:-[a-z0-9]+)*-\d+\b"
        ),
    ),
)


def _json_strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _json_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _json_strings(item)


def _scan_texts(path: Path, text: str) -> list[str]:
    # A JSON escape such as `\n` leaves a word character in front of the next
    # token, which hides it from `\b`; the decoded strings close that gap.
    if path.suffix != ".json":
        return [text]
    try:
        return [text, *_json_strings(json.loads(text))]
    except json.JSONDecodeError:
        return [text]


def scan_telemetry_fixtures(root: Path) -> list[str]:
    if not root.exists():
        return []
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        is_json = path.suffix == ".json"
        if not is_json and path.name != "README.md":
            violations.append(f"{rel}: unexpected file")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"{rel}: not utf-8")
            continue
        texts = _scan_texts(path, text)
        violations.extend(
            f"{rel}: {name}"
            for name, pattern in FORBIDDEN_PATTERNS
            if any(pattern.search(candidate) for candidate in texts)
        )
        if is_json:
            try:
                load_bundle(path)
            except ValueError as exc:
                violations.append(f"{rel}: invalid bundle: {exc}")
    return violations


def _write_bundle(root: Path, files: Mapping[str, str]) -> Path:
    snapshot = {"files": dict(files), "dirs": []}
    document = {
        "format": "hpcmu-telemetry-fixture",
        "version": 1,
        "source": "workstation",
        "snapshots": [snapshot, snapshot],
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bundle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_scan_telemetry_fixtures_real_tree_returns_empty() -> None:
    assert scan_telemetry_fixtures(TELEMETRY_FIXTURES) == []


def test_scan_telemetry_fixtures_missing_root_returns_empty(
    tmp_path: Path,
) -> None:
    assert scan_telemetry_fixtures(tmp_path / "absent") == []


@pytest.mark.parametrize(
    ("token", "name"),
    [
        ("10.0.3.17", "private-ipv4"),
        ("ip-10-0-3-17", "ec2-hostname"),
        ("fs-0123456789abcdef0", "fsx-id"),
        ("x.fsx.us-east-1.amazonaws.com", "aws-dns"),
        ("newave-r7a-48x-dy-newave-r7a-48x-2", "pcluster-node"),
    ],
)
def test_scan_telemetry_fixtures_forbidden_token_names_pattern(
    tmp_path: Path, token: str, name: str
) -> None:
    _write_bundle(tmp_path, {"proc/self/mountstats": f"device {token} x\n"})

    assert scan_telemetry_fixtures(tmp_path) == [f"bundle.json: {name}"]


def test_scan_telemetry_fixtures_fsx_mount_names_id_and_dns(
    tmp_path: Path,
) -> None:
    mount = "device fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com@tcp:/fsx"
    _write_bundle(tmp_path, {"proc/self/mountstats": mount})

    violations = scan_telemetry_fixtures(tmp_path)

    assert "bundle.json: fsx-id" in violations
    assert "bundle.json: aws-dns" in violations


def test_scan_telemetry_fixtures_token_after_json_escape_names_pattern(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path, {"proc/net/arp": "IP address\n10.0.3.17 0x1\n"})

    assert scan_telemetry_fixtures(tmp_path) == ["bundle.json: private-ipv4"]


def test_scan_telemetry_fixtures_documentation_placeholders_return_empty(
    tmp_path: Path,
) -> None:
    text = "192.0.2.10 198.51.100.7 node-0\n"
    _write_bundle(tmp_path, {"proc/self/mountstats": text})

    assert scan_telemetry_fixtures(tmp_path) == []


def test_scan_telemetry_fixtures_invalid_bundle_names_path_and_message(
    tmp_path: Path,
) -> None:
    (tmp_path / "bad.json").write_text('{"format": "other"}', encoding="utf-8")

    violations = scan_telemetry_fixtures(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("bad.json: invalid bundle: ")
    assert "'format' must be" in violations[0]


def test_scan_telemetry_fixtures_malformed_json_still_scans_raw_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "bad.json").write_text("{ 10.0.3.17", encoding="utf-8")

    violations = scan_telemetry_fixtures(tmp_path)

    assert violations[0] == "bad.json: private-ipv4"
    assert violations[1].startswith("bad.json: invalid bundle: ")
    assert len(violations) == 2


def test_scan_telemetry_fixtures_unexpected_file_names_path(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path, {"proc/loadavg": "0.00 0.00 0.00 1/1 1\n"})
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")

    assert scan_telemetry_fixtures(tmp_path) == ["notes.txt: unexpected file"]


def test_scan_telemetry_fixtures_readme_is_allowed(tmp_path: Path) -> None:
    _write_bundle(tmp_path, {"proc/loadavg": "0.00 0.00 0.00 1/1 1\n"})
    (tmp_path / "README.md").write_text("# Bundles\n", encoding="utf-8")

    assert scan_telemetry_fixtures(tmp_path) == []


def test_scan_telemetry_fixtures_forbidden_token_in_readme_names_pattern(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("host ip-10-0-3-17\n", encoding="utf-8")

    assert scan_telemetry_fixtures(tmp_path) == ["README.md: ec2-hostname"]


def test_scan_telemetry_fixtures_non_utf8_json_reports_without_raising(
    tmp_path: Path,
) -> None:
    (tmp_path / "bad.json").write_bytes(b'{"source": "\xff"}')

    assert scan_telemetry_fixtures(tmp_path) == ["bad.json: not utf-8"]


def test_scan_telemetry_fixtures_nested_file_reports_posix_relpath(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path / "sub", {"proc/loadavg": "ip-10-0-3-17\n"})

    assert scan_telemetry_fixtures(tmp_path) == [
        "sub/bundle.json: ec2-hostname"
    ]
