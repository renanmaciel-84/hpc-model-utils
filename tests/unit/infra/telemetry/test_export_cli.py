"""ADR-073: the export entry point, its exit codes, streams and temp files.

The success, usage and cleanup tests call `main` in-process. The isolation
tests run the module in a child interpreter, as `python -I -m` does, and
read whether pyarrow was loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NoReturn

import pyarrow.parquet as pq
import pytest

from hpc_model_utils.infra.telemetry import export
from hpc_model_utils.infra.telemetry.export import (
    ReadWarning,
    discover,
    main,
    read_header,
)
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADER_FIELDS,
    SAMPLES_FILE,
    SUMMARY_FILE,
    WORKSPACE_FILE,
    staging_file_name,
)
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    header,
    sample,
    walk,
    write_staging,
)

MODULE: Final = "hpc_model_utils.infra.telemetry.export"
PROG: Final = "hpcmu-telemetry-export"
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1

SCRIPT: Final = """\
import json
import runpy
import sys

sys.argv = {argv!r}
code = None
try:
    runpy.run_module({module!r}, run_name="__main__")
except SystemExit as exc:
    code = exc.code
print(json.dumps({{"code": code, "pyarrow": "pyarrow" in sys.modules}}))
"""


def _argv(staging: Path, out: Path, *jobs: str) -> list[str]:
    argv = [
        "--staging",
        str(staging),
        "--out",
        str(out),
        "--run-id",
        "run-1",
        "--model",
        "fake",
        "--tool-version",
        "9.9.9",
    ]
    for job in jobs:
        argv += ["--job", job]
    return argv


def _write_valid(directory: Path, head: Mapping[str, JsonValue]) -> Path:
    """A builder file of head with three samples and one walk."""
    directory.mkdir(parents=True, exist_ok=True)
    return write_staging(
        directory,
        head,
        [
            sample(head, 0),
            sample(head, 1),
            walk(head, ts=ORIGIN_MS + 60_000),
            sample(head, 2),
        ],
    )


def _three_files(staging: Path) -> Path:
    """model-1000-r0-0, model-1000-r0-1 and finalize-1001-r0-0."""
    _write_valid(staging, header(nodes=2))
    _write_valid(staging, header(node_index=1, nodes=2))
    _write_valid(staging, header("finalize", "1001"))
    return staging


def _restarts(out: Path) -> list[int]:
    restarts: list[int] = (
        pq.read_table(out / SAMPLES_FILE).column("restart").to_pylist()
    )
    return restarts


def _summary_warnings(out: Path) -> list[dict[str, JsonValue]]:
    warnings: list[dict[str, JsonValue]] = json.loads(
        (out / SUMMARY_FILE).read_text()
    )["warnings"]
    return warnings


def _bad_header(restart: int, message: str) -> dict[str, JsonValue]:
    return {
        "code": "bad_header",
        "phase": "model",
        "job_id": "1000",
        "restart": restart,
        "node_index": 0,
        "count": 1,
        "message": message,
    }


def test_main_success_writes_the_three_files_summary_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging = _three_files(tmp_path / "staging")
    out = tmp_path / "out"
    renamed: list[str] = []
    real_replace = os.replace

    def record(src: Path, dst: Path) -> None:
        renamed.append(dst.name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", record)

    code = main(_argv(staging, out, "model:1000", "finalize:1001"))

    captured = capsys.readouterr()
    assert code == 0
    assert (captured.out, captured.err) == ("", "")
    assert sorted(p.name for p in out.iterdir()) == sorted(EXPORT_FILES)
    assert renamed == list(EXPORT_FILES)
    assert pq.ParquetFile(out / SAMPLES_FILE).metadata.num_rows == 9
    assert pq.ParquetFile(out / WORKSPACE_FILE).metadata.num_rows == 3
    jobs = json.loads((out / SUMMARY_FILE).read_text())["jobs"]
    assert [(job["phase"], job["job_id"]) for job in jobs] == [
        ("model", "1000"),
        ("finalize", "1001"),
    ]


def _child(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the module as `__main__` in a child; its stdout is the probe line."""
    script = SCRIPT.format(argv=["export", *argv], module=MODULE)
    return subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _probe(proc: subprocess.CompletedProcess[str]) -> dict[str, JsonValue]:
    """The probe line, which must be the only line of the child's stdout."""
    assert proc.returncode == 0, proc.stderr
    (line,) = proc.stdout.splitlines()
    probe: dict[str, JsonValue] = json.loads(line)
    return probe


@pytest.mark.parametrize("tree", ["empty", "other_job", "missing"])
def test_main_isolation_no_staging_file_exits_3_without_pyarrow(
    tmp_path: Path, tree: str
) -> None:
    staging = tmp_path / "staging"
    if tree != "missing":
        staging.mkdir()
    if tree == "other_job":
        _write_valid(staging, header(job_id="1234"))
    out = tmp_path / "out"

    proc = _child(_argv(staging, out, "model:1000"))

    assert _probe(proc) == {"code": 3, "pyarrow": False}
    assert proc.stderr.splitlines() == [
        f"{PROG}: no staging file for the given jobs"
    ]
    assert not out.exists()


def test_main_isolation_no_valid_header_exits_1_without_pyarrow(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    name = staging_file_name("model", "1000", 0, 0)
    (staging / name).write_bytes(b'{"kind": "other"}\n')
    out = tmp_path / "out"

    proc = _child(_argv(staging, out, "model:1000"))

    assert _probe(proc) == {"code": 1, "pyarrow": False}
    assert proc.stderr.splitlines() == [f"{PROG}: no valid staging header"]
    assert not out.exists()


def test_main_isolation_unreadable_staging_exits_1_without_pyarrow(
    tmp_path: Path,
) -> None:
    # A regular file where the staging directory should be: discover's
    # OSError is a failure, reported before pyarrow is ever needed.
    staging = tmp_path / "staging"
    staging.write_bytes(b"")
    out = tmp_path / "out"

    proc = _child(_argv(staging, out, "model:1000"))

    assert _probe(proc) == {"code": 1, "pyarrow": False}
    (line,) = proc.stderr.splitlines()
    assert line.startswith(f"{PROG}: NotADirectoryError: ")
    assert not out.exists()


def test_main_isolation_success_loads_pyarrow(tmp_path: Path) -> None:
    # The positive control: the probe does see pyarrow once it is loaded.
    staging = tmp_path / "staging"
    _write_valid(staging, header())
    out = tmp_path / "out"

    proc = _child(_argv(staging, out, "model:1000"))

    assert _probe(proc) == {"code": 0, "pyarrow": True}
    assert proc.stderr == ""
    assert sorted(p.name for p in out.iterdir()) == sorted(EXPORT_FILES)


@pytest.mark.parametrize(
    "jobs",
    [
        pytest.param((), id="no_job"),
        pytest.param(("model",), id="no_job_id"),
        pytest.param(("other:1000",), id="unknown_phase"),
        pytest.param(("model:10a0",), id="non_digit_job_id"),
        pytest.param(("model:1000", "model:1000"), id="duplicate"),
        pytest.param(("model:",), id="empty_job_id"),
        pytest.param((":1000",), id="empty_phase"),
        pytest.param(("model:1000:1",), id="extra_field"),
        pytest.param(("model:١٠٠٠",), id="non_ascii_digits"),
    ],
)
def test_main_usage_error_exits_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    jobs: tuple[str, ...],
) -> None:
    staging = tmp_path / "staging"
    _write_valid(staging, header())
    out = tmp_path / "out"

    with pytest.raises(SystemExit) as info:
        main(_argv(staging, out, *jobs))

    captured = capsys.readouterr()
    assert info.value.code == 2
    assert f"usage: {PROG}" in captured.err
    assert captured.out == ""
    assert not out.exists()


@pytest.mark.parametrize(
    ("target", "error", "line"),
    [
        pytest.param(
            "write_workspace",
            OSError("disk full\nretry"),
            f"{PROG}: OSError: disk full retry",
            id="write_workspace_oserror",
        ),
        pytest.param(
            "render_summary",
            ValueError("Out of range float values are not JSON compliant"),
            f"{PROG}: ValueError: Out of range float values are not JSON "
            "compliant",
            id="render_summary_valueerror",
        ),
    ],
)
def test_main_cleanup_failure_leaves_out_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
    error: Exception,
    line: str,
) -> None:
    staging = _three_files(tmp_path / "staging")
    out = tmp_path / "out"

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(export, target, fail)

    code = main(_argv(staging, out, "model:1000", "finalize:1001"))

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err == f"{line}\n"
    assert list(out.iterdir()) == []


def test_main_cleanup_rename_failure_removes_only_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Decision 3: a rename failing after samples.parquet is in place leaves
    # that final name, never summary.json, and no temp.
    staging = _three_files(tmp_path / "staging")
    out = tmp_path / "out"
    real_replace = os.replace

    def flaky(src: Path, dst: Path) -> None:
        if dst.name == WORKSPACE_FILE:
            raise OSError("rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)

    code = main(_argv(staging, out, "model:1000", "finalize:1001"))

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err == f"{PROG}: OSError: rename failed\n"
    assert sorted(p.name for p in out.iterdir()) == [SAMPLES_FILE]
    assert not (out / SUMMARY_FILE).exists()


def test_main_unreadable_staging_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "staging"
    staging.write_bytes(b"")
    out = tmp_path / "out"

    code = main(_argv(staging, out, "model:1000"))

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    (line,) = captured.err.splitlines()
    assert line.startswith(f"{PROG}: NotADirectoryError: ")
    assert not out.exists()


_INTEGER_FIELDS: Final = tuple(
    field.name for field in HEADER_FIELDS if field.json_type == "integer"
)


@pytest.mark.parametrize("value", [_INT64_MAX + 1, _INT64_MIN - 1])
@pytest.mark.parametrize("name", _INTEGER_FIELDS)
def test_read_header_int64_rejects_any_integer_field_outside_int64(
    tmp_path: Path, name: str, value: int
) -> None:
    path = write_staging(tmp_path, header())
    path.write_bytes(json.dumps({**header(), name: value}).encode() + b"\n")

    (head,) = [read_header(c) for c in discover(tmp_path, {("model", "1000")})]

    assert head.header is None
    assert head.warnings == (
        ReadWarning("bad_header", f"header {name} outside int64", 1),
    )


def test_main_int64_header_past_the_edge_is_skipped_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "staging"
    _write_valid(staging, header())
    _write_valid(staging, header(restart=1, nodes=_INT64_MAX + 1))
    out = tmp_path / "out"

    code = main(_argv(staging, out, "model:1000"))

    captured = capsys.readouterr()
    assert code == 0
    assert (captured.out, captured.err) == ("", "")
    assert _restarts(out) == [0, 0, 0]
    assert _summary_warnings(out) == [
        _bad_header(1, "header nodes outside int64")
    ]


def test_main_int64_header_at_the_edge_is_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "staging"
    _write_valid(staging, header())
    _write_valid(staging, header(restart=1, nodes=_INT64_MAX))
    out = tmp_path / "out"

    code = main(_argv(staging, out, "model:1000"))

    captured = capsys.readouterr()
    assert code == 0
    assert (captured.out, captured.err) == ("", "")
    assert _restarts(out) == [0, 0, 0, 1, 1, 1]
    (job,) = json.loads((out / SUMMARY_FILE).read_text())["jobs"]
    assert [node["header"]["nodes"] for node in job["nodes"]] == [
        1,
        _INT64_MAX,
    ]
    assert "bad_header" not in {w["code"] for w in _summary_warnings(out)}


def test_main_int64_guardian_case_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ticket-043's guardian: two restarts whose headers claim 10**4300 - 1
    # nodes, within json.loads's digit limit, once made render_summary
    # raise. Both are now bad_header files, and the valid restart 2 is
    # exported.
    staging = tmp_path / "staging"
    for restart in (0, 1):
        _write_valid(staging, header(restart=restart, nodes=10**4300 - 1))
    _write_valid(staging, header(restart=2))
    out = tmp_path / "out"

    code = main(_argv(staging, out, "model:1000"))

    captured = capsys.readouterr()
    assert code == 0
    assert (captured.out, captured.err) == ("", "")
    assert _restarts(out) == [2, 2, 2]
    assert _summary_warnings(out) == [
        _bad_header(0, "header nodes outside int64"),
        _bad_header(1, "header nodes outside int64"),
    ]
