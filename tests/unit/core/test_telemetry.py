"""ADR-073/ADR-083/ADR-085/R47/R50/R85 tests for the export runner
(ticket-047).

Most tests run a fake interpreter script in place of ``sys.executable``, so
each child behaviour is exact. One test spawns the real exporter over builder
staging.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from hpc_model_utils.core import telemetry
from hpc_model_utils.core.state import (
    JobRecord,
    RunState,
    ToolInfo,
    current_tool,
)
from hpc_model_utils.core.telemetry import (
    Exported,
    ExportFailed,
    NothingToExport,
    export_telemetry,
    headline_items,
)
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.shell import ShellResult
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADLINE,
    HEADLINE_KEY_PREFIX,
    SUMMARY_FILE,
)
from tests.support.telemetry_staging import header, sample, write_staging

_LOGGER: Final = "hpc_model_utils.core.telemetry"
_SUMMARY_OK: Final = (
    '{"kind": "hpcmu.telemetry.summary", "schema_version": 1, '
    '"headline": {"job_ids": {}, "values": {%s}}}'
)
_FIND_OUT: Final = (
    'while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done\n'
    'mkdir -p "$out"'
)
_WRITE_PARQUET: Final = (
    ': > "$out/samples.parquet"\n: > "$out/workspace.parquet"'
)


def _ws(tmp_path: Path) -> Workspace:
    workspace = Workspace.at(tmp_path)
    workspace.ensure_layout()
    return workspace


def _state(*jobs: tuple[Phase, str]) -> RunState:
    return RunState(
        run_id="run-1",
        plugin="fake",
        tool=ToolInfo(name="hpc-model-utils", version="2.0.0"),
        jobs=tuple(
            JobRecord(phase, job_id, "2000-01-01T00:00:00Z", "log")
            for phase, job_id in jobs
        ),
    )


_TWO_JOBS: Final = ((Phase.MODEL, "1000"), (Phase.FINALIZE, "1001"))


def _fake_interpreter(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake-python"
    script.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _summary_script(tmp_path: Path, payload: bytes) -> str:
    source = tmp_path / "payload.json"
    source.write_bytes(payload)
    return _fake_interpreter(
        tmp_path,
        f'{_FIND_OUT}\n{_WRITE_PARQUET}\ncp "{source}" "$out/summary.json"',
    )


def _stale(ws: Workspace) -> Path:
    ws.telemetry_outputs_dir.mkdir(parents=True)
    (ws.telemetry_outputs_dir / "summary.json").write_text("stale")
    stale_tmp = ws.telemetry_outputs_dir / ".samples.parquet.tmp"
    stale_tmp.write_text("stale")
    return stale_tmp


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER and record.levelno == logging.WARNING
    ]


# Argv and the nothing paths.


def test_export_argv_and_nothing_on_exit_three(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    recorder = tmp_path / "argv.txt"
    interpreter = _fake_interpreter(
        tmp_path, f"printf '%s\\n' \"$0\" \"$@\" > '{recorder}'\nexit 3"
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            ws, _state(*_TWO_JOBS), interpreter=interpreter
        )

    assert result == NothingToExport()
    assert recorder.read_text(encoding="utf-8").splitlines() == [
        interpreter,
        "-I",
        "-m",
        "hpc_model_utils.infra.telemetry.export",
        "--staging",
        str(ws.telemetry_dir),
        "--out",
        str(ws.telemetry_outputs_dir),
        "--run-id",
        "run-1",
        "--model",
        "fake",
        "--tool-version",
        current_tool().version,
        "--job",
        "model:1000",
        "--job",
        "finalize:1001",
    ]
    assert [r.getMessage() for r in caplog.records if r.name == _LOGGER] == [
        "telemetry export: no staging file for the recorded jobs"
    ]


def test_export_nothing_without_jobs_spawns_no_child(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    recorder = tmp_path / "argv.txt"
    interpreter = _fake_interpreter(tmp_path, f": > '{recorder}'\nexit 0")
    stale = _stale(ws)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(ws, _state(), interpreter=interpreter)

    assert result == NothingToExport()
    assert not recorder.exists()
    assert stale.exists()
    assert (ws.telemetry_outputs_dir / SUMMARY_FILE).exists()
    assert [r.getMessage() for r in caplog.records if r.name == _LOGGER] == [
        "telemetry export skipped: no recorded job"
    ]


# Failure mapping.


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        pytest.param(
            "echo 'hpcmu-telemetry-export: boom' >&2\nexit 1",
            "hpcmu-telemetry-export: boom",
            id="stderr-line",
        ),
        pytest.param(
            "echo first\necho 'last line'\necho '   '\necho\nexit 2",
            "last line",
            id="last-non-blank-line",
        ),
        pytest.param("exit 7", "exit status 7", id="silent"),
        pytest.param("kill -KILL $$", "killed by signal 9", id="signal"),
    ],
)
def test_export_failure_mapping_of_the_child_exit(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    body: str,
    reason: str,
) -> None:
    interpreter = _fake_interpreter(tmp_path, body)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            _ws(tmp_path), _state(*_TWO_JOBS), interpreter=interpreter
        )

    assert result == ExportFailed(reason)
    assert _warnings(caplog) == [f"telemetry export failed: {reason}"]


def test_export_failure_mapping_of_a_timeout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    interpreter = _fake_interpreter(tmp_path, "sleep 30")

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            _ws(tmp_path),
            _state(*_TWO_JOBS),
            interpreter=interpreter,
            timeout_s=0.5,
        )
    elapsed = time.monotonic() - started

    assert result == ExportFailed("timed out after 0.5 s")
    assert elapsed < 10
    assert _warnings(caplog) == [
        "telemetry export failed: timed out after 0.5 s"
    ]


def test_export_failure_mapping_of_an_os_error_from_the_runner(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def start_fails(argv: Sequence[str], *, timeout: float) -> ShellResult:
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(telemetry, "shell_run", start_fails)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(_ws(tmp_path), _state(*_TWO_JOBS))

    assert isinstance(result, ExportFailed)
    assert "Resource temporarily unavailable" in result.reason
    assert _warnings(caplog) == [f"telemetry export failed: {result.reason}"]


def test_export_failure_mapping_of_a_missing_interpreter(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            _ws(tmp_path),
            _state(*_TWO_JOBS),
            interpreter=str(tmp_path / "no-such-python"),
        )

    assert isinstance(result, ExportFailed)
    assert result.reason.startswith("cannot execute")
    assert _warnings(caplog) == [f"telemetry export failed: {result.reason}"]


# Clearing and the exit-0 check.


def test_export_clear_removes_only_the_export_names(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.telemetry_outputs_dir.mkdir(parents=True)
    names = [*EXPORT_FILES, *(f".{name}.tmp" for name in EXPORT_FILES)]
    for name in (*names, "keep.txt", ".other.tmp"):
        (ws.telemetry_outputs_dir / name).write_text("stale")
    interpreter = _fake_interpreter(tmp_path, "exit 7")

    export_telemetry(ws, _state(*_TWO_JOBS), interpreter=interpreter)

    assert sorted(p.name for p in ws.telemetry_outputs_dir.iterdir()) == [
        ".other.tmp",
        "keep.txt",
    ]


def test_export_clear_failure_is_a_value_and_spawns_no_child(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    (ws.telemetry_outputs_dir / EXPORT_FILES[0]).mkdir(parents=True)
    recorder = tmp_path / "argv.txt"
    interpreter = _fake_interpreter(tmp_path, f": > '{recorder}'\nexit 0")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            ws, _state(*_TWO_JOBS), interpreter=interpreter
        )

    assert isinstance(result, ExportFailed)
    assert result.reason.startswith(f"cannot clear {ws.telemetry_outputs_dir}")
    assert not recorder.exists()
    assert _warnings(caplog) == [f"telemetry export failed: {result.reason}"]


def test_export_check_names_the_first_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    stale_tmp = _stale(ws)
    interpreter = _fake_interpreter(
        tmp_path, f"{_FIND_OUT}\n{_WRITE_PARQUET}\nexit 0"
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            ws, _state(*_TWO_JOBS), interpreter=interpreter
        )

    assert result == ExportFailed("export exited 0 without summary.json")
    assert not stale_tmp.exists()
    assert _warnings(caplog) == [
        "telemetry export failed: export exited 0 without summary.json"
    ]


@pytest.mark.parametrize(
    ("body", "missing"),
    [
        pytest.param("", "samples.parquet", id="nothing"),
        pytest.param(
            ': > "$out/samples.parquet"', "workspace.parquet", id="samples"
        ),
        pytest.param(
            f'{_WRITE_PARQUET}\n: > "$out/.summary.json.tmp"',
            "summary.json",
            id="temp-only",
        ),
        pytest.param(
            ': > "$out/samples.parquet"\nmkdir "$out/workspace.parquet"\n'
            ': > "$out/summary.json"',
            "workspace.parquet",
            id="directory",
        ),
    ],
)
def test_export_check_requires_three_regular_files(
    tmp_path: Path, body: str, missing: str
) -> None:
    interpreter = _fake_interpreter(tmp_path, f"{_FIND_OUT}\n{body}\nexit 0")

    result = export_telemetry(
        _ws(tmp_path), _state(*_TWO_JOBS), interpreter=interpreter
    )

    assert result == ExportFailed(f"export exited 0 without {missing}")


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses the mode bits")
def test_export_check_of_an_unsearchable_outputs_directory_is_a_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    interpreter = _fake_interpreter(
        tmp_path,
        f'{_FIND_OUT}\n{_WRITE_PARQUET}\n: > "$out/summary.json"\n'
        'chmod 000 "$out"',
    )

    try:
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            result = export_telemetry(
                ws, _state(*_TWO_JOBS), interpreter=interpreter
            )
    finally:
        ws.telemetry_outputs_dir.chmod(0o755)

    assert isinstance(result, ExportFailed)
    assert result.reason.startswith("cannot check samples.parquet: ")
    assert _warnings(caplog) == [f"telemetry export failed: {result.reason}"]


_UNREADABLE: Final = "unreadable summary.json"
_MALFORMED: Final = "malformed summary.json"


@pytest.mark.parametrize(
    ("payload", "prefix"),
    [
        pytest.param(b'{"kind": "x"}', _MALFORMED, id="foreign-kind"),
        pytest.param(b"[" * 100_000, _UNREADABLE, id="deep-nesting"),
        pytest.param(
            (_SUMMARY_OK % '"model.unknown": 1').encode(),
            _MALFORMED,
            id="unknown-key",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": true').encode(),
            _MALFORMED,
            id="bool-value",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": NaN').encode(),
            _UNREADABLE,
            id="nan-token",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": Infinity').encode(),
            _UNREADABLE,
            id="infinity-token",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": -Infinity').encode(),
            _UNREADABLE,
            id="negative-infinity-token",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": 1e999').encode(),
            _MALFORMED,
            id="overflowing-float",
        ),
        pytest.param(
            (_SUMMARY_OK % '"model.cpu_util_mean": -5').encode(),
            _MALFORMED,
            id="negative-value",
        ),
        pytest.param(
            (_SUMMARY_OK % f'"model.cpu_util_mean": {"9" * 400}').encode(),
            _MALFORMED,
            id="out-of-range-integer",
        ),
        pytest.param(
            (_SUMMARY_OK % f'"model.cpu_util_mean": {"9" * 5000}').encode(),
            _UNREADABLE,
            id="over-the-digit-limit",
        ),
        pytest.param(
            b'{"kind": "hpcmu.telemetry.summary", "schema_version": 2, '
            b'"headline": {"values": {}}}',
            _MALFORMED,
            id="foreign-schema-version",
        ),
        pytest.param(
            b'{"kind": "hpcmu.telemetry.summary", "schema_version": 1, '
            b'"headline": {"values": []}}',
            _MALFORMED,
            id="values-not-an-object",
        ),
        pytest.param(b"[]", _MALFORMED, id="not-an-object"),
        pytest.param(b"", _UNREADABLE, id="empty"),
    ],
)
def test_export_check_rejects_a_bad_summary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    payload: bytes,
    prefix: str,
) -> None:
    ws = _ws(tmp_path)
    stale_tmp = _stale(ws)
    interpreter = _summary_script(tmp_path, payload)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            ws, _state(*_TWO_JOBS), interpreter=interpreter
        )

    assert isinstance(result, ExportFailed)
    assert result.reason.startswith(prefix)
    assert not stale_tmp.exists()
    assert _warnings(caplog) == [f"telemetry export failed: {result.reason}"]


def test_export_check_accepts_a_summary_and_formats_its_headline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = _ws(tmp_path)
    values = '"model.cpu_util_mean": 0.5, "model.job_mem_peak_bytes_max": 1234'
    interpreter = _summary_script(tmp_path, (_SUMMARY_OK % values).encode())

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        result = export_telemetry(
            ws, _state(*_TWO_JOBS), interpreter=interpreter
        )

    out = ws.telemetry_outputs_dir
    assert result == Exported(
        (out / EXPORT_FILES[0], out / EXPORT_FILES[1], out / EXPORT_FILES[2]),
        (
            ("telemetry.model.job_mem_peak_bytes_max", "1234"),
            ("telemetry.model.cpu_util_mean", "0.5000"),
        ),
    )
    assert _warnings(caplog) == []


def test_export_check_accepts_an_empty_headline(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    interpreter = _summary_script(tmp_path, (_SUMMARY_OK % "").encode())

    result = export_telemetry(ws, _state(*_TWO_JOBS), interpreter=interpreter)

    assert isinstance(result, Exported)
    assert result.headline == ()


# Real export and headline items.


def test_export_real_exporter_over_builder_staging(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    head = header("model", "1000")
    write_staging(
        ws.telemetry_dir,
        head,
        [
            sample(
                head,
                seq,
                cpu_busy_cores=float(seq + 1),
                job_mem_peak_bytes=1000 * (seq + 1),
            )
            for seq in range(4)
        ],
    )

    result = export_telemetry(
        ws, _state((Phase.MODEL, "1000")), interpreter=sys.executable
    )

    assert isinstance(result, Exported)
    assert result.files == tuple(
        ws.telemetry_outputs_dir / name for name in EXPORT_FILES
    )
    assert all(path.is_file() for path in result.files)
    keys = [key for key, _ in result.headline]
    order = [HEADLINE_KEY_PREFIX + spec.key for spec in HEADLINE]
    assert keys == [key for key in order if key in keys]
    assert "telemetry.model.cpu_busy_cores_mean" in keys
    assert all(key.startswith("telemetry.") for key in keys)
    assert (
        dict(result.headline)["telemetry.model.cpu_busy_cores_mean"] == "2.50"
    )
    assert dict(result.headline)["telemetry.model.job_mem_peak_bytes_max"] == (
        "4000"
    )


def test_export_real_exporter_without_staging_is_nothing(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path)

    result = export_telemetry(ws, _state((Phase.MODEL, "1000")))

    assert result == NothingToExport()


def test_headline_items_follow_headline_order_and_format() -> None:
    items = headline_items(
        {"model.cpu_util_mean": 0.5, "model.job_mem_peak_bytes_max": 1234.5}
    )

    assert items == (
        ("telemetry.model.job_mem_peak_bytes_max", "1234"),
        ("telemetry.model.cpu_util_mean", "0.5000"),
    )


def test_headline_items_of_nothing_and_of_unknown_keys_are_empty() -> None:
    assert headline_items({}) == ()
    assert headline_items({"model.unknown": 1}) == ()


def test_headline_items_propagate_the_format_error() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        headline_items({"model.cpu_util_mean": -1.0})
