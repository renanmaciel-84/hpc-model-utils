"""ADR-073/ADR-083/ADR-085/R47/R50/R85: the publish-side export runner.

The export runs as a timed subprocess so that no exporter failure, a native
crash inside pyarrow included, can stop the rest of publish. Every outcome is
a value: ``export_telemetry`` raises nothing for any child behaviour or any
state of the outputs directory (K1). A failure reason is untrusted child
output; it is only logged.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from hpc_model_utils.core.state import RunState, current_tool
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.errors import ShellCommandError
from hpc_model_utils.infra.shell import ShellResult
from hpc_model_utils.infra.shell import run as shell_run
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADLINE,
    HEADLINE_KEY_PREFIX,
    SCHEMA_VERSION,
    SUMMARY_FILE,
    SUMMARY_KIND,
    format_headline_value,
)

logger = logging.getLogger(__name__)

EXPORT_TIMEOUT_S: Final = 300.0

# core may not import the exporter (it alone imports pyarrow), so the module
# is named, as core/launch.py names the sampler.
_EXPORT_MODULE: Final = "hpc_model_utils.infra.telemetry.export"
_EXIT_NO_STAGING: Final = 3
_HEADLINE_KEYS: Final = frozenset(spec.key for spec in HEADLINE)


@dataclass(frozen=True, slots=True)
class Exported:
    files: tuple[Path, Path, Path]
    headline: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class NothingToExport:
    pass


@dataclass(frozen=True, slots=True)
class ExportFailed:
    reason: str


type ExportResult = Exported | NothingToExport | ExportFailed


def headline_items(
    values: Mapping[str, int | float],
) -> tuple[tuple[str, str], ...]:
    """The (hook key, formatted value) pairs of the headline values, in
    HEADLINE order.

    A value outside the domain of ``format_headline_value`` raises its
    ValueError.
    """
    return tuple(
        (
            HEADLINE_KEY_PREFIX + spec.key,
            format_headline_value(spec.fmt, values[spec.key]),
        )
        for spec in HEADLINE
        if spec.key in values
    )


def export_telemetry(
    ws: Workspace,
    state: RunState,
    *,
    interpreter: str = sys.executable,
    timeout_s: float = EXPORT_TIMEOUT_S,
) -> ExportResult:
    """Export the staging files of ``state.jobs`` into the outputs directory.

    The outputs directory is cleared of the export's own names first (R47).
    An ``Exported`` result means all three files exist and ``summary.json``
    holds a publishable headline.
    """
    if not state.jobs:
        logger.info("telemetry export skipped: no recorded job")
        return NothingToExport()
    result = _export(ws, state, interpreter, timeout_s)
    if isinstance(result, ExportFailed):
        logger.warning("telemetry export failed: %s", result.reason)
    elif isinstance(result, NothingToExport):
        logger.info("telemetry export: no staging file for the recorded jobs")
    return result


def _export(
    ws: Workspace, state: RunState, interpreter: str, timeout_s: float
) -> ExportResult:
    try:
        _clear(ws.telemetry_outputs_dir)
    except OSError as exc:
        return ExportFailed(f"cannot clear {ws.telemetry_outputs_dir}: {exc}")
    try:
        result = shell_run(_argv(ws, state, interpreter), timeout=timeout_s)
    except (ShellCommandError, OSError) as exc:
        return ExportFailed(str(exc))
    return _map(result, ws, timeout_s)


def _clear(directory: Path) -> None:
    for name in EXPORT_FILES:
        (directory / name).unlink(missing_ok=True)
        (directory / f".{name}.tmp").unlink(missing_ok=True)


def _argv(ws: Workspace, state: RunState, interpreter: str) -> list[str]:
    argv = [
        interpreter,
        "-I",
        "-m",
        _EXPORT_MODULE,
        "--staging",
        str(ws.telemetry_dir),
        "--out",
        str(ws.telemetry_outputs_dir),
        "--run-id",
        state.run_id,
        "--model",
        state.plugin,
        "--tool-version",
        current_tool().version,
    ]
    for job in state.jobs:
        argv += ["--job", f"{job.phase.value}:{job.job_id}"]
    return argv


def _map(result: ShellResult, ws: Workspace, timeout_s: float) -> ExportResult:
    if result.timed_out:
        return ExportFailed(f"timed out after {timeout_s:g} s")
    if result.returncode == _EXIT_NO_STAGING:
        return NothingToExport()
    if result.returncode < 0:
        return ExportFailed(f"killed by signal {-result.returncode}")
    if result.returncode != 0:
        last = next((ln for ln in reversed(result.output) if ln.strip()), None)
        return ExportFailed(last or f"exit status {result.returncode}")
    out = ws.telemetry_outputs_dir
    samples, workspace, summary = (out / name for name in EXPORT_FILES)
    files = (samples, workspace, summary)
    for name, path in zip(EXPORT_FILES, files, strict=True):
        try:
            present = path.is_file()
        except OSError as exc:
            return ExportFailed(f"cannot check {name}: {exc}")
        if not present:
            return ExportFailed(f"export exited 0 without {name}")
    headline = _read_summary(summary)
    if isinstance(headline, str):
        return ExportFailed(headline)
    return Exported(files, headline)


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite token {token}")


def _read_summary(path: Path) -> tuple[tuple[str, str], ...] | str:
    """The headline items of a valid summary, else the reason it is not."""
    try:
        data = json.loads(path.read_bytes(), parse_constant=_reject_constant)
    except (OSError, ValueError, RecursionError) as exc:
        return f"unreadable {SUMMARY_FILE}: {exc}"
    if not isinstance(data, dict):
        return f"malformed {SUMMARY_FILE}: not an object"
    if data.get("kind") != SUMMARY_KIND:
        return f"malformed {SUMMARY_FILE}: kind"
    if data.get("schema_version") != SCHEMA_VERSION:
        return f"malformed {SUMMARY_FILE}: schema_version"
    headline = data.get("headline")
    if not isinstance(headline, dict):
        return f"malformed {SUMMARY_FILE}: headline is not an object"
    raw = headline.get("values")
    if not isinstance(raw, dict):
        return f"malformed {SUMMARY_FILE}: headline.values is not an object"
    values: dict[str, int | float] = {}
    for key, value in raw.items():
        if key not in _HEADLINE_KEYS:
            return f"malformed {SUMMARY_FILE}: unknown headline key"
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"malformed {SUMMARY_FILE}: headline value is not a number"
        values[key] = value
    try:
        return headline_items(values)
    except ValueError as exc:
        return f"malformed {SUMMARY_FILE}: {exc}"
