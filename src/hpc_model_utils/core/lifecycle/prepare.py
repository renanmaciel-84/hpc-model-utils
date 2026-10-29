"""ADR-017/ADR-018/ADR-028/ADR-009/R21/R26/R50/R123/R130: the
plugin-agnostic input-preparation lifecycle step (v1's
``extract_sanitize_inputs``/``preprocess``, copied per model in
``app/adapter/repository/{newave,decomp}.py``, ported once, R26).

ADR-017 (E12): the deck's own stale outputs are purged by member name
before the parent archive is ever touched, so a crashed run can never
be masked by a leftover output the deck shipped. ADR-018: sanitization
is pure Python (``infra.encoding``), never a shell pipeline, and never
touches a parent archive -- parents are fetched *after* sanitization.
ADR-028: no deck member name ever reaches a shell; the name normalizer
runs via ``infra.shell.run`` with an argv list, and a broken run is an
absorbed, logged failure (v1 parity) rather than a fatal one.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from hpc_model_utils.core.diagnosis import utc_now_iso
from hpc_model_utils.core.errors import DataError, StateFormatError, UsageError
from hpc_model_utils.core.lifecycle.run import StatusReporter
from hpc_model_utils.core.plugin import ModelPlugin, ParentRun
from hpc_model_utils.core.state import (
    ParentInfo,
    RunState,
    StateStore,
    StepRecord,
    metadata_items,
    write_projections,
)
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.archive import (
    extract,
    validated_directories,
    validated_members,
)
from hpc_model_utils.infra.encoding import TextEncoding, sanitize_file
from hpc_model_utils.infra.errors import ShellCommandError
from hpc_model_utils.infra.shell import run as shell_run

logger = logging.getLogger(__name__)

_CORE_STALE_PATTERNS: tuple[str, ...] = (
    r"(?:[^/]+/)*[^/]+\.modelops",
    r"(?:[^/]+/)*run\.json",
)
_FLATTEN_PATTERN = re.compile("[\x00-\x1f\x7f\x85\u2028\u2029]")


def extract_archive(ws: Workspace, archive: Path) -> tuple[str, ...]:
    """Requirement 1: validate, then reject an ``assets/`` member or a
    member that is the archive itself, before writing anything."""
    members = validated_members(archive)
    archive_resolved = archive.resolve()
    for entry in (*members, *validated_directories(archive)):
        first_segment = PurePosixPath(entry).parts[0]
        if re.fullmatch(r"assets", first_segment, re.IGNORECASE):
            raise DataError(f"deck member under reserved assets/: {entry!r}")
    for member in members:
        if (ws.root / member).resolve() == archive_resolved:
            raise DataError(f"deck member is the archive itself: {member!r}")
    return tuple(extract(archive, ws.root))


def purge_stale_outputs(
    ws: Workspace, plugin: ModelPlugin, members: Sequence[str]
) -> tuple[str, ...]:
    """Requirement 2/ADR-017/E12: delete only members of this archive
    that match an output or core stale pattern, never through a
    symlink."""
    patterns = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (*plugin.output_patterns, *_CORE_STALE_PATTERNS)
    )
    purged: list[str] = []
    for member in members:
        if not any(pattern.fullmatch(member) for pattern in patterns):
            continue
        path = ws.root / member
        if path.is_symlink() or not path.is_file():
            continue
        path.unlink()
        purged.append(member)
    purged.sort()
    logger.info("purged stale outputs: %s", purged)
    return tuple(purged)


def run_name_normalizer(
    ws: Workspace, plugin: ModelPlugin, *, timeout: float = 300.0
) -> None:
    """Requirement 3: a broken normalizer is absorbed (v1 parity),
    never fatal; no error message here carries the workspace-absolute
    executable path."""
    name = plugin.executables.name_normalizer
    if name is None:
        return
    try:
        result = shell_run(
            [str(ws.assets / name)],
            cwd=ws.root,
            timeout=timeout,
            on_line=logger.info,
        )
    except ShellCommandError:
        logger.warning("name normalizer %s failed to start", name)
        return
    if result.timed_out:
        logger.warning(
            "name normalizer %s timed out after %.0fs", name, timeout
        )
    elif result.returncode != 0:
        logger.warning("name normalizer %s exited %d", name, result.returncode)


def sanitize_workspace(ws: Workspace, plugin: ModelPlugin) -> tuple[str, ...]:
    """Requirement 4/ADR-018: top-level only, never through a symlink,
    never into ``assets/``/``.hpcmu/``."""
    return sanitize_files(
        ws, plugin, tuple(entry.name for entry in ws.root.iterdir())
    )


def sanitize_files(
    ws: Workspace, plugin: ModelPlugin, names: Sequence[str]
) -> tuple[str, ...]:
    """``sanitize_workspace`` restricted to the named top-level files.
    A name that is not a bare top-level file name (``a/b``, ``..``) is
    skipped, since deck index files supply these names."""
    if not plugin.sanitize_encoding:
        return ()
    license_files = {name.lower() for name in plugin.executables.license_files}
    exclude_patterns = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in plugin.sanitize_exclude
    )
    converted: list[str] = []
    for name in dict.fromkeys(names):
        if (
            name in {"", ".", ".."}
            or "\\" in name
            or PurePosixPath(name).name != name
        ):
            continue
        entry = ws.root / name
        if entry.is_symlink() or not entry.is_file():
            continue
        if entry == ws.eco_deck_path:
            continue
        if entry.name.lower().endswith(".modelops"):
            continue
        if entry.name.lower() in license_files:
            continue
        if any(pattern.fullmatch(entry.name) for pattern in exclude_patterns):
            continue
        if sanitize_file(entry) is TextEncoding.LATIN1:
            converted.append(entry.name)
    return tuple(sorted(converted))


def move_licences(ws: Workspace, plugin: ModelPlugin) -> tuple[str, ...]:
    """Requirement 5."""
    moved: list[str] = []
    for name in plugin.executables.license_files:
        src = ws.assets / name
        if src.is_file():
            os.replace(src, ws.root / name)
            moved.append(name)
    return tuple(sorted(moved))


def parent_run(
    ws: Workspace, plugin: ModelPlugin, parent: ParentInfo
) -> ParentRun:
    """Requirement 6."""
    archives: list[Path] = []
    for name in plugin.parent_artifacts:
        path = ws.parent_dir / name
        if not path.is_file():
            raise UsageError(
                f"parent archive {name} missing; run "
                "check_and_fetch_inputs first"
            )
        archives.append(path)
    return ParentRun(
        uri=parent.path,
        model_name=parent.model_name,
        starting_date=parent.starting_date,
        archives=tuple(archives),
    )


def _r130_listing(ws: Workspace) -> None:
    for entry in sorted(ws.root.iterdir(), key=lambda p: p.name):
        if entry.name == ".hpcmu":
            continue
        suffix = "/" if entry.is_dir() else ""
        logger.info("%d %s%s", entry.stat().st_size, entry.name, suffix)


def extract_sanitize_inputs(
    ws: Workspace,
    plugin: ModelPlugin,
    store: StateStore,
    reporter: StatusReporter,
) -> RunState:
    """Requirement 7: v1's ``extract_sanitize_inputs``, composed from
    the helpers above in exactly this order."""
    state = store.load_or_create(plugin.name)
    if not ws.eco_deck_path.is_file():
        raise UsageError(
            "eco_deck.zip missing; run check_and_fetch_inputs first"
        )

    started_at = utc_now_iso()
    entry_monotonic = time.monotonic()

    members = extract_archive(ws, ws.eco_deck_path)
    purged = purge_stale_outputs(ws, plugin, members)
    run_name_normalizer(ws, plugin)
    sanitize_workspace(ws, plugin)
    if state.parent is not None:
        plugin.fetch_parent(ws, parent_run(ws, plugin, state.parent))
    move_licences(ws, plugin)
    study = plugin.study_info(ws)

    purged_set = set(purged)
    input_files = tuple(
        member for member in members if member not in purged_set
    )

    step = StepRecord(
        command="extract_sanitize_inputs",
        host=socket.gethostname(),
        started_at=started_at,
        finished_at=utc_now_iso(),
        duration_seconds=time.monotonic() - entry_monotonic,
        outcome="ok",
    )
    new_state = replace(
        state,
        study=study,
        input_files=input_files,
        steps=state.steps + (step,),
    )
    store.save(new_state)
    write_projections(
        ws, new_state, always_write_parent_path=plugin.always_write_parent_path
    )

    for key, value in metadata_items(
        new_state, always_write_parent_path=plugin.always_write_parent_path
    ):
        if key in ("study_starting_date", "study_name"):
            reporter.metadata(key, value)

    _r130_listing(ws)
    return new_state


def flatten_execution_name(name: str) -> str:
    """Requirement 8/ADR-028: printable Unicode is accepted; control
    characters, DEL, NEL and the Unicode line/paragraph separators are
    flattened to a space."""
    return _FLATTEN_PATTERN.sub(" ", name)


def preprocess(
    ws: Workspace,
    plugin: ModelPlugin,
    execution_name: str,
    *,
    store: StateStore,
) -> None:
    """Requirement 9/R123: the state-optional dispatch. A bare
    directory (C2) never gains a ``.hpcmu/``, never gets purged,
    extracted or sanitized."""
    flattened = flatten_execution_name(execution_name)
    if flattened != execution_name:
        logger.warning(
            "execution name flattened: %r -> %r", execution_name, flattened
        )

    if not ws.has_state:
        plugin.prepare(ws, flattened)
        return

    state = store.load()
    if state is None:
        raise StateFormatError("state.json missing after has_state check")
    if state.plugin != plugin.name:
        raise UsageError(
            f"plugin mismatch: workspace has {state.plugin!r}, "
            f"got {plugin.name!r}"
        )

    started_at = utc_now_iso()
    entry_monotonic = time.monotonic()
    plugin.prepare(ws, flattened)
    step = StepRecord(
        command="preprocess",
        host=socket.gethostname(),
        started_at=started_at,
        finished_at=utc_now_iso(),
        duration_seconds=time.monotonic() - entry_monotonic,
        outcome="ok",
    )
    new_state = replace(state, steps=state.steps + (step,))
    store.save(new_state)
    write_projections(
        ws, new_state, always_write_parent_path=plugin.always_write_parent_path
    )
