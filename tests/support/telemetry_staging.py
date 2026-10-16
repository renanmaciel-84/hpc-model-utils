"""ADR-069/ADR-073: build staging files through the real StagingWriter.

The export reader's tests build their inputs here, so an input is always what
ticket-025's writer writes: `write_staging` creates the file with
`StagingWriter.create` and writes every record through it. A shape the writer
cannot write, such as a torn line, a NaN token or an error line past its cap,
is a raw byte edit of a file built here. Placeholders only: `fake-node<i>`,
jobs 1000, 1001 and 1234, and timestamps after `ORIGIN_MS`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hpc_model_utils.infra.telemetry.sampler import JsonValue, StagingWriter
from hpc_model_utils.infra.telemetry.schema import (
    HEADER_FIELDS,
    HEADER_KIND,
    SAMPLE_COLUMNS,
    SCHEMA_VERSION,
    WALK_KIND,
    WORKSPACE_COLUMNS,
    Column,
)

# 2000-01-01T00:00:00Z
ORIGIN_MS: Final = 946_684_800_000


@dataclass(frozen=True, slots=True)
class ErrorLine:
    """One `write_error` call; the writer keeps one per (source, class)."""

    ts_ms: int
    source: str
    exc: Exception


def _record(
    fields: Iterable[tuple[str, bool]], known: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Return known in the order of fields, None for a missing nullable one.

    A name of known outside fields, or a missing field that is not nullable,
    raises KeyError.
    """
    pairs = tuple(fields)
    unknown = known.keys() - {name for name, _ in pairs}
    if unknown:
        raise KeyError(f"unknown names: {sorted(unknown)}")
    return {
        name: known[name] if name in known or not nullable else None
        for name, nullable in pairs
    }


def _identity(
    head: Mapping[str, JsonValue], columns: Iterable[Column]
) -> dict[str, JsonValue]:
    return {
        c.name: head[c.name]
        for c in columns
        if c.source == "identity" and c.name in head
    }


def _int(head: Mapping[str, JsonValue], name: str) -> int:
    value = head[name]
    if not isinstance(value, int):
        raise TypeError(f"header {name} must be an int: {value!r}")
    return value


def _str(head: Mapping[str, JsonValue], name: str) -> str:
    value = head[name]
    if not isinstance(value, str):
        raise TypeError(f"header {name} must be a str: {value!r}")
    return value


def header(
    phase: str = "model",
    job_id: str = "1000",
    restart: int = 0,
    node_index: int = 0,
    *,
    nodes: int | None = 1,
    interval_s: int = 5,
    workspace_interval_s: int = 60,
) -> dict[str, JsonValue]:
    """Return a valid header line with every `HEADER_FIELDS` name, in order.

    The fields not set here are None.
    """
    return _record(
        ((field.name, field.nullable) for field in HEADER_FIELDS),
        {
            "kind": HEADER_KIND,
            "schema_version": SCHEMA_VERSION,
            "tool_version": "9.9.9",
            "phase": phase,
            "job_id": job_id,
            "restart": restart,
            "node": f"fake-node{node_index}",
            "node_index": node_index,
            "nodes": nodes,
            "interval_s": interval_s,
            "workspace_interval_s": workspace_interval_s,
            "started_at": "2000-01-01T00:00:00.000Z",
        },
    )


def sample(
    head: Mapping[str, JsonValue],
    seq: int,
    *,
    ts: int | None = None,
    **values: JsonValue,
) -> dict[str, JsonValue]:
    """Return a sample line of head's file, every `SAMPLE_COLUMNS` name in order.

    The identity comes from head. ts defaults to the grid tick seq + 1 after
    `ORIGIN_MS`, dt_s to interval_s and every other column to None. values
    override any column; an unknown name raises KeyError.
    """
    interval_s = _int(head, "interval_s")
    known: dict[str, JsonValue] = {
        **_identity(head, SAMPLE_COLUMNS),
        "ts": ORIGIN_MS + (seq + 1) * interval_s * 1000 if ts is None else ts,
        "seq": seq,
        "dt_s": float(interval_s),
        **values,
    }
    return _record(((c.name, c.nullable) for c in SAMPLE_COLUMNS), known)


def walk(
    head: Mapping[str, JsonValue], *, ts: int, **values: JsonValue
) -> dict[str, JsonValue]:
    """Return a walk line of head's file: kind, then the `WORKSPACE_COLUMNS`.

    The identity comes from head and every other column defaults to None.
    values override any column; an unknown name raises KeyError.
    """
    known = {**_identity(head, WORKSPACE_COLUMNS), "ts": ts, **values}
    row = _record(((c.name, c.nullable) for c in WORKSPACE_COLUMNS), known)
    return {"kind": WALK_KIND, **row}


def write_staging(
    directory: Path,
    head: Mapping[str, JsonValue],
    records: Iterable[Mapping[str, JsonValue] | ErrorLine] = (),
) -> Path:
    """Write head and records through a `StagingWriter`; return the file path.

    The file is named from head's identity. A mapping whose kind is
    `WALK_KIND` goes to `write_walk`, any other mapping to `write_sample`, and
    an `ErrorLine` to `write_error`. A write that returns False, such as the
    `O_EXCL` open of a file that exists, raises AssertionError.
    """
    writer = StagingWriter.create(
        directory,
        _str(head, "phase"),
        _str(head, "job_id"),
        _int(head, "restart"),
        _int(head, "node_index"),
        monotonic=lambda: 0.0,
    )
    written = writer.write_header(head)
    for record in records:
        if not written:
            break
        if isinstance(record, ErrorLine):
            written = writer.write_error(
                record.ts_ms, record.source, record.exc
            )
        elif record.get("kind") == WALK_KIND:
            written = writer.write_walk(record)
        else:
            written = writer.write_sample(record)
    failure = writer.last_failure
    writer.close()
    if not written:
        raise AssertionError(f"{writer.path}: a write failed: {failure}")
    return writer.path
