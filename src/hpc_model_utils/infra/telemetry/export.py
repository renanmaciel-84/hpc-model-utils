"""ADR-069/ADR-073/ADR-070/ADR-071: the export's reader, writers and roll-up.

The staging files are untrusted input: the model runs as the same user and
can write anything below `.hpcmu/telemetry/`. `discover` keeps a directory
entry only when it is a regular file, its name round-trips through
`staging_file_name`, and its `(phase, job_id)` is an exported job (R86). The
task logs, the srun logs, symlinks and stray names are never opened, and a
file is opened without following a symlink, so no read leaves the directory.

A line is read with a bounded `readline`, so no line costs more than
`STAGING_LINE_LIMIT` bytes, and a problem with one line is a warning on that
line, never an exception (ADR-069). Blank lines, which a zero-byte failed
write leaves, are skipped. `read_header` takes the first line that parses to
a JSON object as the header; the torn header fragments before it are
`bad_line` warnings, and a file whose header is invalid is skipped by the
caller (ADR-073). `read_body` reads the lines after the header, keeps the
valid samples and walks as rows, and turns the error lines into
`sampler_error` warnings. Warnings are counted per file by `(code, message)`
in first-occurrence order, with at most `WARNINGS_CAP` distinct records; the
occurrences of the keys left out are counted in `warnings_omitted`.

Rows keep the parsed JSON values: ints stay ints, also in FLOAT columns. A
dtype accepts exactly what its Arrow type holds, so a value the writer would
refuse or distort breaks the dtype, and its line is a `bad_record`, never an
export failure (ADR-069):
- a STRING cell must be UTF-8 text: a JSON `\\uXXXX` escape can decode to a
  lone surrogate, which has no UTF-8 form;
- a JSON int in a FLOAT column must lie in the float's exact integer range,
  beyond which Arrow refuses it (|v| <= 2**24 for FLOAT32, 2**53 for
  FLOAT64);
- a FLOAT32 float must lie in the float32 range, beyond which Arrow would
  store inf.

A header integer must fit int64 too, or the header is a `bad_header`.

An `OSError` from opening or reading a file propagates, because a file that
cannot be read fails the export (ADR-073).

`write_samples` and `write_workspace` write the ADR-070 Parquet files, and
`parquet_metadata` builds their R52 key-value metadata. This is the one
`infra/telemetry` module allowed a third-party import, pyarrow (ADR-091,
ADR-078). Only the writers and their schema helper import it, inside the
function, so the reader, `parquet_metadata` and the exit-3 path never load
it (ADR-073). pyarrow ships no type information, so its values stay local to
those functions. The writers pin the local filesystem, so a path is never
read as a URI: a URI-shaped relative path raises `pyarrow.ArrowInvalid`.

`Rollup` folds the header passes and the file reads into the ADR-071
`summary.json` roll-up, with the ADR-076 statistics and the ADR-077
headline, and `render_summary` serializes it. The roll-up is stdlib only.
Its statistics take the values as the Parquet publishes them: a FLOAT32
cell rounded to float32, a FLOAT64 cell as a float and an INT cell as an
int. A crafted value can make a statistic non-finite, never raise: the
`ValueError` of `render_summary` then fails the export.

`main` is the ADR-073 entry point, run as `python -I -m` with this module.
It exits 0 once the three `EXPORT_FILES` are in place, `summary.json`
last; 3 when no staging file of the given jobs exists; 2 on a usage error;
and 1 on a failure. stdout stays empty, and the last stderr line is the
reason. pyarrow is imported only after every header is read, so neither
exit 3 nor the exit 1 for no valid header loads it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from array import array
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import BinaryIO, Final, NoReturn, TypeGuard

from hpc_model_utils.infra.telemetry.schema import (
    ERROR_FIELDS,
    ERROR_KIND,
    EXPORT_FILES,
    HEADER_FIELDS,
    HEADER_KIND,
    HEADLINE,
    PARQUET_METADATA_KEYS,
    PHASES,
    SAMPLE_COLUMNS,
    SAMPLE_SORT_KEYS,
    SAMPLE_STATS,
    SAMPLES_FILE,
    SAMPLES_KIND,
    SCHEMA_VERSION,
    STAGING_LINE_LIMIT,
    STAT_AGGREGATES,
    SUMMARY_FILE,
    SUMMARY_KIND,
    WALK_KIND,
    WARNING_CODES,
    WARNINGS_CAP,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    WORKSPACE_KIND,
    WORKSPACE_SORT_KEYS,
    WORKSPACE_STATS,
    Column,
    HeadlineSpec,
    StatSpec,
    staging_file_name,
)

type Cell = int | float | str | None
type HeaderValue = int | str | dict[str, str] | None
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
# A warning occurrence: (code, message).
type _Warning = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    phase: str
    job_id: str
    restart: int
    node_index: int


@dataclass(frozen=True, slots=True)
class ReadWarning:
    """`count` occurrences of one warning; its identity is the file's Candidate."""

    code: str
    message: str
    count: int


@dataclass(frozen=True, slots=True)
class HeaderRead:
    """The header pass of one file; header is None exactly on a bad_header.

    body_offset is the byte offset just after the header line, or where the
    pass stopped when there is no valid header.
    """

    candidate: Candidate
    header: dict[str, HeaderValue] | None
    body_offset: int
    warnings: tuple[ReadWarning, ...]


@dataclass(frozen=True, slots=True)
class FileRead:
    """One file with a valid header: its rows, in column order, and warnings."""

    candidate: Candidate
    header: dict[str, HeaderValue]
    samples: tuple[tuple[Cell, ...], ...]
    walks: tuple[tuple[Cell, ...], ...]
    warnings: tuple[ReadWarning, ...]
    warnings_omitted: int


_CODES: Final = frozenset(code.code for code in WARNING_CODES)


def _code(code: str) -> str:
    """Return code; a name outside `WARNING_CODES` fails the import."""
    if code not in _CODES:
        raise ValueError(f"not a WARNING_CODES code: {code!r}")
    return code


_BAD_HEADER: Final = _code("bad_header")
_TRUNCATED_TAIL: Final = _code("truncated_tail")
_BAD_LINE: Final = _code("bad_line")
_BAD_RECORD: Final = _code("bad_record")
_IDENTITY_MISMATCH: Final = _code("identity_mismatch")
_NON_MONOTONIC_TS: Final = _code("non_monotonic_ts")
_SAMPLER_ERROR: Final = _code("sampler_error")

_NOT_JSON: Final = (_BAD_LINE, "not ASCII JSON")
_TOO_LONG: Final = (_BAD_LINE, f"line exceeds {STAGING_LINE_LIMIT} bytes")
_NO_NEWLINE: Final = (_TRUNCATED_TAIL, "last line has no newline")
_NOT_OBJECT: Final = (_BAD_RECORD, "not a JSON object")
_REPEATED_HEADER: Final = (_BAD_RECORD, "repeated header line")
_UNKNOWN_KIND: Final = (_BAD_RECORD, "unknown kind")
_BAD_ERROR_LINE: Final = (_BAD_RECORD, "error line breaks its fields")

_NAME_PATTERN: Final = re.compile(
    r"(?P<phase>[a-z]+)-(?P<job>[0-9]+)-r(?P<restart>[0-9]+)-(?P<node>[0-9]+)"
    r"\.ndjson"
)
_READ_SIZE: Final = STAGING_LINE_LIMIT + 1
_INT64_MAX: Final = 2**63 - 1
_FLOAT32_MAX: Final = 3.4028234663852886e38
# The exact integer range of each float type, past which Arrow refuses a
# Python int.
_FLOAT32_EXACT_INT: Final = 2**24
_FLOAT64_EXACT_INT: Final = 2**53

_HEADER_NAMES: Final = frozenset(field.name for field in HEADER_FIELDS)
_FILE_IDENTITY: Final = ("phase", "job_id", "restart", "node_index")
_ERROR_KEYS: Final = frozenset(ERROR_FIELDS)
_SAMPLE_KEYS: Final = frozenset(c.name for c in SAMPLE_COLUMNS)
_WALK_KEYS: Final = frozenset(c.name for c in WORKSPACE_COLUMNS) | {"kind"}
_SAMPLE: Final = "sample"
_WALK: Final = "walk"
# The identity columns that the header also carries, checked against it.
_ROW_IDENTITY: Final = {
    noun: tuple(
        c.name
        for c in columns
        if c.source == "identity" and c.name in _HEADER_NAMES
    )
    for noun, columns in ((_SAMPLE, SAMPLE_COLUMNS), (_WALK, WORKSPACE_COLUMNS))
}


def discover(
    staging: Path, jobs: Collection[tuple[str, str]]
) -> list[Candidate]:
    """Return the staging files of jobs below staging, in sample sort order.

    The order is `(phase, job_id, restart, node_index)`, the first two
    compared as strings. A missing directory gives []; any other `OSError`
    propagates.
    """
    try:
        entries = os.scandir(staging)
    except FileNotFoundError:
        return []
    found: list[Candidate] = []
    with entries:
        for entry in entries:
            candidate = _candidate(staging, entry)
            if (
                candidate is not None
                and (candidate.phase, candidate.job_id) in jobs
            ):
                found.append(candidate)
    return sorted(
        found, key=lambda c: (c.phase, c.job_id, c.restart, c.node_index)
    )


def _candidate(staging: Path, entry: os.DirEntry[str]) -> Candidate | None:
    match = _NAME_PATTERN.fullmatch(entry.name)
    if match is None or not entry.is_file(follow_symlinks=False):
        return None
    phase, job_id = match["phase"], match["job"]
    try:
        restart, node_index = int(match["restart"]), int(match["node"])
        canonical = staging_file_name(phase, job_id, restart, node_index)
    except ValueError:
        return None
    if canonical != entry.name:
        return None
    return Candidate(staging / entry.name, phase, job_id, restart, node_index)


def read_header(candidate: Candidate) -> HeaderRead:
    """Find and check the header line of candidate's file."""
    counts: dict[_Warning, int] = {}
    header: dict[str, HeaderValue] | None = None
    end = 0
    with _open(candidate.path) as stream:
        # At most five distinct keys arise here, so _note never drops one.
        for outcome, end in _lines(stream, 0):
            if isinstance(outcome, tuple):
                _note(counts, outcome)
                continue
            checked = _check_header(outcome, candidate)
            if isinstance(checked, str):
                _note(counts, (_BAD_HEADER, checked))
            else:
                header = checked
            break
        else:
            _note(counts, (_BAD_HEADER, "no header line"))
    return HeaderRead(candidate, header, end, _warnings(counts))


def read_body(head: HeaderRead) -> FileRead:
    """Read the lines after a valid header; the header pass warnings carry on.

    A HeaderRead without a header is a caller bug and raises ValueError.
    """
    header = head.header
    if header is None:
        raise ValueError(
            f"{head.candidate.path}: read_body needs a valid header"
        )
    counts = {(w.code, w.message): w.count for w in head.warnings}
    omitted = 0
    samples: list[tuple[Cell, ...]] = []
    walks: list[tuple[Cell, ...]] = []
    last_ts = {_SAMPLE: -1, _WALK: -1}
    with _open(head.candidate.path) as stream:
        for outcome, _ in _lines(stream, head.body_offset):
            problem = (
                outcome
                if isinstance(outcome, tuple)
                else _body_record(outcome, header, samples, walks, last_ts)
            )
            if problem is not None and not _note(counts, problem):
                omitted += 1
    return FileRead(
        head.candidate,
        header,
        tuple(samples),
        tuple(walks),
        _warnings(counts),
        omitted,
    )


def _open(path: Path) -> BinaryIO:
    # O_NOFOLLOW: a file swapped for a symlink after discover fails to open
    # instead of being read from outside the staging directory.
    return open(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")


def _lines(
    stream: BinaryIO, start: int
) -> Iterator[tuple[dict[str, object] | _Warning, int]]:
    """Yield each non-blank line from start, parsed, and the offset after it.

    A line longer than `STAGING_LINE_LIMIT`, its newline included, is one
    `_TOO_LONG` whatever it ends with, its rest skipped in bounded reads. A
    last line without a newline is `_NO_NEWLINE`.
    """
    stream.seek(start)
    end = start
    while line := stream.readline(_READ_SIZE):
        end += len(line)
        if len(line) > STAGING_LINE_LIMIT:
            if not line.endswith(b"\n"):
                end += _skip_line(stream)
            yield _TOO_LONG, end
        elif not line.endswith(b"\n"):
            yield _NO_NEWLINE, end
        elif line != b"\n":
            yield _parse(line), end


def _skip_line(stream: BinaryIO) -> int:
    """Read up to and including the next newline; return the bytes read."""
    skipped = 0
    while chunk := stream.readline(_READ_SIZE):
        skipped += len(chunk)
        if chunk.endswith(b"\n"):
            break
    return skipped


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {token}")


def _parse(line: bytes) -> dict[str, object] | _Warning:
    try:
        value = json.loads(
            line.decode("ascii"), parse_constant=_reject_constant
        )
    # UnicodeDecodeError and JSONDecodeError are ValueErrors; deep nesting
    # raises RecursionError.
    except (ValueError, RecursionError):
        return _NOT_JSON
    if not isinstance(value, dict):
        return _NOT_OBJECT
    return value


def _note(counts: dict[_Warning, int], warning: _Warning) -> bool:
    """Count one occurrence; False when `WARNINGS_CAP` leaves its key out."""
    if warning in counts:
        counts[warning] += 1
    elif len(counts) < WARNINGS_CAP:
        counts[warning] = 1
    else:
        return False
    return True


def _warnings(counts: Mapping[_Warning, int]) -> tuple[ReadWarning, ...]:
    return tuple(
        ReadWarning(code, message, count)
        for (code, message), count in counts.items()
    )


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _text_map(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    texts = {
        key: text
        for key, text in value.items()
        if isinstance(key, str) and isinstance(text, str)
    }
    return texts if len(texts) == len(value) else None


def _check_header(
    record: Mapping[str, object], candidate: Candidate
) -> dict[str, HeaderValue] | str:
    """Return the header in `HEADER_FIELDS` order, or its bad_header message."""
    if record.keys() != _HEADER_NAMES:
        return "header key set differs"
    header: dict[str, HeaderValue] = {}
    for field in HEADER_FIELDS:
        value = record[field.name]
        if value is None and field.nullable:
            header[field.name] = None
        elif field.json_type == "string" and isinstance(value, str):
            header[field.name] = value
        elif field.json_type == "integer" and _is_int(value):
            # ADR-070's int64, as for an INT64 body cell: a crafted header int
            # can neither reach the int-to-str limit of render_summary nor
            # fail the export.
            if not _int64(value):
                return f"header {field.name} outside int64"
            header[field.name] = value
        elif (
            field.json_type == "object"
            and (texts := _text_map(value)) is not None
        ):
            header[field.name] = texts
        else:
            return f"header field {field.name} has the wrong type"
    if header["kind"] != HEADER_KIND:
        return "header kind differs"
    if header["schema_version"] != SCHEMA_VERSION:
        return f"unsupported schema_version {header['schema_version']}"
    for name in ("interval_s", "workspace_interval_s", "nodes"):
        value = header[name]
        if isinstance(value, int) and value < 1:
            return f"header {name} below 1"
    identity = tuple(header[name] for name in _FILE_IDENTITY)
    if identity != (
        candidate.phase,
        candidate.job_id,
        candidate.restart,
        candidate.node_index,
    ):
        return "header identity differs from the file name"
    return header


def _body_record(
    record: Mapping[str, object],
    header: Mapping[str, HeaderValue],
    samples: list[tuple[Cell, ...]],
    walks: list[tuple[Cell, ...]],
    last_ts: dict[str, int],
) -> _Warning | None:
    """Keep the row of a sample or walk, or return the line's warning."""
    if "kind" not in record:
        if record.keys() != _SAMPLE_KEYS:
            return (_BAD_RECORD, "sample key set differs")
        return _keep(_SAMPLE, record, SAMPLE_COLUMNS, header, samples, last_ts)
    kind = record["kind"]
    if kind == HEADER_KIND:
        return _REPEATED_HEADER
    if kind == WALK_KIND:
        if record.keys() != _WALK_KEYS:
            return (_BAD_RECORD, "walk key set differs")
        return _keep(_WALK, record, WORKSPACE_COLUMNS, header, walks, last_ts)
    if kind == ERROR_KIND:
        return _error_warning(record)
    return _UNKNOWN_KIND


def _keep(
    noun: str,
    record: Mapping[str, object],
    columns: tuple[Column, ...],
    header: Mapping[str, HeaderValue],
    rows: list[tuple[Cell, ...]],
    last_ts: dict[str, int],
) -> _Warning | None:
    row = _row(record, columns)
    if isinstance(row, str):
        return (_BAD_RECORD, f"{noun} column {row} breaks its dtype")
    if any(record[name] != header[name] for name in _ROW_IDENTITY[noun]):
        return (_IDENTITY_MISMATCH, f"{noun} identity differs from the header")
    ts = record["ts"]
    # _row has checked ts; the call only narrows it to int.
    if not (_timestamp(ts) and ts > last_ts[noun]):
        return (_NON_MONOTONIC_TS, f"{noun} ts not after the previous {noun}")
    last_ts[noun] = ts
    rows.append(row)
    return None


def _error_warning(record: Mapping[str, object]) -> _Warning:
    if record.keys() != _ERROR_KEYS:
        return _BAD_ERROR_LINE
    ts, source = record["ts"], record["source"]
    error, message = record["error"], record["message"]
    if not (
        _is_int(ts)
        and ts >= 0
        and isinstance(source, str)
        and isinstance(error, str)
        and isinstance(message, str)
    ):
        return _BAD_ERROR_LINE
    return (_SAMPLER_ERROR, f"{source} {error}: {message}")


def _row(
    record: Mapping[str, object], columns: tuple[Column, ...]
) -> tuple[Cell, ...] | str:
    """Return the values in column order, or the first column that breaks."""
    cells: list[Cell] = []
    for column in columns:
        value = record[column.name]
        if value is None and column.nullable:
            cells.append(None)
        elif _DTYPE_RULES[column.dtype](value):
            cells.append(value)
        else:
            return column.name
    return tuple(cells)


def _timestamp(value: object) -> TypeGuard[int]:
    return _is_int(value) and 0 <= value <= _INT64_MAX


def _string(value: object) -> TypeGuard[str]:
    # A lone surrogate from a JSON \uXXXX escape has no UTF-8 form, so
    # Arrow's string type cannot hold it.
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _int16(value: object) -> TypeGuard[int]:
    return _is_int(value) and -(2**15) <= value < 2**15


def _int32(value: object) -> TypeGuard[int]:
    return _is_int(value) and -(2**31) <= value < 2**31


def _int64(value: object) -> TypeGuard[int]:
    return _is_int(value) and -(2**63) <= value < 2**63


def _float64(value: object) -> TypeGuard[int | float]:
    # json.loads reads 1e400 as inf, so a float must also be finite.
    if _is_int(value):
        return abs(value) <= _FLOAT64_EXACT_INT
    return isinstance(value, float) and math.isfinite(value)


def _float32(value: object) -> TypeGuard[int | float]:
    if _is_int(value):
        return abs(value) <= _FLOAT32_EXACT_INT
    return _float64(value) and abs(value) <= _FLOAT32_MAX


_DTYPE_RULES: Final[Mapping[str, Callable[[object], TypeGuard[Cell]]]] = {
    "TIMESTAMP_MS_UTC": _timestamp,
    "STRING": _string,
    "INT16": _int16,
    "INT32": _int32,
    "INT64": _int64,
    "FLOAT32": _float32,
    "FLOAT64": _float64,
}


# ADR-073: one write_table per staging file. A 48 h file at 5 s is 34,560
# rows, so every staging file is one row group.
ROW_GROUP_SIZE: Final = 65_536
_PARQUET_KINDS: Final = (SAMPLES_KIND, WORKSPACE_KIND)


def parquet_metadata(
    kind: str,
    *,
    run_id: str,
    model: str,
    tool_version: str,
    headers: Sequence[Mapping[str, HeaderValue]],
) -> dict[str, str]:
    """Return the R52 key-value metadata of the Parquet file of kind.

    The keys are the `PARQUET_METADATA_KEYS` names, in order. headers are the
    valid staging headers in `discover` order, written as ASCII JSON, so a
    lone surrogate in a header string is escaped. A kind other than
    `SAMPLES_KIND` or `WORKSPACE_KIND` raises ValueError.
    """
    if kind not in _PARQUET_KINDS:
        raise ValueError(f"not a Parquet file kind: {kind!r}")
    # In PARQUET_METADATA_KEYS order.
    values = (
        str(SCHEMA_VERSION),
        kind,
        tool_version,
        run_id,
        model,
        json.dumps(
            [dict(header) for header in headers],
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )
    return {
        key.name: value
        for key, value in zip(PARQUET_METADATA_KEYS, values, strict=True)
    }


def write_samples(
    path: Path, files: Iterable[FileRead], metadata: Mapping[str, str]
) -> int:
    """Write the samples of files to path; return the number of rows.

    files is consumed lazily, one at a time, and the rows go in the order
    given, which the caller makes the `SAMPLE_SORT_KEYS` order. Each file
    with at least one sample is one `write_table` call. metadata comes from
    `parquet_metadata`.

    path is opened on the local filesystem, never read as a URI.
    `OSError`, `pyarrow.ArrowException` and `ValueError` propagate. A
    `UnicodeEncodeError`, the `ValueError` of a metadata value that is not
    UTF-8 text, is raised before the file is opened. Once the writer is
    open, it is closed before any of them leaves, so the caller only removes
    the partial file.
    """
    import pyarrow as pa
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    names = [c.name for c in SAMPLE_COLUMNS]
    writer = pq.ParquetWriter(
        path,
        _arrow_schema(SAMPLE_COLUMNS, metadata),
        filesystem=pafs.LocalFileSystem(),
        sorting_columns=[
            pq.SortingColumn(names.index(key)) for key in SAMPLE_SORT_KEYS
        ],
        **_write_table_options(SAMPLE_COLUMNS),
    )
    written = 0
    try:
        # The _arrow_schema value, read back untyped to reach its fields.
        schema = writer.schema
        for read in files:
            if not read.samples:
                continue
            columns = zip(*read.samples, strict=True)
            arrays = [
                pa.array(values, type=field.type)
                for values, field in zip(columns, schema, strict=True)
            ]
            writer.write_table(
                pa.Table.from_arrays(arrays, schema=schema),
                row_group_size=ROW_GROUP_SIZE,
            )
            written += len(read.samples)
    finally:
        writer.close()
    return written


def write_workspace(
    path: Path, rows: Iterable[tuple[Cell, ...]], metadata: Mapping[str, str]
) -> int:
    """Write the walk rows to path; return the number of rows.

    The rows, in `WORKSPACE_COLUMNS` order, are sorted by
    `WORKSPACE_SORT_KEYS`, phase and job_id compared as strings, and written
    as one table. With no rows the file has the schema and no row group.
    Errors propagate as in `write_samples`.
    """
    import pyarrow as pa
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    names = [c.name for c in WORKSPACE_COLUMNS]
    keys = [names.index(key) for key in WORKSPACE_SORT_KEYS]
    ordered = sorted(rows, key=lambda row: tuple(row[i] for i in keys))
    writer = pq.ParquetWriter(
        path,
        _arrow_schema(WORKSPACE_COLUMNS, metadata),
        filesystem=pafs.LocalFileSystem(),
        sorting_columns=[pq.SortingColumn(i) for i in keys],
        **_write_table_options(WORKSPACE_COLUMNS),
    )
    try:
        if ordered:
            schema = writer.schema
            columns = zip(*ordered, strict=True)
            arrays = [
                pa.array(values, type=field.type)
                for values, field in zip(columns, schema, strict=True)
            ]
            writer.write_table(
                pa.Table.from_arrays(arrays, schema=schema),
                row_group_size=ROW_GROUP_SIZE,
            )
    finally:
        writer.close()
    return len(ordered)


def _arrow_schema(
    columns: Sequence[Column], metadata: Mapping[str, str]
) -> object:
    """Return the ADR-070 `pyarrow.Schema` of columns, carrying metadata.

    It is typed `object` because pyarrow ships no type information; the
    writers only hand it to pyarrow.
    """
    import pyarrow as pa

    types = {
        "TIMESTAMP_MS_UTC": pa.timestamp("ms", tz="UTC"),
        "STRING": pa.string(),
        "INT16": pa.int16(),
        "INT32": pa.int32(),
        "INT64": pa.int64(),
        "FLOAT32": pa.float32(),
        "FLOAT64": pa.float64(),
    }
    return pa.schema(
        [
            pa.field(c.name, types[c.dtype], nullable=c.nullable)
            for c in columns
        ],
        metadata=dict(metadata),
    )


def _write_table_options(
    columns: Sequence[Column],
) -> dict[str, str | list[str]]:
    """Return the ADR-070 `ParquetWriter` options besides schema and sort.

    Dictionary encoding is on the STRING columns only: `use_dictionary=True`
    would also encode the numeric ones.
    """
    return {
        "compression": "zstd",
        "use_dictionary": [c.name for c in columns if c.dtype == "STRING"],
        "version": "2.6",
    }


def _aggregate(name: str) -> str:
    """Return name; a name outside `STAT_AGGREGATES` fails the import."""
    if name not in STAT_AGGREGATES:
        raise ValueError(f"not a STAT_AGGREGATES name: {name!r}")
    return name


_MIN: Final = _aggregate("min")
_MEAN: Final = _aggregate("mean")
_MAX: Final = _aggregate("max")
_TOTAL: Final = _aggregate("total")
_QUANTILES: Final = {_aggregate("p95"): 0.95, _aggregate("p99"): 0.99}
_MISSING_NODE: Final = _code("missing_node")
_WARNINGS_CAPPED: Final = _code("warnings_capped")
_MISSING_MESSAGE: Final = "no valid staging file"
_CAPPED_MESSAGE: Final = f"warning list capped at {WARNINGS_CAP}"

_SAMPLE_INDEX: Final = {c.name: i for i, c in enumerate(SAMPLE_COLUMNS)}
_WALK_INDEX: Final = {c.name: i for i, c in enumerate(WORKSPACE_COLUMNS)}
_TS: Final = _SAMPLE_INDEX["ts"]
_DT_S: Final = _SAMPLE_INDEX["dt_s"]
_FLOAT32: Final = struct.Struct("f")


def _f32(value: float) -> float:
    """value rounded to float32, as a Parquet FLOAT32 cell holds it."""
    return float(_FLOAT32.unpack(_FLOAT32.pack(value))[0])


def quantile7(values: Sequence[float], q: float) -> float:
    """ADR-076: the Hyndman-Fan type 7 quantile q of values.

    values is sorted ascending and non-empty. The quantile interpolates
    linearly between the two values around position (n - 1) * q.
    """
    n = len(values)
    if n == 1:
        return values[0]
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return values[lo] + (h - lo) * (values[hi] - values[lo])


def _fsum(values: Iterable[float]) -> float:
    """`math.fsum` that never raises on finite cells (Decision 9).

    An intermediate overflow, `OverflowError` from `math.fsum`, gives inf.
    Terms holding both inf and -inf, which products x * dt_s of crafted
    cells near the float64 limit can give, make it raise `ValueError` and
    give nan here. `render_summary` refuses either result.
    """
    try:
        return math.fsum(values)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.nan


_ROUNDING: Final[Mapping[str, Callable[[float], float]]] = {
    "FLOAT32": _f32,
    "FLOAT64": float,
}


def _published(cells: Sequence[Cell], dtype: str) -> list[int | float | None]:
    """The cells of one column of dtype as the Parquet holds them.

    A str, which only a hand-built row can hold in a numeric column, raises
    ValueError.
    """
    rounding = _ROUNDING.get(dtype)
    values: list[int | float | None] = []
    for cell in cells:
        if isinstance(cell, str):
            raise ValueError(f"a {dtype} column holds a string: {cell!r}")
        if cell is None or rounding is None:
            values.append(cell)
        else:
            values.append(rounding(cell))
    return values


def _contributions(
    values: Sequence[int | float | None], dts: Sequence[float | None]
) -> tuple[list[int | float], list[float], list[float]]:
    """The non-null values, then x * dt_s and dt_s where both are non-null."""
    xs: list[int | float] = []
    products: list[float] = []
    weights: list[float] = []
    for x, dt in zip(values, dts, strict=True):
        if x is not None:
            xs.append(x)
            if dt is not None:
                products.append(x * dt)
                weights.append(dt)
    return xs, products, weights


def _statistic(
    aggregate: str,
    xs: Sequence[int | float],
    products: Sequence[float],
    weights: Sequence[float],
) -> int | float | None:
    """ADR-076: one aggregate over one column's contributions; None if absent.

    min and max keep the column's Python type, a quantile is over the rows
    and always a float, a mean is sum(x * dt) / sum(dt) and a total is
    sum(x * dt), both over the rows where x and dt_s are non-null.
    """
    if aggregate == _MIN:
        return min(xs) if xs else None
    if aggregate == _MAX:
        return max(xs) if xs else None
    if aggregate in _QUANTILES:
        if not xs:
            return None
        return quantile7(sorted(map(float, xs)), _QUANTILES[aggregate])
    if aggregate == _TOTAL:
        return _fsum(products) if products else None
    if aggregate == _MEAN:
        if not weights:
            return None
        seconds = _fsum(weights)
        return _fsum(products) / seconds if seconds > 0 else None
    raise ValueError(f"not a STAT_AGGREGATES name: {aggregate!r}")


def _stat_names(spec: StatSpec) -> tuple[tuple[str, str], ...]:
    """(aggregate, published name) of spec's aggregates, in order (ADR-071).

    A name is f"{column}_{aggregate}", or total_name for the total.
    """
    names: list[tuple[str, str]] = []
    for aggregate in spec.aggregates:
        if aggregate != _TOTAL:
            names.append((aggregate, f"{spec.column}_{aggregate}"))
        elif spec.total_name is not None:
            names.append((aggregate, spec.total_name))
        else:
            raise ValueError(f"{spec.column}: a total without a total_name")
    return tuple(names)


_STAT_NAMES: Final = {
    spec: _stat_names(spec) for spec in SAMPLE_STATS + WORKSPACE_STATS
}


def _column_stats(
    spec: StatSpec,
    xs: Sequence[int | float],
    products: Sequence[float],
    weights: Sequence[float],
) -> dict[str, Json]:
    """The defined statistics of spec over one column of one file."""
    stats: dict[str, Json] = {}
    for aggregate, name in _STAT_NAMES[spec]:
        value = _statistic(aggregate, xs, products, weights)
        if value is not None:
            stats[name] = value
    return stats


def _rows_by_column(
    specs: Iterable[HeadlineSpec],
) -> dict[str, tuple[HeadlineSpec, ...]]:
    rows: dict[str, list[HeadlineSpec]] = {}
    for spec in specs:
        rows.setdefault(spec.column, []).append(spec)
    return {column: tuple(found) for column, found in rows.items()}


# The headline rows over the samples of each phase's reported job, and over
# the walks of every reported job (ADR-077).
_PHASE_HEADLINE: Final = {
    phase: _rows_by_column(
        h for h in HEADLINE if h.source == SAMPLES_FILE and h.scope == phase
    )
    for phase in PHASES
}
_WALK_HEADLINE: Final = _rows_by_column(
    h for h in HEADLINE if h.source == WORKSPACE_FILE
)


def _columns(
    rows: Sequence[tuple[Cell, ...]], width: int
) -> list[tuple[Cell, ...]]:
    """The cells of rows, column by column."""
    if not rows:
        return [()] * width
    return list(zip(*rows, strict=True))


def _header_int(header: Mapping[str, HeaderValue], name: str) -> int:
    """Return an integer header field, which `read_header` has checked."""
    value = header[name]
    if not _is_int(value):
        raise ValueError(f"header {name} is not an integer: {value!r}")
    return value


def _header_json(header: Mapping[str, HeaderValue]) -> dict[str, Json]:
    """A copy of header, typed as JSON."""
    copy: dict[str, Json] = {}
    for name, value in header.items():
        if isinstance(value, dict):
            mount: dict[str, Json] = dict(value)
            copy[name] = mount
        else:
            copy[name] = value
    return copy


def _warning(
    code: str,
    identity: tuple[str | None, str | None, int | None, int | None],
    count: int,
    message: str,
) -> dict[str, Json]:
    phase, job_id, restart, node_index = identity
    return {
        "code": code,
        "phase": phase,
        "job_id": job_id,
        "restart": restart,
        "node_index": node_index,
        "count": count,
        "message": message,
    }


class Rollup:
    """ADR-071: fold the reads of one export into its summary.json roll-up.

    Call `add_header` for every candidate and `add_file` for every valid
    file, each in `discover` order, then `summary`. `add_file` keeps no
    row: it builds the file's node entry and folds its rows at once, so only
    the pooled headline arrays of the reported jobs span files. Nodes and
    warnings come out in identity order whatever the call order; the call
    order only decides which header is the first valid one of the run and
    of each attempt.

    The reported job of a phase is its last `--job` in argv order (R86,
    ADR-077 as amended by A2). Its rows, across every restart and node,
    feed that phase's headline values, and the walks of the reported jobs
    feed the workspace one. No earlier job of a phase is ever used.
    """

    def __init__(
        self,
        *,
        run_id: str,
        model: str,
        tool_version: str,
        jobs: Sequence[tuple[str, str]],
    ) -> None:
        self._run: dict[str, Json] = {
            "run_id": run_id,
            "model": model,
            "tool_version": tool_version,
        }
        self._jobs = tuple(jobs)
        self._reported: dict[str, str] = dict(self._jobs)
        self._intervals: tuple[int, int] | None = None
        # (phase, job_id, restart) -> nodes of the attempt's first header.
        self._attempt_nodes: dict[tuple[str, str, int], int | None] = {}
        self._present: dict[tuple[str, str, int], set[int]] = {}
        self._entries: dict[
            tuple[str, str], list[tuple[tuple[int, int], dict[str, Json]]]
        ] = {}
        # Candidate identity -> its warnings and its warnings_omitted.
        self._file_warnings: dict[
            tuple[str, str, int, int], tuple[tuple[ReadWarning, ...], int]
        ] = {}
        self._walks = 0
        # The pools, keyed by a headline key or a workspace statistic name.
        # The two never collide: a headline key has a "<scope>." prefix and
        # a statistic name has no dot.
        self._extremes: dict[str, int | float] = {}
        self._values: dict[str, array[float]] = {}
        self._products: dict[str, array[float]] = {}
        self._weights: dict[str, array[float]] = {}

    def add_header(self, head: HeaderRead) -> None:
        """Record the header pass of one candidate.

        Its warnings stand for the file until `add_file` replaces them with
        the file's, which carry them on. A valid header gives the run its
        intervals when it is the first, and its attempt its nodes when it
        is that attempt's first.
        """
        c = head.candidate
        self._file_warnings.setdefault(
            (c.phase, c.job_id, c.restart, c.node_index), (head.warnings, 0)
        )
        header = head.header
        if header is None:
            return
        if self._intervals is None:
            self._intervals = (
                _header_int(header, "interval_s"),
                _header_int(header, "workspace_interval_s"),
            )
        nodes = header["nodes"]
        self._attempt_nodes.setdefault(
            (c.phase, c.job_id, c.restart), nodes if _is_int(nodes) else None
        )

    def add_file(self, read: FileRead) -> None:
        """Build the node entry of one valid file and fold in its rows."""
        c = read.candidate
        self._file_warnings[(c.phase, c.job_id, c.restart, c.node_index)] = (
            read.warnings,
            read.warnings_omitted,
        )
        self._present.setdefault((c.phase, c.job_id, c.restart), set()).add(
            c.node_index
        )
        reported = self._reported.get(c.phase) == c.job_id
        samples = _columns(read.samples, len(SAMPLE_COLUMNS))
        dts = _published(samples[_DT_S], SAMPLE_COLUMNS[_DT_S].dtype)
        headline = _PHASE_HEADLINE.get(c.phase, {}) if reported else {}
        stats: dict[str, Json] = {}
        for spec in SAMPLE_STATS:
            index = _SAMPLE_INDEX[spec.column]
            xs, products, weights = _contributions(
                _published(samples[index], SAMPLE_COLUMNS[index].dtype), dts
            )
            stats.update(_column_stats(spec, xs, products, weights))
            for row in headline.get(spec.column, ()):
                self._fold(row.key, row.aggregate, xs, products, weights)
        self._entries.setdefault((c.phase, c.job_id), []).append(
            ((c.restart, c.node_index), _node_entry(read, dts, stats))
        )
        walks = _columns(read.walks, len(WORKSPACE_COLUMNS))
        no_dt = [None] * len(read.walks)
        self._walks += len(read.walks)
        for spec in WORKSPACE_STATS:
            index = _WALK_INDEX[spec.column]
            xs, products, weights = _contributions(
                _published(walks[index], WORKSPACE_COLUMNS[index].dtype), no_dt
            )
            for aggregate, name in _STAT_NAMES[spec]:
                self._fold(name, aggregate, xs, products, weights)
            if reported:
                for row in _WALK_HEADLINE.get(spec.column, ()):
                    self._fold(row.key, row.aggregate, xs, products, weights)

    def _fold(
        self,
        key: str,
        aggregate: str,
        xs: Sequence[int | float],
        products: Sequence[float],
        weights: Sequence[float],
    ) -> None:
        """Pool one file's contributions to aggregate under key."""
        if aggregate in (_MIN, _MAX):
            value = _statistic(aggregate, xs, products, weights)
            if value is None:
                return
            held = self._extremes.get(key)
            if held is not None:
                value = (
                    min(held, value) if aggregate == _MIN else max(held, value)
                )
            self._extremes[key] = value
        elif aggregate in _QUANTILES:
            self._values.setdefault(key, array("d")).extend(map(float, xs))
        else:
            self._products.setdefault(key, array("d")).extend(products)
            self._weights.setdefault(key, array("d")).extend(weights)

    def _pooled(self, key: str, aggregate: str) -> int | float | None:
        if aggregate in (_MIN, _MAX):
            return self._extremes.get(key)
        return _statistic(
            aggregate,
            self._values.get(key, ()),
            self._products.get(key, ()),
            self._weights.get(key, ()),
        )

    def summary(self) -> dict[str, Json]:
        """Return the summary.json object, its keys in ADR-071 order."""
        interval_s, workspace_interval_s = self._intervals or (None, None)
        jobs: list[Json] = []
        for phase, job_id in self._jobs:
            entries = sorted(
                self._entries.get((phase, job_id), ()), key=lambda e: e[0]
            )
            nodes: list[Json] = [entry for _, entry in entries]
            jobs.append({"phase": phase, "job_id": job_id, "nodes": nodes})
        workspace: dict[str, Json] = {}
        for spec in WORKSPACE_STATS:
            for aggregate, name in _STAT_NAMES[spec]:
                value = self._pooled(name, aggregate)
                if value is not None:
                    workspace[name] = value
        job_ids: dict[str, Json] = {
            phase: self._reported[phase]
            for phase in PHASES
            if phase in self._reported
        }
        values: dict[str, Json] = {}
        for row in HEADLINE:
            value = self._pooled(row.key, row.aggregate)
            if value is not None:
                values[row.key] = value
        return {
            "kind": SUMMARY_KIND,
            "schema_version": SCHEMA_VERSION,
            "run": {
                **self._run,
                "interval_s": interval_s,
                "workspace_interval_s": workspace_interval_s,
            },
            "jobs": jobs,
            "workspace": {"walks": self._walks, "stats": workspace},
            "headline": {"job_ids": job_ids, "values": values},
            "warnings": self._warning_list(),
        }

    def _warning_list(self) -> list[Json]:
        """Decision 8: the file warnings, then the missing nodes, capped.

        At most `WARNINGS_CAP` records are built, and a missing node past
        them is counted, not listed. An attempt whose header claims 10**9
        nodes thus costs its present nodes plus the room left in the list.
        """
        kept: list[tuple[int, dict[str, Json]]] = []
        left_out = 0
        for identity in sorted(self._file_warnings):
            warnings, omitted = self._file_warnings[identity]
            left_out += omitted
            for w in warnings:
                if len(kept) < WARNINGS_CAP:
                    record = _warning(w.code, identity, w.count, w.message)
                    kept.append((w.count, record))
                else:
                    left_out += w.count
        for attempt in sorted(self._attempt_nodes):
            nodes = self._attempt_nodes[attempt]
            if nodes is None:
                continue
            present = self._present.get(attempt, set())
            missing = nodes - sum(1 for index in present if index < nodes)
            absent = (i for i in range(nodes) if i not in present)
            for index in islice(absent, WARNINGS_CAP - len(kept)):
                record = _warning(
                    _MISSING_NODE, (*attempt, index), 1, _MISSING_MESSAGE
                )
                kept.append((1, record))
                missing -= 1
            left_out += missing
        if left_out == 0:
            return [record for _, record in kept]
        left_out += sum(count for count, _ in kept[WARNINGS_CAP - 1 :])
        records: list[Json] = [record for _, record in kept[: WARNINGS_CAP - 1]]
        records.append(
            _warning(
                _WARNINGS_CAPPED,
                (None, None, None, None),
                left_out,
                _CAPPED_MESSAGE,
            )
        )
        return records


def _node_entry(
    read: FileRead, dts: Sequence[float | None], stats: dict[str, Json]
) -> dict[str, Json]:
    """ADR-071: one file's node entry; dts are its published dt_s cells."""
    entry: dict[str, Json] = {"header": _header_json(read.header)}
    if read.samples:
        entry["first_ts"] = read.samples[0][_TS]
        entry["last_ts"] = read.samples[-1][_TS]
    entry["samples"] = len(read.samples)
    present = [dt for dt in dts if dt is not None]
    interval = _header_int(read.header, "interval_s")
    # dt_s > 1.5 * interval_s, compared exactly as 2 * dt_s > 3 * interval_s:
    # the int side never converts to float, so no interval_s can overflow.
    entry["gaps"] = sum(1 for dt in present if 2 * dt > 3 * interval)
    if present:
        entry["max_gap_s"] = max(present)
    entry["stats"] = stats
    return entry


def render_summary(summary: Mapping[str, Json]) -> bytes:
    """The bytes of summary.json: indented ASCII JSON and a final newline.

    A non-finite number raises ValueError (allow_nan=False). Every int stays
    far below the int-to-str digit limit, because `read_header` bounds the
    header ints to int64.
    """
    text = json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


_PROG: Final = "hpcmu-telemetry-export"
_JOB_ID: Final = re.compile(r"[0-9]+")


def _job(text: str) -> tuple[str, str]:
    """Parse a --job value, PHASE:JOB_ID, into (phase, job_id)."""
    phase, _, job_id = text.partition(":")
    if phase not in PHASES or _JOB_ID.fullmatch(job_id) is None:
        raise argparse.ArgumentTypeError(
            f"must be PHASE:JOB_ID with PHASE one of {', '.join(PHASES)} "
            f"and JOB_ID ASCII digits, got {text!r}"
        )
    return phase, job_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Export the staging files of a run's jobs.",
    )
    parser.add_argument(
        "--staging", type=Path, required=True, help="staging directory"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory"
    )
    parser.add_argument("--run-id", required=True, help="run id")
    parser.add_argument("--model", required=True, help="model name")
    parser.add_argument(
        "--tool-version", required=True, help="hpc-model-utils version"
    )
    parser.add_argument(
        "--job",
        type=_job,
        action="append",
        required=True,
        metavar="PHASE:JOB_ID",
        help="a job to export; the last one of a phase is the reported one",
    )
    return parser


def _fail(reason: str) -> None:
    """Write reason to stderr as one line, its whitespace collapsed."""
    sys.stderr.write(f"{_PROG}: {' '.join(reason.split())}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """ADR-073: export the staging files of the --job jobs to --out.

    The files are written to temp names in --out and renamed into place in
    `EXPORT_FILES` order. On a failure only the temps are removed, so a final
    name renamed before the failure stays, without `summary.json`.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    jobs: list[tuple[str, str]] = args.job
    if len(set(jobs)) < len(jobs):
        parser.error("duplicate --job PHASE:JOB_ID")
    staging: Path = args.staging
    out: Path = args.out
    run_id: str = args.run_id
    model: str = args.model
    tool_version: str = args.tool_version
    try:
        candidates = discover(staging, jobs)
    except OSError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return 1
    if not candidates:
        _fail("no staging file for the given jobs")
        return 3
    heads = [read_header(candidate) for candidate in candidates]
    headers = [head.header for head in heads if head.header is not None]
    if not headers:
        _fail("no valid staging header")
        return 1
    rollup = Rollup(
        run_id=run_id, model=model, tool_version=tool_version, jobs=jobs
    )
    for head in heads:
        rollup.add_header(head)
    samples_metadata = parquet_metadata(
        SAMPLES_KIND,
        run_id=run_id,
        model=model,
        tool_version=tool_version,
        headers=headers,
    )
    workspace_metadata = parquet_metadata(
        WORKSPACE_KIND,
        run_id=run_id,
        model=model,
        tool_version=tool_version,
        headers=headers,
    )
    # First load of pyarrow: every header is read and checked by now.
    import pyarrow as pa

    temps = {name: out / f".{name}.tmp" for name in EXPORT_FILES}
    walks: list[tuple[Cell, ...]] = []

    def reads() -> Iterator[FileRead]:
        # One file at a time: read it, fold it into the roll-up, keep its
        # walks, and hand it on unchanged to the samples writer.
        for head in heads:
            if head.header is not None:
                read = read_body(head)
                rollup.add_file(read)
                walks.extend(read.walks)
                yield read

    try:
        out.mkdir(parents=True, exist_ok=True)
        try:
            write_samples(temps[SAMPLES_FILE], reads(), samples_metadata)
            write_workspace(temps[WORKSPACE_FILE], walks, workspace_metadata)
            temps[SUMMARY_FILE].write_bytes(render_summary(rollup.summary()))
            for name in EXPORT_FILES:
                os.replace(temps[name], out / name)
        finally:
            for temp in temps.values():
                temp.unlink(missing_ok=True)
    except (OSError, ValueError, pa.ArrowException) as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
