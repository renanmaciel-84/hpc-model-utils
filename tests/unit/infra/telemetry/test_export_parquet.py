"""ADR-070/ADR-091: the export's Parquet writer over reader-read staging."""

from __future__ import annotations

import dataclasses
import json
import math
import struct
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hpc_model_utils.infra.telemetry.export import (
    ROW_GROUP_SIZE,
    Cell,
    FileRead,
    ReadWarning,
    discover,
    parquet_metadata,
    read_body,
    read_header,
    write_samples,
    write_workspace,
)
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    DTYPES,
    HEADER_KIND,
    PARQUET_METADATA_KEYS,
    SAMPLE_COLUMNS,
    SAMPLE_SORT_KEYS,
    SAMPLES_FILE,
    SAMPLES_KIND,
    SCHEMA_VERSION,
    SUMMARY_KIND,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    WORKSPACE_KIND,
    WORKSPACE_SORT_KEYS,
    Column,
)
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    header,
    sample,
    walk,
    write_staging,
)

# Requirement 3 of ticket-042: the Arrow type of each dtype token.
_ARROW_TYPES: Final = {
    "TIMESTAMP_MS_UTC": pa.timestamp("ms", tz="UTC"),
    "STRING": pa.string(),
    "INT16": pa.int16(),
    "INT32": pa.int32(),
    "INT64": pa.int64(),
    "FLOAT32": pa.float32(),
    "FLOAT64": pa.float64(),
}
_DICTIONARY: Final = "RLE_DICTIONARY"
_KEYS: Final = [key.name for key in PARQUET_METADATA_KEYS]
_KIND_KEY: Final = _KEYS[1]
_HEADERS_KEY: Final = _KEYS[5]
_JOBS: Final = frozenset(
    {("model", "1000"), ("finalize", "1001"), ("model", "1234")}
)
_SAMPLE_NAMES: Final = [c.name for c in SAMPLE_COLUMNS]
_WORKSPACE_NAMES: Final = [c.name for c in WORKSPACE_COLUMNS]

_MODEL: Final = header("model", "1000")
_FINALIZE: Final = header("finalize", "1001")
_HEADER_ONLY: Final = header("model", "1234")
# Lone surrogates, built at run time: a str literal holding one is a
# Literal type that mypy cannot write to its UTF-8 cache.
_LONE_HIGH: Final = chr(0xD800)
_LONE_LOW: Final = chr(0xDC80)
_SURROGATE_NODE: Final = f"fake-node0{_LONE_LOW}"


def _sample(head: dict[str, JsonValue], seq: int) -> dict[str, JsonValue]:
    """A sample whose dt_s, cpu_util and one FLOAT64 cell are JSON ints."""
    return sample(
        head,
        seq,
        dt_s=5,
        cpu_util=1,
        cpu_busy_cores=0.1 * (seq + 1),
        load1=2.5,
        disk_read_bytes_per_s=4096,
        net_rx_bytes_per_s=1e9 / 3,
        mem_used_bytes=2**40 + seq,
        procs_running=3,
        cpu_active_cores=4,
    )


def _walk(head: dict[str, JsonValue], seq: int) -> dict[str, JsonValue]:
    """The walk after sample seq of head's file."""
    return walk(
        head,
        ts=ORIGIN_MS + (seq + 1) * 5_000,
        apparent_bytes=2**33,
        files=3,
        walk_s=0.3,
    )


def _reads(staging: Path) -> list[FileRead]:
    reads: list[FileRead] = []
    for candidate in discover(staging, _JOBS):
        head = read_header(candidate)
        assert head.header is not None
        reads.append(read_body(head))
    return reads


def _read_one(
    tmp_path: Path, records: Sequence[Mapping[str, JsonValue]]
) -> FileRead:
    """The read of one staging file of _MODEL holding records."""
    staging = tmp_path / "staging"
    staging.mkdir()
    write_staging(staging, _MODEL, records)
    (read,) = _reads(staging)
    return read


def _metadata(kind: str, reads: Sequence[FileRead]) -> dict[str, str]:
    return parquet_metadata(
        kind,
        run_id="run-1",
        model="fake",
        tool_version="9.9.9",
        headers=[read.header for read in reads],
    )


def _walks(reads: Sequence[FileRead]) -> list[tuple[Cell, ...]]:
    return [row for read in reads for row in read.walks]


def _cell(value: JsonValue) -> Cell:
    assert value is None or isinstance(value, int | float | str)
    return value


def _walk_row(record: Mapping[str, JsonValue]) -> tuple[Cell, ...]:
    """A builder walk line as the reader keeps it, in column order."""
    return tuple(_cell(record[c.name]) for c in WORKSPACE_COLUMNS)


@dataclass(frozen=True, slots=True)
class _Export:
    reads: list[FileRead]
    samples: Path
    workspace: Path
    sample_rows: int
    walk_rows: int


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    """Two files with samples and walks and one header-only file."""
    directory = tmp_path / "staging"
    directory.mkdir()
    write_staging(
        directory,
        _MODEL,
        [
            _sample(_MODEL, 0),
            _walk(_MODEL, 0),
            _sample(_MODEL, 1),
            _sample(_MODEL, 2),
            _walk(_MODEL, 2),
        ],
    )
    write_staging(
        directory,
        _FINALIZE,
        [_sample(_FINALIZE, 0), _walk(_FINALIZE, 0), _sample(_FINALIZE, 1)],
    )
    write_staging(directory, _HEADER_ONLY)
    return directory


@pytest.fixture
def export(tmp_path: Path, staging: Path) -> _Export:
    reads = _reads(staging)
    out = tmp_path / "out"
    out.mkdir()
    samples = out / SAMPLES_FILE
    workspace = out / WORKSPACE_FILE
    sample_rows = write_samples(
        samples, iter(reads), _metadata(SAMPLES_KIND, reads)
    )
    walk_rows = write_workspace(
        workspace, _walks(reads), _metadata(WORKSPACE_KIND, reads)
    )
    return _Export(reads, samples, workspace, sample_rows, walk_rows)


def _kv(path: Path) -> dict[str, str]:
    metadata = pq.ParquetFile(path).schema_arrow.metadata
    decoded: dict[str, str] = {
        key.decode(): value.decode() for key, value in metadata.items()
    }
    return decoded


def _decoded(path: Path) -> list[tuple[Cell, ...]]:
    """The rows of path, in file order, with ts as epoch milliseconds."""
    table = pq.read_table(path)
    columns: list[list[Cell]] = [
        (
            column.cast(pa.int64())
            if pa.types.is_timestamp(column.type)
            else column
        ).to_pylist()
        for column in table.columns
    ]
    return list(zip(*columns, strict=True))


def _as_float32(value: int | float) -> float:
    rounded: float = struct.unpack("f", struct.pack("f", value))[0]
    return rounded


def _stored(
    row: tuple[Cell, ...], columns: Sequence[Column]
) -> tuple[Cell, ...]:
    """row as Arrow stores it: FLOAT32 cells rounded to single precision."""
    return tuple(
        _as_float32(value)
        if column.dtype == "FLOAT32" and isinstance(value, int | float)
        else value
        for value, column in zip(row, columns, strict=True)
    )


def _sample_rows(reads: Sequence[FileRead]) -> list[tuple[Cell, ...]]:
    return [
        _stored(row, SAMPLE_COLUMNS) for read in reads for row in read.samples
    ]


def _sorting(path: Path) -> list[list[tuple[int, bool, bool]]]:
    meta = pq.ParquetFile(path).metadata
    groups: list[list[tuple[int, bool, bool]]] = [
        [
            (s.column_index, s.descending, s.nulls_first)
            for s in meta.row_group(i).sorting_columns
        ]
        for i in range(meta.num_row_groups)
    ]
    return groups


def _ascending(
    names: Sequence[str], keys: Sequence[str]
) -> list[tuple[int, bool, bool]]:
    return [(names.index(key), False, False) for key in keys]


# Schema and metadata.


def test_schema_type_table_covers_every_dtype() -> None:
    # Every token is used by a sample column, so the sample schema test
    # below reaches every entry of the writer's mapping.
    assert _ARROW_TYPES.keys() == DTYPES
    assert {c.dtype for c in SAMPLE_COLUMNS} == DTYPES


def test_write_samples_schema_is_sample_columns(export: _Export) -> None:
    schema = pq.read_schema(export.samples)

    assert [(f.name, f.type, f.nullable) for f in schema] == [
        (c.name, _ARROW_TYPES[c.dtype], c.nullable) for c in SAMPLE_COLUMNS
    ]


def test_write_workspace_schema_is_workspace_columns(export: _Export) -> None:
    schema = pq.read_schema(export.workspace)

    assert [(f.name, f.type, f.nullable) for f in schema] == [
        (c.name, _ARROW_TYPES[c.dtype], c.nullable) for c in WORKSPACE_COLUMNS
    ]


def test_write_samples_metadata_is_parquet_metadata(export: _Export) -> None:
    expected = _metadata(SAMPLES_KIND, export.reads)

    assert _kv(export.samples) == expected
    assert expected[_KIND_KEY] == SAMPLES_KIND


def test_write_workspace_metadata_is_parquet_metadata(
    export: _Export,
) -> None:
    expected = _metadata(WORKSPACE_KIND, export.reads)

    assert _kv(export.workspace) == expected
    assert expected[_KIND_KEY] == WORKSPACE_KIND


def test_parquet_metadata_values_in_key_order(export: _Export) -> None:
    headers = [read.header for read in export.reads]

    metadata = _metadata(SAMPLES_KIND, export.reads)

    assert list(metadata.items()) == list(
        zip(
            _KEYS,
            (
                str(SCHEMA_VERSION),
                SAMPLES_KIND,
                "9.9.9",
                "run-1",
                "fake",
                json.dumps(
                    headers,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
            ),
            strict=True,
        )
    )
    assert json.loads(metadata[_HEADERS_KEY]) == headers
    assert [h["phase"] for h in headers] == ["finalize", "model", "model"]


def test_parquet_metadata_headers_are_ascii_json(export: _Export) -> None:
    head = {
        **export.reads[1].header,
        "instance_type": "fake-é",
        "node": _SURROGATE_NODE,
    }

    metadata = parquet_metadata(
        WORKSPACE_KIND,
        run_id="run-1",
        model="fake",
        tool_version="9.9.9",
        headers=[head],
    )

    value = metadata[_HEADERS_KEY]
    assert value.isascii()
    assert "\\u00e9" in value
    assert "\\udc80" in value
    assert json.loads(value) == [head]


@pytest.mark.parametrize("kind", [HEADER_KIND, SUMMARY_KIND, "samples", ""])
def test_parquet_metadata_other_kind_raises_value_error(kind: str) -> None:
    with pytest.raises(ValueError, match="not a Parquet file kind"):
        parquet_metadata(
            kind, run_id="run-1", model="fake", tool_version="9.9.9", headers=[]
        )


# Writer options.


def test_write_samples_options_format_version_and_zstd(
    export: _Export,
) -> None:
    meta = pq.ParquetFile(export.samples).metadata

    assert meta.format_version == "2.6"
    assert meta.num_row_groups > 0
    assert {
        meta.row_group(i).column(j).compression
        for i in range(meta.num_row_groups)
        for j in range(meta.num_columns)
    } == {"ZSTD"}


def test_write_samples_options_dictionary_on_string_columns_only(
    export: _Export,
) -> None:
    meta = pq.ParquetFile(export.samples).metadata
    strings = {c.name for c in SAMPLE_COLUMNS if c.dtype == "STRING"}

    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        encoded = {
            _SAMPLE_NAMES[j]
            for j in range(group.num_columns)
            if _DICTIONARY in group.column(j).encodings
        }
        assert encoded == strings == {"phase", "job_id", "node"}


def test_write_samples_options_sorting_columns_are_sample_sort_keys(
    export: _Export,
) -> None:
    expected = _ascending(_SAMPLE_NAMES, SAMPLE_SORT_KEYS)

    assert _sorting(export.samples) == [expected, expected]


def test_write_samples_options_one_row_group_per_file_with_samples(
    export: _Export,
) -> None:
    meta = pq.ParquetFile(export.samples).metadata
    with_samples = [read for read in export.reads if read.samples]

    assert len(export.reads) == 3
    assert meta.num_row_groups == len(with_samples) == 2
    assert [meta.row_group(i).num_rows for i in range(2)] == [
        len(read.samples) for read in with_samples
    ]
    assert ROW_GROUP_SIZE == 65_536


def test_write_workspace_options_match_samples(export: _Export) -> None:
    meta = pq.ParquetFile(export.workspace).metadata
    group = meta.row_group(0)

    assert meta.format_version == "2.6"
    assert meta.num_row_groups == 1
    assert {group.column(j).compression for j in range(meta.num_columns)} == {
        "ZSTD"
    }
    assert {
        _WORKSPACE_NAMES[j]
        for j in range(meta.num_columns)
        if _DICTIONARY in group.column(j).encodings
    } == {c.name for c in WORKSPACE_COLUMNS if c.dtype == "STRING"}
    assert _sorting(export.workspace) == [
        _ascending(_WORKSPACE_NAMES, WORKSPACE_SORT_KEYS)
    ]


# Rows and nulls.


def test_write_samples_rows_ts_round_trip(export: _Export) -> None:
    ts = pq.read_table(export.samples).column("ts").cast(pa.int64())

    assert ts.to_pylist() == [
        row[0] for read in export.reads for row in read.samples
    ]


def test_write_samples_rows_job_pids_all_null_int32(export: _Export) -> None:
    column = pq.read_table(export.samples).column("job_pids")

    assert column.type == pa.int32()
    assert column.null_count == len(column) == export.sample_rows == 5


def test_write_samples_rows_float_cells_read_back_as_floats(
    export: _Export,
) -> None:
    dt_s = _SAMPLE_NAMES.index("dt_s")
    cpu_util = _SAMPLE_NAMES.index("cpu_util")
    floats = [
        i for i, c in enumerate(SAMPLE_COLUMNS) if c.dtype.startswith("FLOAT")
    ]
    written = [row for read in export.reads for row in read.samples]
    assert {type(row[dt_s]) for row in written} == {int}
    assert {type(row[cpu_util]) for row in written} == {int}

    decoded = _decoded(export.samples)

    for row, back in zip(written, decoded, strict=True):
        for i in floats:
            value = row[i]
            if value is None:
                assert back[i] is None
                continue
            assert isinstance(value, int | float)
            assert isinstance(back[i], float)
            if SAMPLE_COLUMNS[i].dtype == "FLOAT32":
                assert back[i] == _as_float32(value)
            else:
                assert back[i] == value


def test_write_samples_rows_equal_the_reads_in_order(export: _Export) -> None:
    assert _decoded(export.samples) == _sample_rows(export.reads)
    assert export.sample_rows == 5


def test_write_workspace_rows_sorted_by_sort_keys(
    tmp_path: Path, export: _Export
) -> None:
    path = tmp_path / WORKSPACE_FILE
    unsorted = _walks(export.reads[::-1])

    count = write_workspace(
        path, unsorted, _metadata(WORKSPACE_KIND, export.reads)
    )

    keys = [_WORKSPACE_NAMES.index(key) for key in WORKSPACE_SORT_KEYS]
    decoded = _decoded(path)
    assert count == export.walk_rows == 3
    assert [tuple(row[i] for i in keys) for row in decoded] == [
        ("finalize", "1001", 0, ORIGIN_MS + 5_000),
        ("model", "1000", 0, ORIGIN_MS + 5_000),
        ("model", "1000", 0, ORIGIN_MS + 15_000),
    ]
    assert decoded == [
        _stored(row, WORKSPACE_COLUMNS) for row in _walks(export.reads)
    ]


def test_write_workspace_rows_sort_job_restart_then_ts(tmp_path: Path) -> None:
    def row(job_id: str, restart: int, ts_s: int) -> tuple[Cell, ...]:
        record = walk(
            header("model", job_id, restart), ts=ORIGIN_MS + ts_s * 1000
        )
        return _walk_row(record)

    rows = [
        row("1234", 0, 5),
        row("1000", 1, 5),
        row("1000", 0, 65),
        row("1000", 0, 5),
    ]
    path = tmp_path / WORKSPACE_FILE

    write_workspace(path, iter(rows), _metadata(WORKSPACE_KIND, []))

    assert _decoded(path) == [rows[3], rows[2], rows[1], rows[0]]


def test_write_workspace_rows_zero_walks_give_an_empty_file_with_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / WORKSPACE_FILE
    metadata = _metadata(WORKSPACE_KIND, [])

    assert write_workspace(path, [], metadata) == 0

    parquet = pq.ParquetFile(path)
    assert parquet.metadata.num_rows == 0
    assert parquet.metadata.num_row_groups == 0
    assert [(f.name, f.type, f.nullable) for f in parquet.schema_arrow] == [
        (c.name, _ARROW_TYPES[c.dtype], c.nullable) for c in WORKSPACE_COLUMNS
    ]
    assert _kv(path) == metadata


def test_write_samples_rows_header_only_input_gives_an_empty_file(
    tmp_path: Path, staging: Path
) -> None:
    (header_only,) = [read for read in _reads(staging) if not read.samples]
    path = tmp_path / SAMPLES_FILE

    count = write_samples(
        path, [header_only], _metadata(SAMPLES_KIND, [header_only])
    )

    parquet = pq.ParquetFile(path)
    assert count == parquet.metadata.num_rows == 0
    assert parquet.metadata.num_row_groups == 0
    assert parquet.schema_arrow.names == _SAMPLE_NAMES


# Crafted values and failures (securitySensitive: no crafted staging value
# may leave the export as an unhandled exception).


def test_lone_surrogate_string_cell_is_a_bad_record_and_the_export_writes(
    tmp_path: Path,
) -> None:
    bad = sample(_MODEL, 1, node=f"fake-node0{_LONE_HIGH}")
    read = _read_one(tmp_path, [sample(_MODEL, 0), bad, sample(_MODEL, 2)])
    path = tmp_path / SAMPLES_FILE

    count = write_samples(path, [read], _metadata(SAMPLES_KIND, [read]))

    assert read.warnings == (
        ReadWarning("bad_record", "sample column node breaks its dtype", 1),
    )
    seq = _SAMPLE_NAMES.index("seq")
    assert count == 2
    assert [row[seq] for row in _decoded(path)] == [0, 2]


def test_lone_surrogate_header_node_drops_its_rows_and_the_export_writes(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    head = {**_MODEL, "node": _SURROGATE_NODE}
    write_staging(
        staging,
        head,
        [sample(head, 0), sample(head, 1), walk(head, ts=ORIGIN_MS + 5_000)],
    )
    (read,) = _reads(staging)
    samples, workspace = tmp_path / SAMPLES_FILE, tmp_path / WORKSPACE_FILE

    assert write_samples(samples, [read], _metadata(SAMPLES_KIND, [read])) == 0
    assert (
        write_workspace(
            workspace, read.walks, _metadata(WORKSPACE_KIND, [read])
        )
        == 0
    )

    assert read.header["node"] == _SURROGATE_NODE
    assert read.warnings == (
        ReadWarning("bad_record", "sample column node breaks its dtype", 2),
        ReadWarning("bad_record", "walk column node breaks its dtype", 1),
    )
    for path in (samples, workspace):
        assert pq.ParquetFile(path).metadata.num_rows == 0
        assert json.loads(_kv(path)[_HEADERS_KEY]) == [read.header]


def test_write_samples_lone_surrogate_cell_raises_value_error_after_closing(
    tmp_path: Path, export: _Export
) -> None:
    good = export.reads[0]
    node = _SAMPLE_NAMES.index("node")
    row = list(good.samples[0])
    row[node] = f"fake-node0{_LONE_HIGH}"
    crafted = dataclasses.replace(good, samples=(tuple(row),))
    path = tmp_path / SAMPLES_FILE

    # The reader never yields this row; a caller that builds one gets a
    # ValueError, which ticket-044 maps to exit 1.
    with pytest.raises(UnicodeEncodeError) as raised:
        write_samples(path, [crafted], _metadata(SAMPLES_KIND, [good]))

    assert isinstance(raised.value, ValueError)
    # A footer is written only by close(), so the file reads back. On
    # success the writer's __del__ would close it too; only a failure, whose
    # traceback keeps the writer alive, shows the finally.
    assert pq.ParquetFile(path).metadata.num_row_groups == 0


def test_write_workspace_lone_surrogate_cell_raises_value_error_after_closing(
    tmp_path: Path,
) -> None:
    record = walk({**_MODEL, "node": _SURROGATE_NODE}, ts=ORIGIN_MS + 5_000)
    crafted = _walk_row(record)
    path = tmp_path / WORKSPACE_FILE

    with pytest.raises(UnicodeEncodeError) as raised:
        write_workspace(path, [crafted], _metadata(WORKSPACE_KIND, []))

    assert isinstance(raised.value, ValueError)
    assert pq.ParquetFile(path).metadata.num_row_groups == 0


def test_write_samples_lone_surrogate_metadata_raises_before_opening(
    tmp_path: Path, export: _Export
) -> None:
    metadata = {
        **_metadata(SAMPLES_KIND, export.reads),
        _KEYS[3]: f"run-{_LONE_HIGH}",
    }
    path = tmp_path / SAMPLES_FILE

    with pytest.raises(UnicodeEncodeError) as raised:
        write_samples(path, export.reads, metadata)

    assert isinstance(raised.value, ValueError)
    assert not path.exists()


def _from_bits(bits: str) -> float:
    value: float = struct.unpack("<f", bytes.fromhex(bits))[0]
    return value


# float32's largest finite value, and the float64 just above it.
_FLOAT32_TOP: Final = _from_bits("ffff7f7f")
_PAST_FLOAT32_TOP: Final = math.nextafter(_FLOAT32_TOP, math.inf)
_KEPT: Final = [
    pytest.param("cpu_util", 2**24, id="float32-int-2**24"),
    pytest.param("cpu_util", -(2**24), id="float32-int--2**24"),
    pytest.param("disk_read_bytes_per_s", 2**53, id="float64-int-2**53"),
    pytest.param("disk_read_bytes_per_s", -(2**53), id="float64-int--2**53"),
    pytest.param("cpu_util", _FLOAT32_TOP, id="float32-max"),
    pytest.param("cpu_util", -_FLOAT32_TOP, id="float32--max"),
]
_BROKEN: Final = [
    pytest.param("cpu_util", 2**24 + 1, id="float32-int-2**24+1"),
    pytest.param("cpu_util", -(2**24 + 1), id="float32-int--(2**24+1)"),
    pytest.param("disk_read_bytes_per_s", 2**53 + 1, id="float64-int-2**53+1"),
    pytest.param(
        "disk_read_bytes_per_s", -(2**53 + 1), id="float64-int--(2**53+1)"
    ),
    pytest.param("cpu_util", _PAST_FLOAT32_TOP, id="float32-past-max"),
    pytest.param("cpu_util", 3.5e38, id="float32-overflow"),
    pytest.param("cpu_util", -3.5e38, id="float32--overflow"),
]


def _finite(rows: Sequence[tuple[Cell, ...]]) -> bool:
    floats = [cell for row in rows for cell in row if isinstance(cell, float)]
    return bool(floats) and all(math.isfinite(cell) for cell in floats)


@pytest.mark.parametrize(("name", "value"), _KEPT)
def test_float_cell_at_its_bound_is_kept_and_written_exactly(
    tmp_path: Path, name: str, value: int | float
) -> None:
    middle = {**sample(_MODEL, 1), name: value}
    read = _read_one(tmp_path, [sample(_MODEL, 0), middle, sample(_MODEL, 2)])
    path = tmp_path / SAMPLES_FILE

    count = write_samples(path, [read], _metadata(SAMPLES_KIND, [read]))

    column = _SAMPLE_NAMES.index(name)
    assert read.warnings == ()
    assert count == 3
    assert [row[column] for row in _decoded(path)] == [None, float(value), None]


@pytest.mark.parametrize(("name", "value"), _BROKEN)
def test_float_cell_past_its_bound_is_a_bad_record_and_the_export_writes(
    tmp_path: Path, name: str, value: int | float
) -> None:
    # Arrow refuses an int past the float's exact range, so without the
    # reader's bound one cell would fail the whole export; a float32 overflow
    # it stores as inf.
    bad = {**sample(_MODEL, 1), name: value}
    read = _read_one(tmp_path, [sample(_MODEL, 0), bad, sample(_MODEL, 2)])
    path = tmp_path / SAMPLES_FILE

    count = write_samples(path, [read], _metadata(SAMPLES_KIND, [read]))

    decoded = _decoded(path)
    seq = _SAMPLE_NAMES.index("seq")
    assert read.warnings == (
        ReadWarning("bad_record", f"sample column {name} breaks its dtype", 1),
    )
    assert count == 2
    assert [row[seq] for row in decoded] == [0, 2]
    assert _finite(decoded)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2**24 + 1, id="int-2**24+1"),
        pytest.param(3.5e38, id="overflow"),
    ],
)
def test_walk_float_cell_past_its_bound_is_a_bad_record_and_the_export_writes(
    tmp_path: Path, value: int | float
) -> None:
    first, last = ORIGIN_MS + 5_000, ORIGIN_MS + 15_000
    read = _read_one(
        tmp_path,
        [
            walk(_MODEL, ts=first, walk_s=0.25),
            walk(_MODEL, ts=ORIGIN_MS + 10_000, walk_s=value),
            walk(_MODEL, ts=last, walk_s=2**24),
        ],
    )
    path = tmp_path / WORKSPACE_FILE

    count = write_workspace(path, read.walks, _metadata(WORKSPACE_KIND, [read]))

    decoded = _decoded(path)
    ts = _WORKSPACE_NAMES.index("ts")
    walk_s = _WORKSPACE_NAMES.index("walk_s")
    assert read.warnings == (
        ReadWarning("bad_record", "walk column walk_s breaks its dtype", 1),
    )
    assert count == 2
    assert [(row[ts], row[walk_s]) for row in decoded] == [
        (first, 0.25),
        (last, float(2**24)),
    ]
    assert _finite(decoded)


def test_writers_open_a_relative_new_path_on_the_local_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export: _Export
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("relative").mkdir()
    samples = Path("relative") / SAMPLES_FILE
    workspace = Path("relative") / WORKSPACE_FILE
    assert not samples.exists()
    assert not workspace.exists()

    sample_rows = write_samples(
        samples, export.reads, _metadata(SAMPLES_KIND, export.reads)
    )
    walk_rows = write_workspace(
        workspace, _walks(export.reads), _metadata(WORKSPACE_KIND, export.reads)
    )

    assert (sample_rows, walk_rows) == (export.sample_rows, export.walk_rows)
    local = tmp_path / "relative"
    assert _decoded(local / SAMPLES_FILE) == _decoded(export.samples)
    assert _decoded(local / WORKSPACE_FILE) == _decoded(export.workspace)


def test_writers_refuse_a_uri_shaped_path_and_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolved as a URI, "mock:" would write to an in-memory filesystem and
    # report success; pinned to the local filesystem it is an Arrow error,
    # which ticket-044 maps to exit 1.
    monkeypatch.chdir(tmp_path)
    samples = Path(f"mock:{SAMPLES_FILE}")
    workspace = Path(f"mock:{WORKSPACE_FILE}")
    refused = "Expected a local filesystem path"

    with pytest.raises(pa.ArrowInvalid, match=refused):
        write_samples(samples, [], _metadata(SAMPLES_KIND, []))
    with pytest.raises(pa.ArrowInvalid, match=refused):
        write_workspace(workspace, [], _metadata(WORKSPACE_KIND, []))

    assert list(tmp_path.iterdir()) == []


def test_write_samples_failing_input_closes_the_writer(
    tmp_path: Path, export: _Export
) -> None:
    pulled: list[FileRead] = []

    def files() -> Iterator[FileRead]:
        pulled.append(export.reads[0])
        yield export.reads[0]
        raise OSError("staging read failed")

    path = tmp_path / SAMPLES_FILE

    with pytest.raises(OSError, match="staging read failed") as raised:
        write_samples(path, files(), _metadata(SAMPLES_KIND, export.reads))

    assert raised.value is not None
    meta = pq.ParquetFile(path).metadata
    assert pulled == [export.reads[0]]
    assert meta.num_row_groups == 1
    assert meta.num_rows == len(export.reads[0].samples)


def test_write_samples_missing_directory_raises_os_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "absent" / SAMPLES_FILE

    with pytest.raises(OSError):
        write_samples(path, [], _metadata(SAMPLES_KIND, []))

    assert not path.parent.exists()


_LAZY_SCRIPT = """\
import json
import sys
from pathlib import Path

from hpc_model_utils.infra.telemetry import export

reads = [
    export.read_body(head)
    for head in map(
        export.read_header,
        export.discover(Path(sys.argv[1]), {("model", "1000")}),
    )
]
export.parquet_metadata(
    export.SAMPLES_KIND,
    run_id="run-1",
    model="fake",
    tool_version="9.9.9",
    headers=[read.header for read in reads],
)
loaded = [m for m in sys.modules if m.partition(".")[0] == "pyarrow"]
print(json.dumps([len(reads), loaded]))
"""


def test_reader_and_parquet_metadata_leave_pyarrow_unloaded(
    staging: Path,
) -> None:
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _LAZY_SCRIPT, str(staging)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [1, []]
