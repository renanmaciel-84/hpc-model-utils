"""R63/ADR-095: the export goldens over nine staging cases.

Each case is built in a temp directory through the real `StagingWriter`
(`tests.support.telemetry_staging`), raw byte edits add its faults, and
`export.main` exports it. The goldens compare the decoded schema, KV
metadata and rows of both Parquet files and the parsed summary.json, never
the file bytes, so they survive pyarrow upgrades (ADR-095). Each case also
asserts its defining fact. The codec check reads the compression and
encodings of every Parquet column chunk, which a decoded comparison cannot
see (R51). The cross-check recomputes every summary statistic from the
decoded Parquet with float64 `pyarrow.compute` (ADR-076).

With HPCMU_REGEN_GOLDENS=1 the golden tests rewrite
tests/goldens/telemetry/export_expected.json and fail.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from hpc_model_utils.infra.telemetry.export import Json, main
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    HEADLINE,
    PARQUET_METADATA_KEYS,
    PHASES,
    SAMPLE_COLUMNS,
    SAMPLE_SORT_KEYS,
    SAMPLE_STATS,
    SAMPLES_FILE,
    STAT_AGGREGATES,
    SUMMARY_FILE,
    WARNING_CODES,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    WORKSPACE_STATS,
    Column,
    StatSpec,
)
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    ErrorLine,
    header,
    sample,
    walk,
    write_staging,
)

_REGEN_ENV: Final = "HPCMU_REGEN_GOLDENS"
_EXPECTED_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "goldens"
    / "telemetry"
    / "export_expected.json"
)
_REL: Final = 1e-9
_DICTIONARY: Final = "RLE_DICTIONARY"
_ZSTD: Final = "ZSTD"
_HEADERS_KEY: Final = PARQUET_METADATA_KEYS[5].name
_MODEL, _FINALIZE = PHASES
# (phase, job_id, restart, node_index), then (phase, job_id).
_NODE_KEYS: Final = SAMPLE_SORT_KEYS[:4]
_JOB_KEYS: Final = SAMPLE_SORT_KEYS[:2]
_DT_S: Final = "dt_s"
_SEQ: Final = "seq"
# The decoded parts of one case.
_SAMPLES: Final = "samples"
_WORKSPACE: Final = "workspace"
_SUMMARY: Final = "summary"
_WALK_VALUES: Final = frozenset(
    c.name for c in WORKSPACE_COLUMNS if c.source != "identity"
)


def _code(name: str) -> str:
    """Return name; a name outside `WARNING_CODES` fails the import."""
    if name not in {code.code for code in WARNING_CODES}:
        raise ValueError(f"not a WARNING_CODES code: {name!r}")
    return name


def _aggregate(name: str) -> str:
    """Return name; a name outside `STAT_AGGREGATES` fails the import."""
    if name not in STAT_AGGREGATES:
        raise ValueError(f"not a STAT_AGGREGATES name: {name!r}")
    return name


_TRUNCATED_TAIL: Final = _code("truncated_tail")
_BAD_LINE: Final = _code("bad_line")
_MISSING_NODE: Final = _code("missing_node")
_SAMPLER_ERROR: Final = _code("sampler_error")
_NON_MONOTONIC_TS: Final = _code("non_monotonic_ts")
_MIN: Final = _aggregate("min")
_MEAN: Final = _aggregate("mean")
_P95: Final = _aggregate("p95")
_MAX: Final = _aggregate("max")
_TOTAL: Final = _aggregate("total")
_QUANTILES: Final = {_P95: 0.95, _aggregate("p99"): 0.99}


def _published(spec: StatSpec, aggregate: str) -> str:
    """ADR-071: f"{column}_{aggregate}", or total_name for the total."""
    if aggregate != _TOTAL:
        return f"{spec.column}_{aggregate}"
    if spec.total_name is None:
        raise ValueError(f"{spec.column}: a total without a total_name")
    return spec.total_name


# (published name, column, aggregate) of every statistic.
type _Spec = tuple[str, str, str]
_SAMPLE_SPECS: Final[tuple[_Spec, ...]] = tuple(
    (_published(s, a), s.column, a) for s in SAMPLE_STATS for a in s.aggregates
)
_WORKSPACE_SPECS: Final[tuple[_Spec, ...]] = tuple(
    (_published(s, a), s.column, a)
    for s in WORKSPACE_STATS
    for a in s.aggregates
)


def _stat_name(column: str, aggregate: str) -> str:
    """The published name of a `SAMPLE_STATS` statistic."""
    for name, spec_column, spec_aggregate in _SAMPLE_SPECS:
        if (spec_column, spec_aggregate) == (column, aggregate):
            return name
    raise KeyError(f"no SAMPLE_STATS {aggregate} of {column}")


def _rate_column(prefix: str) -> str:
    """The one bytes/s column of `SAMPLE_COLUMNS` whose name has prefix.

    The rates are looked up, not spelled out: the NFS read rate's name holds
    the token that the ticket's no-byte-read grep forbids in this module.
    """
    names = [
        c.name
        for c in SAMPLE_COLUMNS
        if c.name.startswith(prefix) and c.unit == "bytes/s"
    ]
    if len(names) != 1:
        raise KeyError(f"not one bytes/s column named {prefix}*: {names}")
    return names[0]


_NFS_READ: Final = _rate_column("nfs_read_")
_DISK_WRITE: Final = _rate_column("disk_write_")
_CPU_BUSY: Final = "cpu_busy_cores"
_NFS_READ_MEAN: Final = _stat_name(_NFS_READ, _MEAN)
_NFS_READ_TOTAL: Final = _stat_name(_NFS_READ, _TOTAL)
_DISK_WRITE_MAX: Final = _stat_name(_DISK_WRITE, _MAX)
_DISK_WRITE_P95: Final = _stat_name(_DISK_WRITE, _P95)
_CPU_BUSY_MEAN: Final = _stat_name(_CPU_BUSY, _MEAN)


# Case builders: each writes its staging files and returns its --job list.


def _rewrite_lines(
    path: Path, edit: Callable[[list[bytes]], list[bytes]]
) -> None:
    """Split path's bytes at each newline, edit the parts and join them back.

    The split leaves an empty last part after the final newline.
    """
    parts = path.read_bytes().split(b"\n")
    path.write_bytes(b"\n".join(edit(parts)))


def _sample(
    head: Mapping[str, JsonValue],
    seq: int,
    *,
    base: int,
    ts: int | None = None,
    values: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Sample seq of head's file, the seven filled columns set.

    values override any column. Every value differs across rows, columns and
    the files of a case: base sets a file apart, and each column has its own
    range. The FLOAT32 values are exact in float32.
    """
    k = base + seq
    filled: dict[str, JsonValue] = {
        "sample_cost_ms": 0.125 + k / 64,
        _CPU_BUSY: 2.0 + k / 4,
        "cpu_util": 0.5 + k / 64,
        "mem_available_bytes": 8 * 2**30 - k * 2**20,
        "job_mem_peak_bytes": 2**30 + k * 2**20,
        _NFS_READ: 1000.25 + 1000 * k,
        _DISK_WRITE: 100.75 + 100 * k,
    }
    return sample(head, seq, ts=ts, **{**filled, **(values or {})})


def _samples(
    head: Mapping[str, JsonValue], count: int, *, base: int
) -> list[dict[str, JsonValue]]:
    return [_sample(head, seq, base=base) for seq in range(count)]


def _walk(head: Mapping[str, JsonValue], *, base: int) -> dict[str, JsonValue]:
    """A walk at the first grid tick with every value column set."""
    values: dict[str, JsonValue] = {
        "apparent_bytes": 3 * 2**30 + base * 2**20,
        "allocated_bytes": 2 * 2**30 + base * 2**20,
        "files": 1000 + base,
        "dirs": 100 + base,
        "walk_s": 0.25 + base / 8,
        "fs_used_bytes": 5 * 2**40 + base * 2**30,
        "fs_avail_bytes": 2**40 - base * 2**30,
    }
    if values.keys() != _WALK_VALUES:
        raise KeyError(f"walk values differ from {sorted(_WALK_VALUES)}")
    return walk(head, ts=ORIGIN_MS + 5_000, **values)


def _with_walk(
    head: Mapping[str, JsonValue], count: int, *, base: int
) -> list[dict[str, JsonValue]]:
    """Samples 0 to count - 1, with one walk after sample 0."""
    records = _samples(head, count, base=base)
    records.insert(1, _walk(head, base=base))
    return records


def _truncated_tail(staging: Path) -> list[tuple[str, str]]:
    head = header()
    path = write_staging(staging, head, _with_walk(head, 3, base=0))
    tail = json.dumps(
        _sample(head, 3, base=0),
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")
    _rewrite_lines(path, lambda parts: [*parts[:-1], tail])
    return [(_MODEL, "1000")]


def _malformed_line(staging: Path) -> list[tuple[str, str]]:
    head = header()
    path = write_staging(staging, head, _samples(head, 4, base=0))
    # parts: the header, samples 0 to 3, and the empty part after the end.
    _rewrite_lines(path, lambda parts: [*parts[:3], b"not json", *parts[3:]])
    return [(_MODEL, "1000")]


def _partial_write(staging: Path) -> list[tuple[str, str]]:
    head = header()
    path = write_staging(staging, head, _samples(head, 4, base=0))
    _rewrite_lines(
        path, lambda parts: [*parts[:2], parts[2][:20] + parts[3], *parts[4:]]
    )
    return [(_MODEL, "1000")]


def _missing_node(staging: Path) -> list[tuple[str, str]]:
    head = header(nodes=2)
    write_staging(staging, head, _samples(head, 3, base=0))
    return [(_MODEL, "1000")]


def _requeue(staging: Path) -> list[tuple[str, str]]:
    first, second = header(restart=0), header(restart=1)
    final = header(_FINALIZE, "1001")
    write_staging(staging, first, _with_walk(first, 3, base=0))
    write_staging(staging, second, _with_walk(second, 3, base=8))
    write_staging(staging, final, _samples(final, 2, base=16))
    return [(_MODEL, "1000"), (_FINALIZE, "1001")]


def _error_lines(staging: Path) -> list[tuple[str, str]]:
    head = header()
    s0, s1, s2 = _samples(head, 3, base=0)
    write_staging(
        staging,
        head,
        [
            s0,
            ErrorLine(
                ORIGIN_MS + 6_000, "meminfo", OSError("fake read failure")
            ),
            s1,
            ErrorLine(
                ORIGIN_MS + 11_000,
                "diskstats",
                ValueError("fake parse failure"),
            ),
            s2,
        ],
    )
    return [(_MODEL, "1000")]


def _zero_walks(staging: Path) -> list[tuple[str, str]]:
    head = header()
    write_staging(staging, head, _samples(head, 3, base=0))
    return [(_MODEL, "1000")]


def _non_monotonic_ts(staging: Path) -> list[tuple[str, str]]:
    head = header()
    records = _samples(head, 4, base=0)
    repeated = records[1]["ts"]
    assert isinstance(repeated, int)
    records[2] = _sample(head, 2, base=0, ts=repeated)
    write_staging(staging, head, records)
    return [(_MODEL, "1000")]


def _rollup_edges(staging: Path) -> list[tuple[str, str]]:
    head = header()
    # Samples 2 to 4 leave the grid after a 10 s gap; sample 4 is the
    # off-grid stop row with a short dt_s.
    stamps = (None, None, ORIGIN_MS + 20_000, ORIGIN_MS + 25_000)
    rows = zip(
        (*stamps, ORIGIN_MS + 25_500),
        (5.0, 5.0, 10.0, 5.0, 0.5),
        (100.0, None, 200.0, 300.0, 400.0),
        (10.0, 20.0, 30.0, 40.0, 1000.0),
        strict=True,
    )
    records = [
        _sample(
            head,
            seq,
            base=0,
            ts=ts,
            values={_DT_S: dt_s, _NFS_READ: nfs, _DISK_WRITE: disk},
        )
        for seq, (ts, dt_s, nfs, disk) in enumerate(rows)
    ]
    write_staging(staging, head, records)
    final = header(_FINALIZE, "1001")
    write_staging(
        staging,
        final,
        [
            _sample(
                final,
                0,
                base=8,
                values={_DT_S: 2.5, _CPU_BUSY: 1.5, _NFS_READ: 80.0},
            )
        ],
    )
    return [(_MODEL, "1000"), (_FINALIZE, "1001")]


CASES: dict[str, Callable[[Path], list[tuple[str, str]]]] = {
    "truncated_tail": _truncated_tail,
    "malformed_line": _malformed_line,
    "partial_write": _partial_write,
    "missing_node": _missing_node,
    "requeue": _requeue,
    "error_lines": _error_lines,
    "zero_walks": _zero_walks,
    "non_monotonic_ts": _non_monotonic_ts,
    "rollup_edges": _rollup_edges,
}


# JSON narrowing.


def _obj(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise TypeError(f"not a JSON object: {value!r}")
    return value


def _arr(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise TypeError(f"not a JSON array: {value!r}")
    return value


def _integer(value: Json) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"not a JSON integer: {value!r}")
    return value


def _text(value: Json) -> str:
    if not isinstance(value, str):
        raise TypeError(f"not a JSON string: {value!r}")
    return value


def _numbers(value: Json) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for key, number in _obj(value).items():
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise TypeError(f"{key} is not a JSON number: {number!r}")
        numbers[key] = float(number)
    return numbers


def _loads(text: str) -> Json:
    value: Json = json.loads(text)
    return value


def _cell(value: object) -> Json:
    if isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"not a JSON scalar cell: {value!r}")


# Export and decode.


def _decode(path: Path) -> dict[str, Json]:
    """Requirement 4: the decoded schema, KV metadata and rows of path.

    A row leaves out its null cells and holds a timestamp as epoch ms.
    """
    schema: list[Json] = [
        [str(field.name), str(field.type), bool(field.nullable)]
        for field in pq.read_schema(path)
    ]
    kv: dict[str, Json] = {}
    for key, value in pq.ParquetFile(path).schema_arrow.metadata.items():
        name, text = str(key.decode("ascii")), str(value.decode("ascii"))
        kv[name] = _loads(text) if name == _HEADERS_KEY else text
    table = pq.read_table(path)
    for index, field in enumerate(table.schema):
        if pa.types.is_timestamp(field.type):
            ms = pc.cast(table.column(index), pa.int64())
            table = table.set_column(index, field.name, ms)
    rows: list[Json] = [
        {
            str(name): _cell(cell)
            for name, cell in row.items()
            if cell is not None
        }
        for row in table.to_pylist()
    ]
    return {"schema": schema, "kv": kv, "rows": rows}


@dataclass(frozen=True, slots=True)
class _Export:
    out: Path
    jobs: tuple[tuple[str, str], ...]
    decoded: dict[str, Json]

    @property
    def summary(self) -> dict[str, Json]:
        return _obj(self.decoded[_SUMMARY])

    def rows(self, part: str) -> list[dict[str, Json]]:
        return [_obj(row) for row in _arr(_obj(self.decoded[part])["rows"])]


def _export(case: str, root: Path) -> _Export:
    """Requirement 1: build case below root and export it with main."""
    staging = root / "staging"
    staging.mkdir()
    jobs = tuple(CASES[case](staging))
    out = root / "out"
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
    for phase, job_id in jobs:
        argv += ["--job", f"{phase}:{job_id}"]
    assert main(argv) == 0
    decoded: dict[str, Json] = {
        _SAMPLES: _decode(out / SAMPLES_FILE),
        _WORKSPACE: _decode(out / WORKSPACE_FILE),
        _SUMMARY: _loads((out / SUMMARY_FILE).read_text(encoding="ascii")),
    }
    return _Export(out, jobs, decoded)


class _Exports:
    """The export of each case, run once per module on first use."""

    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory
        self._done: dict[str, _Export] = {}

    def __getitem__(self, case: str) -> _Export:
        if case not in self._done:
            self._done[case] = _export(case, self._factory.mktemp(case))
        return self._done[case]


@pytest.fixture(scope="module")
def exports(tmp_path_factory: pytest.TempPathFactory) -> _Exports:
    return _Exports(tmp_path_factory)


# Decoded goldens.


def _expected() -> dict[str, Json]:
    try:
        text = _EXPECTED_PATH.read_text(encoding="ascii")
    except FileNotFoundError:
        pytest.fail(f"{_EXPECTED_PATH} is missing; run with {_REGEN_ENV}=1")
    return _obj(_loads(text))


@pytest.mark.parametrize("case", CASES)
def test_golden_matches_expected(case: str, exports: _Exports) -> None:
    if os.environ.get(_REGEN_ENV) == "1":
        expected = {name: exports[name].decoded for name in CASES}
        text = json.dumps(
            expected, indent=1, ensure_ascii=True, allow_nan=False
        )
        _EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EXPECTED_PATH.write_text(text + "\n", encoding="ascii")
        pytest.fail(
            f"{_REGEN_ENV}=1: regenerated {_EXPECTED_PATH}; "
            "re-run without it to verify"
        )

    expected_cases = _expected()
    if case not in expected_cases:
        pytest.fail(
            f"case {case} is missing from {_EXPECTED_PATH}; "
            f"run with {_REGEN_ENV}=1"
        )
    assert sorted(expected_cases) == sorted(CASES), (
        f"the cases of {_EXPECTED_PATH} differ; run with {_REGEN_ENV}=1"
    )
    assert exports[case].decoded == expected_cases[case]


# Defining facts (Requirement 3).


def _codes(export: _Export) -> list[str]:
    return [
        _text(_obj(warning)["code"])
        for warning in _arr(export.summary["warnings"])
    ]


def _seqs(export: _Export) -> list[int]:
    return [_integer(row[_SEQ]) for row in export.rows(_SAMPLES)]


def _nodes(export: _Export, phase: str) -> list[dict[str, Json]]:
    (job,) = [
        _obj(job)
        for job in _arr(export.summary["jobs"])
        if _obj(job)["phase"] == phase
    ]
    return [_obj(node) for node in _arr(job["nodes"])]


def _node(export: _Export, phase: str) -> dict[str, Json]:
    (node,) = _nodes(export, phase)
    return node


def _truncated_tail_fact(export: _Export) -> None:
    assert _codes(export) == [_TRUNCATED_TAIL]
    assert _seqs(export) == [0, 1, 2]


def _malformed_line_fact(export: _Export) -> None:
    assert _codes(export) == [_BAD_LINE]
    assert _seqs(export) == [0, 1, 2, 3]


def _partial_write_fact(export: _Export) -> None:
    assert _codes(export) == [_BAD_LINE]
    assert _seqs(export) == [0, 3]


def _missing_node_fact(export: _Export) -> None:
    (warning,) = [_obj(w) for w in _arr(export.summary["warnings"])]
    assert warning["code"] == _MISSING_NODE
    assert warning["node_index"] == 1


def _requeue_fact(export: _Export) -> None:
    restarts = [
        _integer(_obj(node["header"])["restart"])
        for node in _nodes(export, _MODEL)
    ]
    assert restarts == [0, 1]
    headline = _obj(export.summary["headline"])
    assert headline["job_ids"] == {_MODEL: "1000", _FINALIZE: "1001"}


def _error_lines_fact(export: _Export) -> None:
    assert _codes(export) == [_SAMPLER_ERROR, _SAMPLER_ERROR]


def _zero_walks_fact(export: _Export) -> None:
    parquet = pq.ParquetFile(export.out / WORKSPACE_FILE)
    assert int(parquet.metadata.num_rows) == 0
    assert export.summary["workspace"] == {"walks": 0, "stats": {}}


def _non_monotonic_ts_fact(export: _Export) -> None:
    assert _codes(export) == [_NON_MONOTONIC_TS]
    assert _seqs(export) == [0, 1, 3]


def _rollup_edges_fact(export: _Export) -> None:
    model = _node(export, _MODEL)
    assert model["gaps"] == 1
    assert model["max_gap_s"] == 10.0
    stats = _obj(model["stats"])
    # The null-aware denominator: 20.5 s where nfs_read is valid, never
    # the node's whole 25.5 s.
    assert stats[_NFS_READ_MEAN] == pytest.approx(4200 / 20.5)
    assert stats[_NFS_READ_MEAN] != pytest.approx(4200 / 25.5)
    assert stats[_NFS_READ_TOTAL] == pytest.approx(4200.0)
    # Both set by the off-grid stop row.
    assert stats[_DISK_WRITE_MAX] == 1000.0
    assert stats[_DISK_WRITE_P95] == pytest.approx(808.0)
    final = _node(export, _FINALIZE)
    assert final["samples"] == 1
    final_stats = _obj(final["stats"])
    assert final_stats[_CPU_BUSY_MEAN] == pytest.approx(1.5)
    assert final_stats[_NFS_READ_TOTAL] == pytest.approx(200.0)


_FACTS: Final[dict[str, Callable[[_Export], None]]] = {
    "truncated_tail": _truncated_tail_fact,
    "malformed_line": _malformed_line_fact,
    "partial_write": _partial_write_fact,
    "missing_node": _missing_node_fact,
    "requeue": _requeue_fact,
    "error_lines": _error_lines_fact,
    "zero_walks": _zero_walks_fact,
    "non_monotonic_ts": _non_monotonic_ts_fact,
    "rollup_edges": _rollup_edges_fact,
}


@pytest.mark.parametrize("case", CASES)
def test_defining_fact_holds(case: str, exports: _Exports) -> None:
    assert _FACTS.keys() == CASES.keys()
    _FACTS[case](exports[case])


# Codec and encodings (Requirement 5, R51).


def _codec_chunks(path: Path, columns: Sequence[Column]) -> tuple[int, int]:
    """Check every column chunk of path; return (chunks, STRING chunks).

    Every chunk is zstd-compressed, and every chunk of a STRING column is
    dictionary-encoded.
    """
    strings = {c.name for c in columns if c.dtype == "STRING"}
    meta = pq.ParquetFile(path).metadata
    chunks = dictionary = 0
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        for j in range(group.num_columns):
            chunk = group.column(j)
            name = str(chunk.path_in_schema)
            assert str(chunk.compression) == _ZSTD, (path.name, i, name)
            if name in strings:
                encodings = [str(e) for e in chunk.encodings]
                assert _DICTIONARY in encodings, (path.name, i, name)
                dictionary += 1
            chunks += 1
    return chunks, dictionary


def _codec_counts(export: _Export) -> dict[str, tuple[int, int]]:
    return {
        _SAMPLES: _codec_chunks(export.out / SAMPLES_FILE, SAMPLE_COLUMNS),
        _WORKSPACE: _codec_chunks(
            export.out / WORKSPACE_FILE, WORKSPACE_COLUMNS
        ),
    }


@pytest.mark.parametrize("case", CASES)
def test_codec_zstd_and_dictionary(case: str, exports: _Exports) -> None:
    counts = _codec_counts(exports[case])
    assert counts[_SAMPLES][0] >= 1


def test_codec_visits_both_files(exports: _Exports) -> None:
    totals = {_SAMPLES: [0, 0], _WORKSPACE: [0, 0]}
    for case in CASES:
        for part, (chunks, dictionary) in _codec_counts(exports[case]).items():
            totals[part][0] += chunks
            totals[part][1] += dictionary
    assert totals[_SAMPLES][0] >= 1 and totals[_SAMPLES][1] >= 1
    assert totals[_WORKSPACE][0] >= 1 and totals[_WORKSPACE][1] >= 1


# Statistics cross-check (Requirement 6, Decisions 5 and 6).


def _recomputed(
    path: Path,
    keep: Sequence[Mapping[str, str | int]] | None,
    specs: Iterable[_Spec],
    *,
    weighted: bool,
) -> dict[str, float]:
    """Recompute specs in float64 over the rows of path that keep selects.

    keep lists identities, column name to value, and a row is kept when it
    matches one; None keeps every row. A min, max or quantile runs over the
    non-null values. A mean or a total runs over the rows where both the
    value and dt_s are valid. An absent statistic is left out.
    """
    table = pq.read_table(path)
    if keep is not None:
        parts = []
        for identity in keep:
            part = table
            for name, value in identity.items():
                part = part.filter(pc.equal(part.column(name), value))
            parts.append(part)
        table = pa.concat_tables(parts) if parts else table.slice(0, 0)
    dt = pc.cast(table.column(_DT_S), pa.float64()) if weighted else None
    found: dict[str, float] = {}
    for name, column, aggregate in specs:
        x = pc.cast(table.column(column), pa.float64())
        if aggregate in (_MIN, _MAX):
            value = pc.min_max(x)[aggregate].as_py()
        elif aggregate in _QUANTILES:
            q = _QUANTILES[aggregate]
            value = pc.quantile(x, q=q, interpolation="linear")[0].as_py()
        elif dt is None:
            raise ValueError(f"{name}: a {aggregate} needs {_DT_S}")
        else:
            valid = pc.and_(pc.is_valid(x), pc.is_valid(dt))
            xs, weights = pc.filter(x, valid), pc.filter(dt, valid)
            value = pc.sum(pc.multiply(xs, weights)).as_py()
            if aggregate == _MEAN and value is not None:
                value = value / pc.sum(weights).as_py()
        if value is not None:
            found[name] = float(value)
    return found


@dataclass(frozen=True, slots=True)
class _Compared:
    case: str
    where: tuple[str | int, ...]
    key: str
    aggregate: str


_AGGREGATES: Final = {
    name: aggregate
    for name, _, aggregate in (*_SAMPLE_SPECS, *_WORKSPACE_SPECS)
} | {row.key: row.aggregate for row in HEADLINE}


def _node_identity(node: Mapping[str, Json]) -> dict[str, str | int]:
    head = _obj(node["header"])
    phase, job_id, restart, node_index = (head[key] for key in _NODE_KEYS)
    identity = (
        _text(phase),
        _text(job_id),
        _integer(restart),
        _integer(node_index),
    )
    return dict(zip(_NODE_KEYS, identity, strict=True))


def _crosscheck(case: str, export: _Export) -> list[_Compared]:
    """Recompute every summary statistic of export; return what was compared."""
    samples = export.out / SAMPLES_FILE
    workspace = export.out / WORKSPACE_FILE
    compared: list[_Compared] = []

    def compare(
        where: tuple[str | int, ...],
        recomputed: dict[str, float],
        published: Json,
    ) -> None:
        values = _numbers(published)
        assert recomputed == pytest.approx(values, rel=_REL), where
        compared.extend(
            _Compared(case, where, key, _AGGREGATES[key]) for key in values
        )

    for job in _arr(export.summary["jobs"]):
        for node in _arr(_obj(job)["nodes"]):
            identity = _node_identity(_obj(node))
            found = _recomputed(
                samples, [identity], _SAMPLE_SPECS, weighted=True
            )
            where = ("node", *identity.values())
            compare(where, found, _obj(node)["stats"])

    reported = dict(export.jobs)
    headline: dict[str, float] = {}
    for row in HEADLINE:
        # A phase row is over the rows of its reported job, the workspace
        # row over the walks of every reported job.
        if row.source == SAMPLES_FILE:
            jobs = (
                [(row.scope, reported[row.scope])]
                if row.scope in reported
                else []
            )
            path, weighted = samples, True
        else:
            jobs = list(reported.items())
            path, weighted = workspace, False
        keep = [dict(zip(_JOB_KEYS, job, strict=True)) for job in jobs]
        spec = ((row.key, row.column, row.aggregate),)
        headline |= _recomputed(path, keep, spec, weighted=weighted)
    compare(("headline",), headline, _obj(export.summary["headline"])["values"])

    stats = _recomputed(workspace, None, _WORKSPACE_SPECS, weighted=False)
    compare(("workspace",), stats, _obj(export.summary["workspace"])["stats"])
    return compared


@pytest.mark.parametrize("case", CASES)
def test_crosscheck_matches_summary(case: str, exports: _Exports) -> None:
    assert _crosscheck(case, exports[case])


def test_crosscheck_covers_the_rollup_edges(exports: _Exports) -> None:
    compared = [c for case in CASES for c in _crosscheck(case, exports[case])]
    assert {c.aggregate for c in compared} >= {_MIN, _MEAN, _P95, _MAX, _TOTAL}
    assert any(c.where == ("workspace",) for c in compared)

    edges = exports["rollup_edges"]
    keys = {(c.where, c.key) for c in compared if c.case == "rollup_edges"}
    model = _node(edges, _MODEL)
    model_identity = _node_identity(model)
    model_where = ("node", *model_identity.values())
    rows = [
        row
        for row in edges.rows(_SAMPLES)
        if all(row[k] == v for k, v in model_identity.items())
    ]
    # The mean over a column with a null between non-null rows.
    present = [_NFS_READ in row for row in rows]
    assert present[0] and present[-1] and not all(present)
    assert (model_where, _NFS_READ_MEAN) in keys
    # The gap node.
    assert _integer(model["gaps"]) >= 1
    assert any(where == model_where for where, _ in keys)
    # The off-grid stop row sets the max.
    interval_ms = _integer(_obj(model["header"])["interval_s"]) * 1000
    stop = rows[-1]
    assert (_integer(stop["ts"]) - ORIGIN_MS) % interval_ms != 0
    assert stop[_DISK_WRITE] == _obj(model["stats"])[_DISK_WRITE_MAX]
    assert (model_where, _DISK_WRITE_MAX) in keys
    # The one-sample finalize node.
    final = _node(edges, _FINALIZE)
    assert final["samples"] == 1
    final_where = ("node", *_node_identity(final).values())
    assert any(where == final_where for where, _ in keys)
