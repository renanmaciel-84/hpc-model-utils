from __future__ import annotations

import dataclasses
import math
import re
import sys
from collections import Counter

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from hpc_model_utils.infra.telemetry import schema
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADER_FIELDS,
    HEADLINE,
    HEADLINE_FORMATS,
    HEADLINE_KEY_PREFIX,
    HEADLINE_POOLINGS,
    PARQUET_METADATA_KEYS,
    PHASES,
    SAMPLE_COLUMNS,
    SAMPLE_STATS,
    SAMPLES_FILE,
    STAT_AGGREGATES,
    WARNING_CODES,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    WORKSPACE_STATS,
    Column,
    HeaderField,
    HeadlineSpec,
    MetadataKey,
    StatSpec,
    WarningCode,
    format_headline_value,
    staging_file_name,
)

_EXPECTED_SAMPLE_NAMES = (
    "ts",
    "phase",
    "job_id",
    "restart",
    "node",
    "node_index",
    "seq",
    "dt_s",
    "sample_cost_ms",
    "sampler_maxrss_bytes",
    "cpu_busy_cores",
    "cpu_util",
    "cpu_user_cores",
    "cpu_system_cores",
    "cpu_iowait_cores",
    "cpu_steal_cores",
    "cpu_core_util_min",
    "cpu_core_util_p50",
    "cpu_core_util_max",
    "cpu_active_cores",
    "load1",
    "procs_running",
    "procs_blocked",
    "mem_used_bytes",
    "mem_available_bytes",
    "mem_anon_bytes",
    "mem_cached_bytes",
    "mem_shmem_bytes",
    "mem_dirty_bytes",
    "mem_writeback_bytes",
    "mem_slab_bytes",
    "swap_used_bytes",
    "major_faults_per_s",
    "oom_kills",
    "job_cpu_cores",
    "job_mem_bytes",
    "job_mem_anon_bytes",
    "job_mem_file_bytes",
    "job_mem_shmem_bytes",
    "job_mem_peak_bytes",
    "job_pids",
    "job_oom_kills",
    "psi_cpu_some",
    "psi_mem_some",
    "psi_mem_full",
    "psi_io_some",
    "psi_io_full",
    "disk_read_bytes_per_s",
    "disk_write_bytes_per_s",
    "disk_busy_max",
    "nfs_read_bytes_per_s",
    "nfs_write_bytes_per_s",
    "nfs_server_read_bytes_per_s",
    "nfs_server_write_bytes_per_s",
    "nfs_read_ops_per_s",
    "nfs_write_ops_per_s",
    "nfs_ops_per_s",
    "nfs_read_rtt_ms",
    "nfs_write_rtt_ms",
    "net_rx_bytes_per_s",
    "net_tx_bytes_per_s",
    "net_errors_per_s",
    "efa_rx_bytes_per_s",
    "efa_tx_bytes_per_s",
)

_EXPECTED_WORKSPACE_NAMES = (
    "ts",
    "phase",
    "job_id",
    "restart",
    "node",
    "apparent_bytes",
    "allocated_bytes",
    "files",
    "dirs",
    "walk_s",
    "fs_used_bytes",
    "fs_avail_bytes",
)

_EXPECTED_HEADER = (
    ("kind", "string", False),
    ("schema_version", "integer", False),
    ("tool_version", "string", False),
    ("phase", "string", False),
    ("job_id", "string", False),
    ("restart", "integer", False),
    ("node", "string", False),
    ("node_index", "integer", False),
    ("nodes", "integer", True),
    ("partition", "string", True),
    ("cpus_on_node", "integer", True),
    ("instance_type", "string", True),
    ("cpu_model", "string", True),
    ("ncpus", "integer", True),
    ("mem_total_bytes", "integer", True),
    ("kernel", "string", True),
    ("job_cgroup", "string", True),
    ("own_step_cgroup", "string", True),
    ("workspace_mount", "object", True),
    ("interval_s", "integer", False),
    ("workspace_interval_s", "integer", False),
    ("started_at", "string", False),
)

_TABLES = {"samples": SAMPLE_COLUMNS, "workspace": WORKSPACE_COLUMNS}
_ALL_COLUMNS = SAMPLE_COLUMNS + WORKSPACE_COLUMNS
_FRACTION_NAMES = frozenset(
    {
        "cpu_util",
        "cpu_core_util_min",
        "cpu_core_util_p50",
        "cpu_core_util_max",
        "psi_cpu_some",
        "psi_mem_some",
        "psi_mem_full",
        "psi_io_some",
        "psi_io_full",
        "disk_busy_max",
    }
)


_EXPECTED_METADATA_KEYS = (
    "hpcmu.telemetry.schema_version",
    "hpcmu.telemetry.kind",
    "hpcmu.tool_version",
    "hpcmu.run_id",
    "hpcmu.model",
    "hpcmu.headers",
)

_EXPECTED_WARNING_CODES = (
    "bad_header",
    "truncated_tail",
    "bad_line",
    "bad_record",
    "identity_mismatch",
    "non_monotonic_ts",
    "sampler_error",
    "missing_node",
    "warnings_capped",
)

_EXPECTED_HEADLINE_KEYS = (
    "telemetry.model.job_mem_peak_bytes_max",
    "telemetry.model.mem_available_bytes_min",
    "telemetry.model.cpu_busy_cores_mean",
    "telemetry.model.cpu_busy_cores_p95",
    "telemetry.model.cpu_util_mean",
    "telemetry.model.nfs_read_bytes_total",
    "telemetry.model.nfs_write_bytes_total",
    "telemetry.model.disk_read_bytes_total",
    "telemetry.model.disk_write_bytes_total",
    "telemetry.finalize.job_mem_peak_bytes_max",
    "telemetry.finalize.mem_available_bytes_min",
    "telemetry.finalize.cpu_busy_cores_mean",
    "telemetry.finalize.cpu_busy_cores_p95",
    "telemetry.finalize.cpu_util_mean",
    "telemetry.finalize.nfs_read_bytes_total",
    "telemetry.finalize.nfs_write_bytes_total",
    "telemetry.finalize.disk_read_bytes_total",
    "telemetry.finalize.disk_write_bytes_total",
    "telemetry.workspace.allocated_bytes_max",
)

# (column, aggregate, pooling, fmt) of the 9 rows of each phase.
_EXPECTED_PHASE_HEADLINE_ROWS = (
    ("job_mem_peak_bytes", "max", "node_rows", "bytes"),
    ("mem_available_bytes", "min", "node_rows", "bytes"),
    ("cpu_busy_cores", "mean", "node_rows", "cores"),
    ("cpu_busy_cores", "p95", "node_rows", "cores"),
    ("cpu_util", "mean", "node_rows", "fraction"),
    ("nfs_read_bytes_per_s", "total", "sum_nodes", "bytes"),
    ("nfs_write_bytes_per_s", "total", "sum_nodes", "bytes"),
    ("disk_read_bytes_per_s", "total", "sum_nodes", "bytes"),
    ("disk_write_bytes_per_s", "total", "sum_nodes", "bytes"),
)

_EXPECTED_TOTAL_NAMES = (
    "major_faults_total",
    "disk_read_bytes_total",
    "disk_write_bytes_total",
    "nfs_read_bytes_total",
    "nfs_write_bytes_total",
    "nfs_server_read_bytes_total",
    "nfs_server_write_bytes_total",
    "nfs_read_ops_total",
    "nfs_write_ops_total",
    "nfs_ops_total",
    "net_rx_bytes_total",
    "net_tx_bytes_total",
    "net_errors_total",
    "efa_rx_bytes_total",
    "efa_tx_bytes_total",
)

_EXPECTED_WORKSPACE_STATS = (
    ("apparent_bytes", ("max",), None),
    ("allocated_bytes", ("max",), None),
    ("files", ("max",), None),
    ("dirs", ("max",), None),
    ("walk_s", ("max",), None),
    ("fs_used_bytes", ("max",), None),
    ("fs_avail_bytes", ("min",), None),
)

_RATE_UNITS = frozenset({"bytes/s", "1/s"})
_MAX_ONLY_NAMES = frozenset(
    {
        "sampler_maxrss_bytes",
        "job_mem_peak_bytes",
        "oom_kills",
        "job_oom_kills",
    }
)
_ROLLED_SAMPLE_COLUMNS = tuple(
    c for c in SAMPLE_COLUMNS if c.source != "identity" and c.name != "dt_s"
)
_STAT_TABLES = {
    SAMPLES_FILE: (SAMPLE_COLUMNS, SAMPLE_STATS),
    WORKSPACE_FILE: (WORKSPACE_COLUMNS, WORKSPACE_STATS),
}
_HEADLINE_GRAMMAR = {
    "bytes": "[0-9]+",
    "cores": r"[0-9]+\.[0-9]{2}",
    "fraction": r"[0-9]+\.[0-9]{4}",
}


def _names(columns: tuple[Column, ...]) -> tuple[str, ...]:
    return tuple(column.name for column in columns)


def _expected_aggregates(name: str, unit: str) -> tuple[str, ...]:
    # The class rule of ticket-040 Decision 4, kept independent of
    # SAMPLE_STATS so it guards the literal rows.
    if name in _MAX_ONLY_NAMES:
        return ("max",)
    if name == "sample_cost_ms":
        return ("mean", "p95", "p99", "max")
    if unit in _RATE_UNITS:
        return ("mean", "p95", "max", "total")
    if name.endswith("_rtt_ms"):
        return ("min", "p95", "max")
    return ("min", "mean", "p95", "max")


def _published_names(specs: tuple[StatSpec, ...]) -> list[str]:
    names: list[str] = []
    for spec in specs:
        for aggregate in spec.aggregates:
            if aggregate == "total":
                assert spec.total_name is not None, spec.column
                names.append(spec.total_name)
            else:
                names.append(f"{spec.column}_{aggregate}")
    return names


def _record_id(record: Column | HeaderField | MetadataKey | WarningCode) -> str:
    return record.code if isinstance(record, WarningCode) else record.name


def _expected_unit(name: str) -> str | None:
    if name.endswith("_bytes_per_s"):
        return "bytes/s"
    if name.endswith("_bytes"):
        return "bytes"
    if name == "cpu_active_cores":
        return "count"
    if name.endswith("_cores"):
        return "cores"
    if name.endswith("_ops_per_s") or name in {
        "major_faults_per_s",
        "net_errors_per_s",
    }:
        return "1/s"
    if name.endswith("_ms"):
        return "ms"
    if name.endswith("_s") and not name.endswith("_per_s"):
        return "s"
    if name in _FRACTION_NAMES:
        return "fraction"
    return None


def test_schema_version_and_phases() -> None:
    assert schema.SCHEMA_VERSION == 1
    assert schema.PHASES == ("model", "finalize")


def test_kind_strings() -> None:
    assert schema.HEADER_KIND == "hpcmu.telemetry.header"
    assert schema.WALK_KIND == "hpcmu.telemetry.walk"
    assert schema.ERROR_KIND == "hpcmu.telemetry.error"
    assert schema.SAMPLES_KIND == "hpcmu.telemetry.samples"
    assert schema.WORKSPACE_KIND == "hpcmu.telemetry.workspace"
    assert schema.SUMMARY_KIND == "hpcmu.telemetry.summary"


def test_token_sets() -> None:
    assert schema.DTYPES == frozenset(
        {
            "TIMESTAMP_MS_UTC",
            "STRING",
            "INT16",
            "INT32",
            "INT64",
            "FLOAT32",
            "FLOAT64",
        }
    )
    assert schema.UNITS == frozenset(
        {
            "-",
            "ms",
            "s",
            "cores",
            "fraction",
            "count",
            "1/s",
            "bytes",
            "bytes/s",
        }
    )
    assert schema.SOURCES == frozenset(
        {
            "identity",
            "sampler",
            "stat",
            "loadavg",
            "meminfo",
            "vmstat",
            "cgroup",
            "pressure",
            "diskstats",
            "mountstats",
            "netdev",
            "efa",
            "walk",
            "statvfs",
            "header",
        }
    )


def test_sample_columns_names_in_order() -> None:
    names = _names(SAMPLE_COLUMNS)
    assert len(names) == 64
    assert len(set(names)) == 64
    assert names == _EXPECTED_SAMPLE_NAMES


def test_workspace_columns_names_in_order() -> None:
    names = _names(WORKSPACE_COLUMNS)
    assert len(names) == 12
    assert len(set(names)) == 12
    assert names == _EXPECTED_WORKSPACE_NAMES


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            "samples",
            ("ts", "phase", "job_id", "restart", "node", "node_index", "seq"),
        ),
        ("workspace", ("ts", "phase", "job_id", "restart", "node")),
    ],
)
def test_non_null_names(table: str, expected: tuple[str, ...]) -> None:
    columns = _TABLES[table]
    assert tuple(c.name for c in columns if not c.nullable) == expected


@pytest.mark.parametrize("table", sorted(_TABLES))
def test_column_tokens_are_members(table: str) -> None:
    for column in _TABLES[table]:
        assert column.dtype in schema.DTYPES, column.name
        assert column.unit in schema.UNITS, column.name
        assert column.source in schema.SOURCES, column.name


def test_header_is_the_only_source_without_a_column() -> None:
    used = {column.source for column in _ALL_COLUMNS}
    assert used == schema.SOURCES - {"header"}


def test_cpu_meanings_carry_the_adr093_rule() -> None:
    meanings = {column.name: column.meaning for column in SAMPLE_COLUMNS}
    assert "max(Δidle + Δiowait, 0)" in meanings["cpu_busy_cores"]
    assert "null when Δiowait < 0 or Δidle < 0" in meanings["cpu_iowait_cores"]


@pytest.mark.parametrize("column", _ALL_COLUMNS, ids=lambda c: c.name)
def test_unit_suffix_rules(column: Column) -> None:
    expected = _expected_unit(column.name)
    if expected is not None:
        assert column.unit == expected


@pytest.mark.parametrize(
    "record",
    [*_ALL_COLUMNS, *HEADER_FIELDS, *PARQUET_METADATA_KEYS, *WARNING_CODES],
    ids=_record_id,
)
def test_meaning_fits_a_table_cell(
    record: Column | HeaderField | MetadataKey | WarningCode,
) -> None:
    assert record.meaning
    assert record.meaning == record.meaning.strip()
    assert not {"\n", "|", "`"} & set(record.meaning)


def test_header_fields_in_order() -> None:
    assert (
        tuple((f.name, f.json_type, f.nullable) for f in HEADER_FIELDS)
        == _EXPECTED_HEADER
    )


def test_line_and_sort_constants() -> None:
    assert schema.ERROR_FIELDS == ("kind", "ts", "source", "error", "message")
    assert schema.SAMPLE_SORT_KEYS == (
        "phase",
        "job_id",
        "restart",
        "node_index",
        "ts",
    )
    assert schema.WORKSPACE_SORT_KEYS == ("phase", "job_id", "restart", "ts")


@pytest.mark.parametrize(
    ("keys", "table"),
    [
        (schema.SAMPLE_SORT_KEYS, "samples"),
        (schema.WORKSPACE_SORT_KEYS, "workspace"),
    ],
)
def test_sort_keys_are_non_null_columns(
    keys: tuple[str, ...], table: str
) -> None:
    non_null = {c.name for c in _TABLES[table] if not c.nullable}
    assert set(keys) <= non_null


@pytest.mark.parametrize(
    "record",
    [
        SAMPLE_COLUMNS[0],
        HEADER_FIELDS[0],
        PARQUET_METADATA_KEYS[0],
        SAMPLE_STATS[0],
        HEADLINE[0],
        WARNING_CODES[0],
    ],
    ids=lambda r: type(r).__name__,
)
def test_records_are_frozen_and_slotted(
    record: Column
    | HeaderField
    | MetadataKey
    | StatSpec
    | HeadlineSpec
    | WarningCode,
) -> None:
    first = dataclasses.fields(record)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, first, "other")
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("model", "4711", 0, 1), "model-4711-r0-1.ndjson"),
        (("finalize", "12", 3, 0), "finalize-12-r3-0.ndjson"),
    ],
)
def test_staging_file_name(
    args: tuple[str, str, int, int], expected: str
) -> None:
    assert staging_file_name(*args) == expected


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("other", "4711", 0, 0), "^phase "),
        (("model", "47a", 0, 0), "^job_id "),
        (("model", "", 0, 0), "^job_id "),
        (("model", "4711\n", 0, 0), "^job_id "),
        (("model", "٤٧", 0, 0), "^job_id "),
        (("model", "4711", -1, 0), "^restart "),
        (("model", "4711", 0, -1), "^node_index "),
    ],
)
def test_staging_file_name_rejects_invalid(
    args: tuple[str, str, int, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        staging_file_name(*args)


def test_export_files_in_upload_order() -> None:
    assert schema.SAMPLES_FILE == "samples.parquet"
    assert schema.WORKSPACE_FILE == "workspace.parquet"
    assert schema.SUMMARY_FILE == "summary.json"
    assert EXPORT_FILES == (
        "samples.parquet",
        "workspace.parquet",
        "summary.json",
    )


def test_parquet_metadata_keys_in_order() -> None:
    names = tuple(key.name for key in PARQUET_METADATA_KEYS)
    assert names == _EXPECTED_METADATA_KEYS


def test_warning_codes_in_order() -> None:
    codes = tuple(warning.code for warning in WARNING_CODES)
    assert codes == _EXPECTED_WARNING_CODES


def test_warning_codes_limits() -> None:
    assert schema.WARNINGS_CAP == 500
    assert schema.STAGING_LINE_LIMIT == 65536


def test_stat_specs_aggregate_order() -> None:
    assert STAT_AGGREGATES == ("min", "mean", "p95", "p99", "max", "total")


def test_stat_specs_sample_columns() -> None:
    expected = _names(_ROLLED_SAMPLE_COLUMNS)
    assert len(expected) == 56
    assert tuple(spec.column for spec in SAMPLE_STATS) == expected


def test_stat_specs_sample_class_rule() -> None:
    expected = tuple(
        (column.name, _expected_aggregates(column.name, column.unit))
        for column in _ROLLED_SAMPLE_COLUMNS
    )
    actual = tuple((spec.column, spec.aggregates) for spec in SAMPLE_STATS)
    assert actual == expected


def test_stat_specs_sample_class_counts() -> None:
    assert Counter(spec.aggregates for spec in SAMPLE_STATS) == {
        ("mean", "p95", "p99", "max"): 1,
        ("max",): 4,
        ("mean", "p95", "max", "total"): 15,
        ("min", "p95", "max"): 2,
        ("min", "mean", "p95", "max"): 34,
    }


def test_stat_specs_total_names() -> None:
    rates = [c for c in _ROLLED_SAMPLE_COLUMNS if c.unit in _RATE_UNITS]
    assert len(rates) == 15
    assert all(column.name.endswith("_per_s") for column in rates)
    per_s = [c for c in SAMPLE_COLUMNS if c.name.endswith("_per_s")]
    assert per_s == rates
    for spec in SAMPLE_STATS + WORKSPACE_STATS:
        if "total" in spec.aggregates:
            expected = spec.column.removesuffix("_per_s") + "_total"
            assert spec.total_name == expected, spec.column
        else:
            assert spec.total_name is None, spec.column
    totals = tuple(
        spec.total_name for spec in SAMPLE_STATS if spec.total_name is not None
    )
    assert totals == _EXPECTED_TOTAL_NAMES


def test_stat_specs_p99_only_for_sample_cost() -> None:
    with_p99 = [
        spec.column
        for spec in SAMPLE_STATS + WORKSPACE_STATS
        if "p99" in spec.aggregates
    ]
    assert with_p99 == ["sample_cost_ms"]


@pytest.mark.parametrize(
    "spec", SAMPLE_STATS + WORKSPACE_STATS, ids=lambda s: s.column
)
def test_stat_specs_aggregates_in_canonical_order(spec: StatSpec) -> None:
    assert spec.aggregates
    assert set(spec.aggregates) <= set(STAT_AGGREGATES)
    canonical = tuple(a for a in STAT_AGGREGATES if a in spec.aggregates)
    assert spec.aggregates == canonical


def test_stat_specs_workspace_rows() -> None:
    rows = tuple(
        (spec.column, spec.aggregates, spec.total_name)
        for spec in WORKSPACE_STATS
    )
    assert rows == _EXPECTED_WORKSPACE_STATS
    measured = tuple(
        c.name for c in WORKSPACE_COLUMNS if c.source != "identity"
    )
    assert tuple(spec.column for spec in WORKSPACE_STATS) == measured


def test_stat_specs_published_names() -> None:
    sample_names = _published_names(SAMPLE_STATS)
    workspace_names = _published_names(WORKSPACE_STATS)
    assert len(sample_names) == 4 + 4 + 60 + 6 + 136
    assert len(workspace_names) == 7
    names = sample_names + workspace_names
    assert len(set(names)) == len(names)
    for name in names:
        assert re.fullmatch("[a-z0-9_]+", name), name
        assert not name.endswith("_rtt_ms_mean"), name


def test_headline_table_constants() -> None:
    assert HEADLINE_KEY_PREFIX == "telemetry."
    assert schema.WORKSPACE_SCOPE == "workspace"
    assert HEADLINE_FORMATS == frozenset({"bytes", "cores", "fraction"})
    assert HEADLINE_POOLINGS == frozenset({"node_rows", "sum_nodes", "walks"})


def test_headline_table_keys() -> None:
    keys = tuple(HEADLINE_KEY_PREFIX + spec.key for spec in HEADLINE)
    assert keys == _EXPECTED_HEADLINE_KEYS


def test_headline_table_rows() -> None:
    rows = tuple(
        (h.scope, h.source, h.column, h.aggregate, h.pooling, h.fmt)
        for h in HEADLINE
    )
    expected = (
        *(
            (phase, "samples.parquet", *row)
            for phase in ("model", "finalize")
            for row in _EXPECTED_PHASE_HEADLINE_ROWS
        ),
        (
            "workspace",
            "workspace.parquet",
            "allocated_bytes",
            "max",
            "walks",
            "bytes",
        ),
    )
    assert rows == expected


@pytest.mark.parametrize("spec", HEADLINE, ids=lambda h: h.key)
def test_headline_rows_use_known_tokens(spec: HeadlineSpec) -> None:
    assert spec.scope in (*PHASES, "workspace")
    assert spec.source in (SAMPLES_FILE, WORKSPACE_FILE)
    assert spec.pooling in HEADLINE_POOLINGS
    assert spec.fmt in HEADLINE_FORMATS


@pytest.mark.parametrize("spec", HEADLINE, ids=lambda h: h.key)
def test_headline_rows_key_grammar(spec: HeadlineSpec) -> None:
    if spec.aggregate == "total":
        stem = spec.column.removesuffix("_per_s") + "_total"
    else:
        stem = f"{spec.column}_{spec.aggregate}"
    assert spec.key == f"{spec.scope}.{stem}"


@pytest.mark.parametrize("spec", HEADLINE, ids=lambda h: h.key)
def test_headline_rows_name_a_listed_statistic(spec: HeadlineSpec) -> None:
    columns, stats = _STAT_TABLES[spec.source]
    assert spec.column in _names(columns)
    by_column = {stat.column: stat for stat in stats}
    stat = by_column[spec.column]
    assert spec.aggregate in stat.aggregates
    if spec.aggregate == "total":
        published = stat.total_name
    else:
        published = f"{spec.column}_{spec.aggregate}"
    assert spec.key == f"{spec.scope}.{published}"


def test_headline_rows_match_across_phases() -> None:
    assert [h.scope for h in HEADLINE] == (
        ["model"] * 9 + ["finalize"] * 9 + ["workspace"]
    )
    model, finalize = HEADLINE[:9], HEADLINE[9:18]
    assert [dataclasses.replace(h, scope="", key="") for h in model] == [
        dataclasses.replace(h, scope="", key="") for h in finalize
    ]


def test_headline_workspace_scope_is_not_a_phase() -> None:
    assert "workspace" not in PHASES
    assert schema.WORKSPACE_SCOPE not in PHASES


@pytest.mark.parametrize(
    ("fmt", "value", "expected"),
    [
        ("bytes", 1234.5, "1234"),
        ("bytes", 0.5, "0"),
        ("bytes", 7, "7"),
        ("cores", 3.14159, "3.14"),
        ("fraction", 0.123456, "0.1235"),
        ("bytes", -0.0, "0"),
        ("cores", -0.0, "0.00"),
        ("fraction", -0.0, "0.0000"),
    ],
)
def test_format_headline_value(
    fmt: str, value: int | float, expected: str
) -> None:
    assert format_headline_value(fmt, value) == expected


@pytest.mark.parametrize(
    ("fmt", "value", "match"),
    [
        ("bytes", True, "^value must not be a bool: True$"),
        ("cores", float("nan"), "^value must be finite: nan$"),
        ("fraction", float("inf"), "^value must be finite: inf$"),
        ("bytes", -5, "^value must be >= 0: -5$"),
        ("cores", -0.001, "^value must be >= 0: -0.001$"),
        (
            "bytes",
            10**400,
            "^value must be within the float range: an int of 1329 bits$",
        ),
        ("percent", 1.0, "^fmt must be one of .*: 'percent'$"),
    ],
)
def test_format_headline_value_rejects_outside_domain(
    fmt: str, value: int | float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        format_headline_value(fmt, value)


@pytest.mark.parametrize(
    ("fmt", "value", "match"),
    [
        ("percent", True, "^value must not be a bool"),
        ("percent", math.nan, "^fmt must be one of"),
        ("bytes", -(10**400), "^value must be within the float range"),
        ("bytes", -math.inf, "^value must be finite"),
    ],
)
def test_format_headline_value_check_order(
    fmt: str, value: int | float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        format_headline_value(fmt, value)


def test_format_headline_value_longest_result() -> None:
    assert len(format_headline_value("fraction", sys.float_info.max)) == 314


@settings(deadline=None, max_examples=200)
@given(
    fmt=st.sampled_from(sorted(HEADLINE_FORMATS)),
    value=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)
    | st.integers(min_value=0, max_value=2**1023),
)
@example(fmt="bytes", value=1e22)
@example(fmt="cores", value=1e22)
@example(fmt="fraction", value=1e22)
@example(fmt="bytes", value=0.5)
@example(fmt="cores", value=0.5)
@example(fmt="fraction", value=0.5)
def test_format_headline_value_grammar(fmt: str, value: int | float) -> None:
    result = format_headline_value(fmt, value)
    assert re.fullmatch(_HEADLINE_GRAMMAR[fmt], result), result
    assert len(result) <= 314
