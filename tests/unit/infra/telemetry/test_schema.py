from __future__ import annotations

import dataclasses

import pytest

from hpc_model_utils.infra.telemetry import schema
from hpc_model_utils.infra.telemetry.schema import (
    HEADER_FIELDS,
    SAMPLE_COLUMNS,
    WORKSPACE_COLUMNS,
    Column,
    HeaderField,
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


def _names(columns: tuple[Column, ...]) -> tuple[str, ...]:
    return tuple(column.name for column in columns)


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
    [*_ALL_COLUMNS, *HEADER_FIELDS],
    ids=lambda r: r.name,
)
def test_meaning_fits_a_table_cell(record: Column | HeaderField) -> None:
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


def test_records_are_frozen_and_slotted() -> None:
    column = SAMPLE_COLUMNS[0]
    field = HEADER_FIELDS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(column, "name", "other")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(field, "name", "other")
    assert not hasattr(column, "__dict__")
    assert not hasattr(field, "__dict__")


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
