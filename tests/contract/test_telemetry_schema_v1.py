"""ADR-072/ADR-093/R52: the frozen telemetry contract, schema_version 1.

`tests/goldens/telemetry/schema-v{SCHEMA_VERSION}.json` holds the
`contract_snapshot()` of that version, and `_violations` holds every later
snapshot to its survival rules, so a change that needs a version bump fails
here until it gets one:

- `schema_version`, `kinds`, `headline_key_prefix` and `warnings_cap` stay
  equal;
- the string lists keep every old entry verbatim, in relative order;
- a keyed record (a file, metadata key, column, header field, summary field,
  term, warning code or headline row) keeps its key, its relative order and
  every field. Its meaning may change only through one `_EDITORIAL_FIXES`
  entry that names the exact old and new text. An added column or header
  field must be nullable, and an added summary field must not be required;
- a statistic row keeps its aggregates and published names, and its
  total_name once set. New names and rows are allowed.

With HPCMU_REGEN_GOLDENS=1 a missing golden is created and the test fails;
an existing golden is never rewritten, so a frozen version stays frozen. A
bump to version 2 creates `schema-v2.json` and keeps `schema-v1.json`.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

import pytest

from hpc_model_utils.infra.telemetry.schema import (
    ERROR_KIND,
    HEADER_FIELDS,
    MEANING_CONVENTIONS,
    SAMPLE_COLUMNS,
    SCHEMA_VERSION,
    WALK_KIND,
    WORKSPACE_COLUMNS,
    Column,
    HeaderField,
    SnapshotValue,
    contract_snapshot,
)

_REGEN_ENV: Final = "HPCMU_REGEN_GOLDENS"
_ROOT: Final = Path(__file__).resolve().parents[2]
_GOLDEN_DIR: Final = _ROOT / "tests" / "goldens" / "telemetry"


class EditorialFix(NamedTuple):
    """One reviewed wording fix of a frozen meaning (ADR-072).

    record is the snapshot section and name the record's key; old is the
    golden's meaning and new the text that replaces it. reason says why the
    fix leaves what is measured unchanged.
    """

    record: str
    name: str
    old: str
    new: str
    reason: str


_EDITORIAL_FIXES: Final[tuple[EditorialFix, ...]] = ()


# The golden: render, narrow, load, regenerate.


def _render(snapshot: Mapping[str, SnapshotValue]) -> str:
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"


def _narrow(value: object, path: str) -> SnapshotValue:
    """value as a SnapshotValue; TypeError names the path of a bad node."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, list):
        return [_narrow(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        narrowed: dict[str, SnapshotValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: a key is not a string: {key!r}")
            narrowed[key] = _narrow(item, f"{path}.{key}")
        return narrowed
    raise TypeError(f"{path}: not a snapshot value: {value!r}")


def _load(path: Path) -> dict[str, SnapshotValue]:
    with path.open(encoding="utf-8") as stream:
        parsed: object = json.load(stream)
    try:
        snapshot = _narrow(parsed, "$")
    except TypeError as exc:
        pytest.fail(f"{path}: {exc}")
    if not isinstance(snapshot, dict):
        pytest.fail(f"{path}: $ is not a JSON object")
    return snapshot


def _regen_or_load(directory: Path, version: int) -> dict[str, SnapshotValue]:
    """The golden of version in directory, created when missing under regen.

    Mode "x" creates the file and never replaces one, so the regen path
    cannot rewrite a frozen golden.
    """
    path = directory / f"schema-v{version}.json"
    if path.exists():
        return _load(path)
    if os.environ.get(_REGEN_ENV) != "1":
        pytest.fail(
            f"{path} is missing; run with {_REGEN_ENV}=1 to create {path.name}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(_render(contract_snapshot()))
    pytest.fail(f"{_REGEN_ENV}=1: created {path}; re-run without it to verify")


def _check_version(golden: Mapping[str, SnapshotValue], version: int) -> None:
    found = golden.get("schema_version")
    if found != version:
        pytest.fail(
            f"the golden holds schema_version {found!r}, but SCHEMA_VERSION "
            f"is {version}"
        )


# The guard.

_EQUAL: Final = (
    "schema_version",
    "kinds",
    "headline_key_prefix",
    "warnings_cap",
)
_ORDERED: Final = (
    "phases",
    "stat_aggregates",
    "headline_scopes",
    "meaning_conventions",
)
_KEYED: Final = {
    "files": "name",
    "parquet_metadata": "name",
    "sample_columns": "name",
    "workspace_columns": "name",
    "header_fields": "name",
    "summary_fields": "path",
    "headline_poolings": "name",
    "headline_formats": "name",
    "warning_codes": "code",
    "headline": "key",
}
# section -> (field, value an added record must have, the message word).
_ADDED: Final = {
    "sample_columns": ("nullable", True, "nullable false"),
    "workspace_columns": ("nullable", True, "nullable false"),
    "header_fields": ("nullable", True, "nullable false"),
    "summary_fields": ("required", False, "required true"),
}
_STATS: Final = ("sample_stats", "workspace_stats")

type _Record = dict[str, SnapshotValue]


def _strings(value: SnapshotValue) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, str)]
    return items if len(items) == len(value) else None


def _records(value: SnapshotValue) -> list[_Record] | None:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, dict)]
    return items if len(items) == len(value) else None


def _index(
    records: list[_Record], key_field: str
) -> tuple[dict[str, _Record], list[str]] | None:
    """(records by key, keys in order), or None for a key that is no string.

    A repeated key keeps its first record and appears once in the order.
    """
    by_key: dict[str, _Record] = {}
    for record in records:
        key = record.get(key_field)
        if not isinstance(key, str):
            return None
        by_key.setdefault(key, record)
    return by_key, list(by_key)


def _duplicates(records: list[_Record], key_field: str) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for record in records:
        key = record.get(key_field)
        if isinstance(key, str):
            if key in seen:
                repeated.append(key)
            seen.add(key)
    return repeated


def _is_subsequence(old: Sequence[str], new: Sequence[str]) -> bool:
    remaining: Iterator[str] = iter(new)
    return all(any(item == other for other in remaining) for item in old)


def _same(section: str, old: SnapshotValue, new: SnapshotValue) -> list[str]:
    return [] if old == new else [f"{section} changed"]


def _ordered_list(
    section: str, old: SnapshotValue, new: SnapshotValue
) -> list[str]:
    old_items, new_items = _strings(old), _strings(new)
    if old_items is None or new_items is None:
        return [f"{section} has the wrong shape"]
    lost = [
        f"{section}[{i}] removed or changed: {item!r}"
        for i, item in enumerate(old_items)
        if item not in new_items
    ]
    if lost:
        return lost
    if not _is_subsequence(old_items, new_items):
        return [f"{section} order changed"]
    return []


def _meaning(
    section: str,
    key: str,
    old: SnapshotValue,
    new: SnapshotValue,
    fixes: Sequence[EditorialFix],
    used: set[int],
) -> list[str]:
    if old == new:
        return []
    matches = [
        i
        for i, fix in enumerate(fixes)
        if (fix.record, fix.name, fix.old, fix.new) == (section, key, old, new)
    ]
    if len(matches) != 1:
        return [f"{section}[{key}].meaning changed"]
    used.update(matches)
    return []


def _keyed(
    section: str,
    old: SnapshotValue,
    new: SnapshotValue,
    fixes: Sequence[EditorialFix],
    used: set[int],
) -> list[str]:
    key_field = _KEYED[section]
    old_records, new_records = _records(old), _records(new)
    if old_records is None or new_records is None:
        return [f"{section} has the wrong shape"]
    old_index = _index(old_records, key_field)
    new_index = _index(new_records, key_field)
    if old_index is None or new_index is None:
        return [f"{section} has the wrong shape"]
    (old_by_key, old_keys), (new_by_key, new_keys) = old_index, new_index
    found = [
        f"{section}[{k}] duplicated"
        for k in _duplicates(new_records, key_field)
    ]
    for key in old_keys:
        if key not in new_by_key:
            found.append(f"{section}[{key}] removed")
            continue
        old_record, new_record = old_by_key[key], new_by_key[key]
        for field, value in old_record.items():
            if field == "meaning":
                found += _meaning(
                    section, key, value, new_record.get(field), fixes, used
                )
            elif field not in new_record or new_record[field] != value:
                found.append(f"{section}[{key}].{field} changed")
    surviving = [key for key in new_keys if key in old_by_key]
    if surviving != [key for key in old_keys if key in new_by_key]:
        found.append(f"{section} order changed")
    if section in _ADDED:
        field, required, word = _ADDED[section]
        found += [
            f"{section}[{key}] added with {word}"
            for key in new_keys
            if key not in old_by_key and new_by_key[key].get(field) != required
        ]
    return found


def _stats(section: str, old: SnapshotValue, new: SnapshotValue) -> list[str]:
    old_records, new_records = _records(old), _records(new)
    if old_records is None or new_records is None:
        return [f"{section} has the wrong shape"]
    old_index = _index(old_records, "column")
    new_index = _index(new_records, "column")
    if old_index is None or new_index is None:
        return [f"{section} has the wrong shape"]
    old_by_column, new_by_column = old_index[0], new_index[0]
    found = [
        f"{section}[{c}] duplicated" for c in _duplicates(new_records, "column")
    ]
    for column, old_row in old_by_column.items():
        new_row = new_by_column.get(column)
        if new_row is None:
            found.append(f"{section}[{column}] removed")
            continue
        for field in ("aggregates", "names"):
            old_items = _strings(old_row.get(field))
            new_items = _strings(new_row.get(field))
            if old_items is None or new_items is None:
                found.append(f"{section}[{column}].{field} has the wrong shape")
                continue
            lost = [item for item in old_items if item not in new_items]
            if lost:
                found.append(f"{section}[{column}].{field} lost {lost}")
        total = old_row.get("total_name")
        if total is not None and new_row.get("total_name") != total:
            found.append(f"{section}[{column}].total_name changed")
    return found


def _violations(
    old: Mapping[str, SnapshotValue],
    new: Mapping[str, SnapshotValue],
    fixes: Sequence[EditorialFix],
) -> list[str]:
    """One message per survival-rule violation of new against old."""
    found: list[str] = []
    used: set[int] = set()
    for section, value in old.items():
        if section not in new:
            found.append(f"{section} removed")
        elif section in _EQUAL:
            found += _same(section, value, new[section])
        elif section in _ORDERED:
            found += _ordered_list(section, value, new[section])
        elif section in _KEYED:
            found += _keyed(section, value, new[section], fixes, used)
        elif section in _STATS:
            found += _stats(section, value, new[section])
        else:
            found.append(f"{section} has no survival rule")
    found += [
        f"stale editorial fix {fix.record}[{fix.name}]: no meaning change "
        "from its old text to its new text"
        for i, fix in enumerate(fixes)
        if i not in used
    ]
    return found


# Fixtures and in-memory helpers.


@pytest.fixture(scope="module")
def committed_golden() -> dict[str, SnapshotValue]:
    return _regen_or_load(_GOLDEN_DIR, SCHEMA_VERSION)


@pytest.fixture
def golden(
    committed_golden: dict[str, SnapshotValue],
) -> dict[str, SnapshotValue]:
    """A deep copy of the committed golden, free to mutate."""
    return copy.deepcopy(committed_golden)


def _list(
    snapshot: Mapping[str, SnapshotValue], section: str
) -> list[SnapshotValue]:
    value = snapshot[section]
    assert isinstance(value, list), section
    return value


def _rows(snapshot: Mapping[str, SnapshotValue], section: str) -> list[_Record]:
    rows = _records(snapshot[section])
    assert rows is not None, section
    return rows


def _row(
    snapshot: Mapping[str, SnapshotValue],
    section: str,
    key_field: str,
    key: str,
) -> _Record:
    for row in _rows(snapshot, section):
        if row.get(key_field) == key:
            return row
    raise LookupError(f"{section}[{key}]")


def _strings_of(row: _Record, field: str) -> list[SnapshotValue]:
    value = row[field]
    assert isinstance(value, list), field
    return value


def _text(row: _Record, field: str) -> str:
    value = row[field]
    assert isinstance(value, str), field
    return value


def _all_strings(value: SnapshotValue) -> Iterator[str]:
    """Every key and every string value below value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_strings(item)


# The freeze-time meanings (AC 1).

_NODE: Final = (
    "Slurm node name (SLURMD_NODENAME), or the host name (uname nodename) "
    "when SLURMD_NODENAME is unset"
)
_NO_OWN_STEP: Final = "; no subtraction when own_step_cgroup is null"

_FREEZE_TIME_MEANINGS: Final = {
    ("HEADER_FIELDS", "cpus_on_node"): (
        "SLURM_JOB_CPUS_PER_NODE indexed by SLURM_NODEID, after expanding "
        "Slurm's compressed list (192(x2) is 192, 192); null when "
        "SLURM_JOB_CPUS_PER_NODE is unset or empty, or when SLURM_NODEID is "
        "outside the expanded list"
    ),
    ("HEADER_FIELDS", "node"): _NODE,
    ("HEADER_FIELDS", "nodes"): (
        "SLURM_JOB_NUM_NODES; null when it is unset or empty"
    ),
    ("HEADER_FIELDS", "own_step_cgroup"): (
        "/sys/fs/cgroup path of the sampler's own step, subtracted from the "
        "job values; null when the sampler's cgroup has no numbered step_N "
        "component (a step_batch, step_extern or step_interactive step, or no "
        "step), and then nothing is subtracted"
    ),
    ("HEADER_FIELDS", "partition"): (
        "SLURM_JOB_PARTITION, stripped; null when it is unset or blank"
    ),
    ("HEADER_FIELDS", "workspace_mount"): (
        "{mountpoint, fstype} of the mountstats entry whose mountpoint is the "
        "longest prefix of the workspace path, its symlinks resolved; null "
        "when no entry matches"
    ),
    ("SAMPLE_COLUMNS", "cpu_util"): (
        "cpu_busy_cores / online CPUs; null as cpu_busy_cores"
    ),
    ("SAMPLE_COLUMNS", "disk_read_bytes_per_s"): (
        "Σ Δsectors read × 512 / dt_s over whole local disks (a "
        "/sys/block/<dev>/device entry, and a /sys/block/<dev>/hidden file "
        "that is absent or holds 0) present in both reads"
    ),
    ("SAMPLE_COLUMNS", "job_mem_anon_bytes"): (
        "memory.stat anon of the job cgroup − anon of the own step, clamped "
        "at 0" + _NO_OWN_STEP
    ),
    ("SAMPLE_COLUMNS", "job_mem_bytes"): (
        "memory.current of the job cgroup − memory.current of the own step, "
        "clamped at 0" + _NO_OWN_STEP
    ),
    ("SAMPLE_COLUMNS", "job_mem_file_bytes"): (
        "memory.stat file of the job cgroup − file of the own step, clamped "
        "at 0" + _NO_OWN_STEP
    ),
    ("SAMPLE_COLUMNS", "job_mem_shmem_bytes"): (
        "memory.stat shmem of the job cgroup − shmem of the own step, clamped "
        "at 0" + _NO_OWN_STEP
    ),
    ("SAMPLE_COLUMNS", "job_pids"): (
        "pids.current of the job cgroup, the sampler step included; null when "
        "the job cgroup has no pids controller (pids.current absent)"
    ),
    ("SAMPLE_COLUMNS", "nfs_read_bytes_per_s"): (
        "Δ(normalreadbytes + directreadbytes) / dt_s of the workspace mount "
        "(the mountstats entry whose mountpoint is the longest prefix of the "
        "workspace path, its symlinks resolved)"
    ),
    ("SAMPLE_COLUMNS", "node"): _NODE,
    ("WORKSPACE_COLUMNS", "node"): _NODE,
}

_FREEZE_TIME_CONVENTIONS: Final = (
    "A source whose read fails leaves every column of that source null in "
    "that sample; its rates are also null in the next sample, which has no "
    "baseline.",
    "A source with nothing to read (no job cgroup, a workspace not on an NFS "
    "mount, no EFA device) leaves its columns null and writes no error line.",
    "A float that is not finite is written as null.",
    "Every cgroup column is null when the header's job_cgroup is null.",
    "A header field whose source cannot be read or parsed, a malformed Slurm "
    "variable included, is null.",
)

_TABLES: Final[dict[str, tuple[Column, ...] | tuple[HeaderField, ...]]] = {
    "SAMPLE_COLUMNS": SAMPLE_COLUMNS,
    "WORKSPACE_COLUMNS": WORKSPACE_COLUMNS,
    "HEADER_FIELDS": HEADER_FIELDS,
}


def test_freeze_time_meaning_keys_are_the_16_amended() -> None:
    assert len(_FREEZE_TIME_MEANINGS) == 16
    assert sorted(_FREEZE_TIME_MEANINGS) == [
        ("HEADER_FIELDS", "cpus_on_node"),
        ("HEADER_FIELDS", "node"),
        ("HEADER_FIELDS", "nodes"),
        ("HEADER_FIELDS", "own_step_cgroup"),
        ("HEADER_FIELDS", "partition"),
        ("HEADER_FIELDS", "workspace_mount"),
        ("SAMPLE_COLUMNS", "cpu_util"),
        ("SAMPLE_COLUMNS", "disk_read_bytes_per_s"),
        ("SAMPLE_COLUMNS", "job_mem_anon_bytes"),
        ("SAMPLE_COLUMNS", "job_mem_bytes"),
        ("SAMPLE_COLUMNS", "job_mem_file_bytes"),
        ("SAMPLE_COLUMNS", "job_mem_shmem_bytes"),
        ("SAMPLE_COLUMNS", "job_pids"),
        ("SAMPLE_COLUMNS", "nfs_read_bytes_per_s"),
        ("SAMPLE_COLUMNS", "node"),
        ("WORKSPACE_COLUMNS", "node"),
    ]


@pytest.mark.parametrize(
    ("table", "name"),
    list(_FREEZE_TIME_MEANINGS),
    ids=[f"{t}-{n}" for t, n in _FREEZE_TIME_MEANINGS],
)
def test_freeze_time_meaning_is_the_requirement_text(
    table: str, name: str
) -> None:
    meanings = {record.name: record.meaning for record in _TABLES[table]}
    text = _FREEZE_TIME_MEANINGS[(table, name)]

    assert meanings[name] == text
    assert text == text.strip()
    assert not {"\n", "|", "`"} & set(text)


def test_freeze_time_conventions_are_appended_in_order() -> None:
    assert len(MEANING_CONVENTIONS) == 9
    assert MEANING_CONVENTIONS[4:] == _FREEZE_TIME_CONVENTIONS
    assert [
        t for t in _FREEZE_TIME_CONVENTIONS if {"\n", "|", "`"} & set(t)
    ] == []


# The snapshot and the golden (AC 2).

_TOP_LEVEL_KEYS: Final = (
    "schema_version",
    "kinds",
    "phases",
    "files",
    "parquet_metadata",
    "meaning_conventions",
    "sample_columns",
    "workspace_columns",
    "header_fields",
    "summary_fields",
    "stat_aggregates",
    "sample_stats",
    "workspace_stats",
    "headline_key_prefix",
    "headline_scopes",
    "headline_poolings",
    "headline_formats",
    "headline",
    "warning_codes",
    "warnings_cap",
)


def test_snapshot_survives_the_committed_golden(
    committed_golden: dict[str, SnapshotValue],
) -> None:
    _check_version(committed_golden, SCHEMA_VERSION)

    assert (
        _violations(committed_golden, contract_snapshot(), _EDITORIAL_FIXES)
        == []
    )


def test_snapshot_top_level_keys_in_order() -> None:
    assert tuple(contract_snapshot()) == _TOP_LEVEL_KEYS


@pytest.mark.parametrize(
    ("section", "count"),
    [
        ("sample_columns", 64),
        ("workspace_columns", 12),
        ("header_fields", 22),
        ("summary_fields", 33),
        ("parquet_metadata", 6),
        ("sample_stats", 56),
        ("workspace_stats", 7),
        ("headline", 19),
        ("warning_codes", 9),
    ],
)
def test_snapshot_record_counts(section: str, count: int) -> None:
    assert len(_rows(contract_snapshot(), section)) == count


@pytest.mark.parametrize(
    ("section", "count"), [("sample_stats", 210), ("workspace_stats", 7)]
)
def test_snapshot_stat_rows_carry_the_published_names(
    section: str, count: int
) -> None:
    names = [
        name
        for row in _rows(contract_snapshot(), section)
        for name in _strings_of(row, "names")
    ]

    assert len(names) == len(set(names)) == count


def test_snapshot_records_follow_tuple_order() -> None:
    snapshot = contract_snapshot()

    assert [r["name"] for r in _rows(snapshot, "sample_columns")] == [
        c.name for c in SAMPLE_COLUMNS
    ]
    assert [r["name"] for r in _rows(snapshot, "header_fields")] == [
        f.name for f in HEADER_FIELDS
    ]
    assert snapshot["meaning_conventions"] == list(MEANING_CONVENTIONS)
    assert snapshot["headline_scopes"] == ["model", "finalize", "workspace"]
    assert _rows(snapshot, "files") == [
        {
            "name": "samples.parquet",
            "kind": "hpcmu.telemetry.samples",
            "sort_keys": ["phase", "job_id", "restart", "node_index", "ts"],
        },
        {
            "name": "workspace.parquet",
            "kind": "hpcmu.telemetry.workspace",
            "sort_keys": ["phase", "job_id", "restart", "ts"],
        },
        {
            "name": "summary.json",
            "kind": "hpcmu.telemetry.summary",
            "sort_keys": [],
        },
    ]
    assert _row(snapshot, "sample_stats", "column", "major_faults_per_s") == {
        "column": "major_faults_per_s",
        "aggregates": ["mean", "p95", "max", "total"],
        "total_name": "major_faults_total",
        "names": [
            "major_faults_per_s_mean",
            "major_faults_per_s_p95",
            "major_faults_per_s_max",
            "major_faults_total",
        ],
    }


def test_snapshot_leaves_out_staging_items() -> None:
    staging = (WALK_KIND, ERROR_KIND, "STAGING_LINE_LIMIT")
    texts = list(_all_strings(contract_snapshot()))

    assert staging == (
        "hpcmu.telemetry.walk",
        "hpcmu.telemetry.error",
        "STAGING_LINE_LIMIT",
    )
    assert [t for t in texts if any(item in t for item in staging)] == []


def test_snapshot_calls_return_equal_distinct_objects() -> None:
    first, second = contract_snapshot(), contract_snapshot()

    assert first == second
    assert first is not second
    _list(first, "phases").append("other")
    _strings_of(_rows(first, "sample_stats")[0], "names").clear()
    assert second == contract_snapshot()


def test_snapshot_version_mismatch_names_both_numbers(
    golden: dict[str, SnapshotValue],
) -> None:
    golden["schema_version"] = 2

    with pytest.raises(pytest.fail.Exception, match=r"schema_version 2.* is 1"):
        _check_version(golden, 1)


def test_snapshot_golden_narrowing_names_the_bad_node() -> None:
    with pytest.raises(TypeError, match=r"^\$\.a\[1\]\.b: not a snapshot"):
        _narrow({"a": [1, {"b": 1.5}]}, "$")


def test_snapshot_golden_that_is_no_object_fails(tmp_path: Path) -> None:
    path = tmp_path / "schema-v1.json"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(pytest.fail.Exception, match=r"\$ is not a JSON object"):
        _load(path)


# The guard rules (AC 3).

type _Mutation = Callable[[dict[str, SnapshotValue]], None]


def _drop_p95_name(snapshot: dict[str, SnapshotValue]) -> None:
    row = _row(snapshot, "sample_stats", "column", "cpu_busy_cores")
    _strings_of(row, "names").remove("cpu_busy_cores_p95")


def _pool_by_sum(snapshot: dict[str, SnapshotValue]) -> None:
    _row(snapshot, "headline", "key", "model.cpu_busy_cores_mean")[
        "pooling"
    ] = "sum_nodes"


def _reword_job_pids(snapshot: dict[str, SnapshotValue]) -> None:
    _row(snapshot, "sample_columns", "name", "job_pids")["meaning"] = (
        "pids.current of the job cgroup"
    )


def _add_column(nullable: bool) -> _Mutation:
    def mutate(snapshot: dict[str, SnapshotValue]) -> None:
        _list(snapshot, "sample_columns").append(
            {
                "name": "gpu_util",
                "dtype": "FLOAT32",
                "nullable": nullable,
                "unit": "fraction",
                "source": "stat",
                "meaning": "a test column",
            }
        )

    return mutate


def _swap_columns(snapshot: dict[str, SnapshotValue]) -> None:
    columns = _list(snapshot, "sample_columns")
    columns[1], columns[2] = columns[2], columns[1]


def _add_summary_field(required: bool) -> _Mutation:
    def mutate(snapshot: dict[str, SnapshotValue]) -> None:
        _list(snapshot, "summary_fields").append(
            {
                "path": "run.extra",
                "json_type": "string",
                "required": required,
                "nullable": True,
                "meaning": "a test field",
            }
        )

    return mutate


def _rename_total(snapshot: dict[str, SnapshotValue]) -> None:
    row = _row(snapshot, "sample_stats", "column", "major_faults_per_s")
    row["total_name"] = "major_faults_sum"


def _add_header_field(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "header_fields").append(
        {
            "name": "gpus",
            "json_type": "integer",
            "nullable": False,
            "meaning": "a test field",
        }
    )


def _drop_column(snapshot: dict[str, SnapshotValue]) -> None:
    columns = _list(snapshot, "sample_columns")
    columns.remove(_row(snapshot, "sample_columns", "name", "job_pids"))


def _retype_column(snapshot: dict[str, SnapshotValue]) -> None:
    _row(snapshot, "sample_columns", "name", "job_pids")["dtype"] = "INT64"


def _reword_convention(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "meaning_conventions")[0] = "Δ is a difference."


def _swap_phases(snapshot: dict[str, SnapshotValue]) -> None:
    snapshot["phases"] = ["finalize", "model"]


def _change_kind(snapshot: dict[str, SnapshotValue]) -> None:
    kinds = snapshot["kinds"]
    assert isinstance(kinds, dict)
    kinds["samples"] = "hpcmu.telemetry.rows"


def _change_sort_key(snapshot: dict[str, SnapshotValue]) -> None:
    _strings_of(
        _row(snapshot, "files", "name", "samples.parquet"), "sort_keys"
    ).pop()


def _drop_aggregate(snapshot: dict[str, SnapshotValue]) -> None:
    row = _row(snapshot, "sample_stats", "column", "load1")
    _strings_of(row, "aggregates").remove("min")


def _drop_stat_row(snapshot: dict[str, SnapshotValue]) -> None:
    rows = _list(snapshot, "workspace_stats")
    rows.remove(_row(snapshot, "workspace_stats", "column", "walk_s"))


def _drop_warning_code(snapshot: dict[str, SnapshotValue]) -> None:
    codes = _list(snapshot, "warning_codes")
    codes.remove(_row(snapshot, "warning_codes", "code", "bad_line"))


def _raise_cap(snapshot: dict[str, SnapshotValue]) -> None:
    snapshot["warnings_cap"] = 1000


def _drop_section(snapshot: dict[str, SnapshotValue]) -> None:
    del snapshot["headline_formats"]


def _repeat_column(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "sample_columns").append(
        copy.deepcopy(_row(snapshot, "sample_columns", "name", "ts"))
    )


def _add_stat_name(snapshot: dict[str, SnapshotValue]) -> None:
    row = _row(snapshot, "sample_stats", "column", "cpu_busy_cores")
    _strings_of(row, "aggregates").insert(3, "p99")
    _strings_of(row, "names").insert(3, "cpu_busy_cores_p99")


def _add_warning_code(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "warning_codes").append(
        {"code": "clock_jump", "meaning": "a test warning"}
    )


def _add_stat_row(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "sample_stats").append(
        {
            "column": "gpu_util",
            "aggregates": ["mean"],
            "total_name": None,
            "names": ["gpu_util_mean"],
        }
    )


def _add_convention(snapshot: dict[str, SnapshotValue]) -> None:
    _list(snapshot, "meaning_conventions").append("A test convention.")


def _unchanged(snapshot: dict[str, SnapshotValue]) -> None:
    del snapshot  # the unmutated golden


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_p95_name, "sample_stats[cpu_busy_cores].names lost"),
        (_pool_by_sum, "headline[model.cpu_busy_cores_mean].pooling changed"),
        (_reword_job_pids, "sample_columns[job_pids].meaning changed"),
        (
            _add_column(False),
            "sample_columns[gpu_util] added with nullable false",
        ),
        (_swap_columns, "sample_columns order changed"),
        (
            _add_summary_field(True),
            "summary_fields[run.extra] added with required true",
        ),
        (_rename_total, "sample_stats[major_faults_per_s].total_name changed"),
        (_add_header_field, "header_fields[gpus] added with nullable false"),
        (_drop_column, "sample_columns[job_pids] removed"),
        (_retype_column, "sample_columns[job_pids].dtype changed"),
        (_reword_convention, "meaning_conventions[0] removed or changed"),
        (_swap_phases, "phases order changed"),
        (_change_kind, "kinds changed"),
        (_change_sort_key, "files[samples.parquet].sort_keys changed"),
        (_drop_aggregate, "sample_stats[load1].aggregates lost ['min']"),
        (_drop_stat_row, "workspace_stats[walk_s] removed"),
        (_drop_warning_code, "warning_codes[bad_line] removed"),
        (_raise_cap, "warnings_cap changed"),
        (_drop_section, "headline_formats removed"),
        (_repeat_column, "sample_columns[ts] duplicated"),
    ],
    ids=[
        "p95-name-removed",
        "pooling-changed",
        "job-pids-reworded",
        "non-nullable-column-added",
        "columns-swapped",
        "required-summary-field-added",
        "total-name-changed",
        "non-nullable-header-field-added",
        "column-removed",
        "dtype-changed",
        "convention-reworded",
        "phases-reordered",
        "kind-changed",
        "sort-key-removed",
        "aggregate-removed",
        "stat-row-removed",
        "warning-code-removed",
        "cap-changed",
        "section-removed",
        "column-repeated",
    ],
)
def test_guard_flags_a_violation(
    golden: dict[str, SnapshotValue], mutate: _Mutation, message: str
) -> None:
    new = copy.deepcopy(golden)
    mutate(new)

    found = _violations(golden, new, ())

    assert found != []
    assert [m for m in found if m.startswith(message)] != [], found


@pytest.mark.parametrize(
    "mutate",
    [
        _add_column(True),
        _add_stat_name,
        _add_warning_code,
        _add_summary_field(False),
        _add_stat_row,
        _add_convention,
        _unchanged,
    ],
    ids=[
        "nullable-column-added",
        "statistic-name-added",
        "warning-code-added",
        "optional-summary-field-added",
        "stat-row-added",
        "convention-appended",
        "unmutated",
    ],
)
def test_guard_allows_an_addition(
    golden: dict[str, SnapshotValue], mutate: _Mutation
) -> None:
    new = copy.deepcopy(golden)
    mutate(new)

    assert _violations(golden, new, ()) == []


def test_guard_flags_a_section_without_a_rule(
    golden: dict[str, SnapshotValue],
) -> None:
    extra: list[SnapshotValue] = ["kind"]
    old: dict[str, SnapshotValue] = {**golden, "error_fields": extra}
    new: dict[str, SnapshotValue] = {**golden, "error_fields": list(extra)}

    assert _violations(old, new, ()) == ["error_fields has no survival rule"]


# The editorial path and the regen path (AC 4).


def _job_pids_fix(old: str, new: str) -> EditorialFix:
    return EditorialFix(
        record="sample_columns",
        name="job_pids",
        old=old,
        new=new,
        reason="a test fix",
    )


def test_editorial_fix_accepts_its_exact_meaning_change(
    golden: dict[str, SnapshotValue],
) -> None:
    old = _text(_row(golden, "sample_columns", "name", "job_pids"), "meaning")
    new = copy.deepcopy(golden)
    _reword_job_pids(new)
    reworded = _text(_row(new, "sample_columns", "name", "job_pids"), "meaning")

    assert _violations(golden, new, (_job_pids_fix(old, reworded),)) == []


def test_editorial_fix_with_another_old_text_is_stale(
    golden: dict[str, SnapshotValue],
) -> None:
    new = copy.deepcopy(golden)
    _reword_job_pids(new)
    reworded = _text(_row(new, "sample_columns", "name", "job_pids"), "meaning")

    found = _violations(
        golden, new, (_job_pids_fix("not the golden", reworded),)
    )

    assert "sample_columns[job_pids].meaning changed" in found
    assert [m for m in found if m.startswith("stale editorial fix")] == [
        "stale editorial fix sample_columns[job_pids]: no meaning change from "
        "its old text to its new text"
    ]


def test_editorial_fix_matching_no_change_is_stale(
    golden: dict[str, SnapshotValue],
) -> None:
    old = _text(_row(golden, "sample_columns", "name", "job_pids"), "meaning")

    found = _violations(
        golden, copy.deepcopy(golden), (_job_pids_fix(old, "x"),)
    )

    assert len(found) == 1
    assert found[0].startswith("stale editorial fix sample_columns[job_pids]")


def test_editorial_fix_twice_for_one_change_is_flagged(
    golden: dict[str, SnapshotValue],
) -> None:
    old = _text(_row(golden, "sample_columns", "name", "job_pids"), "meaning")
    new = copy.deepcopy(golden)
    _reword_job_pids(new)
    reworded = _text(_row(new, "sample_columns", "name", "job_pids"), "meaning")
    fix = _job_pids_fix(old, reworded)

    found = _violations(golden, new, (fix, fix))

    assert "sample_columns[job_pids].meaning changed" in found
    assert len([m for m in found if m.startswith("stale editorial fix")]) == 2


def test_editorial_fixes_are_empty() -> None:
    assert _EDITORIAL_FIXES == ()


def test_regen_creates_a_missing_golden_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_REGEN_ENV, "1")
    directory = tmp_path / "goldens"

    with pytest.raises(pytest.fail.Exception, match="created .*schema-v1.json"):
        _regen_or_load(directory, 1)

    path = directory / "schema-v1.json"
    assert path.read_text(encoding="utf-8") == _render(contract_snapshot())
    assert _violations(_load(path), contract_snapshot(), ()) == []


def test_regen_keeps_an_existing_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_REGEN_ENV, "1")
    stale = contract_snapshot()
    _reword_job_pids(stale)
    path = tmp_path / "schema-v1.json"
    path.write_text(_render(stale), encoding="utf-8")
    os.utime(path, ns=(1_000_000_000, 946_684_800_000_000_000))
    before = path.read_bytes()

    loaded = _regen_or_load(tmp_path, 1)

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == 946_684_800_000_000_000
    assert loaded == stale


def test_regen_unset_reports_how_to_create_a_missing_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_REGEN_ENV, raising=False)

    with pytest.raises(
        pytest.fail.Exception,
        match=r"run with HPCMU_REGEN_GOLDENS=1 to create schema-v1\.json$",
    ):
        _regen_or_load(tmp_path, 1)

    assert list(tmp_path.iterdir()) == []
