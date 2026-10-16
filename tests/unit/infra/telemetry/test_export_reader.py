"""ADR-069/ADR-073: the export's staging reader over writer-built files."""

from __future__ import annotations

import errno
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from hpc_model_utils.infra.telemetry.collect import Collector
from hpc_model_utils.infra.telemetry.export import (
    Candidate,
    FileRead,
    ReadWarning,
    discover,
    read_body,
    read_header,
)
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    ERROR_KIND,
    HEADER_FIELDS,
    HEADER_KIND,
    SAMPLE_COLUMNS,
    STAGING_LINE_LIMIT,
    WARNING_CODES,
    WARNINGS_CAP,
    WORKSPACE_COLUMNS,
)
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    ErrorLine,
    header,
    sample,
    walk,
    write_staging,
)
from tests.support.telemetry_tree import load_bundle, materialize

_JOBS = frozenset({("model", "1000"), ("finalize", "1001")})
_CODES = frozenset(code.code for code in WARNING_CODES)
_SAMPLE_NAMES = [c.name for c in SAMPLE_COLUMNS]
_RATE_UNITS = ("1/s", "bytes/s")
_WORKSPACE = Path("/shared-zfs/ws-test")
_PATCH = "@patch@"
_PATCH_BYTES = b'"@patch@"'

_HEAD = header()
_S0 = sample(_HEAD, 0)
_S1 = sample(_HEAD, 1)
_S2 = sample(_HEAD, 2)
_W0 = walk(_HEAD, ts=ORIGIN_MS + 5_000, files=3)
_W2 = walk(_HEAD, ts=ORIGIN_MS + 65_000, files=4)

_NOT_JSON = ReadWarning("bad_line", "not ASCII JSON", 1)
_TOO_LONG = ReadWarning(
    "bad_line", f"line exceeds {STAGING_LINE_LIMIT} bytes", 1
)
_TRUNCATED = ReadWarning("truncated_tail", "last line has no newline", 1)


def _bad(message: str) -> ReadWarning:
    return ReadWarning("bad_record", message, 1)


def _dtype(column: str) -> ReadWarning:
    return _bad(f"sample column {column} breaks its dtype")


def _row(record: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    return tuple(record[c.name] for c in SAMPLE_COLUMNS)


def _walk_row(record: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    return tuple(record[c.name] for c in WORKSPACE_COLUMNS)


def _ts(record: Mapping[str, JsonValue]) -> int:
    value = record["ts"]
    assert isinstance(value, int)
    return value


def _without(
    record: Mapping[str, JsonValue], name: str
) -> dict[str, JsonValue]:
    return {key: value for key, value in record.items() if key != name}


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def _rewrite(path: Path, lines: Iterable[bytes]) -> None:
    path.write_bytes(b"".join(lines))


def _long_line(size: int) -> bytes:
    """A JSON object line of size bytes, its newline included."""
    return b'{"pad":"' + b"a" * (size - 11) + b'"}\n'


def _padded(line: bytes, size: int) -> bytes:
    """line, a JSON object, padded with spaces to size bytes."""
    return line[:1] + b" " * (size - len(line)) + line[1:]


def _only(directory: Path) -> Candidate:
    (candidate,) = discover(directory, _JOBS)
    return candidate


def _read(directory: Path) -> FileRead:
    return read_body(read_header(_only(directory)))


# Discovery.


def test_discover_ignores_logs_and_stray_names(tmp_path: Path) -> None:
    model = write_staging(tmp_path, _HEAD, [_S0])
    restarted = header(restart=1)
    write_staging(tmp_path, restarted, [sample(restarted, 0)])
    for name in (
        "model-1000-r0-0.log",
        "model-1000-r0-srun.log",
        "model-1000-r01-0.ndjson",
        "model-1000-r0-1.ndjson.tmp",
    ):
        (tmp_path / name).write_bytes(model.read_bytes())
    write_staging(tmp_path, header(job_id="1001"))
    (tmp_path / "finalize-1001-r0-1.ndjson").symlink_to(model)
    (tmp_path / "finalize-1001-r0-2.ndjson").mkdir()
    write_staging(tmp_path, header("finalize", "1001"))

    found = discover(tmp_path, _JOBS)

    assert [c.path.name for c in found] == [
        "finalize-1001-r0-0.ndjson",
        "model-1000-r0-0.ndjson",
        "model-1000-r1-0.ndjson",
    ]
    assert found[2] == Candidate(
        tmp_path / "model-1000-r1-0.ndjson", "model", "1000", 1, 0
    )
    assert discover(tmp_path / "absent", _JOBS) == []


@pytest.mark.parametrize(
    "name",
    [
        "model-1000-r00-0.ndjson",
        "model-1000-r0-00.ndjson",
        "model-1000-r-1-0.ndjson",
        "model-1000-r0-0-0.ndjson",
        "other-1000-r0-0.ndjson",
        "Model-1000-r0-0.ndjson",
        "model-1000-r0-0.NDJSON",
        "model-1000-r0-0.ndjson.log",
        ".model-1000-r0-0.ndjson",
    ],
)
def test_discover_rejects_a_name_the_writer_never_writes(
    tmp_path: Path, name: str
) -> None:
    write_staging(tmp_path, _HEAD, [_S0]).rename(tmp_path / name)
    phase, job_id = name.split("-")[:2]

    assert discover(tmp_path, {*_JOBS, (phase, job_id)}) == []


def test_discover_matches_job_id_as_a_string(tmp_path: Path) -> None:
    # A job id is text, as SLURM_JOB_ID is: 01000 is not job 1000.
    write_staging(tmp_path, header(job_id="01000"))

    assert discover(tmp_path, _JOBS) == []
    assert [c.job_id for c in discover(tmp_path, {("model", "01000")})] == [
        "01000"
    ]


def test_discover_sorts_restart_and_node_index_as_ints(
    tmp_path: Path,
) -> None:
    for restart, node_index in ((10, 0), (2, 10), (2, 2)):
        write_staging(tmp_path, header(restart=restart, node_index=node_index))

    found = discover(tmp_path, _JOBS)

    assert [(c.restart, c.node_index) for c in found] == [
        (2, 2),
        (2, 10),
        (10, 0),
    ]


def test_discover_other_os_error_propagates(tmp_path: Path) -> None:
    staged = write_staging(tmp_path, _HEAD)

    with pytest.raises(NotADirectoryError):
        discover(staged, _JOBS)


# Headers.


def test_read_header_torn_prefix_is_one_bad_line(tmp_path: Path) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0])
    torn = b'{"kind":"hpcmu.tel\n'
    header_line = _lines(path)[0]
    path.write_bytes(torn + path.read_bytes())

    head = read_header(_only(tmp_path))
    read = read_body(head)

    assert head.header == _HEAD
    assert head.body_offset == len(torn) + len(header_line)
    assert head.warnings == (_NOT_JSON,)
    assert read.warnings == (_NOT_JSON,)
    assert read.samples == (_row(_S0),)


def test_read_header_repeated_header_line_is_one_bad_record(
    tmp_path: Path,
) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1])
    header_line, *rest = _lines(path)
    _rewrite(path, [header_line, header_line, *rest])

    read = _read(tmp_path)

    assert read.warnings == (_bad("repeated header line"),)
    assert read.samples == (_row(_S0), _row(_S1))


def test_read_header_only_file_reads_no_rows_and_no_warnings(
    tmp_path: Path,
) -> None:
    write_staging(tmp_path, _HEAD)

    read = _read(tmp_path)

    assert read.header == _HEAD
    assert list(read.header) == [field.name for field in HEADER_FIELDS]
    assert (read.samples, read.walks, read.warnings) == ((), (), ())
    assert read.warnings_omitted == 0


def test_read_header_skips_a_json_non_object_before_the_header(
    tmp_path: Path,
) -> None:
    path = write_staging(tmp_path, _HEAD)
    path.write_bytes(b"[1]\n" + path.read_bytes())

    head = read_header(_only(tmp_path))

    assert head.header == _HEAD
    assert head.warnings == (_bad("not a JSON object"),)


def test_read_header_without_newline_is_a_truncated_tail(
    tmp_path: Path,
) -> None:
    path = write_staging(tmp_path, _HEAD)
    path.write_bytes(path.read_bytes()[:-1])

    head = read_header(_only(tmp_path))

    assert head.header is None
    assert head.warnings == (
        _TRUNCATED,
        ReadWarning("bad_header", "no header line", 1),
    )


def test_read_header_does_not_follow_a_symlink_swapped_in(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    path = write_staging(staging, _HEAD, [_S0])
    candidate = _only(staging)
    head = read_header(candidate)
    outside = tmp_path / "outside.ndjson"
    path.rename(outside)
    path.symlink_to(outside)

    with pytest.raises(OSError) as header_error:
        read_header(candidate)
    with pytest.raises(OSError) as body_error:
        read_body(head)

    assert header_error.value.errno == errno.ELOOP
    assert body_error.value.errno == errno.ELOOP


def _replaced(data: bytes) -> Callable[[Path], None]:
    def build(directory: Path) -> None:
        write_staging(directory, _HEAD).write_bytes(data)

    return build


def _edited(old: bytes, new: bytes) -> Callable[[Path], None]:
    def build(directory: Path) -> None:
        path = write_staging(directory, _HEAD)
        data = path.read_bytes()
        assert data.count(old) == 1
        path.write_bytes(data.replace(old, new))

    return build


def _written(head: Mapping[str, JsonValue]) -> Callable[[Path], None]:
    def build(directory: Path) -> None:
        write_staging(directory, head)

    return build


def _sample_first(directory: Path) -> None:
    path = write_staging(directory, _HEAD, [_S0])
    _rewrite(path, _lines(path)[1:])


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(_replaced(b""), "no header line", id="empty"),
        pytest.param(_replaced(b"\n\n"), "no header line", id="blank_lines"),
        pytest.param(
            _written(_without(_HEAD, HEADER_FIELDS[-1].name)),
            "header key set differs",
            id="missing_key",
        ),
        pytest.param(
            _written({**_HEAD, "extra": 1}),
            "header key set differs",
            id="extra_key",
        ),
        pytest.param(
            _edited(HEADER_KIND.encode(), b"hpcmu.telemetry.other"),
            "header kind differs",
            id="wrong_kind",
        ),
        pytest.param(
            _written({**_HEAD, "schema_version": 2}),
            "unsupported schema_version 2",
            id="schema_version_2",
        ),
        pytest.param(
            _written(header(interval_s=0)),
            "header interval_s below 1",
            id="interval_s_0",
        ),
        pytest.param(
            _written(header(workspace_interval_s=0)),
            "header workspace_interval_s below 1",
            id="workspace_interval_s_0",
        ),
        pytest.param(
            _written(header(nodes=0)), "header nodes below 1", id="nodes_0"
        ),
        pytest.param(
            _edited(b'"node_index":0', b'"node_index":1'),
            "header identity differs from the file name",
            id="node_index_1_in_node_0_file",
        ),
        pytest.param(
            _edited(b'"job_id":"1000"', b'"job_id":"1001"'),
            "header identity differs from the file name",
            id="other_job_in_file",
        ),
        pytest.param(
            _edited(b'"restart":0', b'"restart":"0"'),
            "header field restart has the wrong type",
            id="string_restart",
        ),
        pytest.param(
            _written({**_HEAD, "nodes": True}),
            "header field nodes has the wrong type",
            id="bool_nodes",
        ),
        pytest.param(
            _written({**_HEAD, "tool_version": None}),
            "header field tool_version has the wrong type",
            id="null_tool_version",
        ),
        pytest.param(
            _written({**_HEAD, "workspace_mount": {"mountpoint": 1}}),
            "header field workspace_mount has the wrong type",
            id="mount_with_int_value",
        ),
        pytest.param(
            _written({**_HEAD, "workspace_mount": "/shared-zfs"}),
            "header field workspace_mount has the wrong type",
            id="mount_as_string",
        ),
        pytest.param(
            _sample_first, "header key set differs", id="sample_first"
        ),
    ],
)
def test_read_header_bad_header_skips_the_file(
    tmp_path: Path, build: Callable[[Path], None], message: str
) -> None:
    build(tmp_path)

    head = read_header(_only(tmp_path))

    assert head.header is None
    assert head.warnings == (ReadWarning("bad_header", message, 1),)
    with pytest.raises(ValueError, match="needs a valid header"):
        read_body(head)


# Body lines.


_BODY_CASES = [
    pytest.param(_long_line(70_000), None, _TOO_LONG, id="70000_bytes"),
    pytest.param(
        sample(_HEAD, 1, cpu_util=_PATCH), b"NaN", _NOT_JSON, id="nan_token"
    ),
    pytest.param(
        sample(_HEAD, 1, cpu_util=_PATCH),
        b"-Infinity",
        _NOT_JSON,
        id="infinity_token",
    ),
    pytest.param(b"[" * 65_000 + b"\n", None, _NOT_JSON, id="deep_nesting"),
    pytest.param(
        sample(_HEAD, 1, node=_PATCH), b'"\xff"', _NOT_JSON, id="0xff_byte"
    ),
    pytest.param(
        b"1" * 4_400 + b"\n", None, _NOT_JSON, id="int_over_4300_digits"
    ),
    pytest.param(b"  \n", None, _NOT_JSON, id="whitespace_only"),
    pytest.param(b"[1,2]\n", None, _bad("not a JSON object"), id="json_array"),
    pytest.param(
        _without(_S1, SAMPLE_COLUMNS[-1].name),
        None,
        _bad("sample key set differs"),
        id="missing_column",
    ),
    pytest.param(
        {**_S1, "extra": 1},
        None,
        _bad("sample key set differs"),
        id="extra_key",
    ),
    pytest.param(
        sample(_HEAD, 1, procs_running=True),
        None,
        _dtype("procs_running"),
        id="true_in_procs_running",
    ),
    pytest.param({**_S1, "seq": 1.5}, None, _dtype("seq"), id="float_in_seq"),
    pytest.param(
        sample(_HEAD, 1, cpu_util=1e39),
        None,
        _dtype("cpu_util"),
        id="1e39_in_cpu_util",
    ),
    pytest.param(
        sample(_HEAD, 1, disk_read_bytes_per_s=_PATCH),
        b"1e999",
        _dtype("disk_read_bytes_per_s"),
        id="1e999_in_disk_read_bytes_per_s",
    ),
    pytest.param(
        sample(_HEAD, 1, node=None), None, _dtype("node"), id="null_in_node"
    ),
    pytest.param(
        sample(_HEAD, 1, load1=10**400),
        None,
        _dtype("load1"),
        id="int_beyond_float_range",
    ),
    pytest.param(
        sample(_HEAD, 1, cpu_active_cores=2**15),
        None,
        _dtype("cpu_active_cores"),
        id="int16_overflow",
    ),
    pytest.param(
        sample(_HEAD, 1, mem_used_bytes=2**63),
        None,
        _dtype("mem_used_bytes"),
        id="int64_overflow",
    ),
    pytest.param(
        sample(_HEAD, 1, procs_blocked="0"),
        None,
        _dtype("procs_blocked"),
        id="string_in_int_column",
    ),
    pytest.param(sample(_HEAD, 1, ts=-1), None, _dtype("ts"), id="negative_ts"),
    pytest.param(
        sample(_HEAD, 1, job_id="1001"),
        None,
        ReadWarning(
            "identity_mismatch", "sample identity differs from the header", 1
        ),
        id="other_job_id",
    ),
    pytest.param(
        sample(_HEAD, 1, ts=_ts(_S0)),
        None,
        ReadWarning(
            "non_monotonic_ts", "sample ts not after the previous sample", 1
        ),
        id="repeated_ts",
    ),
    pytest.param(
        b'{"kind":"hpcmu.telemetry.other"}\n',
        None,
        _bad("unknown kind"),
        id="unknown_kind",
    ),
    pytest.param(
        b'{"kind":null}\n', None, _bad("unknown kind"), id="null_kind"
    ),
    pytest.param(
        ErrorLine(ORIGIN_MS + 7_000, "cgroup", OSError("boom")),
        None,
        ReadWarning("sampler_error", "cgroup OSError: boom", 1),
        id="error_line",
    ),
    pytest.param(
        ErrorLine(-1, "cgroup", OSError("boom")),
        None,
        _bad("error line breaks its fields"),
        id="error_line_negative_ts",
    ),
    pytest.param(
        b'{"kind":"' + ERROR_KIND.encode() + b'","ts":0}\n',
        None,
        _bad("error line breaks its fields"),
        id="error_line_missing_fields",
    ),
]


@pytest.mark.parametrize(("middle", "patch", "warning"), _BODY_CASES)
def test_read_body_line_is_one_warning_between_kept_samples(
    tmp_path: Path,
    middle: Mapping[str, JsonValue] | ErrorLine | bytes,
    patch: bytes | None,
    warning: ReadWarning,
) -> None:
    if isinstance(middle, bytes):
        path = write_staging(tmp_path, _HEAD, [_S0, _S2])
        header_line, first, last = _lines(path)
        _rewrite(path, [header_line, first, middle, last])
    else:
        path = write_staging(tmp_path, _HEAD, [_S0, middle, _S2])
    if patch is not None:
        data = path.read_bytes()
        assert data.count(_PATCH_BYTES) == 1
        path.write_bytes(data.replace(_PATCH_BYTES, patch))

    read = _read(tmp_path)

    assert read.warnings == (warning,)
    assert warning.code in _CODES
    assert read.samples == (_row(_S0), _row(_S2))


@pytest.mark.parametrize(
    ("middle", "warning"),
    [
        pytest.param(
            _without(walk(_HEAD, ts=ORIGIN_MS + 35_000), "fs_avail_bytes"),
            _bad("walk key set differs"),
            id="missing_column",
        ),
        pytest.param(
            {**walk(_HEAD, ts=ORIGIN_MS + 35_000), "node_index": 0},
            _bad("walk key set differs"),
            id="extra_key",
        ),
        pytest.param(
            walk(_HEAD, ts=ORIGIN_MS + 35_000, apparent_bytes=True),
            _bad("walk column apparent_bytes breaks its dtype"),
            id="true_in_apparent_bytes",
        ),
        pytest.param(
            walk(_HEAD, ts=ORIGIN_MS + 35_000, walk_s=1e39),
            _bad("walk column walk_s breaks its dtype"),
            id="1e39_in_walk_s",
        ),
        pytest.param(
            walk(_HEAD, ts=ORIGIN_MS + 35_000, job_id="1001"),
            ReadWarning(
                "identity_mismatch", "walk identity differs from the header", 1
            ),
            id="other_job_id",
        ),
        pytest.param(
            walk(_HEAD, ts=_ts(_W0)),
            ReadWarning(
                "non_monotonic_ts", "walk ts not after the previous walk", 1
            ),
            id="repeated_ts",
        ),
    ],
)
def test_read_body_walk_line_is_one_warning_between_kept_walks(
    tmp_path: Path, middle: Mapping[str, JsonValue], warning: ReadWarning
) -> None:
    write_staging(tmp_path, _HEAD, [_W0, middle, _W2])

    read = _read(tmp_path)

    assert read.warnings == (warning,)
    assert read.walks == (_walk_row(_W0), _walk_row(_W2))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("phase", "finalize"),
        ("job_id", "1001"),
        ("restart", 1),
        ("node", "fake-node9"),
        ("node_index", 1),
    ],
)
def test_read_body_sample_identity_is_checked_field_by_field(
    tmp_path: Path, name: str, value: JsonValue
) -> None:
    write_staging(tmp_path, _HEAD, [{**_S0, name: value}, _S1])

    read = _read(tmp_path)

    assert read.warnings == (
        ReadWarning(
            "identity_mismatch", "sample identity differs from the header", 1
        ),
    )
    assert read.samples == (_row(_S1),)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("phase", "finalize"),
        ("job_id", "1001"),
        ("restart", 1),
        ("node", "fake-node9"),
    ],
)
def test_read_body_walk_identity_is_checked_field_by_field(
    tmp_path: Path, name: str, value: JsonValue
) -> None:
    other = walk(_HEAD, ts=_ts(_W0), **{name: value})
    write_staging(tmp_path, _HEAD, [other, _W2])

    read = _read(tmp_path)

    assert read.warnings == (
        ReadWarning(
            "identity_mismatch", "walk identity differs from the header", 1
        ),
    )
    assert read.walks == (_walk_row(_W2),)


def test_read_body_last_line_without_newline_is_a_truncated_tail(
    tmp_path: Path,
) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1])
    path.write_bytes(path.read_bytes()[:-1])

    read = _read(tmp_path)

    assert read.warnings == (_TRUNCATED,)
    assert read.samples == (_row(_S0),)


def test_read_body_blank_lines_give_no_warning(tmp_path: Path) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1])
    header_line, first, second = _lines(path)
    _rewrite(path, [b"\n", header_line, b"\n\n", first, b"\n", second, b"\n"])

    read = _read(tmp_path)

    assert read.warnings == ()
    assert read.samples == (_row(_S0), _row(_S1))


def test_read_body_line_at_the_limit_is_kept(tmp_path: Path) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1, _S2])
    header_line, first, middle, last = _lines(path)
    padded = _padded(middle, STAGING_LINE_LIMIT)
    _rewrite(path, [header_line, first, padded, last])

    read = _read(tmp_path)

    assert len(padded) == STAGING_LINE_LIMIT
    assert read.warnings == ()
    assert read.samples == (_row(_S0), _row(_S1), _row(_S2))


@pytest.mark.parametrize(
    "size",
    [
        STAGING_LINE_LIMIT + 1,
        STAGING_LINE_LIMIT + 2,
        2 * (STAGING_LINE_LIMIT + 1),
        2 * (STAGING_LINE_LIMIT + 1) + 1,
        3 * STAGING_LINE_LIMIT,
    ],
)
def test_read_body_over_long_line_skips_to_its_newline(
    tmp_path: Path, size: int
) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1, _S2])
    header_line, first, middle, last = _lines(path)
    _rewrite(path, [header_line, first, _padded(middle, size), last])

    read = _read(tmp_path)

    assert read.warnings == (_TOO_LONG,)
    assert read.samples == (_row(_S0), _row(_S2))


def test_read_body_over_long_last_line_without_newline_is_one_bad_line(
    tmp_path: Path,
) -> None:
    path = write_staging(tmp_path, _HEAD, [_S0, _S1])
    header_line, first, second = _lines(path)
    _rewrite(path, [header_line, first, _padded(second, 70_000)[:-1]])

    read = _read(tmp_path)

    assert read.warnings == (_TOO_LONG,)
    assert read.samples == (_row(_S0),)


def test_read_body_aggregates_warnings_by_code_and_message(
    tmp_path: Path,
) -> None:
    path = write_staging(
        tmp_path,
        _HEAD,
        [_S0, ErrorLine(ORIGIN_MS + 7_000, "cgroup", OSError("boom"))],
    )
    header_line, first, error = _lines(path)
    torn = b'{"kind":"hpcmu.tel\n'
    _rewrite(path, [torn, header_line, first, torn, b"[1]\n", error, torn])

    read = _read(tmp_path)

    assert read.warnings == (
        ReadWarning("bad_line", "not ASCII JSON", 3),
        _bad("not a JSON object"),
        ReadWarning("sampler_error", "cgroup OSError: boom", 1),
    )
    assert read.warnings_omitted == 0


def _error_lines(path: Path, count: int) -> list[bytes]:
    """Raw copies of path's one error line, message boom becoming boom <i>."""
    error = _lines(path)[1]
    assert error.count(b'"boom"') == 1
    return [error.replace(b'"boom"', b'"boom %d"' % i) for i in range(count)]


def test_read_body_caps_distinct_warnings_per_file(tmp_path: Path) -> None:
    error = ErrorLine(ORIGIN_MS, "cgroup", OSError("boom"))
    path = write_staging(tmp_path, _HEAD, [error])
    _rewrite(path, [_lines(path)[0], *_error_lines(path, 600)])

    read = _read(tmp_path)

    assert len(read.warnings) == WARNINGS_CAP == 500
    assert read.warnings_omitted == 100
    assert read.warnings[0] == ReadWarning(
        "sampler_error", "cgroup OSError: boom 0", 1
    )
    assert read.warnings[-1] == ReadWarning(
        "sampler_error", "cgroup OSError: boom 499", 1
    )


def test_read_body_capped_file_still_counts_its_kept_keys(
    tmp_path: Path,
) -> None:
    error = ErrorLine(ORIGIN_MS, "cgroup", OSError("boom"))
    path = write_staging(tmp_path, _HEAD, [error])
    lines = _error_lines(path, 600)
    _rewrite(path, [_lines(path)[0], *lines, lines[0], lines[599]])

    read = _read(tmp_path)

    assert len(read.warnings) == WARNINGS_CAP
    assert read.warnings[0].count == 2
    assert read.warnings_omitted == 101


def test_read_body_leaves_the_staging_directory_unchanged(
    tmp_path: Path,
) -> None:
    write_staging(tmp_path, _HEAD, [_S0, _W0, _S1])
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.iterdir()
    }

    _read(tmp_path)

    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.iterdir()
    } == before


# Real shapes.


def test_read_body_real_shapes_single_sample_with_rates(
    tmp_path: Path,
) -> None:
    rates: dict[str, JsonValue] = {
        c.name: 1.5 for c in SAMPLE_COLUMNS if c.unit in _RATE_UNITS
    }
    final = sample(
        _HEAD,
        0,
        ts=ORIGIN_MS + 2_345,
        dt_s=5.0,
        sample_cost_ms=0.25,
        sampler_maxrss_bytes=12_582_912,
        job_cpu_cores=0,
        **rates,
    )
    write_staging(tmp_path, _HEAD, [final])

    read = _read(tmp_path)

    assert read.warnings == ()
    assert read.samples == (_row(final),)
    cell = read.samples[0][_SAMPLE_NAMES.index("job_cpu_cores")]
    assert type(cell) is int


def test_read_body_real_shapes_single_sample_without_dt(
    tmp_path: Path,
) -> None:
    head = header(nodes=None)
    final = sample(head, 0, ts=ORIGIN_MS + 2_345, dt_s=None)
    write_staging(tmp_path, head, [final])

    read = _read(tmp_path)

    assert read.warnings == ()
    assert read.header["nodes"] is None
    assert read.samples == (_row(final),)
    assert all(
        cell is None
        for column, cell in zip(SAMPLE_COLUMNS, read.samples[0], strict=True)
        if column.unit in _RATE_UNITS or column.name == "dt_s"
    )


def test_read_body_real_shapes_restarts_read_in_restart_order(
    tmp_path: Path,
) -> None:
    first, second = header(restart=0), header(restart=1)
    written = {
        0: [sample(first, 0), sample(first, 1)],
        1: [sample(second, 0, ts=ORIGIN_MS + 60_000)],
    }
    write_staging(tmp_path, second, written[1])
    write_staging(tmp_path, first, written[0])

    reads = [read_body(read_header(c)) for c in discover(tmp_path, _JOBS)]

    assert [r.candidate.restart for r in reads] == [0, 1]
    assert [r.warnings for r in reads] == [(), ()]
    assert [r.samples for r in reads] == [
        tuple(_row(record) for record in written[0]),
        tuple(_row(record) for record in written[1]),
    ]


def test_read_body_real_shapes_node0_interleaves_samples_and_walks(
    tmp_path: Path,
) -> None:
    ticks = [ORIGIN_MS + k * 5_000 for k in (1, 2, 13)]
    samples = [sample(_HEAD, seq, ts=ts) for seq, ts in enumerate(ticks)]
    walks = [
        walk(_HEAD, ts=ticks[0], apparent_bytes=4096, walk_s=0.5),
        walk(_HEAD, ts=ticks[2], apparent_bytes=8192, walk_s=0.75),
    ]
    final = sample(_HEAD, 3, ts=ticks[2] + 1_234)
    write_staging(
        tmp_path,
        _HEAD,
        [samples[0], walks[0], samples[1], samples[2], walks[1], final],
    )

    read = _read(tmp_path)

    assert read.warnings == ()
    assert read.samples == tuple(_row(s) for s in [*samples, final])
    assert read.walks == tuple(_walk_row(w) for w in walks)


def _bundle_read(
    tmp_path: Path, bundle: str
) -> tuple[FileRead, dict[str, JsonValue]]:
    """Read the sample of bundle's snapshot 1 after a snapshot 0 baseline."""
    loaded = load_bundle(bundle)
    root = tmp_path / "root"
    materialize(loaded, 0, root)
    collector = Collector(root, _WORKSPACE)
    collector.baseline()
    materialize(loaded, 1, root)
    values, _ = collector.sample(5.0)
    head: dict[str, JsonValue] = {
        **_HEAD,
        "job_cgroup": collector.job_cgroup,
        "own_step_cgroup": collector.own_step_cgroup,
        "workspace_mount": collector.workspace_mount,
    }
    # The collected columns over the builder's identity and sampler fields.
    record = {**sample(head, 0), **values}
    assert list(record) == _SAMPLE_NAMES
    staging = tmp_path / "staging"
    staging.mkdir()
    write_staging(staging, head, [record])
    read = _read(staging)
    assert read.header == head
    return read, record


@pytest.mark.parametrize("bundle", ["synthetic", "workstation", "prd"])
def test_read_body_bundle_sample_reads_without_warnings(
    tmp_path: Path, bundle: str
) -> None:
    read, record = _bundle_read(tmp_path, bundle)

    assert read.warnings == ()
    assert read.samples == (_row(record),)


def test_read_body_bundle_prd_keeps_null_job_pids(tmp_path: Path) -> None:
    read, _ = _bundle_read(tmp_path, "prd")

    assert read.samples[0][_SAMPLE_NAMES.index("job_pids")] is None


# The builder.


def test_builder_header_has_every_header_field_in_order() -> None:
    assert list(_HEAD) == [field.name for field in HEADER_FIELDS]
    assert _HEAD["node"] == "fake-node0"


def test_builder_unknown_column_raises_key_error() -> None:
    with pytest.raises(KeyError, match="bogus"):
        sample(_HEAD, 0, bogus=1)
    with pytest.raises(KeyError, match="bogus"):
        walk(_HEAD, ts=ORIGIN_MS, bogus=1)


def test_builder_failed_write_raises_assertion_error(tmp_path: Path) -> None:
    write_staging(tmp_path, _HEAD)

    with pytest.raises(AssertionError, match="a write failed"):
        write_staging(tmp_path, _HEAD)
