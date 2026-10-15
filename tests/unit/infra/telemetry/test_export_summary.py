"""ADR-071/ADR-076/ADR-077: the summary.json roll-up over writer-built files.

Every expected number is worked out by hand in the test, never with the
module's own helpers.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hpc_model_utils.infra.telemetry.export import (
    Json,
    Rollup,
    discover,
    quantile7,
    read_body,
    read_header,
    render_summary,
)
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    HEADLINE,
    SCHEMA_VERSION,
    SUMMARY_KIND,
    WARNINGS_CAP,
)
from tests.support.telemetry_staging import (
    ORIGIN_MS,
    ErrorLine,
    header,
    sample,
    walk,
    write_staging,
)

_JOBS = (("model", "1000"), ("finalize", "1001"))
_RECORD_KEYS = [
    "code",
    "phase",
    "job_id",
    "restart",
    "node_index",
    "count",
    "message",
]
_HEADLINE_KEYS = tuple(spec.key for spec in HEADLINE)
_TORN = b'{"kind":"hpcmu.tel\n'
# float32(0.1), exactly 0.100000001490116119384765625.
_F32_TENTH = 0.10000000149011612


def _summarize(
    directory: Path, jobs: Sequence[tuple[str, str]] = _JOBS
) -> dict[str, Json]:
    """Roll up directory as the export does: every header, then every file."""
    heads = [read_header(c) for c in discover(directory, jobs)]
    rollup = Rollup(
        run_id="run-1", model="fake", tool_version="9.9.9", jobs=jobs
    )
    for head in heads:
        rollup.add_header(head)
    for head in heads:
        if head.header is not None:
            rollup.add_file(read_body(head))
    return rollup.summary()


def _obj(value: Json) -> dict[str, Json]:
    assert isinstance(value, dict), value
    return value


def _seq(value: Json) -> list[Json]:
    assert isinstance(value, list), value
    return value


def _jobs(summary: Mapping[str, Json]) -> list[dict[str, Json]]:
    return [_obj(job) for job in _seq(summary["jobs"])]


def _job_nodes(
    summary: Mapping[str, Json], index: int
) -> list[dict[str, Json]]:
    return [_obj(node) for node in _seq(_jobs(summary)[index]["nodes"])]


def _node(summary: Mapping[str, Json]) -> dict[str, Json]:
    """The only node entry of summary."""
    (node,) = [
        _obj(node) for job in _jobs(summary) for node in _seq(job["nodes"])
    ]
    return node


def _stats(node: Mapping[str, Json]) -> dict[str, Json]:
    return _obj(node["stats"])


def _values(summary: Mapping[str, Json]) -> dict[str, Json]:
    return _obj(_obj(summary["headline"])["values"])


def _warnings(summary: Mapping[str, Json]) -> list[dict[str, Json]]:
    records = [_obj(record) for record in _seq(summary["warnings"])]
    for record in records:
        assert list(record) == _RECORD_KEYS
    return records


def _record(
    code: str,
    count: int,
    message: str,
    *,
    phase: str = "model",
    job_id: str = "1000",
    restart: int = 0,
    node_index: int = 0,
) -> dict[str, Json]:
    return {
        "code": code,
        "phase": phase,
        "job_id": job_id,
        "restart": restart,
        "node_index": node_index,
        "count": count,
        "message": message,
    }


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def _error_file(directory: Path, count: int, nodes: int | None = 1) -> None:
    """A model-1000-r0-0 file of count error lines with distinct messages.

    The writer keeps one error line per (source, class), so the copies are
    raw byte edits of its one line, boom becoming boom <i>.
    """
    head = header(nodes=nodes)
    path = write_staging(
        directory, head, [ErrorLine(ORIGIN_MS, "cgroup", OSError("boom"))]
    )
    header_line, error = _lines(path)
    copies = [error.replace(b'"boom"', b'"boom %d"' % i) for i in range(count)]
    path.write_bytes(header_line + b"".join(copies))


# Statistics arithmetic.


def test_summary_arithmetic_of_one_file(tmp_path: Path) -> None:
    head = header(interval_s=5)
    rows = zip(
        (5.0, 5.0, 10.0, 5.0),
        (1.0, 2.0, 3.0, 4.0),
        (100.0, 200.0, None, 300.0),
        (9, 7, 8, 6),
        strict=True,
    )
    write_staging(
        tmp_path,
        head,
        [
            sample(
                head,
                seq,
                dt_s=dt,
                cpu_busy_cores=cpu,
                nfs_read_bytes_per_s=nfs,
                mem_available_bytes=mem,
            )
            for seq, (dt, cpu, nfs, mem) in enumerate(rows)
        ],
    )

    node = _node(_summarize(tmp_path))
    stats = _stats(node)

    # cpu mean: (1 * 5 + 2 * 5 + 3 * 10 + 4 * 5) / 25 = 65 / 25.
    # p95 at (4 - 1) * 0.95 = 2.85: 3 + 0.85 * (4 - 3).
    # nfs over its non-null rows only: total 100 * 5 + 200 * 5 + 300 * 5,
    # mean 3000 / 15 (the whole-file 25 s would give 120), p95 at
    # (3 - 1) * 0.95 = 1.9: 200 + 0.9 * 100.
    # mem mean: (9 * 5 + 7 * 5 + 8 * 10 + 6 * 5) / 25 = 190 / 25, p95 over
    # [6, 7, 8, 9]: 8 + 0.85 * 1.
    assert list(stats) == [
        "cpu_busy_cores_min",
        "cpu_busy_cores_mean",
        "cpu_busy_cores_p95",
        "cpu_busy_cores_max",
        "mem_available_bytes_min",
        "mem_available_bytes_mean",
        "mem_available_bytes_p95",
        "mem_available_bytes_max",
        "nfs_read_bytes_per_s_mean",
        "nfs_read_bytes_per_s_p95",
        "nfs_read_bytes_per_s_max",
        "nfs_read_bytes_total",
    ]
    assert stats["cpu_busy_cores_mean"] == pytest.approx(2.6)
    assert stats["cpu_busy_cores_p95"] == pytest.approx(3.85)
    assert stats["cpu_busy_cores_min"] == 1.0
    assert stats["cpu_busy_cores_max"] == 4.0
    assert stats["nfs_read_bytes_total"] == pytest.approx(3000.0)
    assert stats["nfs_read_bytes_per_s_mean"] == pytest.approx(200.0)
    assert stats["nfs_read_bytes_per_s_p95"] == pytest.approx(290.0)
    assert stats["nfs_read_bytes_per_s_max"] == 300.0
    assert stats["mem_available_bytes_min"] == 6
    assert stats["mem_available_bytes_max"] == 9
    assert stats["mem_available_bytes_mean"] == pytest.approx(7.6)
    assert stats["mem_available_bytes_p95"] == pytest.approx(8.85)
    assert type(stats["mem_available_bytes_min"]) is int
    assert type(stats["mem_available_bytes_max"]) is int
    assert type(stats["mem_available_bytes_p95"]) is float
    assert type(stats["cpu_busy_cores_min"]) is float
    assert node["gaps"] == 1
    assert node["max_gap_s"] == 10.0
    assert node["samples"] == 4
    assert node["first_ts"] == ORIGIN_MS + 5_000
    assert node["last_ts"] == ORIGIN_MS + 20_000


def test_summary_arithmetic_takes_float32_cells_as_published(
    tmp_path: Path,
) -> None:
    head = header()
    write_staging(
        tmp_path,
        head,
        [
            sample(
                head,
                0,
                dt_s=0.1,
                cpu_busy_cores=0.1,
                disk_read_bytes_per_s=0.1,
                nfs_read_bytes_per_s=10.0,
            )
        ],
    )

    node = _node(_summarize(tmp_path))
    stats = _stats(node)

    # FLOAT32 cells, dt_s included, are float32(0.1); the FLOAT64 rate keeps
    # 0.1. The total is 10 * float32(0.1), exact in float64.
    assert stats["cpu_busy_cores_max"] == _F32_TENTH
    assert stats["cpu_busy_cores_mean"] == _F32_TENTH
    assert node["max_gap_s"] == _F32_TENTH
    assert stats["disk_read_bytes_per_s_max"] == 0.1
    assert stats["nfs_read_bytes_total"] == 1.0000000149011612


def test_summary_arithmetic_gap_is_strictly_over_one_and_a_half_intervals(
    tmp_path: Path,
) -> None:
    head = header(interval_s=5)
    # 7.5000001 rounds to the float32 7.5, so it is no gap as published;
    # 7.501 is one.
    dts = (7.5, 7.5000001, 7.501)
    write_staging(
        tmp_path,
        head,
        [sample(head, seq, dt_s=dt) for seq, dt in enumerate(dts)],
    )

    node = _node(_summarize(tmp_path))

    assert node["gaps"] == 1
    assert node["max_gap_s"] == pytest.approx(7.501, rel=1e-6)


def test_summary_arithmetic_gap_rule_takes_any_interval(tmp_path: Path) -> None:
    # interval_s at the int64 edge the reader accepts. The boundary
    # 1.5 * (2**63 - 1) is 3 * 2**62 - 1.5, and float(2**63 - 1) rounds up to
    # 2**63, so a float rule dt_s > 1.5 * interval_s misses the gap at
    # dt_s = 3 * 2**62. The exact rule counts it, and not the float32 just
    # below. The rows come from a header of the same identity, because the
    # builder's default ts and dt_s derive from interval_s.
    interval = 2**63 - 1
    below, above = float(3 * 2**62 - 2**40), float(3 * 2**62)
    head = header(interval_s=interval)
    write_staging(
        tmp_path,
        head,
        [
            sample(header(), seq, dt_s=dt)
            for seq, dt in enumerate((below, above))
        ],
    )

    summary = _summarize(tmp_path)

    node = _node(summary)
    assert node["gaps"] == 1
    assert node["max_gap_s"] == above
    assert _obj(summary["run"])["interval_s"] == interval


def test_quantile7_arithmetic_known_values() -> None:
    assert quantile7([3.0], 0.95) == 3.0
    assert quantile7([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert quantile7([1.0, 2.0, 3.0, 4.0], 0.99) == pytest.approx(3.97)
    assert quantile7([1.0, 2.0], 1.0) == 2.0
    assert quantile7([1.0, 2.0], 0.0) == 1.0


@settings(deadline=None, max_examples=300)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1e12), min_size=2, max_size=200)
)
def test_quantile7_arithmetic_matches_statistics_inclusive(
    xs: list[float],
) -> None:
    cuts = statistics.quantiles(xs, n=100, method="inclusive")

    assert quantile7(sorted(xs), 0.95) == pytest.approx(cuts[94], rel=1e-12)
    assert quantile7(sorted(xs), 0.99) == pytest.approx(cuts[98], rel=1e-12)


# Absence rules.


def test_summary_absent_statistics_of_all_null_columns(tmp_path: Path) -> None:
    head = header()
    # The prd shape: job_pids and job_mem_peak_bytes are null on every row.
    write_staging(
        tmp_path,
        head,
        [
            sample(
                head,
                seq,
                job_pids=None,
                job_mem_peak_bytes=None,
                cpu_busy_cores=1.0,
                efa_rx_bytes_per_s=0.0,
            )
            for seq in range(3)
        ],
    )

    summary = _summarize(tmp_path)

    stats = _stats(_node(summary))
    assert [name for name in stats if name.startswith("job_pids_")] == []
    assert "job_mem_peak_bytes_max" not in stats
    assert "model.job_mem_peak_bytes_max" not in _values(summary)
    # A column of real zeros keeps its statistics: absent is not 0.
    assert stats["efa_rx_bytes_total"] == 0.0
    assert stats["efa_rx_bytes_per_s_max"] == 0.0


def test_summary_absent_none_in_one_sample_file_with_dt(tmp_path: Path) -> None:
    head = header()
    write_staging(
        tmp_path,
        head,
        [
            sample(
                head, 0, dt_s=5.0, cpu_busy_cores=1.5, nfs_read_bytes_per_s=8.0
            )
        ],
    )

    node = _node(_summarize(tmp_path))

    stats = _stats(node)
    assert stats["cpu_busy_cores_mean"] == 1.5
    assert stats["nfs_read_bytes_total"] == 40.0
    assert node["max_gap_s"] == 5.0
    assert node["gaps"] == 0


def test_summary_absent_mean_and_total_in_one_sample_file_without_dt(
    tmp_path: Path,
) -> None:
    head = header()
    write_staging(
        tmp_path,
        head,
        [
            sample(
                head, 0, dt_s=None, cpu_busy_cores=1.5, nfs_read_bytes_per_s=8.0
            )
        ],
    )

    node = _node(_summarize(tmp_path))

    stats = _stats(node)
    assert stats["cpu_busy_cores_min"] == 1.5
    assert stats["cpu_busy_cores_max"] == 1.5
    assert stats["cpu_busy_cores_p95"] == 1.5
    assert "cpu_busy_cores_mean" not in stats
    assert "nfs_read_bytes_total" not in stats
    assert "nfs_read_bytes_per_s_mean" not in stats
    assert stats["nfs_read_bytes_per_s_max"] == 8.0
    assert "max_gap_s" not in node
    assert node["gaps"] == 0


def test_summary_absent_rate_statistics_after_a_failed_baseline_read(
    tmp_path: Path,
) -> None:
    head = header()
    # The sampler's shape: a real dt_s, the failed source's rates null.
    write_staging(
        tmp_path,
        head,
        [sample(head, 0, dt_s=5.0, cpu_busy_cores=1.5)],
    )

    node = _node(_summarize(tmp_path))

    stats = _stats(node)
    assert [name for name in stats if name.startswith("nfs_read_bytes")] == []
    assert stats["cpu_busy_cores_mean"] == 1.5
    assert node["max_gap_s"] == 5.0


def test_summary_absent_samples_in_header_only_file(tmp_path: Path) -> None:
    write_staging(tmp_path, header())

    node = _node(_summarize(tmp_path))

    assert node["samples"] == 0
    assert "first_ts" not in node
    assert "last_ts" not in node
    assert "max_gap_s" not in node
    assert node["gaps"] == 0
    assert node["stats"] == {}


def test_summary_absent_mean_of_latency_columns(tmp_path: Path) -> None:
    head = header()
    write_staging(
        tmp_path,
        head,
        [
            sample(head, seq, nfs_read_rtt_ms=rtt)
            for seq, rtt in enumerate((2.0, None, 4.0))
        ],
    )

    stats = _stats(_node(_summarize(tmp_path)))

    # p95 over [2, 4] at (2 - 1) * 0.95: 2 + 0.95 * 2.
    assert stats == {
        "nfs_read_rtt_ms_min": 2.0,
        "nfs_read_rtt_ms_p95": pytest.approx(3.9),
        "nfs_read_rtt_ms_max": 4.0,
    }


def test_summary_absent_run_intervals_without_a_valid_header(
    tmp_path: Path,
) -> None:
    (tmp_path / "model-1000-r0-0.ndjson").write_bytes(b'{"kind":"other"}\n')

    summary = _summarize(tmp_path)

    run = _obj(summary["run"])
    assert run["interval_s"] is None
    assert run["workspace_interval_s"] is None
    assert [job["nodes"] for job in _jobs(summary)] == [[], []]
    assert summary["workspace"] == {"walks": 0, "stats": {}}
    assert _values(summary) == {}
    assert _warnings(summary) == [
        _record("bad_header", 1, "header key set differs")
    ]


# Jobs and headline scope.


def _restart_files(directory: Path) -> None:
    """model-1000 r0 and r1 with walks on both, and finalize-1001 r0."""
    r0 = header()
    write_staging(
        directory,
        r0,
        [
            sample(
                r0,
                0,
                cpu_busy_cores=1.0,
                job_mem_peak_bytes=300,
                nfs_read_bytes_per_s=10.0,
                mem_available_bytes=70,
            ),
            walk(r0, ts=ORIGIN_MS + 5_000, allocated_bytes=8192),
            sample(
                r0,
                1,
                cpu_busy_cores=2.0,
                job_mem_peak_bytes=900,
                nfs_read_bytes_per_s=20.0,
                mem_available_bytes=60,
            ),
            sample(
                r0,
                2,
                cpu_busy_cores=3.0,
                job_mem_peak_bytes=600,
                nfs_read_bytes_per_s=30.0,
                mem_available_bytes=80,
            ),
        ],
    )
    r1 = header(restart=1)
    write_staging(
        directory,
        r1,
        [
            sample(
                r1,
                0,
                cpu_busy_cores=4.0,
                job_mem_peak_bytes=200,
                nfs_read_bytes_per_s=40.0,
                mem_available_bytes=50,
            ),
            walk(r1, ts=ORIGIN_MS + 5_000, allocated_bytes=4096),
            sample(
                r1,
                1,
                dt_s=10.0,
                cpu_busy_cores=6.0,
                job_mem_peak_bytes=500,
                mem_available_bytes=90,
            ),
        ],
    )
    final = header("finalize", "1001")
    write_staging(
        directory,
        final,
        [
            sample(final, 0, cpu_busy_cores=0.5, job_mem_peak_bytes=50),
            walk(final, ts=ORIGIN_MS + 5_000, allocated_bytes=1024),
        ],
    )


_FINALIZE_VALUES: dict[str, int | float] = {
    "finalize.job_mem_peak_bytes_max": 50,
    "finalize.cpu_busy_cores_mean": 0.5,
    "finalize.cpu_busy_cores_p95": 0.5,
}


def test_summary_headline_pools_every_restart_of_the_reported_job(
    tmp_path: Path,
) -> None:
    _restart_files(tmp_path)

    summary = _summarize(tmp_path, [("model", "1000"), ("finalize", "1001")])

    nodes = _job_nodes(summary, 0)
    assert [_obj(node["header"])["restart"] for node in nodes] == [0, 1]
    headline = _obj(summary["headline"])
    assert headline["job_ids"] == {"model": "1000", "finalize": "1001"}
    values = _values(summary)
    # cpu mean over the five rows: ((1 + 2 + 3 + 4) * 5 + 6 * 10) / 30.
    # cpu p95 over [1, 2, 3, 4, 6] at (5 - 1) * 0.95 = 3.8: 4 + 0.8 * 2.
    # nfs total: (10 + 20 + 30 + 40) * 5; r1's null row adds nothing.
    # The r1 peak of 500 never lowers the r0 peak of 900.
    assert values == pytest.approx(
        {
            "model.job_mem_peak_bytes_max": 900,
            "model.mem_available_bytes_min": 50,
            "model.cpu_busy_cores_mean": 110 / 30,
            "model.cpu_busy_cores_p95": 5.6,
            "model.nfs_read_bytes_total": 500.0,
            **_FINALIZE_VALUES,
            "workspace.allocated_bytes_max": 8192,
        }
    )
    assert type(values["model.job_mem_peak_bytes_max"]) is int
    assert list(values) == [key for key in _HEADLINE_KEYS if key in values]
    # The maximum over the walks of both attempts.
    assert values["workspace.allocated_bytes_max"] == 8192
    assert summary["workspace"] == {
        "walks": 3,
        "stats": {"allocated_bytes_max": 8192},
    }


def test_summary_headline_never_falls_back_to_an_earlier_job(
    tmp_path: Path,
) -> None:
    _restart_files(tmp_path)

    summary = _summarize(
        tmp_path, [("model", "1000"), ("model", "1234"), ("finalize", "1001")]
    )

    headline = _obj(summary["headline"])
    assert headline["job_ids"] == {"model": "1234", "finalize": "1001"}
    values = _values(summary)
    assert [key for key in values if key.startswith("model.")] == []
    # The workspace scope is the walks of the reported jobs: finalize only.
    assert values == pytest.approx(
        {**_FINALIZE_VALUES, "workspace.allocated_bytes_max": 1024}
    )
    assert list(values) == [key for key in _HEADLINE_KEYS if key in values]
    jobs = _jobs(summary)
    assert [(job["phase"], job["job_id"]) for job in jobs] == [
        ("model", "1000"),
        ("model", "1234"),
        ("finalize", "1001"),
    ]
    assert jobs[1]["nodes"] == []
    assert len(_job_nodes(summary, 0)) == 2
    # workspace.stats still spans every kept walk.
    assert summary["workspace"] == {
        "walks": 3,
        "stats": {"allocated_bytes_max": 8192},
    }


def test_summary_headline_reports_only_the_last_job_of_a_phase(
    tmp_path: Path,
) -> None:
    _restart_files(tmp_path)
    later = header(job_id="1234")
    write_staging(tmp_path, later, [sample(later, 0, job_mem_peak_bytes=100)])

    summary = _summarize(
        tmp_path, [("model", "1000"), ("model", "1234"), ("finalize", "1001")]
    )

    values = _values(summary)
    assert values["model.job_mem_peak_bytes_max"] == 100
    assert "model.cpu_busy_cores_mean" not in values
    assert "model.mem_available_bytes_min" not in values


def test_summary_headline_job_ids_follow_phases_and_jobs_follow_argv(
    tmp_path: Path,
) -> None:
    _restart_files(tmp_path)

    summary = _summarize(tmp_path, [("finalize", "1001"), ("model", "1000")])

    job_ids = _obj(_obj(summary["headline"])["job_ids"])
    assert list(job_ids) == ["model", "finalize"]
    assert [job["phase"] for job in _jobs(summary)] == ["finalize", "model"]
    only_finalize = _summarize(tmp_path, [("finalize", "1001")])
    assert _obj(only_finalize["headline"])["job_ids"] == {"finalize": "1001"}
    assert [k for k in _values(only_finalize) if k.startswith("model.")] == []


# Warnings.


def test_summary_warnings_missing_node(tmp_path: Path) -> None:
    head = header(nodes=2)
    write_staging(tmp_path, head, [sample(head, 0)])

    warnings = _warnings(_summarize(tmp_path))

    assert warnings == [
        _record("missing_node", 1, "no valid staging file", node_index=1)
    ]


def test_summary_warnings_merge_two_identical_bad_lines(tmp_path: Path) -> None:
    head = header()
    path = write_staging(tmp_path, head, [sample(head, 0), sample(head, 1)])
    header_line, first, second = _lines(path)
    path.write_bytes(header_line + first + b"oops\n" + second + b"oops\n")

    warnings = _warnings(_summarize(tmp_path))

    assert warnings == [_record("bad_line", 2, "not ASCII JSON")]


def test_summary_warnings_count_header_pass_once_and_first(
    tmp_path: Path,
) -> None:
    head = header()
    path = write_staging(tmp_path, head, [sample(head, 0)])
    header_line, first = _lines(path)
    path.write_bytes(_TORN + header_line + first + b"[1]\n" + _TORN)

    warnings = _warnings(_summarize(tmp_path))

    # The torn header prefix and the torn body line are one bad_line record.
    assert warnings == [
        _record("bad_line", 2, "not ASCII JSON"),
        _record("bad_record", 1, "not a JSON object"),
    ]


def _warning_files(directory: Path) -> None:
    """finalize-1001-r0-0 with a bad line; model-1000-r0 nodes 0 and 2 of 3
    with a bad record each, node 1 with a bad header; model-1000-r1-0."""
    final = header("finalize", "1001", nodes=None)
    path = write_staging(directory, final, [sample(final, 0)])
    path.write_bytes(path.read_bytes() + b"oops\n")
    for node_index in (0, 2):
        head = header(node_index=node_index, nodes=3)
        path = write_staging(directory, head, [sample(head, 0)])
        path.write_bytes(path.read_bytes() + b"[1]\n")
    (directory / "model-1000-r0-1.ndjson").write_bytes(b'{"kind":"other"}\n')
    restarted = header(restart=1)
    write_staging(directory, restarted, [sample(restarted, 0)])


def test_summary_warnings_follow_discover_order_then_missing_nodes(
    tmp_path: Path,
) -> None:
    _warning_files(tmp_path)

    warnings = _warnings(_summarize(tmp_path))

    not_object = "not a JSON object"
    assert warnings == [
        _record(
            "bad_line", 1, "not ASCII JSON", phase="finalize", job_id="1001"
        ),
        _record("bad_record", 1, not_object, node_index=0),
        _record("bad_header", 1, "header key set differs", node_index=1),
        _record("bad_record", 1, not_object, node_index=2),
        _record("missing_node", 1, "no valid staging file", node_index=1),
    ]


def test_summary_warnings_and_nodes_keep_identity_order_in_any_fold_order(
    tmp_path: Path,
) -> None:
    _warning_files(tmp_path)
    heads = [read_header(c) for c in discover(tmp_path, _JOBS)]
    rollup = Rollup(
        run_id="run-1", model="fake", tool_version="9.9.9", jobs=_JOBS
    )

    for head in reversed(heads):
        if head.header is not None:
            rollup.add_file(read_body(head))
    for head in reversed(heads):
        rollup.add_header(head)

    folded = rollup.summary()
    assert render_summary(folded) == render_summary(_summarize(tmp_path))
    restarts = [
        (_obj(node["header"])["restart"], _obj(node["header"])["node_index"])
        for node in _job_nodes(folded, 0)
    ]
    assert restarts == [(0, 0), (0, 2), (1, 0)]


@pytest.mark.parametrize(
    ("first", "second", "missing"),
    [(1, 3, []), (3, 1, [2]), (None, 3, []), (2, None, [])],
)
def test_summary_warnings_missing_nodes_take_the_first_header_nodes(
    tmp_path: Path, first: int | None, second: int | None, missing: list[int]
) -> None:
    for node_index, nodes in ((0, first), (1, second)):
        head = header(node_index=node_index, nodes=nodes)
        write_staging(tmp_path, head, [sample(head, 0)])

    warnings = _warnings(_summarize(tmp_path))

    assert [w["node_index"] for w in warnings] == missing
    assert all(w["code"] == "missing_node" for w in warnings)


def test_summary_warnings_missing_nodes_ignore_indexes_past_nodes(
    tmp_path: Path,
) -> None:
    head = header(node_index=5, nodes=2)
    write_staging(tmp_path, head, [sample(head, 0)])

    warnings = _warnings(_summarize(tmp_path))

    assert [w["node_index"] for w in warnings] == [0, 1]


def test_summary_warnings_cap_at_500(tmp_path: Path) -> None:
    _error_file(tmp_path, 600)

    warnings = _warnings(_summarize(tmp_path))

    # The reader keeps 500 distinct records and omits 100 occurrences; the
    # list keeps 499 and counts the 500th with the 100.
    assert len(warnings) == WARNINGS_CAP == 500
    assert warnings[0] == _record("sampler_error", 1, "cgroup OSError: boom 0")
    assert warnings[498]["message"] == "cgroup OSError: boom 498"
    assert warnings[-1] == {
        "code": "warnings_capped",
        "phase": None,
        "job_id": None,
        "restart": None,
        "node_index": None,
        "count": 101,
        "message": "warning list capped at 500",
    }


@pytest.mark.parametrize(
    ("errors", "nodes", "last", "count"),
    [
        (500, 1, "sampler_error", 1),
        (499, 2, "missing_node", 1),
        (500, 2, "warnings_capped", 2),
    ],
)
def test_summary_warnings_cap_boundary(
    tmp_path: Path, errors: int, nodes: int, last: str, count: int
) -> None:
    # nodes=2 adds one missing_node record after the errors. 500 records and
    # nothing omitted keep the list whole; one more caps it, and the capped
    # record counts the 500th record and the one after it.
    _error_file(tmp_path, errors, nodes=nodes)

    warnings = _warnings(_summarize(tmp_path))

    assert len(warnings) == WARNINGS_CAP
    assert warnings[-1]["code"] == last
    assert warnings[-1]["count"] == count


def test_summary_warnings_bound_a_crafted_node_count(tmp_path: Path) -> None:
    # A naive range(nodes) loop over a header claiming 2**63 - 1 nodes, the
    # int64 edge the reader accepts, never ends; the list is built from the
    # present nodes and the cap instead.
    head = header(nodes=2**63 - 1)
    write_staging(tmp_path, head, [sample(head, 0)])

    summary = _summarize(tmp_path)

    warnings = _warnings(summary)
    assert len(warnings) == WARNINGS_CAP
    assert [w["node_index"] for w in warnings[:-1]] == list(range(1, 500))
    assert {w["code"] for w in warnings[:-1]} == {"missing_node"}
    # 2**63 - 2 missing nodes, 499 of them listed.
    assert warnings[-1]["code"] == "warnings_capped"
    assert warnings[-1]["count"] == 2**63 - 2 - 499
    assert json.loads(render_summary(summary))["warnings"][-1]["count"] == (
        2**63 - 1 - 500
    )


# Rendering.


def test_summary_shape_follows_adr071_order(tmp_path: Path) -> None:
    final = header("finalize", "1001", interval_s=10, workspace_interval_s=120)
    write_staging(tmp_path, final, [sample(final, 0)])
    model: dict[str, JsonValue] = {
        **header(),
        "workspace_mount": {"mountpoint": "/shared-zfs", "fstype": "nfs4"},
    }
    write_staging(
        tmp_path,
        model,
        [sample(model, 0), walk(model, ts=ORIGIN_MS + 5_000, files=3)],
    )

    summary = _summarize(tmp_path)

    assert list(summary) == [
        "kind",
        "schema_version",
        "run",
        "jobs",
        "workspace",
        "headline",
        "warnings",
    ]
    assert summary["kind"] == SUMMARY_KIND
    assert summary["schema_version"] == SCHEMA_VERSION
    # The intervals come from the first valid header in discover order,
    # finalize-1001's.
    assert summary["run"] == {
        "run_id": "run-1",
        "model": "fake",
        "tool_version": "9.9.9",
        "interval_s": 10,
        "workspace_interval_s": 120,
    }
    assert [list(job) for job in _jobs(summary)] == [
        ["phase", "job_id", "nodes"]
    ] * 2
    node = _job_nodes(summary, 0)[0]
    assert list(node) == [
        "header",
        "first_ts",
        "last_ts",
        "samples",
        "gaps",
        "max_gap_s",
        "stats",
    ]
    assert node["header"] == model
    assert summary["workspace"] == {"walks": 1, "stats": {"files_max": 3}}
    assert list(_obj(summary["headline"])) == ["job_ids", "values"]


def test_render_summary_top_level_keys_and_bytes(tmp_path: Path) -> None:
    head = header()
    write_staging(tmp_path, head, [sample(head, 0, cpu_busy_cores=1.0)])
    summary = _summarize(tmp_path)

    rendered = render_summary(summary)

    assert list(json.loads(rendered)) == [
        "kind",
        "schema_version",
        "run",
        "jobs",
        "workspace",
        "headline",
        "warnings",
    ]
    assert json.loads(rendered) == summary
    assert rendered.endswith(b"}\n")
    assert rendered.startswith(b'{\n  "kind": ')


def test_render_summary_escapes_non_ascii() -> None:
    rendered = render_summary({"message": "n\u00f3 \ud800"})

    assert rendered == b'{\n  "message": "n\\u00f3 \\ud800"\n}\n'


def test_render_summary_refuses_an_overflowing_total(tmp_path: Path) -> None:
    head = header()
    # Two finite products of 1e308 whose fsum overflows.
    write_staging(
        tmp_path,
        head,
        [
            sample(head, seq, dt_s=1.0, disk_write_bytes_per_s=1e308)
            for seq in range(2)
        ],
    )

    summary = _summarize(tmp_path)

    stats = _stats(_node(summary))
    assert stats["disk_write_bytes_total"] == math.inf
    assert stats["disk_write_bytes_per_s_max"] == 1e308
    assert _values(summary)["model.disk_write_bytes_total"] == math.inf
    with pytest.raises(ValueError, match="not JSON compliant"):
        render_summary(summary)


def test_render_summary_refuses_opposite_infinite_products(
    tmp_path: Path,
) -> None:
    head = header()
    # 1.7e308 * 5 overflows to inf, and the negated row to -inf.
    write_staging(
        tmp_path,
        head,
        [
            sample(head, seq, dt_s=5.0, disk_write_bytes_per_s=rate)
            for seq, rate in enumerate((1.7e308, -1.7e308))
        ],
    )

    summary = _summarize(tmp_path)

    total = _stats(_node(summary))["disk_write_bytes_total"]
    assert isinstance(total, float)
    assert math.isnan(total)
    with pytest.raises(ValueError, match="not JSON compliant"):
        render_summary(summary)
