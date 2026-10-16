"""ADR-073/R91: the export at the R91 size stays inside its rework trigger.

R91 sizes the head-node export for 8 nodes x 48 h at 5 s, and ADR-073
reopens the design at 150 s of wall time or 1 GiB of peak RSS. The staging
files are generated here from the builder's `header`, `sample` and `walk`
records with buffered writes, because 276,480 `StagingWriter` calls would
dominate the run; the lines are encoded with the arguments of the sampler's
`_encode`.

The export runs as `python -I -m` in a child spawned with `os.posix_spawn`
and reaped once with `os.wait4`, whose `ru_maxrss` is in KiB. It is an upper
bound of the export's own peak: glibc spawns on pytest's address space, and
the exec folds pytest's peak RSS into the child's (measured on Linux 7.0
with glibc 2.39). A pass is thus sound, and a failure needs pytest's own
peak ruled out. The wall time around the spawn and the reap is the test's
one real-clock reading.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import pytest

from hpc_model_utils.infra.telemetry.export import Json
from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    SAMPLE_COLUMNS,
    SAMPLES_FILE,
    SUMMARY_FILE,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    Column,
    staging_file_name,
)
from tests.support.telemetry_staging import header, sample, walk

MODULE: Final = "hpc_model_utils.infra.telemetry.export"
NODES: Final = 8
INTERVAL_S: Final = 5
WORKSPACE_INTERVAL_S: Final = 60
# R91: 48 h at INTERVAL_S.
SAMPLES_PER_NODE: Final = 48 * 3600 // INTERVAL_S
WALK_EVERY: Final = WORKSPACE_INTERVAL_S // INTERVAL_S
R91_NDJSON_BYTES: Final = 300_000_000
WALL_TRIGGER_S: Final = 150.0
RSS_TRIGGER_BYTES: Final = 2**30
_BUFFER: Final = 1 << 20
_OUTPUT_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def _value_columns(
    columns: Sequence[Column],
) -> tuple[tuple[int, Column], ...]:
    """(index, column) of each value column: not identity, not dt_s."""
    return tuple(
        (j, column)
        for j, column in enumerate(columns)
        if column.source != "identity" and column.name != "dt_s"
    )


def _value(dtype: str, seq: int, j: int) -> int | float:
    """A value of seq and column index j that stays inside dtype's range."""
    if dtype.startswith("FLOAT"):
        return (seq % 997) * 1.000001 + j
    if dtype.startswith("INT"):
        return seq % 1000 + j
    raise ValueError(f"no generated value for dtype {dtype}")


def _values(
    columns: Sequence[tuple[int, Column]], seq: int
) -> dict[str, JsonValue]:
    return {column.name: _value(column.dtype, seq, j) for j, column in columns}


def _encode(record: Mapping[str, JsonValue]) -> bytes:
    text = json.dumps(
        record, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    )
    return (text + "\n").encode("ascii")


def _generate(staging: Path) -> tuple[int, int]:
    """Write the R91 staging files; return their sample rows and total bytes.

    Each node file holds its header, then its samples in seq order, and node
    0 also a walk right after every `WALK_EVERY`-th sample, at its ts.
    """
    sample_columns = _value_columns(SAMPLE_COLUMNS)
    walk_columns = _value_columns(WORKSPACE_COLUMNS)
    rows = 0
    for node_index in range(NODES):
        head = header(
            "model",
            "1000",
            0,
            node_index,
            nodes=NODES,
            interval_s=INTERVAL_S,
            workspace_interval_s=WORKSPACE_INTERVAL_S,
        )
        path = staging / staging_file_name("model", "1000", 0, node_index)
        with open(path, "wb", buffering=_BUFFER) as stream:
            stream.write(_encode(head))
            for seq in range(SAMPLES_PER_NODE):
                values = _values(sample_columns, seq)
                # ts=None: the builder's grid tick, bound apart from values.
                record = sample(head, seq, ts=None, **values)
                stream.write(_encode(record))
                rows += 1
                if node_index == 0 and seq % WALK_EVERY == 0:
                    ts = record["ts"]
                    assert isinstance(ts, int), ts
                    line = walk(head, ts=ts, **_values(walk_columns, seq))
                    stream.write(_encode(line))
    return rows, sum(path.stat().st_size for path in staging.iterdir())


def _spawn(
    argv: Sequence[str], out_path: Path, err_path: Path
) -> tuple[int, float, int]:
    """Run argv with fd 1 and 2 on the two paths, and reap it once.

    Return its exit code, its wall seconds and its `ru_maxrss` in KiB. A
    wait that is interrupted, such as by the pytest timeout, kills and reaps
    the child before the interruption propagates.
    """
    actions = [
        (os.POSIX_SPAWN_OPEN, 1, str(out_path), _OUTPUT_FLAGS, 0o644),
        (os.POSIX_SPAWN_OPEN, 2, str(err_path), _OUTPUT_FLAGS, 0o644),
    ]
    start = time.monotonic()
    pid = os.posix_spawn(argv[0], list(argv), os.environ, file_actions=actions)
    reaped = False
    try:
        _, status, usage = os.wait4(pid, 0)
        wall_s = time.monotonic() - start
        reaped = True
    finally:
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), wall_s, usage.ru_maxrss


def _node_counts(summary_path: Path) -> list[list[tuple[Json, Json]]]:
    """The (samples, gaps) of every node entry, one list per summary job."""
    summary: Json = json.loads(summary_path.read_text(encoding="ascii"))
    assert isinstance(summary, dict), summary
    jobs = summary["jobs"]
    assert isinstance(jobs, list), jobs
    counts: list[list[tuple[Json, Json]]] = []
    for job in jobs:
        assert isinstance(job, dict), job
        nodes = job["nodes"]
        assert isinstance(nodes, list), nodes
        entries: list[tuple[Json, Json]] = []
        for node in nodes:
            assert isinstance(node, dict), node
            entries.append((node["samples"], node["gaps"]))
        counts.append(entries)
    return counts


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_export_r91_scale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    err_path = tmp_path / "export.err"
    argv = [
        sys.executable,
        "-I",
        "-m",
        MODULE,
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
        "--job",
        "model:1000",
    ]
    staging.mkdir()
    try:
        rows, ndjson_bytes = _generate(staging)
        assert sorted(path.name for path in staging.iterdir()) == [
            staging_file_name("model", "1000", 0, node_index)
            for node_index in range(NODES)
        ]
        assert ndjson_bytes >= R91_NDJSON_BYTES, ndjson_bytes
        code, wall_s, maxrss_kib = _spawn(
            argv, tmp_path / "export.out", err_path
        )
    finally:
        shutil.rmtree(staging)
    assert not staging.exists()

    report = (
        f"export-scale rows={rows} ndjson_bytes={ndjson_bytes} "
        f"wall_s={wall_s:.1f} maxrss_mib={maxrss_kib // 1024}"
    )
    with capsys.disabled():
        # The leading newline starts the line at column 0 after any
        # progress output of the terminal reporter.
        sys.stdout.write(f"\n{report}\n")

    assert code == 0, (
        f"{report}\n{err_path.read_text(encoding='utf-8', errors='replace')}"
    )
    assert wall_s < WALL_TRIGGER_S, f"ADR-073 wall trigger: {report}"
    assert maxrss_kib * 1024 < RSS_TRIGGER_BYTES, (
        f"ADR-073 RSS trigger: {report}"
    )
    samples_rows = int(pq.ParquetFile(out / SAMPLES_FILE).metadata.num_rows)
    assert samples_rows == 276_480
    walk_rows = int(pq.ParquetFile(out / WORKSPACE_FILE).metadata.num_rows)
    assert walk_rows == 2_880
    assert _node_counts(out / SUMMARY_FILE) == [[(34_560, 0)] * 8]
