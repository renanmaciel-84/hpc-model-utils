"""R35/R36/R62/ADR-079: the sampler as a real process, with real signals."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn, Protocol

import pytest

from hpc_model_utils.infra.telemetry.sampler import JsonValue
from hpc_model_utils.infra.telemetry.schema import (
    HEADER_KIND,
    SAMPLE_COLUMNS,
    WALK_KIND,
)
from tests.support.telemetry_tree import load_bundle, materialize

pytestmark = pytest.mark.slow

type Row = dict[str, JsonValue]

_POLL_INTERVAL_S = 0.1
_POLL_TIMEOUT_S = 15
_STOP_TIMEOUT_S = 5
_FAILURE_EXIT_TIMEOUT_S = 15
_USAGE_EXIT_TIMEOUT_S = 10
_MIN_SAMPLES = 3
_SLURM_ENV = {
    "SLURM_JOB_ID": "4711",
    "SLURM_NODEID": "0",
    "SLURMD_NODENAME": "fake-node0",
    "SLURM_JOB_NUM_NODES": "1",
}


class Launch(Protocol):
    def __call__(self, *, omit: str | None = None) -> subprocess.Popen[str]: ...


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    root = materialize(load_bundle("synthetic"), 0, tmp_path / "root")
    workspace = tmp_path / "ws"
    staging = tmp_path / "staging"
    workspace.mkdir()
    staging.mkdir()
    argv = [
        sys.executable,
        "-I",
        "-m",
        "hpc_model_utils.infra.telemetry.sampler",
        "--staging",
        str(staging),
        "--phase",
        "model",
        "--workspace",
        str(workspace),
        "--interval",
        "1",
        "--workspace-interval",
        "60",
        "--tool-version",
        "0.0.0-test",
        "--root",
        str(root),
    ]

    def start(*, omit: str | None = None) -> subprocess.Popen[str]:
        env = {**os.environ, **_SLURM_ENV}
        env.pop("SLURM_RESTART_COUNT", None)
        if omit is not None:
            env.pop(omit)
        return subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    return start


def _reap(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.kill()
    return proc.communicate()[1]


def _fail(proc: subprocess.Popen[str], message: str) -> NoReturn:
    pytest.fail(f"{message}; stderr: {_reap(proc)!r}")


def _wait_exit(
    proc: subprocess.Popen[str], timeout_s: float
) -> tuple[int, str]:
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _fail(proc, f"sampler still running after {timeout_s} s")
    return code, _reap(proc)


def _read_rows(path: Path) -> list[Row]:
    try:
        text = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return []
    rows: list[Row] = [json.loads(line) for line in text.split("\n")[:-1]]
    return rows


def _samples(rows: list[Row]) -> list[Row]:
    return [row for row in rows if "kind" not in row]


def _timestamps(rows: list[Row]) -> list[int]:
    stamps: list[int] = []
    for row in rows:
        stamp = row["ts"]
        assert isinstance(stamp, int)
        stamps.append(stamp)
    return stamps


def _await_samples(proc: subprocess.Popen[str], path: Path, count: int) -> int:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while True:
        seen = len(_samples(_read_rows(path)))
        if seen >= count:
            return seen
        if proc.poll() is not None:
            _fail(proc, f"sampler exited with {proc.returncode} early")
        if time.monotonic() >= deadline:
            _fail(proc, f"{seen} of {count} samples in {_POLL_TIMEOUT_S} s")
        time.sleep(_POLL_INTERVAL_S)


def test_sigterm_ends_with_final_sample(launch: Launch, tmp_path: Path) -> None:
    staging_file = tmp_path / "staging" / "model-4711-r0-0.ndjson"
    proc = launch()
    try:
        polled = _await_samples(proc, staging_file, _MIN_SAMPLES)
        proc.send_signal(signal.SIGTERM)
        code, stderr = _wait_exit(proc, _STOP_TIMEOUT_S)
    finally:
        _reap(proc)

    rows = _read_rows(staging_file)
    samples = _samples(rows)
    names = {column.name for column in SAMPLE_COLUMNS}
    stamps = _timestamps(samples)
    headers = [i for i, r in enumerate(rows) if r.get("kind") == HEADER_KIND]
    walk_nodes = [r["node"] for r in rows if r.get("kind") == WALK_KIND]
    assert code == 0, stderr
    assert headers == [0]
    assert len(samples) > polled
    assert [sorted(set(r) ^ names) for r in samples if set(r) != names] == []
    assert stamps == sorted(set(stamps))
    assert walk_nodes == ["fake-node0"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the directory mode")
def test_unwritable_staging_exits_one(launch: Launch, tmp_path: Path) -> None:
    (tmp_path / "staging").chmod(0o555)
    proc = launch()
    try:
        code, stderr = _wait_exit(proc, _FAILURE_EXIT_TIMEOUT_S)
    finally:
        _reap(proc)

    assert code == 1, stderr
    assert "5 consecutive staging failures" in stderr


def test_missing_job_id_exits_two(launch: Launch) -> None:
    proc = launch(omit="SLURM_JOB_ID")
    try:
        code, stderr = _wait_exit(proc, _USAGE_EXIT_TIMEOUT_S)
    finally:
        _reap(proc)

    assert code == 2, stderr
    assert "SLURM_JOB_ID" in stderr
