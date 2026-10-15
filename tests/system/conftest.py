"""Pytest fixture for the fake SLURM test harness (ticket-022, ADR-025,
R81). Installs the controller under a per-test state directory, prepends
its `bin/` to PATH, and at teardown kills every live job process group,
any lingering runner process and every `srun --ntasks` step task (ticket-031:
each leads its own session, so the job's group kill never reaches it), so
`pgrep -f fake_slurm` finds nothing once the suite ends.

`fake_slurm_pair` (job-telemetry ticket-038) installs two controllers, off
and on, under one test and leaves `PATH` alone: each arm reaches its own
`srun` through `Slurm(bin_dir)` and the job prelude's `PATH`.

The teardown tolerates a job file that is missing or already exited:
ticket-023's `purge()` can remove a job's state entirely, and this
module is not among the files it modifies.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm

_REAP_TIMEOUT = 1.0
_REAP_INTERVAL = 0.02


def _wait_gone(pid: int) -> None:
    deadline = time.time() + _REAP_TIMEOUT
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(_REAP_INTERVAL)


def _kill_live_jobs(controller: FakeSlurm) -> None:
    for entry in controller.submitted():
        try:
            job = controller.job(entry["id"])
        except FileNotFoundError:
            continue
        pgid = job["pgid"]
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            _wait_gone(pgid)
        runner_pid = job["runner_pid"]
        if runner_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(runner_pid, signal.SIGKILL)
            _wait_gone(runner_pid)
        # `.get`: test_run_follow.py fabricates job records without the key.
        for task in job.get("step_tasks", []):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(task["pid"], signal.SIGKILL)
            _wait_gone(task["pid"])


@pytest.fixture
def fake_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeSlurm]:
    controller = FakeSlurm.install(tmp_path)
    monkeypatch.setenv(
        "PATH", f"{controller.bin_dir}{os.pathsep}{os.environ['PATH']}"
    )
    try:
        yield controller
    finally:
        _kill_live_jobs(controller)


@pytest.fixture
def fake_slurm_pair(tmp_path: Path) -> Iterator[tuple[FakeSlurm, FakeSlurm]]:
    pair = (
        FakeSlurm.install(tmp_path / "off"),
        FakeSlurm.install(tmp_path / "on"),
    )
    try:
        yield pair
    finally:
        for controller in pair:
            _kill_live_jobs(controller)
