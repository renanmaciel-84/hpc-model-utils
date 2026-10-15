"""Subprocess CLI wrappers and generic job-state and step-task polls shared
by the fake SLURM system tests (tests/system/test_fake_slurm_queue.py,
test_fake_slurm_accounting.py and test_fake_slurm_steps.py), kept alongside
the harness they drive.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from tests.support.fake_slurm import FakeSlurm
from tests.support.fake_slurm.shim import StepTask


def write_script(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(f"#!/bin/bash\n{body}\n")
    script.chmod(0o755)
    return script


def write_executable_stub(bin_dir: Path, name: str, body: str) -> None:
    stub = bin_dir / name
    stub.write_text(f"#!/bin/bash\n{body}\n")
    stub.chmod(0o755)


def sbatch(
    *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sbatch", *args], capture_output=True, text=True, check=False, cwd=cwd
    )


def squeue(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["squeue", *args], capture_output=True, text=True, check=False
    )


def scancel(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["scancel", *args], capture_output=True, text=True, check=False
    )


def sacct(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sacct", *args], capture_output=True, text=True, check=False
    )


def scontrol(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["scontrol", *args], capture_output=True, text=True, check=False
    )


def wait_for_state(
    fake_slurm: FakeSlurm, job_id: int, state: str, timeout: float = 5.0
) -> None:
    deadline = time.time() + timeout
    while fake_slurm.job(job_id)["state"] != state:
        if time.time() > deadline:
            raise AssertionError(f"job {job_id} never reached {state}")
        time.sleep(0.05)


def wait_for_step_tasks(
    fake_slurm: FakeSlurm, job_id: int, count: int, timeout: float = 5.0
) -> list[StepTask]:
    deadline = time.time() + timeout
    while True:
        step_tasks: list[StepTask] = fake_slurm.job(job_id)["step_tasks"]
        if len(step_tasks) >= count:
            return step_tasks
        if time.time() > deadline:
            raise AssertionError(
                f"job {job_id} never recorded {count} step tasks"
            )
        time.sleep(0.05)
