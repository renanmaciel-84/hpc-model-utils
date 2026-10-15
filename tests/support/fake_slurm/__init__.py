"""Controller for the fake SLURM test harness (ticket-022/023, ADR-025,
R81). `FakeSlurm.install()` writes real `sbatch`/`squeue`/`scancel`/
`sacct`/`scontrol`/`mpiexec`/`srun` shims that really execute rendered
job scripts as detached bash processes, so system tests can drive
production code through the existing `--slurm-path`/PATH seams. The
controller also exposes `force`, `lag_sacct`, `disable_accounting`,
`purge` and `configuring` (ticket-023) for deterministic accounting
scenarios, and `fail_step_start` (ticket-032) to make every `srun
--ntasks` step fail to start.

Per the 2026-10-01 operator amendment, `squeue` lists a job in every
state until it is purged; `purge()` is the only operation that removes
a job from both `squeue` and `scontrol`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.support.fake_slurm.shim import (
    FAIL_STEP_START_FLAG,
    TERMINAL_STATES,
    list_job_ids,
    purge_job,
    read_job,
    set_accounting,
    set_force,
    update_job,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIRST_JOB_ID = 1000
_POLL_INTERVAL = 0.05

# Must stay in sync with shim.py's `_DISPATCH`.
_SHIMMED_COMMANDS = (
    "sbatch",
    "squeue",
    "scancel",
    "sacct",
    "scontrol",
    "mpiexec",
    "srun",
)

_SHIM_TEMPLATE = """\
#!{python}
import os
import sys

sys.path[0:0] = [{repo_root!r}]
os.environ["FAKE_SLURM_STATE_DIR"] = {state_dir!r}

from tests.support.fake_slurm.shim import main

raise SystemExit(main())
"""


def _install_shim(bin_dir: Path, state_dir: Path, command: str) -> None:
    shim_path = bin_dir / command
    shim_path.write_text(
        _SHIM_TEMPLATE.format(
            python=sys.executable,
            repo_root=str(_REPO_ROOT),
            state_dir=str(state_dir),
        )
    )
    shim_path.chmod(0o755)


@dataclass(frozen=True)
class FakeSlurm:
    bin_dir: Path
    state_dir: Path

    @staticmethod
    def install(tmp_dir: Path) -> FakeSlurm:
        bin_dir = tmp_dir / "bin"
        state_dir = tmp_dir / "state"
        bin_dir.mkdir(parents=True)
        (state_dir / "jobs").mkdir(parents=True)
        (state_dir / "counter").write_text(str(_FIRST_JOB_ID))
        (state_dir / "invocations.jsonl").touch()
        for command in _SHIMMED_COMMANDS:
            _install_shim(bin_dir, state_dir, command)
        return FakeSlurm(bin_dir=bin_dir, state_dir=state_dir)

    def submitted(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for job_id in list_job_ids(self.state_dir):
            job = read_job(self.state_dir, job_id)
            jobs.append(
                {
                    "id": job["id"],
                    "argv": job["argv"],
                    "directives": job["directives"],
                    "dependency": job["dependency"],
                    "script": job["script"],
                }
            )
        return jobs

    def job(self, job_id: int) -> dict[str, Any]:
        return read_job(self.state_dir, job_id)

    def wait_idle(self, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ids = list_job_ids(self.state_dir)
            if all(
                read_job(self.state_dir, job_id)["state"] in TERMINAL_STATES
                for job_id in ids
            ):
                return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError("fake_slurm: jobs still active after timeout")

    def hold(self, job_id: int) -> None:
        update_job(self.state_dir, job_id, lambda j: {**j, "held": True})

    def release(self, job_id: int) -> None:
        update_job(self.state_dir, job_id, lambda j: {**j, "held": False})

    def set_pending_reason(self, job_id: int, reason: str) -> None:
        update_job(self.state_dir, job_id, lambda j: {**j, "reason": reason})

    def force(
        self,
        job_id_or_phase: int | str,
        *,
        state: str,
        exit_code: str = "0:0",
        after: float = 0.0,
        step_oom: bool = False,
        reason: str | None = None,
    ) -> None:
        set_force(
            self.state_dir,
            str(job_id_or_phase),
            {
                "state": state,
                "exit_code": exit_code,
                "after": after,
                "step_oom": step_oom,
                "reason": reason,
            },
        )

    def lag_sacct(self, n: int) -> None:
        set_accounting(self.state_dir, lambda a: {**a, "lag": n})

    def disable_accounting(self) -> None:
        set_accounting(self.state_dir, lambda a: {**a, "disabled": True})

    def purge(self, job_id: int) -> None:
        purge_job(self.state_dir, job_id)

    def configuring(self, job_id: int, seconds: float) -> None:
        update_job(
            self.state_dir,
            job_id,
            lambda j: {**j, "configuring_seconds": seconds},
        )

    def fail_step_start(self) -> None:
        (self.state_dir / FAIL_STEP_START_FLAG).touch()
