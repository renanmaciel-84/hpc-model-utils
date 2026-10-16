"""System smoke tests for the fake SLURM accounting harness (ticket-023,
ADR-025, ADR-022, R81, R47, R110): forced outcomes, `sacct`/`scontrol`,
`purge`, the `CONFIGURING` window and the `mpiexec`/`srun` shims.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from tests.support.fake_slurm import FakeSlurm
from tests.support.fake_slurm.cli import (
    sacct,
    sbatch,
    scancel,
    scontrol,
    squeue,
    wait_for_state,
    write_script,
)


def _submit(tmp_path: Path, name: str, body: str) -> int:
    script = write_script(tmp_path, name, body)
    result = sbatch("--parsable", str(script), cwd=tmp_path)
    assert result.returncode == 0
    return int(result.stdout.strip())


def test_sacct_force_timeout_kills_job_and_reports_timeout_row(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    log_dir = tmp_path / "logs"
    fake_slurm.force("model", state="TIMEOUT", after=0.5)
    job_id = _submit(
        tmp_path,
        "model.sh",
        f"#SBATCH --output={log_dir}/model-%j.out\nsleep 30",
    )

    fake_slurm.wait_idle()

    result = sacct(
        "-j",
        str(job_id),
        "--noheader",
        "--parsable2",
        "--format=JobID,State,ExitCode,Timelimit",
    )
    assert result.returncode == 0
    first_row = result.stdout.strip().splitlines()[0]
    assert first_row.startswith(f"{job_id}|TIMEOUT|0:15|")

    rows = (
        sacct(
            "-j",
            str(job_id),
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode",
        )
        .stdout.strip()
        .splitlines()
    )
    assert rows == [
        f"{job_id}|TIMEOUT|0:15",
        f"{job_id}.batch|CANCELLED|0:15",
        f"{job_id}.extern|COMPLETED|0:0",
    ]


def test_sacct_force_node_fail_kills_job_and_reports_node_fail_rows(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "sleep 30")
    fake_slurm.hold(job_id)
    fake_slurm.force(job_id, state="NODE_FAIL", after=0.3)
    fake_slurm.release(job_id)

    fake_slurm.wait_idle()

    result = sacct(
        "-j",
        str(job_id),
        "--noheader",
        "--parsable2",
        "--format=JobID,State,ExitCode",
    )
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [
        f"{job_id}|NODE_FAIL|0:0",
        f"{job_id}.batch|NODE_FAIL|0:9",
        f"{job_id}.extern|COMPLETED|0:0",
    ]


def test_sacct_force_cancelled_kills_job_and_reports_cancelled_by_0_rows(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "sleep 30")
    fake_slurm.hold(job_id)
    fake_slurm.force(job_id, state="CANCELLED", after=0.3)
    fake_slurm.release(job_id)

    fake_slurm.wait_idle()

    result = sacct(
        "-j",
        str(job_id),
        "--noheader",
        "--parsable2",
        "--format=JobID,State,ExitCode",
    )
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [
        f"{job_id}|CANCELLED by 0|0:0",
        f"{job_id}.batch|CANCELLED|0:15",
        f"{job_id}.extern|COMPLETED|0:0",
    ]


def test_sacct_natural_completed_job_reports_single_row(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()

    result = sacct(
        "-j", str(job_id), "--noheader", "--parsable2", "--format=JobID,State"
    )
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [f"{job_id}|COMPLETED"]


def test_sacct_force_step_oom_reports_failed_job_row_and_oom_step_row(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.hold(job_id)
    fake_slurm.force(job_id, state="FAILED", exit_code="1:0", step_oom=True)
    fake_slurm.release(job_id)

    fake_slurm.wait_idle()

    result = sacct(
        "-j", str(job_id), "--noheader", "--parsable2", "--format=JobID,State"
    )
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [
        f"{job_id}|FAILED",
        f"{job_id}.batch|OUT_OF_MEMORY",
    ]


def test_sacct_and_scontrol_report_scancelled_job_with_uid_reason(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "sleep 60")
    fake_slurm.hold(job_id)

    cancel_result = scancel(str(job_id))
    assert cancel_result.returncode == 0

    wait_for_state(fake_slurm, job_id, "CANCELLED", timeout=2.0)

    sacct_result = sacct(
        "-j", str(job_id), "--noheader", "--parsable2", "--format=JobID,State"
    )
    assert sacct_result.stdout.strip() == f"{job_id}|CANCELLED"

    scontrol_result = scontrol("show", "job", "-o", str(job_id))
    assert scontrol_result.returncode == 0
    assert f"Reason=cancelled by {os.getuid()}" in scontrol_result.stdout
    assert "JobState=CANCELLED " in scontrol_result.stdout


def test_sacct_lag_sacct_returns_no_rows_then_recovers(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()
    fake_slurm.lag_sacct(2)

    def _query() -> subprocess.CompletedProcess[str]:
        return sacct(
            "-j",
            str(job_id),
            "--noheader",
            "--parsable2",
            "--format=JobID,State",
        )

    first, second, third = _query(), _query(), _query()

    assert (first.returncode, first.stdout) == (0, "")
    assert (second.returncode, second.stdout) == (0, "")
    assert third.returncode == 0
    assert third.stdout.strip() == f"{job_id}|COMPLETED"


def test_sacct_disable_accounting_exits_1_with_prd_stderr_line(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()
    fake_slurm.disable_accounting()

    result = sacct(
        "-j", str(job_id), "--noheader", "--parsable2", "--format=JobID,State"
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Slurm accounting storage is disabled"


def test_scontrol_show_job_reports_completed_before_purge_and_invalid_after(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()

    before = scontrol("show", "job", "-o", str(job_id))
    assert before.returncode == 0
    assert "JobState=COMPLETED" in before.stdout

    fake_slurm.purge(job_id)

    after = scontrol("show", "job", "-o", str(job_id))
    assert after.returncode == 1
    assert "Invalid job id" in after.stderr


def test_scontrol_requeue_finished_job_reruns_with_restart_count(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", 'echo "restart=$SLURM_RESTART_COUNT"')
    fake_slurm.wait_idle()

    result = scontrol("requeue", str(job_id))
    fake_slurm.wait_idle()

    assert result.returncode == 0
    output = (tmp_path / f"slurm-{job_id}.out").read_text()
    assert output.splitlines() == ["restart=0", "restart=1"]
    assert fake_slurm.job(job_id)["restart_count"] == 1


def test_scontrol_requeue_running_job_is_refused(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "sleep 5")
    wait_for_state(fake_slurm, job_id, "RUNNING")

    result = scontrol("requeue", str(job_id))

    assert result.returncode == 1
    assert (
        result.stderr.strip()
        == "scontrol: error: fake-slurm requeue needs a finished job"
    )
    assert fake_slurm.job(job_id)["restart_count"] == 0


def test_squeue_after_purge_single_id_exits_1_invalid_job_id(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()
    fake_slurm.purge(job_id)

    result = squeue("-h", "-j", str(job_id), "-o", "%T")
    assert result.returncode == 1
    assert "Invalid job id" in result.stderr


def test_squeue_after_purge_multi_id_skips_purged_job(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_a = _submit(tmp_path, "a.sh", "exit 0")
    job_b = _submit(tmp_path, "b.sh", "exit 0")
    fake_slurm.wait_idle()
    fake_slurm.purge(job_a)

    result = squeue("-h", "-j", f"{job_a},{job_b}", "-o", "%i")
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [str(job_b)]


def test_squeue_configuring_window_reports_configuring_before_running(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.hold(job_id)
    fake_slurm.configuring(job_id, 0.3)
    fake_slurm.release(job_id)

    deadline = time.time() + 2.0
    seen_configuring = False
    while time.time() < deadline:
        state = squeue("-h", "-j", str(job_id), "-o", "%T").stdout.strip()
        if state == "CONFIGURING":
            seen_configuring = True
        if state == "COMPLETED":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("job never reached COMPLETED")

    assert seen_configuring
    assert fake_slurm.job(job_id)["state"] == "COMPLETED"


def test_mpiexec_runs_command_once_and_logs_argv(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    marker = tmp_path / "ran.txt"
    fake_model = write_script(tmp_path, "fake-model", f"echo ran >> {marker}")
    job_id = _submit(tmp_path, "job.sh", f"mpiexec -np 4 {fake_model}")

    fake_slurm.wait_idle()

    assert fake_slurm.job(job_id)["state"] == "COMPLETED"
    assert marker.read_text().splitlines() == ["ran"]
    records = [
        json.loads(line)
        for line in (fake_slurm.state_dir / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    mpi_records = [r for r in records if r["cmd"] == "mpiexec"]
    assert len(mpi_records) == 1
    assert mpi_records[0]["argv"][1:] == ["-np", "4", str(fake_model)]


def test_srun_runs_command_once_and_sets_slurm_env(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    out_file = tmp_path / "env.txt"
    helper = write_script(
        tmp_path,
        "print_env.sh",
        f'echo "$SLURM_PROCID $SLURM_NNODES" >> {out_file}',
    )
    job_id = _submit(tmp_path, "job.sh", f"#SBATCH --nodes=2\nsrun {helper}")

    fake_slurm.wait_idle()

    assert fake_slurm.job(job_id)["state"] == "COMPLETED"
    assert out_file.read_text().splitlines() == ["0 2"]
    records = [
        json.loads(line)
        for line in (fake_slurm.state_dir / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    srun_records = [r for r in records if r["cmd"] == "srun"]
    assert len(srun_records) == 1


def test_sacct_invalid_format_field_exits_1_mirroring_real_sacct(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    job_id = _submit(tmp_path, "job.sh", "exit 0")
    fake_slurm.wait_idle()

    result = sacct(
        "-j", str(job_id), "--noheader", "--parsable2", "--format=Bogus"
    )

    assert result.returncode == 1
    assert (
        result.stderr.strip()
        == 'sacct: error: Invalid field requested: "Bogus"'
    )
