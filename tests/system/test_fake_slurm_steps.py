"""System tests for the fake SLURM `srun --ntasks` step path (ticket-031,
ADR-087, R88): per-task environment, `%j`/`%t` outputs, session separation
from the batch script, exit-status aggregation, the usage errors, and the
unchanged single-task `exec` path. Ticket-032 adds the job-end teardown of
live step tasks, the srun client's signal behaviour and the step-start
failure knob.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.support.fake_slurm.cli import (
    sbatch,
    scancel,
    wait_for_state,
    wait_for_step_tasks,
    write_script,
)
from tests.support.fake_slurm.shim import (
    _expand_output,
    _parse_launcher_args,
    _step_status,
)


def _submit(tmp_path: Path, name: str, body: str) -> int:
    script = write_script(tmp_path, name, body)
    result = sbatch("--parsable", str(script), cwd=tmp_path)
    assert result.returncode == 0
    return int(result.stdout.strip())


def _batch_log(tmp_path: Path, job_id: int) -> str:
    return (tmp_path / f"slurm-{job_id}.out").read_text()


def _proc_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat.rsplit(")", 1)[1].split()[0]


def _wait_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            if _proc_state(pid) == "Z":
                return
        except (FileNotFoundError, ProcessLookupError):
            return
        if time.time() > deadline:
            raise AssertionError(f"pid {pid} is still alive")
        time.sleep(0.05)


def _wait_runner_gone(fake_slurm: FakeSlurm, job_id: int) -> None:
    _wait_gone(fake_slurm.job(job_id)["runner_pid"], timeout=10.0)


def _wait_for_files(*paths: Path, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not all(path.exists() for path in paths):
        if time.time() > deadline:
            raise AssertionError(f"never appeared: {paths}")
        time.sleep(0.05)


def _cancelled_pattern(job_id: int, step: int) -> str:
    return (
        rf"slurmstepd: error: \*\*\* STEP {job_id}\.{step} ON fake-node0"
        r" CANCELLED AT \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d \*\*\*"
    )


def _trap_task(tmp_path: Path, prefix: str = "") -> Path:
    """A task that records TERM or INT in `<prefix>term-<n>`/`<prefix>int-<n>`
    and exits 0, after writing `<prefix>ready-<n>`. Its sleeper ignores TERM
    so the trap, not the sleeper's death, ends the wait."""
    return write_script(
        tmp_path,
        f"{prefix}trap-task.sh",
        "(trap '' TERM; exec sleep 30) &\n"
        "sleeper=$!\n"
        f"trap 'touch {tmp_path}/{prefix}term-$SLURM_PROCID;"
        " kill -9 $sleeper; exit 0' TERM\n"
        f"trap 'touch {tmp_path}/{prefix}int-$SLURM_PROCID;"
        " kill -9 $sleeper; exit 0' INT\n"
        f"touch {tmp_path}/{prefix}ready-$SLURM_PROCID\n"
        "wait $sleeper",
    )


def _stubborn_task(tmp_path: Path) -> Path:
    return write_script(
        tmp_path,
        "stubborn-task.sh",
        f"trap '' TERM\ntouch {tmp_path}/ready-$SLURM_PROCID\nsleep 30",
    )


def _detached_step(tmp_path: Path, task: Path, tail: str) -> str:
    """Backgrounds a two-task overlap step, waits for both tasks to be
    ready (bounded), then runs `tail`."""
    return (
        f"srun --overlap --ntasks=2 --output={tmp_path}/t-%j-%t.log {task}"
        f" > {tmp_path}/srun.log 2>&1 & disown\n"
        "for _ in $(seq 200); do\n"
        f"  [ -e {tmp_path}/ready-0 ] && [ -e {tmp_path}/ready-1 ] && break\n"
        "  sleep 0.05\n"
        "done\n"
        f"{tail}"
    )


def test_srun_ntasks_writes_one_output_per_task_with_task_env(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    helper = write_script(
        tmp_path,
        "task-env.sh",
        'echo "$SLURMD_NODENAME $SLURM_NODEID $SLURM_PROCID $SLURM_NNODES'
        ' $SLURM_JOB_ID"',
    )
    job_id = _submit(
        tmp_path,
        "job.sh",
        "srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1"
        " --cpus-per-task=1 --cpu-bind=none --mpi=none"
        " --kill-on-bad-exit=0 --input=none --job-name=hpcmu-telemetry"
        f" --output={telemetry}/model-%j-r0-%t.log {helper}",
    )

    fake_slurm.wait_idle()

    job = fake_slurm.job(job_id)
    assert job["state"] == "COMPLETED"
    assert (telemetry / f"model-{job_id}-r0-0.log").read_text() == (
        f"fake-node0 0 0 2 {job_id}\n"
    )
    assert (telemetry / f"model-{job_id}-r0-1.log").read_text() == (
        f"fake-node1 1 1 2 {job_id}\n"
    )
    batch_log = _batch_log(tmp_path, job_id)
    assert "fake-node0" not in batch_log
    assert "fake-node1" not in batch_log
    assert [t["task"] for t in job["step_tasks"]] == [0, 1]
    assert len({t["pid"] for t in job["step_tasks"]}) == 2


def test_srun_ntasks_output_file_captures_stdout_and_stderr(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    log = tmp_path / "step.log"
    helper = write_script(tmp_path, "both.sh", "echo out\necho err >&2")
    job_id = _submit(
        tmp_path, "job.sh", f"srun --ntasks=1 --output={log} {helper}"
    )

    fake_slurm.wait_idle()

    assert log.read_text() == "out\nerr\n"
    assert "out" not in _batch_log(tmp_path, job_id)


def test_srun_ntasks_shared_output_is_truncated_and_opened_once(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    log = tmp_path / "shared.log"
    log.write_text("stale\n")
    helper = write_script(tmp_path, "task.sh", 'echo "task $SLURM_PROCID"')
    _submit(tmp_path, "job.sh", f"srun --ntasks=2 --output={log} {helper}")

    fake_slurm.wait_idle()

    assert sorted(log.read_text().splitlines()) == ["task 0", "task 1"]


def test_srun_ntasks_without_output_inherits_batch_stdout_and_stderr(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    helper = write_script(
        tmp_path,
        "task.sh",
        'echo "out $SLURM_PROCID"\necho "err $SLURM_PROCID" >&2',
    )
    job_id = _submit(tmp_path, "job.sh", f"srun --ntasks=2 {helper}")

    fake_slurm.wait_idle()

    assert sorted(_batch_log(tmp_path, job_id).splitlines()) == [
        "err 0",
        "err 1",
        "out 0",
        "out 1",
    ]


def test_srun_ntasks_gives_tasks_devnull_stdin(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    helper = write_script(tmp_path, "task.sh", 'echo "stdin=[$(cat)]"')
    job_id = _submit(
        tmp_path, "job.sh", f"echo data | srun --ntasks=1 {helper}"
    )

    fake_slurm.wait_idle()

    assert _batch_log(tmp_path, job_id) == "stdin=[]\n"


def test_srun_ntasks_records_each_task_pid_with_its_task_index(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    out = tmp_path / "pids.txt"
    helper = write_script(
        tmp_path, "task.sh", f'echo "$$ $SLURM_PROCID" >> {out}'
    )
    job_id = _submit(tmp_path, "job.sh", f"srun --ntasks=3 {helper}")

    fake_slurm.wait_idle()

    seen = {
        (int(pid), int(task))
        for pid, task in (line.split() for line in out.read_text().splitlines())
    }
    recorded = {
        (t["pid"], t["task"]) for t in fake_slurm.job(job_id)["step_tasks"]
    }
    assert recorded == seen
    assert len(recorded) == 3


def test_srun_ntasks_appends_step_tasks_across_steps(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    helper = write_script(tmp_path, "task.sh", "exit 0")
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --ntasks=2 {helper}\nsrun --ntasks=1 {helper}",
    )

    fake_slurm.wait_idle()

    step_tasks = fake_slurm.job(job_id)["step_tasks"]
    assert [t["task"] for t in step_tasks] == [0, 1, 0]
    assert [t["step"] for t in step_tasks] == [0, 0, 1]


def test_batch_exit_terms_live_step_tasks_and_logs_cancelled_in_task_zero(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    task = _trap_task(tmp_path)
    job_id = _submit(
        tmp_path, "job.sh", _detached_step(tmp_path, task, "exit 0")
    )

    fake_slurm.wait_idle()

    job = fake_slurm.job(job_id)
    assert job["state"] == "COMPLETED"
    assert job["exit_code"] == "0:0"
    assert (tmp_path / "term-0").exists()
    assert (tmp_path / "term-1").exists()
    [line] = (tmp_path / f"t-{job_id}-0.log").read_text().splitlines()
    assert re.fullmatch(_cancelled_pattern(job_id, 0), line)
    assert (tmp_path / f"t-{job_id}-1.log").read_text() == ""
    assert (tmp_path / "srun.log").read_text() == ""
    assert _batch_log(tmp_path, job_id) == ""
    for entry in job["step_tasks"]:
        _wait_gone(entry["pid"])


def test_teardown_does_not_overwrite_or_get_overwritten_by_task_output(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    task = write_script(
        tmp_path,
        "late-writer.sh",
        "(trap '' TERM; exec sleep 30) &\n"
        "sleeper=$!\n"
        "trap 'sleep 0.5; echo bye; kill -9 $sleeper; exit 0' TERM\n"
        f"touch {tmp_path}/ready-$SLURM_PROCID\n"
        "wait $sleeper",
    )
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --overlap --ntasks=1 --output={tmp_path}/t-%j-%t.log {task}"
        " > /dev/null 2>&1 & disown\n"
        f"for _ in $(seq 200); do [ -e {tmp_path}/ready-0 ] && break;"
        " sleep 0.05; done",
    )

    fake_slurm.wait_idle()

    lines = (tmp_path / f"t-{job_id}-0.log").read_text().splitlines()
    assert len(lines) == 2
    assert "bye" in lines
    [cancelled] = [line for line in lines if line != "bye"]
    assert re.fullmatch(_cancelled_pattern(job_id, 0), cancelled)


def test_batch_exit_logs_one_cancelled_line_per_live_step(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    first = _trap_task(tmp_path, "a-")
    second = _trap_task(tmp_path, "b-")
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --overlap --ntasks=1 --output={tmp_path}/a-%j.log {first}"
        " > /dev/null 2>&1 & disown\n"
        f"for _ in $(seq 200); do [ -e {tmp_path}/a-ready-0 ] && break;"
        " sleep 0.05; done\n"
        f"srun --overlap --ntasks=1 --output={tmp_path}/b-%j.log {second}"
        " > /dev/null 2>&1 & disown\n"
        f"for _ in $(seq 200); do [ -e {tmp_path}/b-ready-0 ] && break;"
        " sleep 0.05; done",
    )

    fake_slurm.wait_idle()

    [first_line] = (tmp_path / f"a-{job_id}.log").read_text().splitlines()
    [second_line] = (tmp_path / f"b-{job_id}.log").read_text().splitlines()
    assert re.fullmatch(_cancelled_pattern(job_id, 0), first_line)
    assert re.fullmatch(_cancelled_pattern(job_id, 1), second_line)
    assert (tmp_path / "a-term-0").exists()
    assert (tmp_path / "b-term-0").exists()


@pytest.mark.parametrize("cancel", [False, True], ids=["batch-exit", "scancel"])
def test_teardown_kills_batch_leftovers_and_terms_them_only_on_scancel(
    fake_slurm: FakeSlurm, tmp_path: Path, cancel: bool
) -> None:
    leftover = write_script(
        tmp_path,
        "leftover.sh",
        f"trap 'sleep 0.5; touch {tmp_path}/leftover-term' TERM\n"
        f"echo $$ > {tmp_path}/leftover.pid\n"
        "while :; do sleep 0.05; done",
    )
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"{leftover} &\n"
        f"for _ in $(seq 200); do [ -s {tmp_path}/leftover.pid ] && break;"
        " sleep 0.05; done\n" + ("wait" if cancel else "exit 0"),
    )
    _wait_for_files(tmp_path / "leftover.pid")
    pid = int((tmp_path / "leftover.pid").read_text())
    if cancel:
        assert scancel(str(job_id)).returncode == 0

    _wait_runner_gone(fake_slurm, job_id)

    _wait_gone(pid)
    assert (tmp_path / "leftover-term").exists() is cancel
    state = "CANCELLED" if cancel else "COMPLETED"
    assert fake_slurm.job(job_id)["state"] == state


def test_teardown_continues_a_stopped_task_before_terming_it(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    task = _trap_task(tmp_path)
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --overlap --ntasks=1 --output={tmp_path}/t-%j-%t.log {task}"
        " > /dev/null 2>&1 & disown\n"
        f"for _ in $(seq 200); do [ -e {tmp_path}/stopped ] && break;"
        " sleep 0.05; done",
    )
    [entry] = wait_for_step_tasks(fake_slurm, job_id, 1)
    _wait_for_files(tmp_path / "ready-0")
    os.killpg(entry["pid"], signal.SIGSTOP)
    deadline = time.time() + 5.0
    while _proc_state(entry["pid"]) != "T":
        assert time.time() < deadline
        time.sleep(0.05)

    (tmp_path / "stopped").touch()
    fake_slurm.wait_idle()

    assert (tmp_path / "term-0").exists()
    _wait_gone(entry["pid"])


@pytest.mark.parametrize(
    ("forced", "terminal"),
    [
        pytest.param(False, "COMPLETED", id="batch-exit"),
        pytest.param(True, "TIMEOUT", id="forced-timeout"),
    ],
)
def test_teardown_escalates_to_kill_for_tasks_that_ignore_term(
    fake_slurm: FakeSlurm, tmp_path: Path, forced: bool, terminal: str
) -> None:
    task = _stubborn_task(tmp_path)
    if forced:
        fake_slurm.force("model", state="TIMEOUT", after=2.0)
        job_id = _submit(
            tmp_path,
            "model.sh",
            f"#SBATCH --output={tmp_path}/model-%j.out\n"
            + _detached_step(tmp_path, task, "sleep 30"),
        )
    else:
        job_id = _submit(
            tmp_path,
            "job.sh",
            _detached_step(
                tmp_path,
                task,
                f"date +%s.%N > {tmp_path}/exit-at\nexit 0",
            ),
        )
    tasks = wait_for_step_tasks(fake_slurm, job_id, 2)
    _wait_for_files(tmp_path / "ready-0", tmp_path / "ready-1")

    job = fake_slurm.job(job_id)
    assert job["state"] == "RUNNING"
    for entry in tasks:
        pid = entry["pid"]
        assert os.getpgid(pid) == pid
        assert os.getsid(pid) == pid
        assert pid != job["pgid"]
    wait_for_state(fake_slurm, job_id, terminal, timeout=15.0)

    job = fake_slurm.job(job_id)
    for entry in tasks:
        _wait_gone(entry["pid"])
    if not forced:
        exit_at = float((tmp_path / "exit-at").read_text())
        assert job["end_time"] - exit_at >= 2.0
    [line] = (tmp_path / f"t-{job_id}-0.log").read_text().splitlines()
    assert re.fullmatch(_cancelled_pattern(job_id, 0), line)


_CLIENT_ABORT_LINES = [
    "srun: forcing job termination",
    "srun: Job step aborted: Waiting up to 4 seconds for job step to finish.",
    "srun: error: fake-node1: task 1: Killed",
    "srun: error: fake-node0: task 0: Killed",
]


@pytest.mark.parametrize("action", ["term-client", "scancel", "int-client"])
def test_srun_client_signals_are_converted_or_forwarded(
    fake_slurm: FakeSlurm, tmp_path: Path, action: str
) -> None:
    task = _trap_task(tmp_path)
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --ntasks=2 --output={tmp_path}/t-%t.log {task}"
        f" 2> {tmp_path}/srun.err &\n"
        f"echo $! > {tmp_path}/client.pid\n"
        "wait $!\n"
        'echo "rc=$?"',
    )
    wait_for_step_tasks(fake_slurm, job_id, 2)
    _wait_for_files(
        tmp_path / "ready-0", tmp_path / "ready-1", tmp_path / "client.pid"
    )
    client = int((tmp_path / "client.pid").read_text())

    if action == "term-client":
        os.kill(client, signal.SIGTERM)
    elif action == "scancel":
        assert scancel(str(job_id)).returncode == 0
    else:
        os.kill(client, signal.SIGINT)
    _wait_runner_gone(fake_slurm, job_id)

    stderr = (tmp_path / "srun.err").read_text()
    last_line = _batch_log(tmp_path, job_id).splitlines()[-1:]
    if action == "int-client":
        assert stderr == ""
        assert (tmp_path / "int-0").exists()
        assert (tmp_path / "int-1").exists()
        assert last_line == ["rc=0"]
        return
    assert stderr.splitlines() == _CLIENT_ABORT_LINES
    assert not (tmp_path / "term-0").exists()
    assert not (tmp_path / "term-1").exists()
    [line] = (tmp_path / "t-0.log").read_text().splitlines()
    assert re.fullmatch(_cancelled_pattern(job_id, 0), line)
    if action == "term-client":
        assert last_line == ["rc=137"]
    else:
        assert fake_slurm.job(job_id)["state"] == "CANCELLED"


def test_fail_step_start_fails_the_step_before_spawning_or_opening_outputs(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    fake_slurm.fail_step_start()
    task = write_script(tmp_path, "task.sh", "exit 0")
    job_id = _submit(
        tmp_path,
        "job.sh",
        f"srun --ntasks=2 --output={tmp_path}/t-%t.log {task}"
        f" 2> {tmp_path}/srun.err\n"
        'echo "rc=$?"\n'
        "srun echo hi",
    )

    fake_slurm.wait_idle()

    assert _batch_log(tmp_path, job_id) == "rc=1\nhi\n"
    assert (tmp_path / "srun.err").read_text() == (
        f"srun: error: Unable to create step for job {job_id}:"
        " fake-slurm step-start failure\n"
    )
    assert list(tmp_path.glob("t-*.log")) == []
    assert fake_slurm.job(job_id)["step_tasks"] == []


@pytest.mark.parametrize(
    ("helper_body", "expected_rc"),
    [
        pytest.param(
            '[ "$SLURM_PROCID" = 0 ] && exit 3\nexit 5', "rc=5", id="highest"
        ),
        pytest.param(
            '[ "$SLURM_PROCID" = 0 ] && exit 3\nkill -TERM $$',
            "rc=143",
            id="signal",
        ),
    ],
)
def test_srun_ntasks_returns_highest_task_status(
    fake_slurm: FakeSlurm, tmp_path: Path, helper_body: str, expected_rc: str
) -> None:
    helper = write_script(tmp_path, "task.sh", helper_body)
    job_id = _submit(
        tmp_path, "job.sh", f'srun --ntasks=2 {helper}\necho "rc=$?"'
    )

    fake_slurm.wait_idle()

    assert _batch_log(tmp_path, job_id) == f"{expected_rc}\n"


def test_srun_ntasks_zero_prints_usage_error_and_spawns_nothing(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    marker = tmp_path / "marker"
    helper = write_script(tmp_path, "task.sh", f"touch {marker}")
    job_id = _submit(
        tmp_path, "job.sh", f'srun --ntasks=0 {helper}\necho "rc=$?"'
    )

    fake_slurm.wait_idle()

    lines = _batch_log(tmp_path, job_id).splitlines()
    assert lines[0].startswith(
        "srun: error: --ntasks must be a positive integer"
    )
    assert lines[-1] == "rc=1"
    assert not marker.exists()
    assert fake_slurm.job(job_id)["step_tasks"] == []


def test_srun_ntasks_missing_output_dir_fails_without_creating_it(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    marker = tmp_path / "marker"
    missing = tmp_path / "missing"
    helper = write_script(tmp_path, "task.sh", f"touch {marker}")
    job_id = _submit(
        tmp_path,
        "job.sh",
        f'srun --ntasks=1 --output={missing}/t-%t.log {helper}\necho "rc=$?"',
    )

    fake_slurm.wait_idle()

    assert _batch_log(tmp_path, job_id).splitlines()[-1] == "rc=1"
    assert not marker.exists()
    assert not missing.exists()
    assert fake_slurm.job(job_id)["step_tasks"] == []


@pytest.mark.parametrize(
    "raw", ["0", "-1", "+2", "two", "", "2.0", " 2", "\u0662"]
)
def test_srun_ntasks_rejects_non_positive_integer_values(
    fake_slurm: FakeSlurm, raw: str
) -> None:
    result = subprocess.run(
        ["srun", f"--ntasks={raw}", "true"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == (
        f"srun: error: --ntasks must be a positive integer, got '{raw}'\n"
    )


@pytest.mark.parametrize("job_id", [None, ""], ids=["unset", "empty"])
def test_srun_ntasks_without_job_id_prints_usage_error(
    fake_slurm: FakeSlurm, job_id: str | None
) -> None:
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    if job_id is not None:
        env["SLURM_JOB_ID"] = job_id

    result = subprocess.run(
        ["srun", "--ntasks=1", "true"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("srun: error: --ntasks needs SLURM_JOB_ID")


def test_srun_ntasks_record_failure_kills_started_tasks_and_exits_1(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    pids = tmp_path / "pids"
    helper = write_script(tmp_path, "task.sh", f"echo $$ >> {pids}\nsleep 30")
    env = {**os.environ, "SLURM_JOB_ID": "4242"}

    result = subprocess.run(
        ["srun", "--ntasks=1", str(helper)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=15,
    )

    assert result.returncode == 1
    assert "FileNotFoundError" in result.stderr
    started = pids.read_text().split() if pids.exists() else []
    assert all(not Path(f"/proc/{pid}").exists() for pid in started)


def test_srun_without_ntasks_execs_in_place_with_procid_zero(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    out = tmp_path / "helper.out"
    pid_file = tmp_path / "srun.pid"
    helper = write_script(
        tmp_path, "task.sh", f'echo "$$ $SLURM_PROCID" > {out}'
    )
    job_id = _submit(
        tmp_path,
        "job.sh",
        "srun --mpi=pmix --ntasks-per-node=1 --cpus-per-task=1"
        f" --kill-on-bad-exit=1 {helper} &\n"
        f'echo "$!" > {pid_file}\nwait',
    )

    fake_slurm.wait_idle()

    pid, procid = out.read_text().split()
    assert pid == pid_file.read_text().strip()
    assert procid == "0"
    assert fake_slurm.job(job_id)["step_tasks"] == []


def test_srun_overlap_without_ntasks_runs_command_once(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    out = tmp_path / "ran.txt"
    helper = write_script(tmp_path, "task.sh", f"echo ran >> {out}")
    job_id = _submit(tmp_path, "job.sh", f"srun --overlap {helper}")

    fake_slurm.wait_idle()

    assert out.read_text().splitlines() == ["ran"]
    assert fake_slurm.job(job_id)["step_tasks"] == []


def test_srun_short_n_keeps_single_task_exec_path(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    out = tmp_path / "ran.txt"
    helper = write_script(tmp_path, "task.sh", f'echo "$SLURM_PROCID" >> {out}')
    job_id = _submit(tmp_path, "job.sh", f"srun -n 2 {helper}")

    fake_slurm.wait_idle()

    assert out.read_text().splitlines() == ["0"]
    assert fake_slurm.job(job_id)["step_tasks"] == []


def test_parse_launcher_args_splits_options_from_command() -> None:
    options, cmd = _parse_launcher_args(
        [
            "--overlap",
            "--ntasks=2",
            "-np",
            "4",
            "--x=1",
            "--x=2",
            "prog",
            "--not-an-option",
        ],
        frozenset({"--overlap"}),
    )

    assert options == {"overlap": "", "ntasks": "2", "np": "4", "x": "2"}
    assert cmd == ["prog", "--not-an-option"]


def test_parse_launcher_args_without_flags_consumes_next_token() -> None:
    assert _parse_launcher_args(["--overlap", "prog"], frozenset()) == (
        {"overlap": "prog"},
        [],
    )
    assert _parse_launcher_args(["--ntasks"], frozenset()) == (
        {"ntasks": ""},
        [],
    )


def test_expand_output_replaces_job_and_task_only() -> None:
    assert _expand_output("a-%N-%j-%t-%s-%j", 7, 3) == "a-%N-7-3-%s-7"


@pytest.mark.parametrize(
    ("returncodes", "expected"),
    [
        ([0, 0], 0),
        ([3, 5], 5),
        ([5, 3], 5),
        ([0, -15], 143),
        ([-9, 2], 137),
        ([3, -2, 0], 130),
    ],
)
def test_step_status_is_highest_with_signals_as_128_plus_n(
    returncodes: list[int], expected: int
) -> None:
    assert _step_status(returncodes) == expected
