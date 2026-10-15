"""Busybox-style dispatcher for the fake SLURM `sbatch`/`squeue`/
`scancel`/`sacct`/`scontrol`/`mpiexec`/`srun` shims and their detached
runner (ticket-022/023, ADR-025, R81). Forced outcomes and accounting
knobs (ticket-023) reuse the `locked`/`read_job`/`write_job_atomic`/
`update_job` primitives below, stored under `state/forces.json` and
`state/accounting.json`.

Per the 2026-10-01 operator amendment, `squeue` lists a job in every
state until it is purged; `purge_job` is the only thing that removes a
job's state.

A job that ignores SIGTERM is escalated to SIGKILL after `_KILL_WAIT`
(2s), faking Slurm's 30s KillWait. A forced TIMEOUT/NODE_FAIL/CANCELLED
is killed directly instead, since it emulates an externally-triggered
termination rather than a user `scancel`. This module imports nothing
from the production package: it must stay independent of the code
under test.

`srun` has two paths (ticket-031, ADR-087, R88). Without `--ntasks` it
`exec`s the command once, as before. With `--ntasks N` it is a step: it
launches N background tasks, one per fake node, each leading its own
session with `SLURMD_NODENAME`/`SLURM_NODEID`/`SLURM_PROCID`/
`SLURM_NNODES` set, expands `%j`/`%t` in `--output` (opened for append),
records the task pids, step id and output path under the job's
`step_tasks`, and waits in the foreground for all of them. The tasks sit
outside the batch script's process group, so the runner's `_teardown`
ends them (ticket-032, ADR-087, R88) once the batch script exits, is
cancelled or times out: SIGCONT then SIGTERM, one `slurmstepd: ... CANCELLED
AT ...` line appended to task 0's output, SIGKILL after `_KILL_WAIT`.
The step's srun client turns SIGTERM into a KILL of the step (the four
`srun:` lines, rc 137) and forwards SIGINT to the tasks. The
`fail-step-start` flag file makes every `--ntasks` step fail to start.

`scontrol requeue <id>` (job-telemetry ticket-039, ADR-069) re-runs a
finished job: it clears the run's pids, times and steps, bumps the job's
`restart_count` (the `SLURM_RESTART_COUNT` of its next attempt) and
spawns a new runner. Real Slurm also requeues a running job; the fake
refuses it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any, TypedDict

_REPO_ROOT = Path(__file__).resolve().parents[3]

TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "CANCELLED by 0",
    }
)

_KILL_WAIT = 2.0
_POLL_INTERVAL = 0.05

FAIL_STEP_START_FLAG = "fail-step-start"

_SHORT_STATE = {
    "PENDING": "PD",
    "RUNNING": "R",
    "COMPLETED": "CD",
    "FAILED": "F",
    "CANCELLED": "CA",
}

_SRUN_FLAGS = frozenset({"--overlap"})


class StepTask(TypedDict):
    pid: int
    task: int
    step: int
    output: str | None


# --- state primitives, shared with the controller
# (tests/support/fake_slurm/__init__.py) ---


@contextlib.contextmanager
def locked(state_dir: Path) -> Iterator[None]:
    fd = os.open(state_dir / "lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _job_path(state_dir: Path, job_id: int) -> Path:
    return state_dir / "jobs" / f"{job_id}.json"


def read_job(state_dir: Path, job_id: int) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_job_path(state_dir, job_id).read_text())
    return data


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def write_job_atomic(state_dir: Path, job_id: int, job: dict[str, Any]) -> None:
    _atomic_write(_job_path(state_dir, job_id), json.dumps(job))


def create_job(state_dir: Path, job_id: int, job: dict[str, Any]) -> None:
    with locked(state_dir):
        write_job_atomic(state_dir, job_id, job)


def update_job(
    state_dir: Path,
    job_id: int,
    mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Apply ``mutator`` under the lock and return the *pre-mutation*
    job, so a caller that must act on the job's state (e.g. ``_scancel``
    signalling its pgid) can do so only after the new state is already
    durably written, instead of signalling first and mutating after."""
    with locked(state_dir):
        job = read_job(state_dir, job_id)
        updated = mutator(job)
        if updated is not None:
            write_job_atomic(state_dir, job_id, updated)
    return job


def next_job_id(state_dir: Path) -> int:
    counter_path = state_dir / "counter"
    with locked(state_dir):
        current = int(counter_path.read_text())
        _atomic_write(counter_path, str(current + 1))
    return current


def list_job_ids(state_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in (state_dir / "jobs").glob("*.json"))


def log_invocation(state_dir: Path, cmd: str, argv: list[str]) -> None:
    record = json.dumps({"cmd": cmd, "argv": argv, "t": time.time()})
    with locked(state_dir):
        with (state_dir / "invocations.jsonl").open("a") as f:
            f.write(record + "\n")


def purge_job(state_dir: Path, job_id: int) -> None:
    with locked(state_dir):
        _job_path(state_dir, job_id).unlink(missing_ok=True)


# --- forced-outcome and accounting state, written by the controller's
# `force`/`lag_sacct`/`disable_accounting` and read by the runner/`sacct` ---

_ACCOUNTING_DEFAULTS: dict[str, Any] = {"disabled": False, "lag": 0}


def _read_json_or(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def set_force(state_dir: Path, key: str, record: dict[str, Any]) -> None:
    path = state_dir / "forces.json"
    with locked(state_dir):
        forces = _read_json_or(path, {})
        forces[key] = record
        _atomic_write(path, json.dumps(forces))


def _read_forces(state_dir: Path) -> dict[str, Any]:
    return _read_json_or(state_dir / "forces.json", {})


def set_accounting(
    state_dir: Path, mutator: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    path = state_dir / "accounting.json"
    with locked(state_dir):
        current = _read_json_or(path, dict(_ACCOUNTING_DEFAULTS))
        _atomic_write(path, json.dumps(mutator(current)))


def _read_accounting(state_dir: Path) -> dict[str, Any]:
    return _read_json_or(
        state_dir / "accounting.json", dict(_ACCOUNTING_DEFAULTS)
    )


def _consume_lag(state_dir: Path) -> bool:
    path = state_dir / "accounting.json"
    with locked(state_dir):
        current = _read_json_or(path, dict(_ACCOUNTING_DEFAULTS))
        if current.get("lag", 0) > 0:
            current["lag"] -= 1
            _atomic_write(path, json.dumps(current))
            return True
        return False


# --- option parsing, shared by sbatch's CLI argv and its #SBATCH lines ---

_SBATCH_DIRECTIVE = re.compile(r"^#SBATCH\s+(.*)$")


def _parse_option_token(token: str) -> tuple[str, str] | None:
    if not token.startswith("--"):
        return None
    key, sep, value = token[2:].partition("=")
    return (key, value) if sep else (key, "")


def _parse_sbatch_lines(script_path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in script_path.read_text().splitlines():
        match = _SBATCH_DIRECTIVE.match(line.strip())
        if match is None:
            continue
        for token in match.group(1).split():
            parsed = _parse_option_token(token)
            if parsed is not None:
                directives[parsed[0]] = parsed[1]
    return directives


# --- sbatch ---


def _spawn_runner(job_id: int) -> None:
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.fake_slurm.shim",
            "__run__",
            str(job_id),
        ],
        cwd=str(_REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _sbatch(args: list[str], state_dir: Path) -> int:
    parsable = False
    cli: dict[str, str] = {}
    script_arg: str | None = None
    for token in args:
        if token == "--parsable":
            parsable = True
            continue
        parsed = _parse_option_token(token)
        if parsed is not None:
            cli[parsed[0]] = parsed[1]
        else:
            script_arg = token
    if script_arg is None:
        print("sbatch: error: missing batch script", file=sys.stderr)
        return 1
    script_path = Path(script_arg).resolve()
    directives: dict[str, str] = {**_parse_sbatch_lines(script_path), **cli}
    job_id = next_job_id(state_dir)
    dependency: int | None = None
    dep_spec = directives.get("dependency")
    if dep_spec is not None and ":" in dep_spec:
        dependency = int(dep_spec.rsplit(":", 1)[1])
    job: dict[str, Any] = {
        "id": job_id,
        "argv": args,
        "directives": directives,
        "dependency": dependency,
        "script": str(script_path),
        "state": "PENDING",
        "reason": "Dependency" if dependency is not None else None,
        "held": False,
        "cwd": directives.get("chdir", os.getcwd()),
        "submit_dir": os.getcwd(),
        "start_time": None,
        "end_time": None,
        "exit_code": None,
        "pid": None,
        "pgid": None,
        "runner_pid": None,
        "configuring_seconds": 0.0,
        "steps": [],
        "step_tasks": [],
        "restart_count": 0,
    }
    spawn = True
    if (
        dependency is not None
        and directives.get("kill-on-invalid-dep") == "yes"
        and not _job_path(state_dir, dependency).exists()
    ):
        job["state"] = "CANCELLED"
        job["reason"] = "DependencyNeverSatisfied"
        job["end_time"] = time.time()
        spawn = False
    create_job(state_dir, job_id, job)
    if spawn:
        _spawn_runner(job_id)
    print(str(job_id) if parsable else f"Submitted batch job {job_id}")
    return 0


# --- the detached runner (`__run__`) ---


def _wait_until_runnable(state_dir: Path, job_id: int) -> bool:
    """Blocks while held or the dependency is not terminal. Returns False
    if the job was cancelled before it got a chance to run."""
    while True:
        job = read_job(state_dir, job_id)
        if job["state"] == "CANCELLED":
            return False
        dependency = job["dependency"]
        dep_ready = True
        if dependency is not None:
            try:
                dep_ready = read_job(state_dir, dependency)["state"] in (
                    TERMINAL_STATES
                )
            except FileNotFoundError:
                dep_ready = False
        if not job["held"] and dep_ready:
            return True
        time.sleep(_POLL_INTERVAL)


def _resolve_output(job: dict[str, Any], job_id: int) -> Path:
    raw = str(job["directives"].get("output", f"slurm-{job_id}.out"))
    raw = raw.replace("%j", str(job_id))
    path = Path(raw)
    return path if path.is_absolute() else Path(job["cwd"]) / path


def _build_env(
    bin_dir: Path, job: dict[str, Any], job_id: int
) -> dict[str, str]:
    env = dict(os.environ)
    directives = job["directives"]
    cpus = str(directives.get("cpus-per-task", "1"))
    env["SLURM_JOB_ID"] = str(job_id)
    env["SLURM_NTASKS"] = str(directives.get("ntasks", "1"))
    env["SLURM_CPUS_PER_TASK"] = cpus
    env["SLURM_JOB_NUM_NODES"] = str(directives.get("nodes", "1"))
    env["SLURM_CPUS_ON_NODE"] = cpus
    env["SLURM_RESTART_COUNT"] = str(job["restart_count"])
    env["SLURM_SUBMIT_DIR"] = str(job["submit_dir"])
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


_KILLING_STATES = frozenset({"TIMEOUT", "NODE_FAIL", "CANCELLED"})

# G2 decision (2026-10-01, operator): the `.batch` step's state/exit code
# for a forced kill, and the trailing `.extern` step (always COMPLETED,
# exit 0:0), mirroring real sacct's output for TIMEOUT/NODE_FAIL/an
# admin CANCELLED.
_BATCH_STEP_STATE = {
    "TIMEOUT": "CANCELLED",
    "NODE_FAIL": "NODE_FAIL",
    "CANCELLED": "CANCELLED",
}
_BATCH_STEP_EXIT_CODE = {
    "TIMEOUT": "0:15",
    "NODE_FAIL": "0:9",
    "CANCELLED": "0:15",
}


def _match_force(
    state_dir: Path, job: dict[str, Any], job_id: int
) -> dict[str, Any] | None:
    """Reads the force record the runner should apply, keyed by job id or
    else by the `model`/`finalize` phase matched against the `--output`
    basename prefix (Requirement 1 and 5 of ticket-023)."""
    forces = _read_forces(state_dir)
    record = forces.get(str(job_id))
    if record is not None:
        return record  # type: ignore[no-any-return]
    output_name = _resolve_output(job, job_id).name
    for phase in ("model", "finalize"):
        if output_name.startswith(phase) and phase in forces:
            return forces[phase]  # type: ignore[no-any-return]
    return None


def _wait_for_exit(
    state_dir: Path,
    job_id: int,
    proc: subprocess.Popen[bytes],
    pgid: int,
    force: dict[str, Any] | None,
) -> int:
    cancelled_at: float | None = None
    start = time.time()
    forced_killed = False
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        if read_job(state_dir, job_id)["state"] == "CANCELLED":
            if cancelled_at is None:
                cancelled_at = time.time()
            elif time.time() - cancelled_at >= _KILL_WAIT:
                with locked(state_dir), contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
        elif (
            force is not None
            and not forced_killed
            and force["state"] in _KILLING_STATES
            and time.time() - start >= force["after"]
        ):
            with locked(state_dir), contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            forced_killed = True
        time.sleep(_POLL_INTERVAL)


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _append_cancelled_line(path: str, job_id: int, step: int) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with Path(path).open("a") as out:
        out.write(
            f"slurmstepd: error: *** STEP {job_id}.{step} ON fake-node0"
            f" CANCELLED AT {stamp} ***\n"
        )


def _teardown(state_dir: Path, job_id: int, pgid: int, cancelled: bool) -> None:
    """Slurm's job-end step cleanup (ADR-087, T8b): batch leftovers get
    SIGKILL with no TERM, step tasks still alive get SIGCONT then SIGTERM,
    task 0's output gets one CANCELLED line per step, and survivors are
    SIGKILLed after `_KILL_WAIT`. Runs before `_finalize`, prints nothing."""
    if cancelled:
        deadline = time.time() + _KILL_WAIT
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(_POLL_INTERVAL)
    with locked(state_dir), contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    tasks: list[StepTask] = read_job(state_dir, job_id)["step_tasks"]
    live = [t for t in tasks if _alive(t["pid"])]
    for t in live:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(t["pid"], signal.SIGCONT)
            os.killpg(t["pid"], signal.SIGTERM)
    for step in sorted({t["step"] for t in live}):
        [output] = [
            t["output"] for t in tasks if t["step"] == step and t["task"] == 0
        ]
        if output is not None:
            _append_cancelled_line(output, job_id, step)
    start = time.time()
    killed = False
    while survivors := [t for t in live if _alive(t["pid"])]:
        elapsed = time.time() - start
        if elapsed >= 2 * _KILL_WAIT:
            break
        if elapsed >= _KILL_WAIT and not killed:
            for t in survivors:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(t["pid"], signal.SIGKILL)
            killed = True
        time.sleep(_POLL_INTERVAL)


def _finalize(
    rc: int, force: dict[str, Any] | None
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _mutate(job: dict[str, Any]) -> dict[str, Any] | None:
        if job["state"] in TERMINAL_STATES:
            return None
        if force is None:
            return {
                **job,
                "state": "COMPLETED" if rc == 0 else "FAILED",
                "exit_code": f"{rc}:0",
                "end_time": time.time(),
            }
        state = force["state"]
        exit_code = "0:15" if state == "TIMEOUT" else force["exit_code"]
        updated: dict[str, Any] = {
            **job,
            "state": "CANCELLED by 0" if state == "CANCELLED" else state,
            "exit_code": exit_code,
            "end_time": time.time(),
        }
        if force["reason"] is not None:
            updated["reason"] = force["reason"]
        if state in _KILLING_STATES:
            updated["steps"] = [
                *job["steps"],
                {
                    "id": f"{job['id']}.batch",
                    "state": _BATCH_STEP_STATE[state],
                    "exit_code": _BATCH_STEP_EXIT_CODE[state],
                },
                {
                    "id": f"{job['id']}.extern",
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                },
            ]
        elif force["step_oom"]:
            updated["steps"] = [
                *job["steps"],
                {
                    "id": f"{job['id']}.batch",
                    "state": "OUT_OF_MEMORY",
                    "exit_code": "0:9",
                },
            ]
        return updated

    return _mutate


def _fail_job(job: dict[str, Any]) -> dict[str, Any] | None:
    if job["state"] in TERMINAL_STATES:
        return None
    return {
        **job,
        "state": "FAILED",
        "exit_code": "1:0",
        "end_time": time.time(),
    }


def _run(state_dir: Path, job_id: int) -> int:
    bin_dir = state_dir.parent / "bin"
    update_job(state_dir, job_id, lambda j: {**j, "runner_pid": os.getpid()})
    try:
        if not _wait_until_runnable(state_dir, job_id):
            return 0
        job = read_job(state_dir, job_id)
        force = _match_force(state_dir, job, job_id)
        if job["configuring_seconds"]:
            update_job(
                state_dir, job_id, lambda j: {**j, "state": "CONFIGURING"}
            )
            time.sleep(job["configuring_seconds"])
        # The re-read, the CANCELLED check, and the spawn itself must all
        # happen under one `locked()` so a racing scancel (whose own
        # update_job() takes the same lock) can't slip in between the
        # check and the spawn: a cancel landing CONFIGURING either
        # finishes first (we see CANCELLED and never spawn) or finishes
        # after we have already spawned and recorded RUNNING, in which
        # case the existing RUNNING-kill path in _wait_for_exit applies.
        with locked(state_dir):
            job = read_job(state_dir, job_id)
            if job["state"] == "CANCELLED":
                return 0
            output_path = _resolve_output(job, job_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            env = _build_env(bin_dir, job, job_id)
            with output_path.open("a") as out:
                proc = subprocess.Popen(
                    ["bash", job["script"]],
                    cwd=job["cwd"],
                    env=env,
                    stdout=out,
                    stderr=out,
                    start_new_session=True,
                )
            pgid = os.getpgid(proc.pid)
            write_job_atomic(
                state_dir,
                job_id,
                {
                    **job,
                    "state": "RUNNING",
                    "reason": None,
                    "start_time": time.time(),
                    "pid": proc.pid,
                    "pgid": pgid,
                },
            )
        rc = _wait_for_exit(state_dir, job_id, proc, pgid, force)
        cancelled = read_job(state_dir, job_id)["state"] == "CANCELLED"
        _teardown(state_dir, job_id, pgid, cancelled)
        update_job(state_dir, job_id, _finalize(rc, force))
    except Exception:
        with contextlib.suppress(Exception):
            update_job(state_dir, job_id, _fail_job)
    return 0


# --- squeue ---


def _format_epoch(value: float | None) -> str:
    if value is None:
        return "N/A"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(value))


def _format_elapsed(job: dict[str, Any]) -> str:
    start = job["start_time"]
    if start is None:
        return "0:00"
    end = job["end_time"] or time.time()
    total = int(end - start)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


_FORMAT_CODES: dict[str, Callable[[dict[str, Any]], str]] = {
    "i": lambda j: str(j["id"]),
    "T": lambda j: str(j["state"]),
    "t": lambda j: _SHORT_STATE.get(j["state"], j["state"][:2]),
    "r": lambda j: str(j["reason"]) if j["reason"] is not None else "None",
    "l": lambda j: str(j["directives"].get("time", "UNLIMITED")),
    "S": lambda j: _format_epoch(j["start_time"]),
    "M": lambda j: _format_elapsed(j),
    "Z": lambda j: str(j["cwd"]),
}


def _render_format(fmt: str, job: dict[str, Any]) -> str:
    def _repl(match: re.Match[str]) -> str:
        fn = _FORMAT_CODES.get(match.group(1))
        return fn(job) if fn is not None else match.group(0)

    return re.sub(r"%(\w)", _repl, fmt)


def _squeue(args: list[str], state_dir: Path) -> int:
    ids: list[int] = []
    fmt = ""
    i = 0
    while i < len(args):
        token = args[i]
        if token in ("-h", "--noheader"):
            i += 1
        elif token == "-j":
            ids = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif token in ("-o", "--format"):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1
    if len(ids) == 1:
        try:
            job = read_job(state_dir, ids[0])
        except FileNotFoundError:
            print(
                "slurm_load_jobs error: Invalid job id specified",
                file=sys.stderr,
            )
            return 1
        print(_render_format(fmt, job))
        return 0
    for job_id in ids:
        try:
            job = read_job(state_dir, job_id)
        except FileNotFoundError:
            continue
        print(_render_format(fmt, job))
    return 0


# --- scancel ---


def _cancel_mutator(
    uid: int,
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    def _mutate(job: dict[str, Any]) -> dict[str, Any] | None:
        if job["state"] not in ("PENDING", "CONFIGURING", "RUNNING"):
            return None
        return {
            **job,
            "state": "CANCELLED",
            "reason": f"cancelled by {uid}",
            "end_time": time.time(),
        }

    return _mutate


def _scancel(args: list[str], state_dir: Path) -> int:
    uid = os.getuid()
    exit_code = 0
    for raw_id in args:
        job_id = int(raw_id)
        # Mirrors real Slurm: slurmctld records CANCELLED *before* the
        # signal goes out, so a job that exits (or completes) in the
        # window between the two always ends CANCELLED, never
        # COMPLETED/FAILED. `update_job` returns the pre-mutation job,
        # read under the same lock that just wrote the new state, so
        # the "was it RUNNING, and what was its pgid" check below is
        # the one atomic snapshot that preceded the write -- not a
        # separate, pre-lock read that could race the mutation.
        try:
            before = update_job(state_dir, job_id, _cancel_mutator(uid))
        except FileNotFoundError:
            print(
                "scancel: error: Kill job error on job id "
                f"{job_id}: Invalid job id specified",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        if before["state"] == "RUNNING":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(before["pgid"], signal.SIGTERM)
    return exit_code


# --- sacct ---

_SACCT_FIELDS: dict[str, Callable[[dict[str, Any]], str]] = {
    "JobID": lambda r: str(r["id"]),
    "JobIDRaw": lambda r: str(r["id"]),
    "State": lambda r: str(r["state"]),
    "ExitCode": lambda r: (
        str(r["exit_code"]) if r["exit_code"] is not None else "0:0"
    ),
    "Elapsed": _format_elapsed,
    "Timelimit": lambda r: (
        "" if r["is_step"] else str(r["directives"].get("time", "UNLIMITED"))
    ),
    "Start": lambda r: _format_epoch(r["start_time"]),
    "End": lambda r: _format_epoch(r["end_time"]),
    "MaxRSS": lambda r: str(r.get("max_rss", "")),
    "NodeList": lambda r: str(r["directives"].get("nodelist", "fake-node1")),
}


def _sacct_job_row(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(job["id"]),
        "state": job["state"],
        "exit_code": job["exit_code"],
        "start_time": job["start_time"],
        "end_time": job["end_time"],
        "directives": job["directives"],
        "is_step": False,
    }


def _sacct_step_row(
    job: dict[str, Any], step: dict[str, Any]
) -> dict[str, Any]:
    return {
        **_sacct_job_row(job),
        "id": step["id"],
        "state": step["state"],
        "exit_code": step["exit_code"],
        "is_step": True,
    }


def _parse_sacct_args(args: list[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    fmt: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "-j":
            ids = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif token.startswith("--format="):
            fmt = token[len("--format=") :].split(",")
            i += 1
        elif token == "--format":
            fmt = args[i + 1].split(",")
            i += 2
        else:
            i += 1
    return ids, fmt


def _sacct(args: list[str], state_dir: Path) -> int:
    ids, fmt = _parse_sacct_args(args)
    for field in fmt:
        if field not in _SACCT_FIELDS:
            print(
                f'sacct: error: Invalid field requested: "{field}"',
                file=sys.stderr,
            )
            return 1
    if _read_accounting(state_dir).get("disabled"):
        # Prod-faithful: no `sacct: error: ` prefix (cluster fact F1).
        print("Slurm accounting storage is disabled", file=sys.stderr)
        return 1
    if _consume_lag(state_dir):
        return 0
    for job_id in ids:
        try:
            job = read_job(state_dir, job_id)
        except FileNotFoundError:
            continue
        print("|".join(_SACCT_FIELDS[f](_sacct_job_row(job)) for f in fmt))
        for step in job["steps"]:
            row = _sacct_step_row(job, step)
            print("|".join(_SACCT_FIELDS[f](row) for f in fmt))
    return 0


# --- scontrol ---


def _requeue(state_dir: Path, job_id: int) -> int:
    with locked(state_dir):
        try:
            job = read_job(state_dir, job_id)
        except FileNotFoundError:
            print(
                "slurm_load_jobs error: Invalid job id specified",
                file=sys.stderr,
            )
            return 1
        if job["state"] not in TERMINAL_STATES:
            print(
                "scontrol: error: fake-slurm requeue needs a finished job",
                file=sys.stderr,
            )
            return 1
        write_job_atomic(
            state_dir,
            job_id,
            {
                **job,
                "state": "PENDING",
                "reason": None,
                "pid": None,
                "pgid": None,
                "runner_pid": None,
                "start_time": None,
                "end_time": None,
                "exit_code": None,
                "steps": [],
                "step_tasks": [],
                "restart_count": job["restart_count"] + 1,
            },
        )
    _spawn_runner(job_id)
    return 0


def _scontrol(args: list[str], state_dir: Path) -> int:
    if args[:1] == ["requeue"]:
        return _requeue(state_dir, int(args[1]))
    job_id = int(args[-1])
    try:
        job = read_job(state_dir, job_id)
    except FileNotFoundError:
        print(
            "slurm_load_jobs error: Invalid job id specified", file=sys.stderr
        )
        return 1
    reason = job["reason"] if job["reason"] is not None else "None"
    exit_code = job["exit_code"] if job["exit_code"] is not None else "0:0"
    time_limit = str(job["directives"].get("time", "UNLIMITED"))
    print(
        f"JobId={job['id']} JobState={job['state']} Reason={reason} "
        f"ExitCode={exit_code} TimeLimit={time_limit} "
        f"StartTime={_format_epoch(job['start_time'])} "
        f"EndTime={_format_epoch(job['end_time'])} WorkDir={job['cwd']}"
    )
    return 0


# --- mpiexec / srun ---


def _parse_launcher_args(
    args: list[str], flags: frozenset[str]
) -> tuple[dict[str, str], list[str]]:
    """Splits launcher options from the wrapped command. A token in
    `flags` is a value-less option; any other `-`-prefixed token with `=`
    is self-contained, and one without `=` consumes the next token as its
    value. The first non-option token starts the command. Option keys are
    the token's name without its leading dashes; a later duplicate
    overrides an earlier one."""
    options: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("-"):
            return options, args[i:]
        name, sep, value = token.lstrip("-").partition("=")
        if token in flags:
            options[name] = ""
            i += 1
        elif sep:
            options[name] = value
            i += 1
        else:
            options[name] = args[i + 1] if i + 1 < len(args) else ""
            i += 2
    return options, []


def _mpiexec(args: list[str], state_dir: Path) -> int:
    _, cmd = _parse_launcher_args(args, frozenset())
    os.execvp(cmd[0], cmd)


def _parse_ntasks(raw: str) -> int | None:
    if raw.isascii() and raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return None


def _expand_output(pattern: str, job_id: int, task: int) -> str:
    values = {"%j": str(job_id), "%t": str(task)}
    return re.sub(r"%[jt]", lambda m: values[m.group()], pattern)


def _task_env(env: dict[str, str], task: int, ntasks: int) -> dict[str, str]:
    return {
        **env,
        "SLURMD_NODENAME": f"fake-node{task}",
        "SLURM_NODEID": str(task),
        "SLURM_PROCID": str(task),
        "SLURM_NNODES": str(ntasks),
    }


def _step_status(returncodes: list[int]) -> int:
    return max(128 - rc if rc < 0 else rc for rc in returncodes)


def _abort_step(
    procs: list[subprocess.Popen[bytes]],
    output: str | None,
    job_id: int,
    step: int,
) -> int:
    """The srun client's reaction to SIGTERM (probe T8): it turns it into a
    KILL of the step, so no task sees a TERM."""
    print("srun: forcing job termination", file=sys.stderr)
    print(
        f"srun: Job step aborted: Waiting up to {int(_KILL_WAIT) + 2}"
        " seconds for job step to finish.",
        file=sys.stderr,
    )
    for proc in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    killed = [
        task
        for task, proc in enumerate(procs)
        if proc.wait() == -signal.SIGKILL
    ]
    for task in reversed(killed):
        print(
            f"srun: error: fake-node{task}: task {task}: Killed",
            file=sys.stderr,
        )
    if killed and output is not None:
        _append_cancelled_line(output, job_id, step)
    return 128 + signal.SIGKILL


def _wait_step(
    procs: list[subprocess.Popen[bytes]],
    received: list[int],
    output: str | None,
    job_id: int,
    step: int,
) -> int:
    handled = 0
    while True:
        pending = received[handled:]
        handled += len(pending)
        if signal.SIGTERM in pending:
            return _abort_step(procs, output, job_id, step)
        for _ in pending:
            for proc in procs:
                if proc.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGINT)
        if all(proc.poll() is not None for proc in procs):
            return _step_status([proc.wait() for proc in procs])
        time.sleep(_POLL_INTERVAL)


def _run_step(state_dir: Path, options: dict[str, str], cmd: list[str]) -> int:
    ntasks = _parse_ntasks(options["ntasks"])
    if ntasks is None:
        print(
            "srun: error: --ntasks must be a positive integer, got "
            f"'{options['ntasks']}'",
            file=sys.stderr,
        )
        return 1
    raw_job_id = os.environ.get("SLURM_JOB_ID")
    if not raw_job_id:
        print(
            "srun: error: --ntasks needs SLURM_JOB_ID; run srun inside a "
            "fake-Slurm job",
            file=sys.stderr,
        )
        return 1
    job_id = int(raw_job_id)
    if (state_dir / FAIL_STEP_START_FLAG).exists():
        print(
            f"srun: error: Unable to create step for job {job_id}: "
            "fake-slurm step-start failure",
            file=sys.stderr,
        )
        return 1
    received: list[int] = []

    def _record(signum: int, _frame: FrameType | None) -> None:
        received.append(signum)

    # Before any spawn: an INT inherited as ignored from a `&` start would
    # otherwise survive `exec` into the tasks.
    signal.signal(signal.SIGTERM, _record)
    signal.signal(signal.SIGINT, _record)
    pattern = options.get("output")
    paths = (
        [_expand_output(pattern, job_id, task) for task in range(ntasks)]
        if pattern is not None
        else []
    )
    procs: list[subprocess.Popen[bytes]] = []
    try:
        with locked(state_dir), contextlib.ExitStack() as stack:
            fds: dict[str, int] = {}
            for path in dict.fromkeys(paths):
                fds[path] = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND,
                    0o666,
                )
                stack.callback(os.close, fds[path])
            for task in range(ntasks):
                fd = fds[paths[task]] if paths else None
                procs.append(
                    subprocess.Popen(
                        cmd,
                        env=_task_env(dict(os.environ), task, ntasks),
                        stdin=subprocess.DEVNULL,
                        stdout=fd,
                        stderr=fd,
                        start_new_session=True,
                    )
                )
            job = read_job(state_dir, job_id)
            step = len({t["step"] for t in job["step_tasks"]})
            records: list[StepTask] = [
                {
                    "pid": proc.pid,
                    "task": task,
                    "step": step,
                    "output": paths[task] if paths else None,
                }
                for task, proc in enumerate(procs)
            ]
            write_job_atomic(
                state_dir,
                job_id,
                {**job, "step_tasks": [*job["step_tasks"], *records]},
            )
    except OSError:
        for proc in procs:
            os.killpg(proc.pid, signal.SIGKILL)
        for proc in procs:
            proc.wait()
        raise
    return _wait_step(procs, received, records[0]["output"], job_id, step)


def _srun(args: list[str], state_dir: Path) -> int:
    options, cmd = _parse_launcher_args(args, _SRUN_FLAGS)
    if "ntasks" in options:
        return _run_step(state_dir, options, cmd)
    env = dict(os.environ)
    env["SLURM_PROCID"] = "0"
    env["SLURM_NNODES"] = env.get("SLURM_JOB_NUM_NODES", "1")
    os.execvpe(cmd[0], cmd, env)


# --- dispatch ---

_DISPATCH: dict[str, Callable[[list[str], Path], int]] = {
    "sbatch": _sbatch,
    "squeue": _squeue,
    "scancel": _scancel,
    "sacct": _sacct,
    "scontrol": _scontrol,
    "mpiexec": _mpiexec,
    "srun": _srun,
}


def _state_dir() -> Path:
    return Path(os.environ["FAKE_SLURM_STATE_DIR"])


def main() -> int:
    argv = sys.argv
    state_dir = _state_dir()
    if len(argv) >= 3 and argv[1] == "__run__":
        log_invocation(state_dir, "__run__", argv)
        return _run(state_dir, int(argv[2]))
    command = Path(argv[0]).name
    log_invocation(state_dir, command, argv)
    handler = _DISPATCH.get(command)
    if handler is None:
        print(f"fake-slurm: unknown command {command!r}", file=sys.stderr)
        return 1
    return handler(argv[1:], state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
