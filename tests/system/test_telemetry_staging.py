"""Job-telemetry ticket-039 (ADR-069, ADR-079, R36, R37, R62): the fake-Slurm
proof of the sidecar's staging files. A rendered 2-node model script and a
rendered finalize script start the real sampler through the fake ``srun``
step path, and each attempt leaves one ``<phase>-<job>-r<restart>-<node>``
staging file per node: a header, and an off-grid final sample taken on the
job-end SIGTERM (R36). The per-task and ``srun`` logs hold nothing but the
teardown's ``CANCELLED`` line. A requeued attempt writes new ``r1`` files
and leaves the ``r0`` files untouched (``O_EXCL``, R62).

The stubs that stand in for the model and the CLI wait for the staging files
instead of sleeping, so the sampler's start-up never races the batch exit.
Every test has its own workspace, because ``O_EXCL`` refuses a second writer
of the same name.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

import pytest

from hpc_model_utils.core.launch import (
    FinalizeInvocation,
    Launcher,
    LaunchSpec,
    Resources,
    TelemetryConfig,
    Toolchain,
    write_finalize_script,
    write_model_script,
)
from hpc_model_utils.core.workspace import Workspace
from tests.support.cli_shim import write_cli_shim, write_python_sibling
from tests.support.fake_slurm import FakeSlurm
from tests.support.fake_slurm.cli import sbatch, scontrol

pytestmark = pytest.mark.slow

_TELEMETRY = TelemetryConfig(
    enabled=True, interval=30, workspace_interval=60, tool_version="9.9.9"
)
_GRID_MS = _TELEMETRY.interval * 1000
_HEADER_KIND = "hpcmu.telemetry.header"


def _write_waiting_stub(
    path: Path, phase: str, nodes: int, ws: Workspace
) -> None:
    path.write_text(
        "#!/bin/bash\n"
        f"WS={shlex.quote(str(ws.root))}\n"
        "for _ in $(seq 200); do\n"
        "  ready=1\n"
        f"  for i in $(seq 0 {nodes - 1}); do\n"
        '    [ -s "$WS/.hpcmu/telemetry/'
        f'{phase}-$SLURM_JOB_ID-r${{SLURM_RESTART_COUNT}}-$i.ndjson"'
        " ] || ready=0\n"
        "  done\n"
        '  [ "$ready" = 1 ] && exit 0\n'
        "  sleep 0.05\n"
        "done\n"
        "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "study-staging"
    root.mkdir()
    return Workspace.at(root)


def _toolchain(fake_slurm: FakeSlurm, tmp_path: Path) -> Toolchain:
    return Toolchain(
        mpich_bin=tmp_path / "mpich" / "bin",
        slurm_bin=fake_slurm.bin_dir,
        cli_bin=write_cli_shim(tmp_path),
    )


def _submit(script: Path, tmp_path: Path) -> int:
    result = sbatch("--parsable", str(script), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip())


def _run_model_job(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> tuple[Workspace, int]:
    ws = _make_workspace(tmp_path)
    stub = tmp_path / "model-stub"
    _write_waiting_stub(stub, "model", 2, ws)
    script = write_model_script(
        ws,
        LaunchSpec(Launcher.SRUN_PMIX, (str(stub),), nodes=2),
        Resources(queue="batch", cores=2),
        _toolchain(fake_slurm, tmp_path),
        telemetry=_TELEMETRY,
    )
    job_id = _submit(script, tmp_path)
    fake_slurm.wait_idle()
    return ws, job_id


def _run_finalize_job(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> tuple[Workspace, int]:
    ws = _make_workspace(tmp_path)
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    write_python_sibling(cli_dir)
    _write_waiting_stub(cli_dir / "hpcmu", "finalize", 1, ws)
    script = write_finalize_script(
        ws,
        Resources(queue="batch", cores=1),
        replace(_toolchain(fake_slurm, tmp_path), cli_bin=cli_dir / "hpcmu"),
        FinalizeInvocation("fake", None, 1, None),
        telemetry=_TELEMETRY,
    )
    job_id = _submit(script, tmp_path)
    fake_slurm.wait_idle()
    return ws, job_id


def _parse(line: str) -> dict[str, object]:
    record = json.loads(line)
    assert isinstance(record, dict)
    return record


def _assert_attempt(
    ws: Workspace, phase: str, job_id: int, restart: int, nodes: int
) -> None:
    stem = f"{phase}-{job_id}-r{restart}"
    for node in range(nodes):
        staged = ws.telemetry_dir / f"{stem}-{node}.ndjson"
        assert staged.is_file(), f"missing staging file {staged}"
        lines = staged.read_text(encoding="ascii").splitlines()
        header = _parse(lines[0])
        assert header["kind"] == _HEADER_KIND
        assert header["job_id"] == str(job_id)
        assert header["restart"] == restart
        assert header["node_index"] == node
        samples = [
            record for record in map(_parse, lines[1:]) if "kind" not in record
        ]
        assert samples, f"no sample line in {staged}"
        stop_ts = samples[-1]["ts"]
        assert isinstance(stop_ts, int)
        assert stop_ts % _GRID_MS != 0, f"final sample on the grid in {staged}"

    cancelled = re.compile(
        rf"slurmstepd: error: \*\*\* STEP {job_id}\.\d+ ON fake-node0"
        r" CANCELLED AT \S+ \*\*\*\n"
    )
    task0_log = ws.telemetry_dir / f"{stem}-0.log"
    assert cancelled.fullmatch(task0_log.read_text(encoding="utf-8"))
    quiet = [
        ws.telemetry_dir / f"{stem}-{node}.log" for node in range(1, nodes)
    ]
    quiet.append(ws.telemetry_dir / f"{stem}-srun.log")
    for log in quiet:
        assert log.read_text(encoding="utf-8") == "", log
    for log in ws.telemetry_dir.glob("*.log"):
        text = log.read_text(encoding="utf-8")
        assert "hpcmu-telemetry:" not in text, log
        assert "Traceback" not in text, log


def _digests(ws: Workspace, pattern: str) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ws.telemetry_dir.glob(pattern))
    }


@pytest.mark.timeout(90)
def test_staging_model_two_nodes_writes_one_file_per_node(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws, job_id = _run_model_job(fake_slurm, tmp_path)

    _assert_attempt(ws, "model", job_id, 0, 2)


@pytest.mark.timeout(90)
def test_staging_finalize_writes_its_own_file(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws, job_id = _run_finalize_job(fake_slurm, tmp_path)

    _assert_attempt(ws, "finalize", job_id, 0, 1)


@pytest.mark.timeout(90)
def test_staging_requeue_writes_new_attempt_files(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    ws, job_id = _run_model_job(fake_slurm, tmp_path)
    before = _digests(ws, f"model-{job_id}-r0-*")
    assert before

    result = scontrol("requeue", str(job_id))
    assert result.returncode == 0, result.stderr
    fake_slurm.wait_idle()

    _assert_attempt(ws, "model", job_id, 1, 2)
    assert _digests(ws, f"model-{job_id}-r0-*") == before
