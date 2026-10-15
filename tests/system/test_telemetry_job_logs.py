"""Job-telemetry ticket-038 (R80, R9, ADR-094, ADR-061): the fake-Slurm proof
of K2. With telemetry on, the model and finalize job logs are the logs of a
telemetry-off run, apart from the strict comparator's closed set of volatile
tokens (``deploy/modelops/compare_job_logs.py``, ticket-037, is the
definition of "same").

"Every plugin" is read as every ``Launcher`` a plugin reaches: ``DIRECT``,
``MPIEXEC_HYDRA`` (NEWAVE, DECOMP) and ``SRUN_PMIX`` (cobre). The sidecar
line is the only on/off difference in a rendered script and depends only on
the Launcher's body seam, so one fake plugin per Launcher covers every
rendered body shape. The real plugin logs are ticket-059's (prd mode).

Each test runs an off arm and an on arm of the same job under its own fake
Slurm (``fake_slurm_pair``), then compares the two model logs and the two
finalize logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from deploy.modelops.compare_job_logs import main as compare_job_logs
from hpc_model_utils.core.diagnosis import RunStatus
from hpc_model_utils.core.launch import (
    TELEMETRY_OFF,
    Launcher,
    LaunchSpec,
    Resources,
    TelemetryConfig,
    Toolchain,
)
from hpc_model_utils.core.lifecycle.finalize import legacy_synthesis_bin
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import StateStore, current_tool
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.slurm import Slurm
from hpc_model_utils.platform.modelops import Reporter
from tests.support.cli_shim import write_cli_shim
from tests.support.fake_plugin import FakePlugin, install_fake_model
from tests.support.fake_slurm import FakeSlurm

pytestmark = pytest.mark.slow

_TELEMETRY_ON = replace(
    TELEMETRY_OFF, enabled=True, tool_version=current_tool().version
)
_CRASHING_SAMPLER = """\
#!/bin/sh
echo 'Traceback (most recent call last):' >&2
echo 'RuntimeError: injected sampler crash' >&2
exit 1
"""
_CANCELLED_TASK_LOG = re.compile(
    r"^slurmstepd: error: \*\*\* STEP 1000\.\d+ ON fake-node0"
    r" CANCELLED AT \S+ \*\*\*\n$"
)
_STEP_START_FAILURE = (
    "srun: error: Unable to create step for job 1000:"
    " fake-slurm step-start failure"
)
_INJECTED_LINE = "srun: error: fake-node0: task 0: Killed\n"
_SETTINGS = EngineSettings(
    poll_interval=0.05,
    settle_window=0.2,
    outcome_attempts=3,
    outcome_backoff=0.1,
    cancel_timeout=5.0,
)


@pytest.fixture(autouse=True)
def _chdir_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Small module-level helpers, copied and adapted from
# tests/system/test_engine_end_to_end.py (its helpers are module-private).
# ---------------------------------------------------------------------------


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _make_workspace(arm_dir: Path) -> Workspace:
    root = arm_dir / "study-fake"
    root.mkdir()
    return Workspace.at(root)


def _toolchain(arm_dir: Path, bin_dir: Path) -> Toolchain:
    return Toolchain(
        cli_bin=write_cli_shim(arm_dir),
        mpich_bin=arm_dir / "mpich" / "bin",
        slurm_bin=bin_dir,
    )


def _write_legacy_sintetizador(ws: Workspace) -> Path:
    # The leading sleep keeps the finalize job alive past the sampler's
    # start-up, so the job-end teardown reaches a live step.
    stub = legacy_synthesis_bin(ws, FakePlugin())
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        "#!/bin/bash\nsleep 1\nmkdir -p sintese\n: > sintese/x.parquet\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


# ---------------------------------------------------------------------------
# The plugin variants: only `launch` differs, the name stays "fake" so the
# finalize job resolves the base plugin through the CLI shim's registry.
# ---------------------------------------------------------------------------


class _HydraPlugin(FakePlugin):
    def launch(self, ws: Workspace, res: Resources) -> LaunchSpec:
        return LaunchSpec(
            Launcher.MPIEXEC_HYDRA,
            (str(ws.assets / self.executables.entrypoint),),
            ntasks=2,
        )


class _PmixPlugin(FakePlugin):
    def launch(self, ws: Workspace, res: Resources) -> LaunchSpec:
        return LaunchSpec(
            Launcher.SRUN_PMIX,
            (str(ws.assets / self.executables.entrypoint),),
            nodes=2,
        )


@dataclass(frozen=True)
class _Arm:
    ws: Workspace
    status: RunStatus
    model_log: Path
    finalize_log: Path


def _run_arm(
    arm_dir: Path,
    controller: FakeSlurm,
    plugin: FakePlugin,
    telemetry: TelemetryConfig,
    *,
    sibling: str | None = None,
) -> _Arm:
    """``sibling`` replaces the ``python`` the toolchain writes, after
    ``_toolchain`` has created it and before ``run`` submits."""
    ws = _make_workspace(arm_dir)
    install_fake_model(ws, sleep=1.0)
    _write_legacy_sintetizador(ws)
    tools = _toolchain(arm_dir, controller.bin_dir)
    if sibling is not None:
        tools.python_bin.write_text(sibling, encoding="utf-8")
    state = run(
        ws,
        plugin,
        Slurm(controller.bin_dir),
        Reporter(_ListChannel(), enabled=True),
        StateStore(ws),
        SubmitRequest(
            resources=Resources(queue="batch", cores=2),
            tools=tools,
            skip_model=False,
            synthesis_bin=None,
            telemetry=telemetry,
        ),
        JobLedger(),
        settings=_SETTINGS,
        emit=lambda _line: None,
    )
    controller.wait_idle()
    assert state.diagnosis is not None
    return _Arm(
        ws,
        state.diagnosis.status,
        ws.log_path(Phase.MODEL, "1000"),
        ws.log_path(Phase.FINALIZE, "1001"),
    )


def _compare(off: _Arm, on: _Arm, off_log: Path, on_log: Path) -> int:
    return compare_job_logs(
        [
            "--mode",
            "strict",
            "--off-root",
            str(off.ws.root),
            "--on-root",
            str(on.ws.root),
            str(off_log),
            str(on_log),
        ]
    )


def _same(off: _Arm, on: _Arm, log: Literal["model", "finalize"]) -> int:
    if log == "model":
        return _compare(off, on, off.model_log, on.model_log)
    return _compare(off, on, off.finalize_log, on.finalize_log)


def _ndjson(arm: _Arm) -> list[Path]:
    return sorted(arm.ws.telemetry_dir.glob("*.ndjson"))


# ---------------------------------------------------------------------------
# R80: the normal path, every Launcher.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("plugin", "model_nodes"),
    [
        pytest.param(FakePlugin(), ("0",), id="direct"),
        pytest.param(_HydraPlugin(), ("0",), id="hydra"),
        pytest.param(_PmixPlugin(), ("0", "1"), id="pmix"),
    ],
)
def test_job_logs_normal_match(
    fake_slurm_pair: tuple[FakeSlurm, FakeSlurm],
    tmp_path: Path,
    plugin: FakePlugin,
    model_nodes: tuple[str, ...],
) -> None:
    off = _run_arm(tmp_path / "off", fake_slurm_pair[0], plugin, TELEMETRY_OFF)
    on = _run_arm(tmp_path / "on", fake_slurm_pair[1], plugin, _TELEMETRY_ON)

    assert _same(off, on, "model") == 0
    assert _same(off, on, "finalize") == 0
    assert off.status is on.status is RunStatus.SUCCESS
    staged = on.ws.telemetry_dir
    for node in model_nodes:
        assert (staged / f"model-1000-r0-{node}.ndjson").is_file()
    assert (staged / "finalize-1001-r0-0.ndjson").is_file()
    assert _CANCELLED_TASK_LOG.fullmatch(
        (staged / "model-1000-r0-0.log").read_text(encoding="utf-8")
    )
    assert not list(off.ws.telemetry_dir.glob("*"))


# ---------------------------------------------------------------------------
# R80: a sampler that crashes at start.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_job_logs_crash_at_start_match(
    fake_slurm_pair: tuple[FakeSlurm, FakeSlurm], tmp_path: Path
) -> None:
    plugin = _PmixPlugin()
    off = _run_arm(tmp_path / "off", fake_slurm_pair[0], plugin, TELEMETRY_OFF)
    on = _run_arm(
        tmp_path / "on",
        fake_slurm_pair[1],
        plugin,
        _TELEMETRY_ON,
        sibling=_CRASHING_SAMPLER,
    )

    assert _same(off, on, "model") == 0
    assert _same(off, on, "finalize") == 0
    assert off.status is on.status
    task_log = on.ws.telemetry_dir / "model-1000-r0-0.log"
    assert "Traceback (most recent call last):" in task_log.read_text(
        encoding="utf-8"
    )
    assert _ndjson(on) == []


# ---------------------------------------------------------------------------
# R88: a step that cannot start.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_job_logs_step_start_failure_match(
    fake_slurm_pair: tuple[FakeSlurm, FakeSlurm], tmp_path: Path
) -> None:
    plugin = _PmixPlugin()
    fake_slurm_pair[1].fail_step_start()
    off = _run_arm(tmp_path / "off", fake_slurm_pair[0], plugin, TELEMETRY_OFF)
    on = _run_arm(tmp_path / "on", fake_slurm_pair[1], plugin, _TELEMETRY_ON)

    assert _same(off, on, "model") == 0
    assert _same(off, on, "finalize") == 0
    assert off.status is on.status
    srun_log = on.ws.telemetry_dir / "model-1000-r0-srun.log"
    assert _STEP_START_FAILURE in srun_log.read_text(encoding="utf-8")
    assert _ndjson(on) == []


# ---------------------------------------------------------------------------
# The controls: off vs off passes, an injected `srun:` line fails.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_job_logs_control_off_vs_off_and_injected_line(
    fake_slurm_pair: tuple[FakeSlurm, FakeSlurm], tmp_path: Path
) -> None:
    plugin = FakePlugin()
    first = _run_arm(
        tmp_path / "off", fake_slurm_pair[0], plugin, TELEMETRY_OFF
    )
    second = _run_arm(
        tmp_path / "on", fake_slurm_pair[1], plugin, TELEMETRY_OFF
    )

    assert _same(first, second, "model") == 0
    assert _same(first, second, "finalize") == 0
    injected = tmp_path / "model-injected.out"
    injected.write_text(
        second.model_log.read_text(encoding="utf-8") + _INJECTED_LINE,
        encoding="utf-8",
    )
    assert _compare(first, second, first.model_log, injected) == 1
