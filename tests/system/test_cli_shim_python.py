"""ADR-081 / job-telemetry ticket-036: the ``python`` sibling the shim
CLIs gain, so ``Toolchain.python_bin`` (``cli_bin.parent / "python"``)
exists under fake Slurm. The sibling is an exec wrapper around
``sys.executable``; these tests run it to prove it keeps the venv, the
argv and the environment, and starts the sampler's entry point.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import pytest

from hpc_model_utils.core.launch import Toolchain
from tests.support.cli_shim import write_cli_shim, write_python_sibling

_PROBE = """
import importlib.util
import json
import os
import sys

print(
    json.dumps(
        {
            "prefix": sys.prefix,
            "isolated": sys.flags.isolated,
            "argv": sys.argv[1:],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "probe_var": os.environ.get("HPCMU_PROBE_VAR"),
            "path": sys.path,
            "importable": importlib.util.find_spec("hpc_model_utils") is not None,
        }
    )
)
"""


class _ProbeReport(TypedDict):
    prefix: str
    isolated: int
    argv: list[str]
    slurm_job_id: str | None
    probe_var: str | None
    path: list[str]
    importable: bool


@pytest.fixture
def probe_report(tmp_path: Path) -> _ProbeReport:
    sibling = write_python_sibling(tmp_path)
    result = subprocess.run(
        [str(sibling), "-I", "-c", _PROBE, "a b", "c"],
        env={
            **os.environ,
            "SLURM_JOB_ID": "4242",
            "HPCMU_PROBE_VAR": "kept",
            "PYTHONPATH": "/nonexistent-probe",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    report: _ProbeReport = json.loads(result.stdout)
    return report


def test_write_cli_shim_python_sibling_is_regular_executable_with_exact_text(
    tmp_path: Path,
) -> None:
    write_cli_shim(tmp_path)

    sibling = tmp_path / "python"
    assert sibling.is_file()
    assert not sibling.is_symlink()
    assert sibling.stat().st_mode & 0o111 == 0o111
    assert sibling.read_text() == (
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )


def test_write_cli_shim_python_sibling_is_the_toolchain_python_bin(
    tmp_path: Path,
) -> None:
    shim = write_cli_shim(tmp_path)

    assert (
        Toolchain(Path("/m"), Path("/s"), shim).python_bin
        == tmp_path / "python"
    )


def test_write_python_sibling_second_call_returns_same_path_and_text(
    tmp_path: Path,
) -> None:
    first = write_python_sibling(tmp_path)
    text = first.read_text()

    second = write_python_sibling(tmp_path)

    assert second == first == tmp_path / "python"
    assert second.read_text() == text


def test_write_python_sibling_isolated_run_keeps_venv_prefix(
    probe_report: _ProbeReport,
) -> None:
    assert probe_report["prefix"] == sys.prefix
    assert probe_report["isolated"] == 1
    assert probe_report["importable"]


def test_write_python_sibling_isolated_run_keeps_argv_arguments_whole(
    probe_report: _ProbeReport,
) -> None:
    assert probe_report["argv"] == ["a b", "c"]


def test_write_python_sibling_isolated_run_passes_environment_not_pythonpath(
    probe_report: _ProbeReport,
) -> None:
    assert probe_report["slurm_job_id"] == "4242"
    assert probe_report["probe_var"] == "kept"
    assert "/nonexistent-probe" not in probe_report["path"]


def test_write_python_sibling_sampler_help_prints_usage(tmp_path: Path) -> None:
    sibling = write_python_sibling(tmp_path)

    result = subprocess.run(
        [
            str(sibling),
            "-I",
            "-m",
            "hpc_model_utils.infra.telemetry.sampler",
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: hpcmu-telemetry")
