"""R84/ADR-078: running the sampler loads no third-party module."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

PACKAGE = "hpc_model_utils"
TELEMETRY_PACKAGE = f"{PACKAGE}.infra.telemetry"
PACKAGE_CHAIN = frozenset({PACKAGE, f"{PACKAGE}.infra", TELEMETRY_PACKAGE})
FORBIDDEN_PACKAGES = frozenset(
    {
        "pyarrow",
        "click",
        "boto3",
        "botocore",
        "pandas",
        "inewave",
        "idecomp",
        "cfinterface",
    }
)

SCRIPT = f"""\
import json
import runpy
import sys

before = set(sys.modules)
sys.argv = ["sampler", "--help"]
try:
    runpy.run_module("{TELEMETRY_PACKAGE}.sampler", run_name="__main__")
except SystemExit:
    pass
print(json.dumps(sorted(set(sys.modules) - before)))
"""


@pytest.fixture(scope="module")
def loaded_modules() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    try:
        modules: list[str] = json.loads(proc.stdout.strip().rpartition("\n")[2])
    except json.JSONDecodeError:
        pytest.fail(f"unparsable output {proc.stdout!r}\n{proc.stderr}")
    return modules


def test_sampler_module_loads_only_stdlib(loaded_modules: list[str]) -> None:
    allowed_top_level = sys.stdlib_module_names | {PACKAGE}

    third_party = [
        name
        for name in loaded_modules
        if name.partition(".")[0] not in allowed_top_level
    ]
    stray_package_modules = [
        name
        for name in loaded_modules
        if name.partition(".")[0] == PACKAGE
        and name not in PACKAGE_CHAIN
        and not name.startswith(f"{TELEMETRY_PACKAGE}.")
    ]

    assert third_party == []
    assert stray_package_modules == []


def test_sampler_module_loads_no_forbidden_package(
    loaded_modules: list[str],
) -> None:
    top_level = {name.partition(".")[0] for name in loaded_modules}

    assert sorted(top_level & FORBIDDEN_PACKAGES) == []
