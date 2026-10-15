"""ADR-077/ADR-095/R77: the frozen telemetry hook-key contract (ticket-050).

The 19 headline hook keys are a one-way door. FROZEN_KEYS is written out
literally and is never derived from HEADLINE, so a change to the headline
table fails here until the literal is edited too. This module changes no
source: a failing assertion is a defect in schema.py, core/telemetry.py or
platform/modelops.py, never a reason to relax the literal.
"""

from __future__ import annotations

import math
import re
import sys
from typing import Final

import pytest

from hpc_model_utils.core.telemetry import headline_items
from hpc_model_utils.core.workspace import Phase
from hpc_model_utils.infra.telemetry.schema import (
    HEADLINE,
    HEADLINE_KEY_PREFIX,
    PHASES,
    WORKSPACE_SCOPE,
    format_headline_value,
)
from hpc_model_utils.platform.encoding import METADATA_VALUE_CAP
from hpc_model_utils.platform.modelops import Reporter
from tests.support.hooks import Hook, parse_hooks

FROZEN_KEYS: Final = (
    "telemetry.model.job_mem_peak_bytes_max",
    "telemetry.model.mem_available_bytes_min",
    "telemetry.model.cpu_busy_cores_mean",
    "telemetry.model.cpu_busy_cores_p95",
    "telemetry.model.cpu_util_mean",
    "telemetry.model.nfs_read_bytes_total",
    "telemetry.model.nfs_write_bytes_total",
    "telemetry.model.disk_read_bytes_total",
    "telemetry.model.disk_write_bytes_total",
    "telemetry.finalize.job_mem_peak_bytes_max",
    "telemetry.finalize.mem_available_bytes_min",
    "telemetry.finalize.cpu_busy_cores_mean",
    "telemetry.finalize.cpu_busy_cores_p95",
    "telemetry.finalize.cpu_util_mean",
    "telemetry.finalize.nfs_read_bytes_total",
    "telemetry.finalize.nfs_write_bytes_total",
    "telemetry.finalize.disk_read_bytes_total",
    "telemetry.finalize.disk_write_bytes_total",
    "telemetry.workspace.allocated_bytes_max",
)

_KEY_PATTERN: Final = re.compile(
    r"telemetry\.(model|finalize|workspace)\.[a-z0-9_]+_(min|mean|p95|max|total)"
)


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def test_frozen_keys_are_19_and_unique() -> None:
    assert len(FROZEN_KEYS) == 19
    assert len(set(FROZEN_KEYS)) == 19


def test_frozen_keys_equal_derived_headline_keys() -> None:
    derived = tuple(HEADLINE_KEY_PREFIX + spec.key for spec in HEADLINE)

    assert derived == FROZEN_KEYS


def test_frozen_keys_match_the_key_grammar() -> None:
    assert [k for k in FROZEN_KEYS if not _KEY_PATTERN.fullmatch(k)] == []


@pytest.mark.parametrize("key", FROZEN_KEYS)
def test_reporter_accepts_key(key: str) -> None:
    channel = _ListChannel()

    Reporter(channel, enabled=True).metadata(key, "1")

    assert len(channel.lines) == 1
    assert parse_hooks("\n".join(channel.lines)) == [
        Hook("SetMetadata", (key, "1"))
    ]


def test_scope_set_is_phases_plus_workspace() -> None:
    scopes = {key.split(".")[1] for key in FROZEN_KEYS}

    assert scopes == set(PHASES) | {WORKSPACE_SCOPE}


def test_scope_workspace_is_reserved_and_never_a_phase() -> None:
    assert WORKSPACE_SCOPE == "workspace"
    assert WORKSPACE_SCOPE not in PHASES
    assert WORKSPACE_SCOPE not in {phase.value for phase in Phase}


def test_scope_phase_values_equal_schema_phases() -> None:
    assert {phase.value for phase in Phase} == set(PHASES)


def test_length_longest_key_is_pinned() -> None:
    longest = max(FROZEN_KEYS, key=len)

    assert longest == "telemetry.finalize.mem_available_bytes_min"
    assert len(longest) == 42


@pytest.mark.parametrize(
    ("fmt", "value", "expected"),
    [
        ("bytes", 1234.5, "1234"),
        ("bytes", 1235.5, "1236"),
        ("bytes", 7, "7"),
        ("cores", 2.0, "2.00"),
        ("cores", 3.14159, "3.14"),
        ("fraction", 0.5, "0.5000"),
        ("fraction", 0.123456, "0.1235"),
        ("cores", -0.0, "0.00"),
    ],
)
def test_format_value(fmt: str, value: float, expected: str) -> None:
    assert format_headline_value(fmt, value) == expected


@pytest.mark.parametrize(
    ("fmt", "value"),
    [
        ("bytes", True),
        ("cores", math.nan),
        ("fraction", math.inf),
        ("percent", 1.0),
        ("bytes", -5),
        ("bytes", 10**400),
    ],
    ids=["bool", "nan", "inf", "unknown-format", "negative", "out-of-range"],
)
def test_format_value_outside_the_domain_raises(fmt: str, value: float) -> None:
    with pytest.raises(ValueError):
        format_headline_value(fmt, value)


def test_format_longest_value_fits_under_the_metadata_cap() -> None:
    longest = format_headline_value("fraction", sys.float_info.max)

    assert len(longest) == 314
    assert len(longest) < METADATA_VALUE_CAP


def test_items_of_no_values_is_empty() -> None:
    assert headline_items({}) == ()


def test_items_of_the_workspace_value() -> None:
    items = headline_items({"workspace.allocated_bytes_max": 4096})

    assert items == (("telemetry.workspace.allocated_bytes_max", "4096"),)
