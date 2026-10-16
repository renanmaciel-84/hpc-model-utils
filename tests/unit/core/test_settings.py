"""ADR-025/ADR-080: EngineSettings.from_env and its internal HPCMU_* overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.settings import EngineSettings


def test_from_env_no_overrides_returns_defaults() -> None:
    assert EngineSettings.from_env({}) == EngineSettings()


def test_from_env_valid_overrides_parses_every_field(
    tmp_path: Path,
) -> None:
    cli_bin = tmp_path / "bin" / "hpcmu"
    env = {
        "HPCMU_POLL_INTERVAL": "1.5",
        "HPCMU_HEARTBEAT_AFTER": "300",
        "HPCMU_FOLLOW_MARGIN": "900",
        "HPCMU_PENDING_CAP": "43200",
        "HPCMU_SETTLE_WINDOW": "20",
        "HPCMU_MISSING_LOG_GRACE": "120",
        "HPCMU_SQUEUE_FAILURE_BUDGET": "3",
        "HPCMU_OUTCOME_ATTEMPTS": "4",
        "HPCMU_OUTCOME_BACKOFF": "5",
        "HPCMU_CANCEL_TIMEOUT": "30",
        "HPCMU_CLI_BIN": str(cli_bin),
        "HPCMU_TELEMETRY": "off",
        "HPCMU_TELEMETRY_INTERVAL": "30",
        "HPCMU_TELEMETRY_WORKSPACE_INTERVAL": "3600",
    }
    settings = EngineSettings.from_env(env)
    assert settings == EngineSettings(
        poll_interval=1.5,
        heartbeat_after=300.0,
        follow_margin=900.0,
        pending_cap=43200.0,
        settle_window=20.0,
        missing_log_grace=120.0,
        squeue_failure_budget=3,
        outcome_attempts=4,
        outcome_backoff=5.0,
        cancel_timeout=30.0,
        cli_bin=cli_bin,
        telemetry=False,
        telemetry_interval=30,
        telemetry_workspace_interval=3600,
    )


def test_from_env_no_overrides_telemetry_defaults() -> None:
    settings = EngineSettings.from_env({})
    assert settings.telemetry is True
    assert settings.telemetry_interval == 5
    assert settings.telemetry_workspace_interval == 60


def test_from_env_empty_string_override_is_treated_as_unset() -> None:
    settings = EngineSettings.from_env({"HPCMU_POLL_INTERVAL": ""})
    assert settings.poll_interval == EngineSettings().poll_interval


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "0", "-1", "abc"])
def test_from_env_invalid_float_raises_usage_error(raw: str) -> None:
    with pytest.raises(UsageError, match="HPCMU_POLL_INTERVAL"):
        EngineSettings.from_env({"HPCMU_POLL_INTERVAL": raw})


def test_from_env_missing_log_grace_override_parses_to_float() -> None:
    settings = EngineSettings.from_env({"HPCMU_MISSING_LOG_GRACE": "120"})
    assert settings.missing_log_grace == 120.0


def test_from_env_missing_log_grace_absent_defaults_to_90_seconds() -> None:
    assert EngineSettings.from_env({}).missing_log_grace == 90.0


@pytest.mark.parametrize("raw", ["0", "-1", "inf", "x"])
def test_from_env_missing_log_grace_invalid_raises_usage_error(
    raw: str,
) -> None:
    with pytest.raises(UsageError, match="HPCMU_MISSING_LOG_GRACE"):
        EngineSettings.from_env({"HPCMU_MISSING_LOG_GRACE": raw})


def test_from_env_relative_cli_bin_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="HPCMU_CLI_BIN"):
        EngineSettings.from_env({"HPCMU_CLI_BIN": "bin/hpcmu"})


def test_from_env_absolute_cli_bin_parses_to_path() -> None:
    settings = EngineSettings.from_env({"HPCMU_CLI_BIN": "/usr/bin/hpcmu"})
    assert settings.cli_bin == Path("/usr/bin/hpcmu")


def test_from_env_non_integer_budget_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="HPCMU_SQUEUE_FAILURE_BUDGET"):
        EngineSettings.from_env({"HPCMU_SQUEUE_FAILURE_BUDGET": "5.5"})


def test_from_env_negative_budget_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="HPCMU_SQUEUE_FAILURE_BUDGET"):
        EngineSettings.from_env({"HPCMU_SQUEUE_FAILURE_BUDGET": "-1"})


def test_from_env_zero_budget_is_allowed() -> None:
    settings = EngineSettings.from_env({"HPCMU_SQUEUE_FAILURE_BUDGET": "0"})
    assert settings.squeue_failure_budget == 0


def test_from_env_outcome_attempts_zero_raises_usage_error() -> None:
    with pytest.raises(UsageError, match="HPCMU_OUTCOME_ATTEMPTS"):
        EngineSettings.from_env({"HPCMU_OUTCOME_ATTEMPTS": "0"})


def test_from_env_outcome_attempts_one_is_allowed() -> None:
    settings = EngineSettings.from_env({"HPCMU_OUTCOME_ATTEMPTS": "1"})
    assert settings.outcome_attempts == 1


def test_from_env_usage_error_message_names_variable_and_value() -> None:
    with pytest.raises(UsageError, match=r"HPCMU_FOLLOW_MARGIN.*nan"):
        EngineSettings.from_env({"HPCMU_FOLLOW_MARGIN": "nan"})


@pytest.mark.parametrize("raw", ["4", "31", "5.0", "x"])
def test_from_env_telemetry_interval_invalid_raises_usage_error(
    raw: str,
) -> None:
    with pytest.raises(UsageError, match="HPCMU_TELEMETRY_INTERVAL"):
        EngineSettings.from_env({"HPCMU_TELEMETRY_INTERVAL": raw})


@pytest.mark.parametrize("raw", [5, 30])
def test_from_env_telemetry_interval_boundaries_parse(raw: int) -> None:
    settings = EngineSettings.from_env({"HPCMU_TELEMETRY_INTERVAL": str(raw)})
    assert settings.telemetry_interval == raw


@pytest.mark.parametrize("raw", ["59", "3601", "60s"])
def test_from_env_telemetry_workspace_interval_invalid_raises_usage_error(
    raw: str,
) -> None:
    with pytest.raises(UsageError, match="HPCMU_TELEMETRY_WORKSPACE_INTERVAL"):
        EngineSettings.from_env({"HPCMU_TELEMETRY_WORKSPACE_INTERVAL": raw})


@pytest.mark.parametrize("raw", [60, 3600])
def test_from_env_telemetry_workspace_interval_boundaries_parse(
    raw: int,
) -> None:
    settings = EngineSettings.from_env(
        {"HPCMU_TELEMETRY_WORKSPACE_INTERVAL": str(raw)}
    )
    assert settings.telemetry_workspace_interval == raw


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("4", "HPCMU_TELEMETRY_INTERVAL must be >= 5: '4'"),
        ("31", "HPCMU_TELEMETRY_INTERVAL must be <= 30: '31'"),
        ("5.0", "HPCMU_TELEMETRY_INTERVAL must be an integer: '5.0'"),
    ],
)
def test_from_env_telemetry_interval_usage_error_message(
    raw: str, message: str
) -> None:
    with pytest.raises(UsageError) as excinfo:
        EngineSettings.from_env({"HPCMU_TELEMETRY_INTERVAL": raw})
    assert str(excinfo.value) == message


@pytest.mark.parametrize("raw", ["ON", "true", "1", "0", "no"])
def test_from_env_telemetry_invalid_switch_raises_usage_error(
    raw: str,
) -> None:
    with pytest.raises(
        UsageError, match="HPCMU_TELEMETRY must be on or off"
    ) as excinfo:
        EngineSettings.from_env({"HPCMU_TELEMETRY": raw})
    assert str(excinfo.value) == f"HPCMU_TELEMETRY must be on or off: {raw!r}"


@pytest.mark.parametrize(("raw", "expected"), [("on", True), ("off", False)])
def test_from_env_telemetry_switch_parses(raw: str, expected: bool) -> None:
    settings = EngineSettings.from_env({"HPCMU_TELEMETRY": raw})
    assert settings.telemetry is expected


def test_from_env_telemetry_empty_switch_is_treated_as_unset() -> None:
    settings = EngineSettings.from_env({"HPCMU_TELEMETRY": ""})
    assert settings.telemetry is True
