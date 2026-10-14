"""ADR-025: the follow loop's test seams, routed through one settings
object with internal ``HPCMU_*`` environment overrides (never new
public CLI flags).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hpc_model_utils.core.errors import UsageError


@dataclass(frozen=True)
class EngineSettings:
    poll_interval: float = 2.0
    heartbeat_after: float = 600.0
    follow_margin: float = 1800.0
    pending_cap: float = 86400.0
    settle_window: float = 10.0
    # How long _settle keeps polling a log that was never opened. Exceeds
    # F15's acdirmax (60 s) plus one poll, so a cached NFS "no such file"
    # lookup can expire.
    missing_log_grace: float = 90.0
    squeue_failure_budget: int = 5
    outcome_attempts: int = 6
    outcome_backoff: float = 10.0
    cancel_timeout: float = 60.0
    cli_bin: Path | None = None
    telemetry: bool = True
    telemetry_interval: int = 5
    telemetry_workspace_interval: int = 60

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> EngineSettings:
        defaults = cls()
        return cls(
            poll_interval=_float_override(
                env, "HPCMU_POLL_INTERVAL", defaults.poll_interval
            ),
            heartbeat_after=_float_override(
                env, "HPCMU_HEARTBEAT_AFTER", defaults.heartbeat_after
            ),
            follow_margin=_float_override(
                env, "HPCMU_FOLLOW_MARGIN", defaults.follow_margin
            ),
            pending_cap=_float_override(
                env, "HPCMU_PENDING_CAP", defaults.pending_cap
            ),
            settle_window=_float_override(
                env, "HPCMU_SETTLE_WINDOW", defaults.settle_window
            ),
            missing_log_grace=_float_override(
                env, "HPCMU_MISSING_LOG_GRACE", defaults.missing_log_grace
            ),
            squeue_failure_budget=_int_override(
                env,
                "HPCMU_SQUEUE_FAILURE_BUDGET",
                defaults.squeue_failure_budget,
                minimum=0,
            ),
            outcome_attempts=_int_override(
                env,
                "HPCMU_OUTCOME_ATTEMPTS",
                defaults.outcome_attempts,
                minimum=1,
            ),
            outcome_backoff=_float_override(
                env, "HPCMU_OUTCOME_BACKOFF", defaults.outcome_backoff
            ),
            cancel_timeout=_float_override(
                env, "HPCMU_CANCEL_TIMEOUT", defaults.cancel_timeout
            ),
            cli_bin=_cli_bin_override(env, "HPCMU_CLI_BIN", defaults.cli_bin),
            telemetry=_switch_override(
                env, "HPCMU_TELEMETRY", defaults.telemetry
            ),
            telemetry_interval=_int_override(
                env,
                "HPCMU_TELEMETRY_INTERVAL",
                defaults.telemetry_interval,
                minimum=5,
                maximum=30,
            ),
            telemetry_workspace_interval=_int_override(
                env,
                "HPCMU_TELEMETRY_WORKSPACE_INTERVAL",
                defaults.telemetry_workspace_interval,
                minimum=60,
                maximum=3600,
            ),
        )


def _float_override(env: Mapping[str, str], var: str, default: float) -> float:
    raw = env.get(var, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise UsageError(f"{var} must be a number: {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise UsageError(f"{var} must be finite and > 0: {raw!r}")
    return value


def _int_override(
    env: Mapping[str, str],
    var: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = env.get(var, "")
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise UsageError(f"{var} must be an integer: {raw!r}") from None
    if value < minimum:
        raise UsageError(f"{var} must be >= {minimum}: {raw!r}")
    if maximum is not None and value > maximum:
        raise UsageError(f"{var} must be <= {maximum}: {raw!r}")
    return value


def _cli_bin_override(
    env: Mapping[str, str], var: str, default: Path | None
) -> Path | None:
    raw = env.get(var, "")
    if raw == "":
        return default
    path = Path(raw)
    if not path.is_absolute():
        raise UsageError(f"{var} must be an absolute path: {raw!r}")
    return path


def _switch_override(env: Mapping[str, str], var: str, default: bool) -> bool:
    raw = env.get(var, "")
    if raw == "":
        return default
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise UsageError(f"{var} must be on or off: {raw!r}")
