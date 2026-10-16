"""ticket-053 AC2-AC5: ``ModelArg``, ``tests.support.command_lines``,
the workflow command surface, ``resolve_cli_bin``, the ``run`` wiring,
and the ``object_store`` factory.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from hpc_model_utils.cli import workflow
from hpc_model_utils.cli.params import ModelArg
from hpc_model_utils.cli.root import AppContext, HpcmuCommand, cli
from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.launch import (
    TELEMETRY_OFF,
    Resources,
    TelemetryConfig,
    Toolchain,
)
from hpc_model_utils.core.lifecycle.run import SubmitRequest
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import ToolInfo, current_tool
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.s3 import Boto3ObjectStore, ObjectStore
from hpc_model_utils.models.newave import NewavePlugin
from hpc_model_utils.platform.modelops import Reporter
from tests.support.command_lines import load_command_lines, render
from tests.support.fake_plugin import FakePlugin, fake_registered
from tests.support.object_store import RecordingObjectStore

_WORKFLOW_COMMANDS: tuple[str, ...] = (
    "check_and_fetch_executables",
    "check_and_fetch_inputs",
    "extract_sanitize_inputs",
    "preprocess",
    "run",
    "result_upload",
    "cancel_run",
    "ingest_offline_run",
)
_LEGACY_ONLY_COMMANDS: tuple[str, ...] = (
    "output_compression_and_cleanup",
    "download_executed_run",
    "fetch_extract_raw_outputs",
)


@dataclass
class _Call:
    args: tuple[object, ...]
    kwargs: dict[str, object]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[_Call] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append(_Call(args, kwargs))


class _ListChannel:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _app_context(cli_bin: Path = Path("/c/bin/hpc-model-utils")) -> AppContext:
    return AppContext(
        reporter=Reporter(_ListChannel(), enabled=True),
        channels=None,
        settings=EngineSettings(cli_bin=cli_bin),
    )


def _hpcmu_command(name: str) -> HpcmuCommand:
    cmd = cli.commands[name]
    assert isinstance(cmd, HpcmuCommand)
    return cmd


# ---------------------------------------------------------------------------
# ModelArg (Requirement 1).
# ---------------------------------------------------------------------------


def _click_probe() -> tuple[click.Context, click.Parameter]:
    return click.Context(click.Command("probe")), click.Argument(["model"])


@pytest.mark.parametrize("value", ["NEWAVE", "newave", "Newave"])
def test_model_arg_convert_any_case_returns_newave_plugin(value: str) -> None:
    ctx, param = _click_probe()

    result = ModelArg().convert(value, param, ctx)

    assert isinstance(result, NewavePlugin)


def test_model_arg_convert_unknown_name_raises_usage_error() -> None:
    ctx, param = _click_probe()

    with pytest.raises(click.UsageError, match="unknown model 'NOSUCH'"):
        ModelArg().convert("NOSUCH", param, ctx)


def test_model_arg_get_metavar_lists_sorted_registry_keys() -> None:
    ctx, param = _click_probe()

    assert ModelArg().get_metavar(param, ctx) == "cobre|decomp|newave"


# ---------------------------------------------------------------------------
# tests.support.command_lines (Requirement 6).
# ---------------------------------------------------------------------------


def test_load_command_lines_duplicate_label_raises_value_error(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "dup.txt"
    fixture.write_text("a: one\na: two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate label: 'a'"):
        load_command_lines(fixture)


def test_load_command_lines_missing_separator_raises_value_error(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "malformed.txt"
    fixture.write_text("not-a-valid-row\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed command-line"):
        load_command_lines(fixture)


def test_render_unknown_placeholder_raises_key_error() -> None:
    with pytest.raises(KeyError, match="nosuch"):
        render("foo {nosuch}", {})


# ---------------------------------------------------------------------------
# Surface AC: --help, emits_terminal_status, unknown model, fake registry.
# ---------------------------------------------------------------------------


def test_help_lists_all_eight_workflow_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    for name in _WORKFLOW_COMMANDS:
        assert name in result.output


def test_help_excludes_finalize_and_legacy_only_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert "finalize" not in result.output
    for name in _LEGACY_ONLY_COMMANDS:
        assert name not in result.output


def test_cancel_run_command_emits_terminal_status_is_false() -> None:
    assert _hpcmu_command("cancel_run").emits_terminal_status is False


@pytest.mark.parametrize(
    "name", [c for c in _WORKFLOW_COMMANDS if c != "cancel_run"]
)
def test_other_workflow_command_emits_terminal_status_true(name: str) -> None:
    assert _hpcmu_command(name).emits_terminal_status is True


def test_run_command_unknown_model_exits_2_lists_valid_models() -> None:
    result = CliRunner().invoke(
        cli, ["run", "NOSUCH", "q", "1"], obj=_app_context()
    )

    assert result.exit_code == 2
    assert "unknown model 'NOSUCH'" in result.output
    assert "['cobre', 'decomp', 'newave']" in result.output


def test_extract_sanitize_inputs_command_fake_model_resolves_fake_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(workflow, "extract_sanitize_inputs", recorder)

    with fake_registered():
        result = CliRunner().invoke(
            cli, ["extract_sanitize_inputs", "fake"], obj=_app_context()
        )

    assert result.exit_code == 0, result.output
    assert isinstance(recorder.calls[0].args[1], FakePlugin)


# ---------------------------------------------------------------------------
# resolve_cli_bin (Requirement 2).
# ---------------------------------------------------------------------------


def test_resolve_cli_bin_settings_override_returns_as_is() -> None:
    settings = EngineSettings(cli_bin=Path("/x/y"))

    assert workflow.resolve_cli_bin(settings) == Path("/x/y")


def test_resolve_cli_bin_unresolved_venv_symlink_returns_sibling_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_python = tmp_path / "real" / "python3"
    real_python.parent.mkdir(parents=True)
    real_python.write_text("", encoding="utf-8")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    symlinked_python = venv_bin / "python"
    symlinked_python.symlink_to(real_python)
    (venv_bin / "hpc-model-utils").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(symlinked_python))

    result = workflow.resolve_cli_bin(EngineSettings())

    assert result == venv_bin / "hpc-model-utils"


def test_resolve_cli_bin_missing_script_raises_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))

    with pytest.raises(UsageError, match="HPCMU_CLI_BIN"):
        workflow.resolve_cli_bin(EngineSettings())


# ---------------------------------------------------------------------------
# run wiring (Requirement 3).
# ---------------------------------------------------------------------------


def test_run_command_explicit_options_records_submit_request_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(workflow, "run", recorder)
    app_ctx = _app_context(cli_bin=Path("/c/bin/hpc-model-utils"))

    result = CliRunner().invoke(
        cli,
        [
            "run",
            "newave",
            "q-a",
            "64",
            "--max-cores-per-node",
            "32",
            "--max-job-time-hours",
            "24",
            "--mpich-path",
            "/m/bin",
            "--slurm-path",
            "/s/bin",
            "--synthesis-bin",
            "/t/bin/sint",
        ],
        obj=app_ctx,
    )

    assert result.exit_code == 0, result.output
    call = recorder.calls[0]
    req = call.args[5]
    assert isinstance(req, SubmitRequest)
    assert req.resources == Resources("q-a", 64, 32, 24)
    assert req.tools == Toolchain(
        Path("/m/bin"), Path("/s/bin"), Path("/c/bin/hpc-model-utils")
    )
    assert req.skip_model is False
    assert req.synthesis_bin == Path("/t/bin/sint")
    assert call.kwargs["settings"] is app_ctx.settings


def test_run_command_minimal_args_default_mpich_and_slurm_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(workflow, "run", recorder)
    app_ctx = _app_context(cli_bin=Path("/c/bin/hpc-model-utils"))

    result = CliRunner().invoke(cli, ["run", "newave", "q-a", "4"], obj=app_ctx)

    assert result.exit_code == 0, result.output
    req = recorder.calls[0].args[5]
    assert isinstance(req, SubmitRequest)
    assert req.tools.mpich_bin == Path("/usr/local/mpich-4.2.0/bin")
    assert req.tools.slurm_bin == Path("/opt/slurm/bin")


def test_run_command_max_cores_per_node_zero_exits_2() -> None:
    result = CliRunner().invoke(
        cli,
        ["run", "newave", "q", "1", "--max-cores-per-node", "0"],
        obj=_app_context(),
    )

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# telemetry launch config (ticket-033, ADR-082).
# ---------------------------------------------------------------------------


def test_telemetry_config_default_settings_maps_all_four_fields() -> None:
    config = workflow.telemetry_config(EngineSettings())

    assert config == TelemetryConfig(True, 5, 60, current_tool().version)


def test_telemetry_config_env_off_maps_all_four_fields() -> None:
    settings = EngineSettings.from_env(
        {
            "HPCMU_TELEMETRY": "off",
            "HPCMU_TELEMETRY_INTERVAL": "30",
            "HPCMU_TELEMETRY_WORKSPACE_INTERVAL": "3600",
        }
    )

    config = workflow.telemetry_config(settings)

    assert config == TelemetryConfig(False, 30, 3600, current_tool().version)


def test_run_command_telemetry_settings_record_config_on_submit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(workflow, "run", recorder)
    monkeypatch.setattr(
        workflow, "current_tool", lambda: ToolInfo("hpc-model-utils", "9.9.9")
    )
    app_ctx = AppContext(
        reporter=Reporter(_ListChannel(), enabled=True),
        channels=None,
        settings=EngineSettings(
            cli_bin=Path("/c/bin/hpc-model-utils"),
            telemetry=False,
            telemetry_interval=12,
            telemetry_workspace_interval=120,
        ),
    )

    result = CliRunner().invoke(cli, ["run", "newave", "q-a", "4"], obj=app_ctx)

    assert result.exit_code == 0, result.output
    req = recorder.calls[0].args[5]
    assert isinstance(req, SubmitRequest)
    assert req.telemetry == TelemetryConfig(False, 12, 120, "9.9.9")


def test_submit_request_telemetry_omitted_defaults_to_telemetry_off() -> None:
    req = SubmitRequest(
        Resources("q", 4, None, None),
        Toolchain(Path("/m"), Path("/s"), Path("/c/bin/hpc-model-utils")),
        False,
        None,
    )

    assert req.telemetry is TELEMETRY_OFF


def test_telemetry_off_matches_engine_settings_cadence_defaults() -> None:
    defaults = EngineSettings()

    assert TELEMETRY_OFF == TelemetryConfig(
        False,
        defaults.telemetry_interval,
        defaults.telemetry_workspace_interval,
        "",
    )


def test_telemetry_config_assignment_raises_frozen_instance_error() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(TELEMETRY_OFF, "interval", 6)


# ---------------------------------------------------------------------------
# object_store factory (Requirement 2/AC "Store factory").
# ---------------------------------------------------------------------------


def test_object_store_default_returns_boto3_object_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    assert isinstance(workflow.object_store(), Boto3ObjectStore)


@pytest.mark.parametrize(
    "command_args,lifecycle_name",
    [
        (
            ["check_and_fetch_executables", "newave", "s3://b/v/"],
            "fetch_executables",
        ),
        (
            ["check_and_fetch_inputs", "newave", "s3://b/k.zip"],
            "fetch_inputs",
        ),
        (["result_upload", "newave", "s3://b/saidas/"], "publish"),
        (
            [
                "ingest_offline_run",
                "newave",
                "s3://b/i.zip",
                "s3://b/o.zip",
                "s3://b/c.zip",
            ],
            "ingest_offline_run",
        ),
    ],
)
def test_store_commands_call_object_store_exactly_once(
    command_args: list[str],
    lifecycle_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _counting_object_store() -> ObjectStore:
        nonlocal calls
        calls += 1
        return RecordingObjectStore()

    monkeypatch.setattr(workflow, "object_store", _counting_object_store)
    monkeypatch.setattr(workflow, lifecycle_name, _Recorder())

    result = CliRunner().invoke(cli, command_args, obj=_app_context())

    assert result.exit_code == 0, result.output
    assert calls == 1


# ---------------------------------------------------------------------------
# Every command builds Workspace.at(Path.cwd()).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command_args,lifecycle_name",
    [
        (
            ["check_and_fetch_executables", "newave", "s3://b/v/"],
            "fetch_executables",
        ),
        (
            ["check_and_fetch_inputs", "newave", "s3://b/k.zip"],
            "fetch_inputs",
        ),
        (["extract_sanitize_inputs", "newave"], "extract_sanitize_inputs"),
        (["preprocess", "newave"], "preprocess"),
        (["run", "newave", "q", "1"], "run"),
        (["result_upload", "newave", "s3://b/saidas/"], "publish"),
        (["cancel_run", "newave"], "cancel"),
        (
            [
                "ingest_offline_run",
                "newave",
                "s3://b/i.zip",
                "s3://b/o.zip",
                "s3://b/c.zip",
            ],
            "ingest_offline_run",
        ),
    ],
)
def test_workflow_command_builds_workspace_at_cwd(
    command_args: list[str],
    lifecycle_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    monkeypatch.setattr(workflow, lifecycle_name, recorder)

    result = CliRunner().invoke(cli, command_args, obj=_app_context())

    assert result.exit_code == 0, result.output
    assert recorder.calls[0].args[0] == Workspace.at(tmp_path)
