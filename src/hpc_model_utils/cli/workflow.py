"""ADR-004/ADR-005/ADR-006/ADR-020/ADR-024/ADR-082/R10/R11/R44/R69/R91/R101:
the C1 workflow commands -- thin click wrappers over the lifecycle
steps (ticket-053). No command body catches anything: a lifecycle
failure propagates to ``cli.root.main``'s single fatal-path
chokepoint. Every lifecycle name below is a plain module-level import,
never aliased, so a test can monkeypatch ``hpc_model_utils.cli.
workflow.<name>`` to a recorder.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from hpc_model_utils.cli.params import ModelArg
from hpc_model_utils.cli.root import AppContext, HpcmuCommand, cli
from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.launch import Resources, TelemetryConfig, Toolchain
from hpc_model_utils.core.lifecycle.cancel import cancel
from hpc_model_utils.core.lifecycle.fetch import fetch_executables, fetch_inputs
from hpc_model_utils.core.lifecycle.ingest import ingest_offline_run
from hpc_model_utils.core.lifecycle.prepare import (
    extract_sanitize_inputs,
    preprocess,
)
from hpc_model_utils.core.lifecycle.publish import publish
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.plugin import ModelPlugin
from hpc_model_utils.core.settings import EngineSettings
from hpc_model_utils.core.state import StateStore, current_tool
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.s3 import Boto3ObjectStore, ObjectStore
from hpc_model_utils.infra.slurm import Slurm

_DEFAULT_MPICH_PATH = Path("/usr/local/mpich-4.2.0/bin")
_DEFAULT_SLURM_PATH = Path("/opt/slurm/bin")


def object_store() -> ObjectStore:
    return Boto3ObjectStore()


def resolve_cli_bin(settings: EngineSettings) -> Path:
    if settings.cli_bin is not None:
        return settings.cli_bin
    candidate = Path(sys.executable).parent / "hpc-model-utils"
    if not candidate.is_absolute() or not candidate.is_file():
        raise UsageError(
            "cannot locate the hpc-model-utils console script next to "
            "the interpreter; set HPCMU_CLI_BIN"
        )
    return candidate


def telemetry_config(settings: EngineSettings) -> TelemetryConfig:
    return TelemetryConfig(
        settings.telemetry,
        settings.telemetry_interval,
        settings.telemetry_workspace_interval,
        current_tool().version,
    )


@cli.command("check_and_fetch_executables", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.argument("path")
@click.pass_obj
def check_and_fetch_executables_command(
    app_ctx: AppContext, model: ModelPlugin, path: str
) -> None:
    ws = Workspace.at(Path.cwd())
    fetch_executables(
        ws, model, object_store(), app_ctx.reporter, StateStore(ws), path
    )


@cli.command("check_and_fetch_inputs", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.argument("path")
@click.option("--parent-path", default="")
@click.option("--delete", is_flag=True, default=False)
@click.pass_obj
def check_and_fetch_inputs_command(
    app_ctx: AppContext,
    model: ModelPlugin,
    path: str,
    parent_path: str,
    delete: bool,
) -> None:
    ws = Workspace.at(Path.cwd())
    fetch_inputs(
        ws,
        model,
        object_store(),
        app_ctx.reporter,
        StateStore(ws),
        path,
        parent_path=parent_path,
        delete=delete,
    )


@cli.command("extract_sanitize_inputs", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.pass_obj
def extract_sanitize_inputs_command(
    app_ctx: AppContext, model: ModelPlugin
) -> None:
    ws = Workspace.at(Path.cwd())
    extract_sanitize_inputs(ws, model, StateStore(ws), app_ctx.reporter)


@cli.command("preprocess", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.option("--execution-name", default="")
@click.pass_obj
def preprocess_command(
    app_ctx: AppContext, model: ModelPlugin, execution_name: str
) -> None:
    ws = Workspace.at(Path.cwd())
    preprocess(ws, model, execution_name, store=StateStore(ws))


@cli.command("run", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.argument("queue")
@click.argument("cores", type=click.IntRange(min=1))
@click.option("--max-cores-per-node", type=click.IntRange(min=1))
@click.option("--max-job-time-hours", type=click.IntRange(min=1))
@click.option(
    "--mpich-path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_MPICH_PATH,
)
@click.option(
    "--slurm-path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_SLURM_PATH,
)
@click.option("--skip", is_flag=True, default=False)
@click.option("--synthesis-bin", type=click.Path(path_type=Path))
@click.pass_obj
def run_command(
    app_ctx: AppContext,
    model: ModelPlugin,
    queue: str,
    cores: int,
    max_cores_per_node: int | None,
    max_job_time_hours: int | None,
    mpich_path: Path,
    slurm_path: Path,
    skip: bool,
    synthesis_bin: Path | None,
) -> None:
    ws = Workspace.at(Path.cwd())
    run(
        ws,
        model,
        Slurm(slurm_path),
        app_ctx.reporter,
        StateStore(ws),
        SubmitRequest(
            Resources(queue, cores, max_cores_per_node, max_job_time_hours),
            Toolchain(
                mpich_path, slurm_path, resolve_cli_bin(app_ctx.settings)
            ),
            skip,
            synthesis_bin,
            telemetry=telemetry_config(app_ctx.settings),
        ),
        JobLedger(),
        settings=app_ctx.settings,
        emit=logging.getLogger("hpc_model_utils.job").info,
    )


@cli.command("result_upload", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.argument("path")
@click.pass_obj
def result_upload_command(
    app_ctx: AppContext, model: ModelPlugin, path: str
) -> None:
    ws = Workspace.at(Path.cwd())
    publish(ws, model, object_store(), app_ctx.reporter, StateStore(ws), path)


@cli.command("cancel_run", cls=HpcmuCommand, emits_terminal_status=False)
@click.argument("model", type=ModelArg())
@click.option("--job-id", default="")
@click.option(
    "--slurm-path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_SLURM_PATH,
)
@click.pass_obj
def cancel_run_command(
    app_ctx: AppContext, model: ModelPlugin, job_id: str, slurm_path: Path
) -> None:
    ws = Workspace.at(Path.cwd())
    cancel(
        ws,
        Slurm(slurm_path),
        StateStore(ws),
        job_id=job_id or None,
        settings=app_ctx.settings,
    )


@cli.command("ingest_offline_run", cls=HpcmuCommand)
@click.argument("model", type=ModelArg())
@click.argument("inputs")
@click.argument("outputs")
@click.argument("cuts")
@click.pass_obj
def ingest_offline_run_command(
    app_ctx: AppContext,
    model: ModelPlugin,
    inputs: str,
    outputs: str,
    cuts: str,
) -> None:
    ws = Workspace.at(Path.cwd())
    ingest_offline_run(
        ws,
        model,
        object_store(),
        app_ctx.reporter,
        StateStore(ws),
        (inputs, outputs, cuts),
    )
