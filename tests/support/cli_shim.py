"""v2-plan ticket-036: writes the ``hpcmu-test-cli`` shim used as a
system test's ``Toolchain.cli_bin``. The shim registers the fake plugin
and then calls the real v2 CLI entry point; it is generated at test
time, never committed, because its shebang is machine-specific.

job-telemetry ticket-036 (ADR-081): ``Toolchain.python_bin`` is the
``python`` next to ``cli_bin``, so both shim writers (this module's
``write_cli_shim`` and the cobre scenarios' ``_write_cli_shim``) place
that sibling through ``write_python_sibling``. The sibling is an exec
wrapper around ``sys.executable``, not a symlink: a symlink outside the
venv loses venv detection (``pyvenv.cfg`` is looked up relative to the
link's own location), so the venv's site-packages and
``hpc_model_utils`` would not load. For the same reason neither the
helper nor the shims ever resolve the sibling or ``sys.executable``.

ticket-043: the generated shim also registers the test-only
``engine-run-test`` command (``register_engine_run_test_command``
below) when ``HPCMU_TEST_ENGINE_COMMAND=1`` is set in its own
environment at run time. That check lives in the generated script, not
here, so a plain ``import tests.support.cli_shim`` -- or production
code, which never imports this test-support module at all -- never
registers the command on the shared ``cli`` group.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

import click

from hpc_model_utils.cli.root import AppContext, HpcmuCommand, cli
from hpc_model_utils.core.launch import Resources, Toolchain
from hpc_model_utils.core.lifecycle.run import JobLedger, SubmitRequest, run
from hpc_model_utils.core.state import StateStore
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.slurm import Slurm
from hpc_model_utils.models import get_plugin

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SHIM_TEMPLATE = """\
#!{python}
import os
import sys

sys.path[0:0] = [{repo_root!r}]

import hpc_model_utils.cli
import tests.support.fake_plugin

tests.support.fake_plugin.register_fake()
if os.environ.get("HPCMU_TEST_ENGINE_COMMAND") == "1":
    import tests.support.cli_shim

    tests.support.cli_shim.register_engine_run_test_command()

raise SystemExit(hpc_model_utils.cli.main())
"""


def write_python_sibling(dest_dir: Path) -> Path:
    sibling = dest_dir / "python"
    sibling.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    sibling.chmod(0o755)
    return sibling


def write_cli_shim(dest_dir: Path) -> Path:
    shim_path = dest_dir / "hpcmu-test-cli"
    shim_path.write_text(
        _SHIM_TEMPLATE.format(python=sys.executable, repo_root=str(_REPO_ROOT))
    )
    shim_path.chmod(0o755)
    write_python_sibling(dest_dir)
    return shim_path


def register_engine_run_test_command() -> None:
    """Registers ``engine-run-test`` on the shared ``cli`` group: a
    thin wrapper around ``core.lifecycle.run.run`` for
    ``tests/system/test_run_signals.py`` (ticket-043), against a
    pre-built fake-plugin ``workspace`` and a ``--slurm-bin`` directory
    the caller controls. ``run``'s own ``EngineSettings`` still come
    from the usual ``HPCMU_*`` environment overrides (``app_ctx.
    settings``), so the test paces polling/settling/cancellation by
    setting those in the subprocess's environment, same as any other
    command.
    """

    @cli.command("engine-run-test", cls=HpcmuCommand, hidden=True)
    @click.argument("workspace", type=click.Path(path_type=Path))
    @click.option("--slurm-bin", required=True, type=click.Path(path_type=Path))
    @click.pass_obj
    def _engine_run_test(
        app_ctx: AppContext, workspace: Path, slurm_bin: Path
    ) -> None:
        ws = Workspace.at(workspace)
        plugin = get_plugin("fake")
        slurm = Slurm(slurm_bin)
        store = StateStore(ws)
        ledger = JobLedger()
        tools = Toolchain(
            cli_bin=Path(sys.argv[0]).resolve(),
            mpich_bin=workspace / "mpich" / "bin",
            slurm_bin=slurm_bin,
        )
        req = SubmitRequest(
            resources=Resources(queue="batch", cores=1),
            tools=tools,
            skip_model=False,
            synthesis_bin=None,
        )
        run(
            ws,
            plugin,
            slurm,
            app_ctx.reporter,
            store,
            req,
            ledger,
            settings=app_ctx.settings,
            emit=logging.getLogger("hpc_model_utils.engine_run_test").info,
        )
