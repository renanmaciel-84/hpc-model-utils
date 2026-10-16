"""ADR-003/ADR-038/ADR-057/R73/R74/R78/R105: ``CobrePlugin``, the first
plugin for a new model (ticket-068, reworked by ticket-068a).

The deck concerns (case detection, study info, input files) live in
``case`` and the static ``novomodelo-mpi`` check lives here; the comm and
solver facts are verified in the model job (tickets 069 and 070).
``launch``, ``diagnose`` and ``outputs`` delegate to the sibling
``launch``, ``diagnosis`` and ``outputs`` modules (tickets 069/070/071).
``synthesis_args`` delegates to ``outputs.dashboard_args``, the
novomodelo-bridge dashboard (ADR-059, ticket-080c). ``prepare``,
``postprocess``, ``fetch_parent`` and ``ingest_offline`` keep the ABC
defaults. The plugin is registered in ``models.PLUGINS`` (ticket-072).
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path
from typing import ClassVar

from hpc_model_utils.core.diagnosis import JobReport, Verdict
from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.launch import LaunchSpec, Resources
from hpc_model_utils.core.outputs import OutputPlan
from hpc_model_utils.core.plugin import ExecutableSpec, ModelPlugin
from hpc_model_utils.core.state import StudyInfo
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models.cobre import case, diagnosis, launch, outputs

logger = logging.getLogger(__name__)

# The pre-fork binary name, kept only to explain a stale versions upload.
_LEGACY_BINARY = "cobre-mpi"


def _read_head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


class CobrePlugin(ModelPlugin):
    name = "cobre"
    executables = ExecutableSpec(launch.MPI_BINARY)
    sanitize_encoding = False
    output_patterns: ClassVar[tuple[str, ...]] = case.OUTPUT_PATTERNS
    log_patterns = diagnosis.LOG_PATTERNS
    parent_model = None
    parent_artifacts: ClassVar[tuple[str, ...]] = ()
    always_write_parent_path = False

    def study_info(self, ws: Workspace) -> StudyInfo:
        return case.study_info(ws)

    def input_files(self, ws: Workspace) -> tuple[str, ...]:
        return case.input_files(ws)

    def launch(self, ws: Workspace, res: Resources) -> LaunchSpec:
        return launch.launch_spec(ws, res)

    def primary_evidence(self, ws: Workspace) -> tuple[Path, ...]:
        return diagnosis.primary_evidence(ws)

    def diagnose(self, ws: Workspace, job: JobReport) -> Verdict:
        return diagnosis.diagnose(ws, job)

    def outputs(self, ws: Workspace) -> OutputPlan:
        return outputs.output_plan(ws)

    def synthesis_args(
        self, ws: Workspace, cpus: int
    ) -> tuple[str, ...] | None:
        return outputs.dashboard_args(ws)

    def check_executables(self, ws: Workspace) -> None:
        name = launch.MPI_BINARY
        mpi = ws.assets / name
        if not (mpi.exists() or mpi.is_symlink()):
            legacy = ws.assets / _LEGACY_BINARY
            if legacy.exists() or legacy.is_symlink():
                raise DataError(
                    f"{name} missing from the fetched versoes files "
                    f"(found {_LEGACY_BINARY}: re-upload this version "
                    f"from a {name} build)"
                )
            raise DataError(f"{name} missing from the fetched versoes files")
        if (
            mpi.is_symlink()
            or not mpi.is_file()
            or not mpi.stat().st_mode & stat.S_IXUSR
        ):
            raise DataError(f"{name} is not an executable regular file")
        try:
            head = _read_head(mpi)
        except OSError as err:
            raise DataError(f"{name} cannot be read") from err
        if head != b"\x7fELF":
            raise DataError(f"{name} is not an ELF executable")
        logger.info(
            "%s: regular ELF executable; "
            "comm and solver are verified in the model job",
            name,
        )
