"""ADR-038/ADR-040/R78/R116: the cobre ``OutputPlan`` (ticket-071).

Each phase archive keeps the Hive layout: members are named relative to
``output/`` through a ``Tree`` layout, because Hive members share
basenames (``part-0000.parquet``) and a ``Flat`` layout would reject
them. A phase that did not run has no directory, so it gets no archive
and no raw file. ``output/stochastic/`` and ``output/hydro_models/``
(opt-in exports) have no archive in ADR-038 and are never uploaded.

``dashboard_args`` (ticket-080c, ADR-059) is the cobre synthesis step:
the ``novomodelo-bridge dashboard`` arguments finalize runs in ``ws.root``
after a SUCCESS run. Its output ``sintese/dashboard.html`` is published
under the same key by the ``sintese/`` loop of ``publish``. Without
``output/simulation/`` the command would only exit 1, so there are no
arguments and no step runs.
"""

from __future__ import annotations

import re

from hpc_model_utils.core.outputs import (
    OutputPlan,
    RawFile,
    Selector,
    Tree,
    ZipGroup,
)
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models.cobre import case

_ARCHIVED_PHASES: tuple[str, ...] = ("training", "policy", "simulation")
_RAW_METADATA_PHASES: tuple[str, ...] = ("training", "simulation")
_DASHBOARD = "sintese/dashboard.html"


def output_plan(ws: Workspace) -> OutputPlan:
    out = f"{case.case_prefix(ws)}output"
    return OutputPlan(
        deck_inputs=case.input_files(ws),
        groups=tuple(
            ZipGroup(
                f"{phase}.zip",
                Selector(
                    patterns=(rf"{re.escape(out)}/{phase}/.+",),
                    recursive=True,
                ),
                Tree(out),
            )
            for phase in _ARCHIVED_PHASES
        ),
        raw=tuple(
            RawFile(
                Selector(
                    names=(f"{out}/{phase}/metadata.json",), recursive=True
                ),
                dest=phase,
            )
            for phase in _RAW_METADATA_PHASES
        ),
        deck_layout=Tree(case.top_folder(ws) or ""),
    )


def dashboard_args(ws: Workspace) -> tuple[str, ...] | None:
    if not (case.case_root(ws) / "output" / "simulation").is_dir():
        return None
    return ("dashboard", case.top_folder(ws) or ".", "--output", _DASHBOARD)
