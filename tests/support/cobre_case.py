"""ADR-025/R105: synthetic cobre cases, output trees and ELF stubs.

No cobre fixture is vendored: the case is built in code, shaped after
cobre's ``examples/1dtoy`` and the metadata structs of
``crates/cobre-io/src/output/manifest.rs`` (novomodelo:
``crates/novomodelo-io/src/output/manifest.rs``; v0.18.0 by default,
v0.17.0 on request). Every builder writes only under the caller's
``tmp_path`` (or the directory it is handed).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from hpc_model_utils.core.workspace import Workspace
from tests.support.decks import DeckWorkspace

_PLACEHOLDER_PARQUET = b"PAR1 placeholder PAR1"
COBRE_RELEASE = "0.18.0"
_PLACEHOLDER_CUTS = b"cobre-cuts placeholder"

COBRE_EXECUTION_LINES: tuple[str, ...] = (
    "Execution",
    "  Solver:    HiGHS 1.11.0",
    "  Backend:   MPI (MPICH 4.2.3, MPI 4.1)",
)
"""The ``Execution`` block rank 0 prints to stderr before loading the case:
cobre v0.17.0 and v0.18.0
``crates/cobre-cli/src/summary.rs::print_execution_topology`` (novomodelo:
``crates/novomodelo-cli/src/summary.rs::print_execution_topology``) for an
MPI backend (the model-log lines ticket-070 reads)."""


def _json_bytes(data: object) -> bytes:
    return json.dumps(data, indent=2).encode("utf-8")


def case_members(
    *,
    training: bool = True,
    simulation: bool = True,
    start_dates: Sequence[str] = ("2024-01-01",),
) -> dict[str, bytes]:
    config = {
        "training": {
            "enabled": training,
            "selection": {"method": "sampled", "forward_passes": 1},
            "stopping_rules": [{"type": "iteration_limit", "limit": 128}],
        },
        "simulation": {
            "enabled": simulation,
            "selection": {"method": "sampled", "num_scenarios": 100},
        },
    }
    stages = {
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.12,
        },
        "stages": [
            {
                "id": index,
                "start_date": start_date,
                "blocks": [{"id": 0, "name": "SINGLE", "hours": 744}],
                "num_openings": 10,
            }
            for index, start_date in enumerate(start_dates)
        ],
    }
    return {
        "config.json": _json_bytes(config),
        "stages.json": _json_bytes(stages),
        "initial_conditions.json": _json_bytes(
            {
                "storage": [{"hydro_id": 0, "value_hm3": 83.222}],
                "filling_storage": [],
            }
        ),
        "penalties.json": _json_bytes(
            {
                "bus": {
                    "deficit_segments": [{"depth_mw": None, "cost": 7500.0}],
                    "excess_cost": 100.0,
                },
                "line": {"exchange_cost": 2.0},
            }
        ),
        "system/buses.json": _json_bytes({"buses": [{"id": 0, "name": "SIN"}]}),
        "system/hydros.json": _json_bytes(
            {"hydros": [{"id": 0, "name": "UHE1", "downstream_id": None}]}
        ),
        "system/thermals.json": _json_bytes(
            {"thermals": [{"id": 0, "name": "UTE1", "bus_id": 0}]}
        ),
    }


def cobre_workspace(
    tmp_path: Path,
    *,
    top: str | None = "caso_cobre",
    members: Mapping[str, bytes] | None = None,
    stale_outputs: Sequence[str] = (),
) -> DeckWorkspace:
    """``members`` and ``stale_outputs`` are named relative to the case
    root; both land under ``top/`` in the zip when ``top`` is set."""
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    prefix = "" if top is None else f"{top}/"
    entries = {
        f"{prefix}{name}": data
        for name, data in (
            case_members() if members is None else members
        ).items()
    }
    for name in stale_outputs:
        entries[f"{prefix}{name}"] = b"STALE " + name.encode()
    archive_path = root / "eco_deck.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    (root / "assets").mkdir(exist_ok=True)
    return DeckWorkspace(Workspace.at(root), tuple(sorted(entries)))


def _merge(
    defaults: dict[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(defaults)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _distribution() -> dict[str, object]:
    return {
        "backend": "local",
        "world_size": 1,
        "ranks_participated": 1,
        "num_hosts": 1,
        "threads_per_rank": 1,
        "hosts": [],
    }


def _software(release: str) -> dict[str, object]:
    """The version header: v0.18.0 replaced ``cobre_version`` with
    ``software`` and ``software_version``."""
    if tuple(int(part) for part in release.split(".")) < (0, 18, 0):
        return {"cobre_version": release}
    return {"software": "cobre", "software_version": release}


def training_metadata(
    *, release: str = COBRE_RELEASE, **fields: object
) -> dict[str, object]:
    """cobre ``TrainingMetadata`` as ``release`` writes it; nested
    mappings in ``fields`` are merged into the defaults, any other value
    replaces its key."""
    defaults: dict[str, object] = {
        **_software(release),
        "hostname": "node001",
        "solver": "highs",
        "solver_version": "1.11.0",
        "started_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:10:00Z",
        "duration_seconds": 600.0,
        "status": "complete",
        "configuration": {
            "seed": 42,
            "max_iterations": 128,
            "forward_passes": 1,
            "stopping_mode": "any",
            "policy_mode": "fresh",
        },
        "problem_dimensions": {
            "num_stages": 4,
            "num_hydros": 1,
            "num_thermals": 2,
            "num_buses": 1,
            "num_lines": 0,
        },
        "iterations": {"completed": 128, "converged_at": None},
        "convergence": {
            "achieved": False,
            "final_gap_percent": None,
            "termination_reason": "iteration_limit",
        },
        "row_pool": {
            "total_generated": 1000,
            "total_active": 800,
            "peak_active": 900,
        },
        "bounds": {
            "final_lower_bound": 1234.5,
            "final_upper_bound": None,
            "final_upper_bound_std": None,
            "final_upper_bound_kind": "statistical",
        },
        "solve_stats": {},
        "distribution": _distribution(),
    }
    return _merge(defaults, fields)


def simulation_metadata(
    *, release: str = COBRE_RELEASE, **fields: object
) -> dict[str, object]:
    """cobre ``SimulationMetadata`` as ``release`` writes it; merge rules
    as in ``training_metadata``."""
    defaults: dict[str, object] = {
        **_software(release),
        "hostname": "node001",
        "solver": "highs",
        "solver_version": "1.11.0",
        "started_at": "2024-01-01T00:10:00Z",
        "completed_at": "2024-01-01T00:12:00Z",
        "duration_seconds": 120.0,
        "status": "complete",
        "scenarios": {"total": 100, "completed": 100, "failed": 0},
        "cost": {"mean_cost": 1500.0, "std_cost": 25.0},
        "solve_stats": {},
        "distribution": _distribution(),
    }
    return _merge(defaults, fields)


type _Metadata = Mapping[str, object] | None | Literal["default"]


def write_outputs(
    case_dir: Path,
    *,
    training: _Metadata = "default",
    simulation: _Metadata = "default",
    policy: bool = True,
) -> tuple[str, ...]:
    """Write ``<case_dir>/output/`` and return the sorted file names
    relative to ``output/``. A ``None`` metadata skips that phase's whole
    directory; ``"default"`` writes the builder's default metadata."""
    files: dict[str, bytes] = {}
    train = training_metadata() if isinstance(training, str) else training
    sim = simulation_metadata() if isinstance(simulation, str) else simulation
    if train is not None:
        files.update(
            {
                "training/metadata.json": _json_bytes(train),
                "training/_SUCCESS": b"",
                "training/convergence.parquet": _PLACEHOLDER_PARQUET,
                "training/dictionaries/codes.json": _json_bytes({"codes": {}}),
                "training/timing/iterations.parquet": _PLACEHOLDER_PARQUET,
            }
        )
    if policy:
        files["policy/cuts/stage_000.bin"] = _PLACEHOLDER_CUTS
    if sim is not None:
        files.update(
            {
                "simulation/metadata.json": _json_bytes(sim),
                "simulation/_SUCCESS": b"",
                "simulation/costs/scenario_id=0000/part-0000.parquet": (
                    _PLACEHOLDER_PARQUET
                ),
            }
        )
    output = case_dir / "output"
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return tuple(sorted(files))


def write_elf_stub(assets: Path, *, name: str = "novomodelo-mpi") -> Path:
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / name
    path.write_bytes(b"\x7fELF\x02\x01\x01" + bytes(57))
    path.chmod(0o755)
    return path
