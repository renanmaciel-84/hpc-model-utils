"""ADR-014/ADR-052/ADR-057/R73/R77/R107/R112/R134: the cobre L3 outcome
mapping (ticket-070).

``LOG_PATTERNS`` is empty because every cobre condition needs the exit
code or a fact absent from a line. The L3 table never returns
``INFEASIBLE`` (an LP infeasibility is exit 3). The output metadata is
read at exit 0 and at exit 5 (cobre v0.18.0 and later: a SIGTERM or
SIGINT stopped training after its outputs and policy checkpoint were
written), and ``primary_evidence`` is the exit file alone, so exits 1
and 2 (which write no metadata) stay ``DATA_ERROR``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hpc_model_utils.core.diagnosis import (
    EvidenceItem,
    JobReport,
    LogPattern,
    RunStatus,
    Verdict,
)
from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models.cobre import case, launch

LOG_PATTERNS: tuple[LogPattern, ...] = ()

# Verified against cobre v0.14.0-v0.18.0 and novomodelo v0.18.0-v0.18.1:
# TERMINATION_REASONS are the RULE_* constants of
# crates/novomodelo-sddp/src/convergence/stopping_rule.rs
# (crates/cobre-sddp/... before the fork), and the Backend/Solver patterns
# match the labels written by
# crates/novomodelo-cli/src/summary.rs::print_execution_topology
# (crates/cobre-cli/... before the fork). HPCMU_SHAPE comes from
# core.launch.render_prelude. A novomodelo version that changes any of them
# needs a change here before it is offered.
TERMINATION_REASONS = frozenset(
    {
        "iteration_limit",
        "time_limit",
        "bound_stalling",
        "gap",
        "graceful_shutdown",
    }
)
SHAPE_LINE = re.compile(r"\bHPCMU_SHAPE nodes=([0-9]+)\b")
BACKEND_LINE = re.compile(r"^\s*Backend:\s+(\S.*?)\s*$")
SOLVER_LINE = re.compile(r"^\s*Solver:\s+(\S.*?)\s*$")
BACKEND_MPI = re.compile(r"^MPI\b")

# cobre's exits: 0 completed, 1-4 the CliError classes, and from v0.18.0
# 5, a stop on request
# (crates/novomodelo-cli/src/commands/run/mod.rs::RunOutcome).
# A signal that arrives while the training outputs are written still exits
# 5, with the metadata's own termination_reason. novomodelo v0.18.1 exits
# 101 on a panic (PANIC_EXIT_CODE), which is not a known exit.
_STOPPED_ON_REQUEST = 5
_KNOWN_EXITS = range(_STOPPED_ON_REQUEST + 1)
_OUTPUT_EXITS = frozenset({0, _STOPPED_ON_REQUEST})
# v0.18.0 renamed the metadata key cobre_version to software_version.
_VERSION_KEYS = ("software_version", "cobre_version")
_GRACEFUL_SHUTDOWN = "graceful_shutdown"
_TIMEOUT_STATES = frozenset({"TIMEOUT", "DEADLINE"})
_TRAINING_FILE = "training/metadata.json"
_SIMULATION_FILE = "simulation/metadata.json"


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    status: str
    iterations: int
    termination_reason: str
    lower_bound: int | float
    world_size: int
    cobre_version: str


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    total: int
    completed: int
    failed: int
    world_size: int


@dataclass(frozen=True, slots=True)
class LogFacts:
    read: int
    nodes: int | None
    backend: str | None
    solver: str | None
    backend_mpi: bool


@dataclass(frozen=True, slots=True)
class RunOutputs:
    phases: case.Phases | None = None
    training: TrainingSummary | None = None
    simulation: SimulationSummary | None = None
    training_missing: bool = False
    simulation_missing: bool = False
    problem: tuple[str, str] | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CobreEvidence:
    code: int | None
    state: str | None
    logs: LogFacts
    outputs: RunOutputs


def _field(data: Mapping[str, object], dotted: str) -> object:
    node: object = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise DataError(f"{dotted} is missing")
        node = node[key]
    return node


def _str_field(data: Mapping[str, object], dotted: str) -> str:
    value = _field(data, dotted)
    if not isinstance(value, str):
        raise DataError(f"{dotted} is not a string")
    return value


def _int_field(data: Mapping[str, object], dotted: str) -> int:
    value = _field(data, dotted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"{dotted} is not an integer")
    return value


def _number_field(data: Mapping[str, object], dotted: str) -> int | float:
    value = _field(data, dotted)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataError(f"{dotted} is not a number")
    return value


def _cobre_version(data: Mapping[str, object]) -> str:
    for key in _VERSION_KEYS:
        if key in data:
            return _str_field(data, key)
    raise DataError(
        "software_version (cobre_version before v0.18.0) is missing"
    )


def _training_summary(data: Mapping[str, object]) -> TrainingSummary:
    return TrainingSummary(
        status=_str_field(data, "status"),
        iterations=_int_field(data, "iterations.completed"),
        termination_reason=_str_field(data, "convergence.termination_reason"),
        lower_bound=_number_field(data, "bounds.final_lower_bound"),
        world_size=_int_field(data, "distribution.world_size"),
        cobre_version=_cobre_version(data),
    )


def _simulation_summary(data: Mapping[str, object]) -> SimulationSummary:
    return SimulationSummary(
        total=_int_field(data, "scenarios.total"),
        completed=_int_field(data, "scenarios.completed"),
        failed=_int_field(data, "scenarios.failed"),
        world_size=_int_field(data, "distribution.world_size"),
    )


def _load_object(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as err:
        raise DataError("cannot be read as UTF-8 text") from err
    except json.JSONDecodeError as err:
        raise DataError(f"invalid JSON: {err}") from err
    if not isinstance(data, dict):
        raise DataError("not a JSON object")
    return data


def _scan_logs(paths: Sequence[Path]) -> LogFacts:
    read = 0
    nodes: int | None = None
    backend: str | None = None
    solver: str | None = None
    backend_mpi = False
    for path in paths:
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    line = raw.decode("utf-8", errors="replace")
                    if (shape := SHAPE_LINE.search(line)) is not None:
                        nodes = int(shape.group(1))
                    if (found := BACKEND_LINE.match(line)) is not None:
                        backend = found.group(1)
                        backend_mpi = backend_mpi or bool(
                            BACKEND_MPI.match(backend)
                        )
                    if (found := SOLVER_LINE.match(line)) is not None:
                        solver = found.group(1)
        except OSError:
            continue
        read += 1
    return LogFacts(read, nodes, backend, solver, backend_mpi)


def _read_outputs(ws: Workspace) -> RunOutputs:
    try:
        phases = case.phases(ws)
        output = case.case_root(ws) / "output"
    except DataError as err:
        return RunOutputs(problem=("config.json", str(err)))
    problems: list[tuple[str, str]] = []
    training: TrainingSummary | None = None
    simulation: SimulationSummary | None = None
    training_missing = simulation_missing = False
    if phases.training:
        path = output / _TRAINING_FILE
        training_missing = not path.exists()
        if not training_missing:
            try:
                training = _training_summary(_load_object(path))
            except DataError as err:
                problems.append((_TRAINING_FILE, str(err)))
    if phases.simulation:
        path = output / _SIMULATION_FILE
        simulation_missing = not path.exists()
        if not simulation_missing:
            try:
                simulation = _simulation_summary(_load_object(path))
            except DataError as err:
                problems.append((_SIMULATION_FILE, str(err)))
    notes: tuple[str, ...] = ()
    if (
        training is not None
        and training.termination_reason not in TERMINATION_REASONS
    ):
        notes = (
            f"termination_reason {training.termination_reason!r} is outside "
            "the verified cobre vocabulary",
        )
    return RunOutputs(
        phases=phases,
        training=training,
        simulation=simulation,
        training_missing=training_missing,
        simulation_missing=simulation_missing,
        problem=problems[0] if problems else None,
        notes=notes,
    )


def extract_evidence(ws: Workspace, job: JobReport) -> CobreEvidence:
    return CobreEvidence(
        code=job.process_exit,
        state=None if job.outcome is None else job.outcome.state,
        logs=_scan_logs(job.log_paths),
        outputs=(
            _read_outputs(ws)
            if job.process_exit in _OUTPUT_EXITS
            else RunOutputs()
        ),
    )


def _world_size(evidence: CobreEvidence) -> int | None:
    if evidence.outputs.training is not None:
        return evidence.outputs.training.world_size
    if evidence.outputs.simulation is not None:
        return evidence.outputs.simulation.world_size
    return None


def _mpi_not_started(evidence: CobreEvidence) -> bool:
    return (
        evidence.code is not None
        and evidence.logs.read > 0
        and not evidence.logs.backend_mpi
    )


def _unexpected_exit(evidence: CobreEvidence) -> bool:
    return evidence.code is not None and evidence.code not in _KNOWN_EXITS


def _stopped_on_request(evidence: CobreEvidence) -> bool:
    training = evidence.outputs.training
    return evidence.code == _STOPPED_ON_REQUEST or (
        training is not None
        and training.termination_reason == _GRACEFUL_SHUTDOWN
    )


def _shutdown_timeout(evidence: CobreEvidence) -> bool:
    return _stopped_on_request(evidence) and evidence.state in _TIMEOUT_STATES


def _shutdown_cancelled(evidence: CobreEvidence) -> bool:
    return _stopped_on_request(evidence) and not _shutdown_timeout(evidence)


def _training_partial(evidence: CobreEvidence) -> bool:
    training = evidence.outputs.training
    return training is not None and training.status != "complete"


def _scenarios_failed(evidence: CobreEvidence) -> bool:
    simulation = evidence.outputs.simulation
    return simulation is not None and simulation.failed > 0


def _simulation_incomplete(evidence: CobreEvidence) -> bool:
    simulation = evidence.outputs.simulation
    return (
        simulation is not None
        and simulation.completed + simulation.failed != simulation.total
    )


def _rank_count_mismatch(evidence: CobreEvidence) -> bool:
    world_size = _world_size(evidence)
    nodes = evidence.logs.nodes
    return world_size is not None and nodes is not None and world_size != nodes


def _no_phase_enabled(evidence: CobreEvidence) -> bool:
    phases = evidence.outputs.phases
    return phases is not None and not phases.training and not phases.simulation


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    status: RunStatus
    template: str
    applies: Callable[[CobreEvidence], bool]


# ADR-052 L3 table: the first row whose ``applies`` predicate is true
# wins. Rows 8-17 read only exit-0 and exit-5 evidence (``RunOutputs`` is
# empty at any other exit). ``cobre.completed`` is the fallback -- true
# exactly when no row of ``_DIAGNOSTIC_RULES`` applies.
_DIAGNOSTIC_RULES: tuple[Rule, ...] = (
    Rule(
        "cobre.exit_unknown",
        RunStatus.RUNTIME_ERROR,
        "model exit code unreadable (.hpcmu/jobs/model.exit)",
        lambda ev: ev.code is None,
    ),
    Rule(
        "cobre.mpi_not_started",
        RunStatus.RUNTIME_ERROR,
        launch.MPI_BINARY
        + " exited {code} without reporting an MPI backend ({seen}); "
        "check the cobre MPICH, PMIx and EFA",
        _mpi_not_started,
    ),
    Rule(
        "cobre.validation_error",
        RunStatus.DATA_ERROR,
        "cobre exited 1: case validation failed",
        lambda ev: ev.code == 1,
    ),
    Rule(
        "cobre.io_error",
        RunStatus.DATA_ERROR,
        "cobre exited 2: case I/O error",
        lambda ev: ev.code == 2,
    ),
    Rule(
        "cobre.solver_error",
        RunStatus.RUNTIME_ERROR,
        "cobre exited 3: solver error",
        lambda ev: ev.code == 3,
    ),
    Rule(
        "cobre.internal_error",
        RunStatus.RUNTIME_ERROR,
        "cobre exited 4: internal error",
        lambda ev: ev.code == 4,
    ),
    Rule(
        "cobre.unexpected_exit",
        RunStatus.RUNTIME_ERROR,
        "cobre exited {code}",
        _unexpected_exit,
    ),
    Rule(
        "cobre.metadata_unreadable",
        RunStatus.RUNTIME_ERROR,
        "{file}: {problem}",
        lambda ev: ev.outputs.problem is not None,
    ),
    Rule(
        "cobre.training_metadata_missing",
        RunStatus.RUNTIME_ERROR,
        "training enabled but training/metadata.json is missing",
        lambda ev: ev.outputs.training_missing,
    ),
    Rule(
        "cobre.simulation_metadata_missing",
        RunStatus.RUNTIME_ERROR,
        "simulation enabled but simulation/metadata.json is missing",
        lambda ev: ev.outputs.simulation_missing,
    ),
    Rule(
        "cobre.shutdown_timeout",
        RunStatus.TIMEOUT,
        "training stopped on request after {iterations} iterations "
        "(Slurm {state})",
        _shutdown_timeout,
    ),
    Rule(
        "cobre.shutdown_cancelled",
        RunStatus.CANCELLED,
        "training stopped on request after {iterations} iterations",
        _shutdown_cancelled,
    ),
    Rule(
        "cobre.training_partial",
        RunStatus.RUNTIME_ERROR,
        "training {status} after {iterations} iterations",
        _training_partial,
    ),
    Rule(
        "cobre.scenarios_failed",
        RunStatus.RUNTIME_ERROR,
        "{failed} of {total} simulation scenarios failed "
        "({completed} completed)",
        _scenarios_failed,
    ),
    Rule(
        "cobre.simulation_incomplete",
        RunStatus.RUNTIME_ERROR,
        "simulation incomplete: {completed} completed, {failed} failed, "
        "{total} total",
        _simulation_incomplete,
    ),
    Rule(
        "cobre.rank_count_mismatch",
        RunStatus.RUNTIME_ERROR,
        "cobre ran {world_size} MPI rank(s) on {nodes} node(s); "
        "one rank per node is required",
        _rank_count_mismatch,
    ),
    Rule(
        "cobre.no_phase_enabled",
        RunStatus.SUCCESS,
        "training and simulation both disabled: nothing ran",
        _no_phase_enabled,
    ),
)


def _completed(evidence: CobreEvidence) -> bool:
    return not any(rule.applies(evidence) for rule in _DIAGNOSTIC_RULES)


RULES: tuple[Rule, ...] = (
    *_DIAGNOSTIC_RULES,
    Rule(
        "cobre.completed",
        RunStatus.SUCCESS,
        "{training_part}; {simulation_part}",
        _completed,
    ),
)


def _fields(evidence: CobreEvidence) -> dict[str, object]:
    outputs = evidence.outputs
    training, simulation = outputs.training, outputs.simulation
    fields: dict[str, object] = {
        "code": evidence.code,
        "state": evidence.state,
        "seen": (
            "no Backend line"
            if evidence.logs.backend is None
            else f"last Backend line: {evidence.logs.backend}"
        ),
        "nodes": evidence.logs.nodes,
        "world_size": _world_size(evidence),
        "iterations": "?",
        "training_part": "training disabled",
        "simulation_part": "simulation disabled",
    }
    if outputs.problem is not None:
        fields["file"], fields["problem"] = outputs.problem
    if training is not None:
        fields["status"] = training.status
        fields["iterations"] = training.iterations
        fields["training_part"] = (
            f"training {training.termination_reason} after "
            f"{training.iterations} iterations, "
            f"lower bound {training.lower_bound:.6g}"
        )
    if simulation is not None:
        fields["completed"] = simulation.completed
        fields["failed"] = simulation.failed
        fields["total"] = simulation.total
        fields["simulation_part"] = (
            f"simulation {simulation.completed}/{simulation.total} scenarios"
        )
    return fields


def _plugin(source: str, detail: str) -> EvidenceItem:
    return EvidenceItem("plugin", source, detail)


def _evidence_items(evidence: CobreEvidence) -> tuple[EvidenceItem, ...]:
    outputs, logs = evidence.outputs, evidence.logs
    items: list[EvidenceItem] = []
    if evidence.code is not None:
        items.append(_plugin("model.exit", str(evidence.code)))
    if (training := outputs.training) is not None:
        items.append(
            _plugin(
                _TRAINING_FILE,
                f"status {training.status}; "
                f"termination_reason {training.termination_reason}; "
                f"iterations {training.iterations}; "
                f"final_lower_bound {training.lower_bound}; "
                f"cobre_version {training.cobre_version}; "
                f"world_size {training.world_size}",
            )
        )
    if (simulation := outputs.simulation) is not None:
        items.append(
            _plugin(
                _SIMULATION_FILE,
                f"completed {simulation.completed}; "
                f"failed {simulation.failed}; total {simulation.total}; "
                f"world_size {simulation.world_size}",
            )
        )
    if logs.read:
        nodes = "?" if logs.nodes is None else logs.nodes
        items.append(
            _plugin(
                "model log",
                f"HPCMU_SHAPE nodes={nodes}; "
                f"Backend: {logs.backend or '?'}; Solver: {logs.solver or '?'}",
            )
        )
        if logs.nodes is None:
            items.append(
                _plugin(
                    "model log",
                    "rank check skipped: no HPCMU_SHAPE nodes value",
                )
            )
    else:
        items.append(
            _plugin("model log", "unreadable: backend and rank checks skipped")
        )
    items.extend(_plugin(_TRAINING_FILE, note) for note in outputs.notes)
    return tuple(items)


def primary_evidence(ws: Workspace) -> tuple[Path, ...]:
    return (ws.model_exit_path,)


def diagnose(ws: Workspace, job: JobReport) -> Verdict:
    evidence = extract_evidence(ws, job)
    applicable = [rule for rule in RULES if rule.applies(evidence)]
    winner = applicable[0]
    return Verdict(
        status=winner.status,
        rule_id=winner.rule_id,
        reason=winner.template.format(**_fields(evidence)),
        evidence=_evidence_items(evidence),
        matched=tuple(rule.rule_id for rule in applicable),
    )
