"""ADR-014/ADR-052/ADR-057/R73/R77/R107/R112/R134 tests for the cobre L3
outcome mapping (ticket-070)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest

from hpc_model_utils.core.diagnosis import (
    Diagnosis,
    EvidenceItem,
    JobReport,
    RunStatus,
    Verdict,
)
from hpc_model_utils.core.lifecycle.finalize import diagnose_workspace
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.slurm import JobOutcome
from hpc_model_utils.models.cobre import CobrePlugin, case, diagnosis
from tests.support.cobre_case import (
    COBRE_EXECUTION_LINES,
    case_members,
    cobre_workspace,
    simulation_metadata,
    training_metadata,
    write_outputs,
)

type _Metadata = Mapping[str, object] | None | Literal["default"]

_SHAPE = "HPCMU_SHAPE nodes=1 cpus_on_node=4"
_LOG_OK = (_SHAPE, *COBRE_EXECUTION_LINES)
_MPI_VALUE = "MPI (MPICH 4.2.3, MPI 4.1)"
_ABSENT = object()
_MAX_EVIDENCE = 6


def _workspace(
    tmp_path: Path,
    *,
    enabled: case.Phases = case.Phases(training=True, simulation=True),
    training: _Metadata = "default",
    simulation: _Metadata = "default",
    top: str | None = "caso_cobre",
) -> Workspace:
    members = case_members(
        training=enabled.training, simulation=enabled.simulation
    )
    ws = cobre_workspace(tmp_path, top=top, members=members).ws
    write_outputs(case.case_root(ws), training=training, simulation=simulation)
    ws.ensure_layout()
    return ws


def _outcome(state: str) -> JobOutcome:
    return JobOutcome(
        job_id="1",
        state=state,
        exit_code=0,
        signal=None,
        elapsed="00:10:00",
        time_limit="01:00:00",
        oom=False,
        source="sacct",
        raw="",
    )


def _write_log(
    ws: Workspace, lines: Sequence[str], *, job_id: str = "1"
) -> Path:
    path = ws.log_path(Phase.MODEL, job_id)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _job(
    ws: Workspace,
    code: int | None,
    *,
    lines: Sequence[str] | None = _LOG_OK,
    state: str | None = None,
) -> JobReport:
    if code is not None:
        ws.model_exit_path.write_text(f"{code}\n", encoding="utf-8")
    logs = () if lines is None else (_write_log(ws, lines),)
    outcome = None if state is None else _outcome(state)
    return JobReport(outcome=outcome, process_exit=code, log_paths=logs)


def _check_invariants(
    ws: Workspace,
    status: RunStatus,
    texts: Sequence[str],
    evidence: Sequence[EvidenceItem],
) -> None:
    assert status is not RunStatus.INFEASIBLE
    assert len(evidence) <= _MAX_EVIDENCE
    root = str(ws.root.parent)
    for item in evidence:
        assert item.layer in {"plugin", "guard", "slurm", "log"}
        assert root not in item.source
        assert root not in item.detail
    for text in texts:
        assert root not in text


def _diagnose(ws: Workspace, job: JobReport) -> Diagnosis:
    diag = diagnose_workspace(ws, CobrePlugin(), job, job_id="1")
    _check_invariants(
        ws, diag.status, (diag.reason, diag.annotation()), diag.evidence
    )
    return diag


def _verdict(ws: Workspace, job: JobReport) -> Verdict:
    verdict = diagnosis.diagnose(ws, job)
    _check_invariants(ws, verdict.status, (verdict.reason,), verdict.evidence)
    return verdict


def _details(diag: Diagnosis | Verdict, source: str) -> list[str]:
    return [i.detail for i in diag.evidence if i.source == source]


def _mutated(
    base: dict[str, object], dotted: str, value: object
) -> dict[str, object]:
    data = copy.deepcopy(base)
    *parents, leaf = dotted.split(".")
    node: dict[str, object] = data
    for key in parents:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    if value is _ABSENT:
        del node[leaf]
    else:
        node[leaf] = value
    return data


def _write_raw(ws: Workspace, relative: str, data: bytes) -> None:
    path = case.case_root(ws) / "output" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# -- exit codes and the in-job backend rows ----------------------------


@pytest.mark.parametrize(
    ("code", "status", "rule_id", "reason"),
    [
        (
            1,
            RunStatus.DATA_ERROR,
            "cobre.validation_error",
            "cobre exited 1: case validation failed",
        ),
        (
            2,
            RunStatus.DATA_ERROR,
            "cobre.io_error",
            "cobre exited 2: case I/O error",
        ),
        (
            3,
            RunStatus.RUNTIME_ERROR,
            "cobre.solver_error",
            "cobre exited 3: solver error",
        ),
        (
            4,
            RunStatus.RUNTIME_ERROR,
            "cobre.internal_error",
            "cobre exited 4: internal error",
        ),
        (
            6,
            RunStatus.RUNTIME_ERROR,
            "cobre.unexpected_exit",
            "cobre exited 6",
        ),
        (
            101,
            RunStatus.RUNTIME_ERROR,
            "cobre.unexpected_exit",
            "cobre exited 101",
        ),
        (
            137,
            RunStatus.RUNTIME_ERROR,
            "cobre.unexpected_exit",
            "cobre exited 137",
        ),
        (
            -9,
            RunStatus.RUNTIME_ERROR,
            "cobre.unexpected_exit",
            "cobre exited -9",
        ),
    ],
)
def test_exit_code_rows_with_an_mpi_backend_line(
    tmp_path: Path, code: int, status: RunStatus, rule_id: str, reason: str
) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, code))
    assert (diag.status, diag.rule_id) == (status, rule_id)
    assert diag.reason == reason
    assert diag.annotation() == f"{status}: {reason} [{rule_id}]"
    assert diag.matched == (rule_id,)


def test_exit_nonzero_ignores_the_output_metadata(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, training="default", simulation="default")
    _write_raw(ws, "training/metadata.json", b"{")
    diag = _diagnose(ws, _job(ws, 1))
    assert diag.rule_id == "cobre.validation_error"
    assert [i.source for i in diag.evidence] == ["model.exit", "model log"]


@pytest.mark.parametrize("code", [1, 4, 0])
def test_backend_missing_line_is_mpi_not_started(
    tmp_path: Path, code: int
) -> None:
    ws = _workspace(tmp_path)
    lines = (_SHAPE, "Execution", "  Solver:    HiGHS 1.11.0")
    diag = _diagnose(ws, _job(ws, code, lines=lines))
    assert diag.status is RunStatus.RUNTIME_ERROR
    assert diag.rule_id == "cobre.mpi_not_started"
    assert "no Backend line" in diag.annotation()
    assert diag.reason == (
        f"novomodelo-mpi exited {code} without reporting an MPI backend "
        "(no Backend line); check the cobre MPICH, PMIx and EFA"
    )


def test_backend_local_line_is_mpi_not_started(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    lines = (
        _SHAPE,
        "Execution",
        "  Solver:    HiGHS 1.11.0",
        "  Backend:   local",
    )
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert diag.rule_id == "cobre.mpi_not_started"
    assert "last Backend line: local" in diag.annotation()
    assert _details(diag, "model log") == [
        "HPCMU_SHAPE nodes=1; Backend: local; Solver: HiGHS 1.11.0"
    ]


@pytest.mark.parametrize(
    ("value", "started"),
    [
        ("MPI", True),
        (_MPI_VALUE, True),
        ("MPIX", False),
        ("mpi", False),
        ("local", False),
        ("not MPI", False),
    ],
)
def test_backend_value_must_start_with_mpi(
    tmp_path: Path, value: str, started: bool
) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 0, lines=(_SHAPE, f"  Backend:   {value}")))
    assert (diag.rule_id != "cobre.mpi_not_started") is started


def test_backend_annotation_names_the_last_backend_line(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    lines = (_SHAPE, "  Backend:   local", "  Backend:   tcp")
    diag = _diagnose(ws, _job(ws, 4, lines=lines))
    assert diag.rule_id == "cobre.mpi_not_started"
    assert "last Backend line: tcp" in diag.annotation()


def test_backend_any_read_line_with_mpi_is_enough(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    lines = (_SHAPE, f"  Backend:   {_MPI_VALUE}", "  Backend:   local")
    assert _diagnose(ws, _job(ws, 3, lines=lines)).rule_id == (
        "cobre.solver_error"
    )


def test_backend_second_log_can_hold_the_line(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    first = _write_log(ws, (_SHAPE,), job_id="1")
    second = _write_log(ws, COBRE_EXECUTION_LINES, job_id="2")
    ws.model_exit_path.write_text("2\n", encoding="utf-8")
    diag = _diagnose(ws, JobReport(None, 2, (first, second)))
    assert diag.rule_id == "cobre.io_error"


def test_backend_bytes_that_are_not_utf8_do_not_stop_the_scan(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    log = ws.log_path(Phase.MODEL, "1")
    body = b"\xff\xfe broken line\n" + "\n".join(_LOG_OK).encode() + b"\n"
    log.write_bytes(body)
    ws.model_exit_path.write_text("1\n", encoding="utf-8")
    diag = _diagnose(ws, JobReport(None, 1, (log,)))
    assert diag.rule_id == "cobre.validation_error"


def test_backend_without_a_log_keeps_the_exit_row(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 1, lines=None))
    assert diag.rule_id == "cobre.validation_error"
    assert _details(diag, "model log") == [
        "unreadable: backend and rank checks skipped"
    ]


def test_backend_unreadable_log_is_skipped(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    ws.model_exit_path.write_text("1\n", encoding="utf-8")
    missing = ws.log_path(Phase.MODEL, "7")
    diag = _diagnose(ws, JobReport(None, 1, (missing,)))
    assert diag.rule_id == "cobre.validation_error"
    assert _details(diag, "model log") == [
        "unreadable: backend and rank checks skipped"
    ]


def test_backend_readable_log_after_an_unreadable_one_is_used(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    missing = ws.log_path(Phase.MODEL, "7")
    good = _write_log(ws, (_SHAPE,))
    ws.model_exit_path.write_text("1\n", encoding="utf-8")
    diag = _diagnose(ws, JobReport(None, 1, (missing, good)))
    assert diag.rule_id == "cobre.mpi_not_started"


def test_backend_matched_lists_every_applicable_row_in_order(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(status="partial"),
        simulation=simulation_metadata(
            scenarios={"total": 10, "completed": 8, "failed": 2}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0, lines=(_SHAPE,)))
    assert diag.matched == (
        "cobre.mpi_not_started",
        "cobre.training_partial",
        "cobre.scenarios_failed",
    )


def test_exit_malformed_file_is_exit_unknown(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    ws.model_exit_path.write_text("not a number\n", encoding="utf-8")
    diag = _diagnose(ws, _job(ws, None))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.exit_unknown",
    )
    assert diag.reason == "model exit code unreadable (.hpcmu/jobs/model.exit)"
    assert diag.matched == ("cobre.exit_unknown",)
    assert "model.exit" not in [i.source for i in diag.evidence]


def test_exit_unknown_comes_before_the_backend_check(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    ws.model_exit_path.write_text("", encoding="utf-8")
    diag = _diagnose(ws, _job(ws, None, lines=(_SHAPE,)))
    assert diag.rule_id == "cobre.exit_unknown"
    assert diag.matched == ("cobre.exit_unknown",)


def test_exit_missing_file_is_core_missing_output(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, None))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "core.missing_output",
    )
    assert diag.reason == "missing output file(s): model.exit"


@pytest.mark.parametrize("code", [0, 1, 3])
def test_exit_slurm_timeout_comes_before_every_cobre_row(
    tmp_path: Path, code: int
) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, code, lines=(_SHAPE,), state="TIMEOUT"))
    assert (diag.status, diag.rule_id) == (RunStatus.TIMEOUT, "slurm.timeout")


def test_exit_primary_evidence_is_the_exit_file_alone(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    assert CobrePlugin().primary_evidence(ws) == (ws.model_exit_path,)
    assert diagnosis.primary_evidence(ws) == (ws.model_exit_path,)
    assert diagnosis.LOG_PATTERNS == ()


# -- phase-aware rows (R112) -------------------------------------------


def test_phase_missing_training_metadata(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, training=None)
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.training_metadata_missing",
    )
    assert diag.reason == (
        "training enabled but training/metadata.json is missing"
    )


def test_phase_missing_simulation_metadata(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, simulation=None)
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.simulation_metadata_missing",
    )
    assert diag.reason == (
        "simulation enabled but simulation/metadata.json is missing"
    )


def test_phase_training_disabled_with_complete_simulation_succeeds(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=False, simulation=True),
        training=None,
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.SUCCESS,
        "cobre.completed",
    )
    assert diag.reason.startswith(
        "training disabled; simulation 100/100 scenarios"
    )
    assert [i.source for i in diag.evidence] == [
        "model.exit",
        "simulation/metadata.json",
        "model log",
    ]


def test_phase_simulation_disabled_reports_training_only(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=True, simulation=False),
        simulation=None,
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.completed"
    assert diag.reason == (
        "training iteration_limit after 128 iterations, lower bound 1234.5; "
        "simulation disabled"
    )


def test_phase_both_disabled_is_no_phase_enabled(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=False, simulation=False),
        training=None,
        simulation=None,
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.SUCCESS,
        "cobre.no_phase_enabled",
    )
    assert diag.reason == "training and simulation both disabled: nothing ran"
    assert diag.matched == ("cobre.no_phase_enabled",)


def test_phase_disabled_metadata_is_never_read(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=True, simulation=False),
        simulation=None,
    )
    _write_raw(ws, "simulation/metadata.json", b"{")
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.completed"


_TRAINING_DEFECTS = [
    pytest.param(b"{", "invalid JSON", id="invalid-json"),
    pytest.param(
        b"\xff\xfe\x00", "cannot be read as UTF-8 text", id="not-utf8"
    ),
    pytest.param(b"[]", "not a JSON object", id="not-an-object"),
    pytest.param(b"", "invalid JSON", id="empty"),
]
_TRAINING_FIELD_DEFECTS = [
    pytest.param("status", _ABSENT, "status is missing", id="status-absent"),
    pytest.param("status", 1, "status is not a string", id="status-int"),
    pytest.param(
        "iterations.completed",
        _ABSENT,
        "iterations.completed is missing",
        id="iterations-absent",
    ),
    pytest.param(
        "iterations",
        [],
        "iterations.completed is missing",
        id="iterations-list",
    ),
    pytest.param(
        "iterations.completed",
        True,
        "iterations.completed is not an integer",
        id="iterations-bool",
    ),
    pytest.param(
        "iterations.completed",
        "128",
        "iterations.completed is not an integer",
        id="iterations-str",
    ),
    pytest.param(
        "iterations.completed",
        128.0,
        "iterations.completed is not an integer",
        id="iterations-float",
    ),
    pytest.param(
        "convergence.termination_reason",
        None,
        "convergence.termination_reason is not a string",
        id="reason-null",
    ),
    pytest.param(
        "bounds.final_lower_bound",
        _ABSENT,
        "bounds.final_lower_bound is missing",
        id="bound-absent",
    ),
    pytest.param(
        "bounds.final_lower_bound",
        False,
        "bounds.final_lower_bound is not a number",
        id="bound-bool",
    ),
    pytest.param(
        "bounds.final_lower_bound",
        None,
        "bounds.final_lower_bound is not a number",
        id="bound-null",
    ),
    pytest.param(
        "distribution.world_size",
        True,
        "distribution.world_size is not an integer",
        id="world-size-bool",
    ),
    pytest.param(
        "software_version",
        _ABSENT,
        "software_version (cobre_version before v0.18.0) is missing",
        id="version-absent",
    ),
    pytest.param(
        "software_version",
        18,
        "software_version is not a string",
        id="version-int",
    ),
]


@pytest.mark.parametrize(("data", "problem"), _TRAINING_DEFECTS)
def test_phase_unparseable_training_metadata_is_metadata_unreadable(
    tmp_path: Path, data: bytes, problem: str
) -> None:
    ws = _workspace(tmp_path)
    _write_raw(ws, "training/metadata.json", data)
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.metadata_unreadable",
    )
    assert diag.reason.startswith(f"training/metadata.json: {problem}")


@pytest.mark.parametrize(
    ("dotted", "value", "problem"), _TRAINING_FIELD_DEFECTS
)
def test_phase_training_field_defect_is_metadata_unreadable(
    tmp_path: Path, dotted: str, value: object, problem: str
) -> None:
    ws = _workspace(
        tmp_path, training=_mutated(training_metadata(), dotted, value)
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.metadata_unreadable"
    assert diag.reason == f"training/metadata.json: {problem}"
    assert [i.source for i in diag.evidence] == [
        "model.exit",
        "simulation/metadata.json",
        "model log",
    ]


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        pytest.param(
            _ABSENT,
            "software_version (cobre_version before v0.18.0) is missing",
            id="absent",
        ),
        pytest.param(None, "cobre_version is not a string", id="null"),
    ],
)
def test_phase_v017_version_defect_is_metadata_unreadable(
    tmp_path: Path, value: object, problem: str
) -> None:
    base = training_metadata(release="0.17.0")
    ws = _workspace(tmp_path, training=_mutated(base, "cobre_version", value))
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.metadata_unreadable"
    assert diag.reason == f"training/metadata.json: {problem}"


@pytest.mark.parametrize(
    ("dotted", "value", "problem"),
    [
        pytest.param(
            "scenarios", _ABSENT, "scenarios.total is missing", id="scenarios"
        ),
        pytest.param(
            "scenarios.failed",
            _ABSENT,
            "scenarios.failed is missing",
            id="failed-absent",
        ),
        pytest.param(
            "scenarios.completed",
            True,
            "scenarios.completed is not an integer",
            id="completed-bool",
        ),
        pytest.param(
            "scenarios.total",
            100.0,
            "scenarios.total is not an integer",
            id="total-float",
        ),
        pytest.param(
            "distribution.world_size",
            "1",
            "distribution.world_size is not an integer",
            id="world-size-str",
        ),
    ],
)
def test_phase_simulation_field_defect_is_metadata_unreadable(
    tmp_path: Path, dotted: str, value: object, problem: str
) -> None:
    ws = _workspace(
        tmp_path, simulation=_mutated(simulation_metadata(), dotted, value)
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.metadata_unreadable"
    assert diag.reason == f"simulation/metadata.json: {problem}"


def test_phase_unparseable_simulation_metadata_keeps_training_evidence(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    _write_raw(ws, "simulation/metadata.json", b"[]")
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.reason == "simulation/metadata.json: not a JSON object"
    assert [i.source for i in diag.evidence] == [
        "model.exit",
        "training/metadata.json",
        "model log",
    ]


def test_phase_training_defect_is_reported_before_the_simulation_one(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    _write_raw(ws, "training/metadata.json", b"{")
    _write_raw(ws, "simulation/metadata.json", b"{")
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.reason.startswith("training/metadata.json: invalid JSON")


def test_phase_metadata_that_is_a_directory_is_metadata_unreadable(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path, training=None)
    (case.case_root(ws) / "output" / "training" / "metadata.json").mkdir(
        parents=True
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.metadata_unreadable"
    assert diag.reason == (
        "training/metadata.json: cannot be read as UTF-8 text"
    )


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        pytest.param(b"{", "caso_cobre/config.json: invalid JSON", id="json"),
        pytest.param(
            b"[]", "caso_cobre/config.json: not a JSON object", id="list"
        ),
        pytest.param(
            b'{"training": {"enabled": "yes"}}',
            "caso_cobre/config.json: training.enabled is not a boolean",
            id="enabled-str",
        ),
    ],
)
def test_phase_unreadable_config_is_metadata_unreadable(
    tmp_path: Path, config: bytes, problem: str
) -> None:
    members = {**case_members(), "config.json": config}
    ws = cobre_workspace(tmp_path, members=members).ws
    write_outputs(case.case_root(ws))
    ws.ensure_layout()
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.metadata_unreadable",
    )
    assert diag.reason.startswith(f"config.json: {problem}")


def test_phase_missing_config_and_deck_are_metadata_unreadable(
    tmp_path: Path,
) -> None:
    members = case_members()
    del members["config.json"]
    ws = cobre_workspace(tmp_path / "a", members=members).ws
    ws.ensure_layout()
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.reason == "config.json: missing caso_cobre/config.json"
    other = _workspace(tmp_path / "b")
    other.eco_deck_path.unlink()
    diag = _diagnose(other, _job(other, 0))
    assert diag.rule_id == "cobre.metadata_unreadable"
    assert diag.reason.startswith("config.json: eco_deck.zip missing")


def test_phase_flat_zip_reads_the_output_at_the_workspace_root(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path, top=None)
    assert case.case_root(ws) == ws.root
    assert _diagnose(ws, _job(ws, 0)).rule_id == "cobre.completed"


# -- count, partial and shutdown rows (R107, R134) ---------------------


def test_scenarios_failed_row(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        simulation=simulation_metadata(
            scenarios={"total": 10, "completed": 8, "failed": 2}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.scenarios_failed",
    )
    assert "2 of 10 simulation scenarios failed (8 completed)" in (
        diag.annotation()
    )
    assert diag.matched == ("cobre.scenarios_failed",)


@pytest.mark.parametrize(
    ("completed", "failed", "total"), [(7, 0, 10), (11, 0, 10), (0, 0, 10)]
)
def test_incomplete_simulation_row(
    tmp_path: Path, completed: int, failed: int, total: int
) -> None:
    scenarios = {"total": total, "completed": completed, "failed": failed}
    ws = _workspace(
        tmp_path, simulation=simulation_metadata(scenarios=scenarios)
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.simulation_incomplete",
    )
    assert (
        f"simulation incomplete: {completed} completed, {failed} failed, "
        f"{total} total"
    ) in diag.annotation()


def test_incomplete_failed_scenarios_come_first_when_both_apply(
    tmp_path: Path,
) -> None:
    scenarios = {"total": 10, "completed": 5, "failed": 3}
    ws = _workspace(
        tmp_path, simulation=simulation_metadata(scenarios=scenarios)
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.scenarios_failed"
    assert diag.matched == (
        "cobre.scenarios_failed",
        "cobre.simulation_incomplete",
    )


def test_partial_training_status_row(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, training=training_metadata(status="partial"))
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.training_partial",
    )
    assert diag.reason == "training partial after 128 iterations"


def test_partial_training_comes_before_the_scenario_rows(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(status="partial"),
        simulation=simulation_metadata(
            scenarios={"total": 10, "completed": 8, "failed": 2}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.training_partial"
    assert diag.matched == ("cobre.training_partial", "cobre.scenarios_failed")


def _shutdown_workspace(tmp_path: Path) -> Workspace:
    return _workspace(
        tmp_path,
        training=training_metadata(
            iterations={"completed": 37},
            convergence={"termination_reason": "graceful_shutdown"},
        ),
    )


@pytest.mark.parametrize(
    ("state", "status", "rule_id"),
    [
        ("TIMEOUT", RunStatus.TIMEOUT, "cobre.shutdown_timeout"),
        ("DEADLINE", RunStatus.TIMEOUT, "cobre.shutdown_timeout"),
        ("COMPLETED", RunStatus.CANCELLED, "cobre.shutdown_cancelled"),
        ("CANCELLED", RunStatus.CANCELLED, "cobre.shutdown_cancelled"),
        (None, RunStatus.CANCELLED, "cobre.shutdown_cancelled"),
    ],
)
def test_shutdown_is_split_by_the_slurm_state(
    tmp_path: Path, state: str | None, status: RunStatus, rule_id: str
) -> None:
    ws = _shutdown_workspace(tmp_path)
    verdict = _verdict(ws, _job(ws, 0, state=state))
    assert (verdict.status, verdict.rule_id) == (status, rule_id)
    assert verdict.matched == (rule_id,)
    expected = "training stopped on request after 37 iterations"
    if status is RunStatus.TIMEOUT:
        expected += f" (Slurm {state})"
    assert verdict.reason == expected


def _stopped_workspace(tmp_path: Path) -> Workspace:
    """What cobre v0.18.0 writes before it exits 5: partial training,
    and a simulation that never ran a scenario."""
    return _workspace(
        tmp_path,
        training=training_metadata(
            status="partial",
            iterations={"completed": 37},
            convergence={"termination_reason": "graceful_shutdown"},
        ),
        simulation=simulation_metadata(
            status="partial",
            scenarios={"total": 100, "completed": 0, "failed": 0},
        ),
    )


@pytest.mark.parametrize(
    ("state", "status", "rule_id"),
    [
        ("TIMEOUT", RunStatus.TIMEOUT, "cobre.shutdown_timeout"),
        ("FAILED", RunStatus.CANCELLED, "cobre.shutdown_cancelled"),
        (None, RunStatus.CANCELLED, "cobre.shutdown_cancelled"),
    ],
)
def test_shutdown_exit_five_is_split_by_the_slurm_state(
    tmp_path: Path, state: str | None, status: RunStatus, rule_id: str
) -> None:
    ws = _stopped_workspace(tmp_path)
    verdict = _verdict(ws, _job(ws, 5, state=state))
    assert (verdict.status, verdict.rule_id) == (status, rule_id)
    assert verdict.matched == (
        rule_id,
        "cobre.training_partial",
        "cobre.simulation_incomplete",
    )
    expected = "training stopped on request after 37 iterations"
    if status is RunStatus.TIMEOUT:
        expected += f" (Slurm {state})"
    assert verdict.reason == expected


def test_shutdown_exit_five_through_the_lifecycle_reads_the_outputs(
    tmp_path: Path,
) -> None:
    ws = _stopped_workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 5, state="FAILED"))
    assert diag.annotation() == (
        "CANCELLED: training stopped on request after 37 iterations "
        "[cobre.shutdown_cancelled]"
    )
    assert [i.source for i in diag.evidence] == [
        "model.exit",
        "training/metadata.json",
        "simulation/metadata.json",
        "model log",
    ]


def test_shutdown_exit_five_after_the_training_writes_is_cancelled(
    tmp_path: Path,
) -> None:
    """A signal during the output writes still exits 5, but the training
    metadata keeps its own termination_reason."""
    ws = _workspace(
        tmp_path,
        simulation=simulation_metadata(
            status="partial",
            scenarios={"total": 100, "completed": 0, "failed": 0},
        ),
    )
    diag = _diagnose(ws, _job(ws, 5))
    assert diag.rule_id == "cobre.shutdown_cancelled"
    assert diag.reason == "training stopped on request after 128 iterations"
    assert diag.matched == (
        "cobre.shutdown_cancelled",
        "cobre.simulation_incomplete",
    )


def test_shutdown_exit_five_without_training_metadata_is_missing(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path, training=None, simulation=None)
    diag = _diagnose(ws, _job(ws, 5))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.training_metadata_missing",
    )
    assert diag.matched == (
        "cobre.training_metadata_missing",
        "cobre.simulation_metadata_missing",
        "cobre.shutdown_cancelled",
    )


def test_shutdown_exit_five_with_training_disabled_has_no_count(
    tmp_path: Path,
) -> None:
    """cobre exits 5 only after training; the reason still renders."""
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=False, simulation=True),
        training=None,
        simulation=simulation_metadata(
            status="partial",
            scenarios={"total": 100, "completed": 40, "failed": 0},
        ),
    )
    diag = _diagnose(ws, _job(ws, 5))
    assert diag.rule_id == "cobre.shutdown_cancelled"
    assert diag.reason == "training stopped on request after ? iterations"


def test_shutdown_through_the_lifecycle_entry_point_is_cancelled(
    tmp_path: Path,
) -> None:
    ws = _shutdown_workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.shutdown_cancelled"
    assert diag.annotation() == (
        "CANCELLED: training stopped on request after 37 iterations "
        "[cobre.shutdown_cancelled]"
    )


def test_shutdown_comes_before_the_scenario_rows(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(
            convergence={"termination_reason": "graceful_shutdown"}
        ),
        simulation=simulation_metadata(
            scenarios={"total": 10, "completed": 8, "failed": 2}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.shutdown_cancelled"
    assert diag.matched == (
        "cobre.shutdown_cancelled",
        "cobre.scenarios_failed",
    )


# -- success, rank and evidence ----------------------------------------


def test_completed_annotation_and_model_log_evidence(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.status is RunStatus.SUCCESS
    assert diag.annotation() == (
        "SUCCESS: training iteration_limit after 128 iterations, lower bound "
        "1234.5; simulation 100/100 scenarios [cobre.completed]"
    )
    assert diag.matched == ("cobre.completed",)
    assert _details(diag, "model log") == [
        "HPCMU_SHAPE nodes=1; Backend: MPI (MPICH 4.2.3, MPI 4.1); "
        "Solver: HiGHS 1.11.0"
    ]
    assert [(i.layer, i.source, i.detail) for i in diag.evidence] == [
        ("plugin", "model.exit", "0"),
        (
            "plugin",
            "training/metadata.json",
            "status complete; termination_reason iteration_limit; "
            "iterations 128; final_lower_bound 1234.5; "
            "cobre_version 0.18.0; world_size 1",
        ),
        (
            "plugin",
            "simulation/metadata.json",
            "completed 100; failed 0; total 100; world_size 1",
        ),
        (
            "plugin",
            "model log",
            "HPCMU_SHAPE nodes=1; Backend: MPI (MPICH 4.2.3, MPI 4.1); "
            "Solver: HiGHS 1.11.0",
        ),
    ]


@pytest.mark.parametrize("release", ["0.17.0", "0.18.0"])
def test_completed_reads_either_version_key(
    tmp_path: Path, release: str
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(release=release),
        simulation=simulation_metadata(release=release),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.completed"
    assert (
        f"cobre_version {release};"
        in (_details(diag, "training/metadata.json")[0])
    )


@pytest.mark.parametrize(
    ("bound", "text"),
    [(1234.5, "1234.5"), (5, "5"), (1234567.891, "1.23457e+06"), (0.0, "0")],
)
def test_completed_lower_bound_uses_six_significant_digits(
    tmp_path: Path, bound: float, text: str
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(bounds={"final_lower_bound": bound}),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.reason.startswith(
        f"training iteration_limit after 128 iterations, lower bound {text}; "
    )


def test_rank_world_size_one_on_four_nodes_is_a_mismatch(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    lines = ("HPCMU_SHAPE nodes=4 cpus_on_node=100", *COBRE_EXECUTION_LINES)
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.rank_count_mismatch",
    )
    assert diag.reason == (
        "cobre ran 1 MPI rank(s) on 4 node(s); one rank per node is required"
    )


def test_rank_world_size_four_on_four_nodes_succeeds(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(distribution={"world_size": 4}),
        simulation=simulation_metadata(distribution={"world_size": 4}),
    )
    lines = ("HPCMU_SHAPE nodes=4 cpus_on_node=100", *COBRE_EXECUTION_LINES)
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert (diag.status, diag.rule_id) == (
        RunStatus.SUCCESS,
        "cobre.completed",
    )


def test_rank_training_world_size_takes_precedence(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path, simulation=simulation_metadata(distribution={"world_size": 4})
    )
    assert _diagnose(ws, _job(ws, 0)).rule_id == "cobre.completed"


def test_rank_simulation_world_size_is_used_without_training(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        enabled=case.Phases(training=False, simulation=True),
        training=None,
        simulation=simulation_metadata(distribution={"world_size": 2}),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.rank_count_mismatch"
    assert diag.reason == (
        "cobre ran 2 MPI rank(s) on 1 node(s); one rank per node is required"
    )


def test_rank_last_shape_line_wins(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    lines = (
        "HPCMU_SHAPE nodes=2 cpus_on_node=4",
        _SHAPE,
        *COBRE_EXECUTION_LINES,
    )
    assert _diagnose(ws, _job(ws, 0, lines=lines)).rule_id == "cobre.completed"


@pytest.mark.parametrize(
    "shape",
    [None, "HPCMU_SHAPE nodes=? cpus_on_node=4", "XHPCMU_SHAPE nodes=4"],
    ids=["absent", "question-mark", "glued-prefix"],
)
def test_rank_without_a_shape_value_skips_the_check(
    tmp_path: Path, shape: str | None
) -> None:
    ws = _workspace(tmp_path)
    lines = (
        COBRE_EXECUTION_LINES
        if shape is None
        else (shape, *COBRE_EXECUTION_LINES)
    )
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert diag.rule_id == "cobre.completed"
    assert _details(diag, "model log") == [
        "HPCMU_SHAPE nodes=?; Backend: MPI (MPICH 4.2.3, MPI 4.1); "
        "Solver: HiGHS 1.11.0",
        "rank check skipped: no HPCMU_SHAPE nodes value",
    ]


def test_rank_missing_log_skips_both_checks(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    diag = _diagnose(ws, _job(ws, 0, lines=None))
    assert (diag.status, diag.rule_id) == (
        RunStatus.SUCCESS,
        "cobre.completed",
    )
    assert _details(diag, "model log") == [
        "unreadable: backend and rank checks skipped"
    ]


def test_evidence_solver_line_is_recorded_not_judged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    lines = (_SHAPE, "  Solver:    CLP 1.17.9", "  Backend:   MPI")
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert diag.rule_id == "cobre.completed"
    assert _details(diag, "model log") == [
        "HPCMU_SHAPE nodes=1; Backend: MPI; Solver: CLP 1.17.9"
    ]


def test_evidence_absent_solver_line_reads_a_question_mark(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    lines = (_SHAPE, "  Backend:   MPI")
    diag = _diagnose(ws, _job(ws, 0, lines=lines))
    assert _details(diag, "model log") == [
        "HPCMU_SHAPE nodes=1; Backend: MPI; Solver: ?"
    ]


def test_vocabulary_unknown_reason_adds_a_note_and_keeps_success(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(
            convergence={"termination_reason": "bogus_rule"}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert (diag.status, diag.rule_id) == (
        RunStatus.SUCCESS,
        "cobre.completed",
    )
    assert diag.reason.startswith("training bogus_rule after 128 iterations")
    assert _details(diag, "training/metadata.json")[-1] == (
        "termination_reason 'bogus_rule' is outside the verified cobre "
        "vocabulary"
    )


@pytest.mark.parametrize(
    "reason", ["iteration_limit", "time_limit", "bound_stalling", "gap"]
)
def test_vocabulary_known_reasons_add_no_note(
    tmp_path: Path, reason: str
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(convergence={"termination_reason": reason}),
    )
    diag = _diagnose(ws, _job(ws, 0))
    assert diag.rule_id == "cobre.completed"
    assert len(_details(diag, "training/metadata.json")) == 1


def test_vocabulary_is_the_pinned_cobre_set() -> None:
    assert diagnosis.TERMINATION_REASONS == frozenset(
        {
            "iteration_limit",
            "time_limit",
            "bound_stalling",
            "gap",
            "graceful_shutdown",
        }
    )


def test_evidence_holds_at_most_six_items_with_every_kind(
    tmp_path: Path,
) -> None:
    ws = _workspace(
        tmp_path,
        training=training_metadata(
            convergence={"termination_reason": "bogus_rule"}
        ),
    )
    diag = _diagnose(ws, _job(ws, 0, lines=("  Backend:   MPI",)))
    assert [i.source for i in diag.evidence] == [
        "model.exit",
        "training/metadata.json",
        "simulation/metadata.json",
        "model log",
        "model log",
        "training/metadata.json",
    ]
    assert {i.layer for i in diag.evidence} == {"plugin"}


def test_evidence_exit_three_is_never_infeasible(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    lines = (*_LOG_OK, "error: LP infeasible at stage 3")
    diag = _diagnose(ws, _job(ws, 3, lines=lines))
    assert (diag.status, diag.rule_id) == (
        RunStatus.RUNTIME_ERROR,
        "cobre.solver_error",
    )
