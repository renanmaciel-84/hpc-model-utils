"""ADR-009/R23/R41/R123: the Workspace value and the .hpcmu layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpc_model_utils.core.errors import UsageError
from hpc_model_utils.core.workspace import Phase, Workspace
from hpc_model_utils.infra.telemetry import schema


def test_phase_members_are_model_and_finalize() -> None:
    assert {member.value for member in Phase} == {"model", "finalize"}


def test_phase_values_equal_telemetry_schema_phases() -> None:
    assert tuple(p.value for p in Phase) == schema.PHASES


def test_workspace_at_property_access_does_not_create_hpcmu_dir(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    _ = (
        ws.assets,
        ws.eco_deck_path,
        ws.hpcmu_dir,
        ws.state_path,
        ws.finalize_path,
        ws.jobs_dir,
        ws.logs_dir,
        ws.outputs_dir,
        ws.telemetry_dir,
        ws.telemetry_outputs_dir,
        ws.model_exit_path,
        ws.legacy_status_path,
        ws.legacy_metadata_path,
        ws.parent_dir,
        ws.has_state,
    )
    assert (tmp_path / ".hpcmu").exists() is False


def test_properties_return_expected_paths(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    assert ws.assets == tmp_path / "assets"
    assert ws.eco_deck_path == ws.root / "eco_deck.zip"
    assert ws.hpcmu_dir == tmp_path / ".hpcmu"
    assert ws.parent_dir == ws.hpcmu_dir / "parent"
    assert ws.state_path == tmp_path / ".hpcmu" / "state.json"
    assert ws.finalize_path == tmp_path / ".hpcmu" / "finalize.json"
    assert ws.jobs_dir == tmp_path / ".hpcmu" / "jobs"
    assert ws.logs_dir == tmp_path / ".hpcmu" / "logs"
    assert ws.outputs_dir == tmp_path / ".hpcmu" / "outputs"
    assert ws.telemetry_dir == tmp_path / ".hpcmu" / "telemetry"
    assert (
        ws.telemetry_outputs_dir
        == tmp_path / ".hpcmu" / "outputs" / "telemetry"
    )
    assert ws.model_exit_path == ws.jobs_dir / "model.exit"
    assert ws.legacy_status_path == tmp_path / "status.modelops"
    assert ws.legacy_metadata_path == tmp_path / "metadata.modelops"


def test_ensure_layout_called_twice_creates_layout_without_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    ws.ensure_layout()
    ws.ensure_layout()
    assert ws.jobs_dir.is_dir()
    assert ws.logs_dir.is_dir()
    assert ws.outputs_dir.is_dir()
    assert ws.telemetry_dir.is_dir()
    assert ws.telemetry_outputs_dir.exists() is False
    assert ws.parent_dir.is_dir() is False


def test_job_script_returns_jobs_dir_sbatch_path(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    assert ws.job_script(Phase.MODEL) == ws.jobs_dir / "model.sbatch"
    assert ws.job_script(Phase.FINALIZE) == ws.jobs_dir / "finalize.sbatch"


def test_log_path_valid_job_id_returns_expected_suffix(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    model_log = ws.log_path(Phase.MODEL, "6208")
    finalize_log = ws.log_path(Phase.FINALIZE, "6209")
    assert str(model_log).endswith(
        str(Path(".hpcmu") / "logs" / "model-6208.out")
    )
    assert str(finalize_log).endswith(
        str(Path(".hpcmu") / "logs" / "finalize-6209.out")
    )


def test_log_path_non_numeric_job_id_raises_value_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(ValueError):
        ws.log_path(Phase.MODEL, "62;rm")


@pytest.mark.parametrize("job_id", ["12\n", "12\r\n", "\n12"])
def test_log_path_trailing_newline_job_id_raises_value_error(
    tmp_path: Path, job_id: str
) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(ValueError):
        ws.log_path(Phase.MODEL, job_id)


def test_log_pattern_returns_percent_j_template(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    assert ws.log_pattern(Phase.MODEL) == ws.logs_dir / "model-%j.out"
    assert ws.log_pattern(Phase.FINALIZE) == ws.logs_dir / "finalize-%j.out"


def test_relative_path_inside_root_returns_posix_string(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    target = ws.hpcmu_dir / "state.json"
    assert ws.relative(target) == ".hpcmu/state.json"


def test_relative_path_outside_root_raises_value_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError):
        ws.relative(outside)


def test_relative_nested_path_returns_posix_string(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    target = tmp_path / "a" / "b"
    assert ws.relative(target) == "a/b"


def test_relative_dotdot_escape_raises_value_error(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(ValueError):
        ws.relative(tmp_path / ".." / "etc")
    with pytest.raises(ValueError):
        ws.relative(tmp_path / "a" / ".." / ".." / "x")


def test_has_state_no_state_file_returns_false(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    assert ws.has_state is False


def test_has_state_with_state_file_returns_true(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    ws.hpcmu_dir.mkdir()
    ws.state_path.write_text("{}", encoding="utf-8")
    assert ws.has_state is True


def test_at_missing_path_raises_usage_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(UsageError):
        Workspace.at(missing)


def test_at_file_path_raises_usage_error(tmp_path: Path) -> None:
    file_path = tmp_path / "a-file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(UsageError):
        Workspace.at(file_path)


def test_at_path_with_whitespace_raises_usage_error_matching_whitespace(
    tmp_path: Path,
) -> None:
    spaced = tmp_path / "with space"
    spaced.mkdir()
    with pytest.raises(UsageError, match="whitespace"):
        Workspace.at(spaced)


def test_workspace_construction_relative_root_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Workspace(Path("relative/root"))
