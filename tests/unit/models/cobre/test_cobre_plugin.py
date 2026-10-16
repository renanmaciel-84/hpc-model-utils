"""ADR-003/ADR-038/R73/R74/R105 tests for the cobre plugin skeleton
(ticket-068)."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hpc_model_utils.core.errors import DataError, UsageError
from hpc_model_utils.core.lifecycle.prepare import purge_stale_outputs
from hpc_model_utils.core.plugin import ExecutableSpec, ParentRun
from hpc_model_utils.core.state import StudyInfo
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models import cobre
from hpc_model_utils.models.cobre import CobrePlugin, case, outputs
from hpc_model_utils.models.cobre import plugin as cobre_plugin_module
from tests.support.cobre_case import (
    case_members,
    cobre_workspace,
    write_elf_stub,
    write_outputs,
)

_STUDY = "caso_cobre"


def _config_members(config: object) -> dict[str, bytes]:
    members = case_members()
    members["config.json"] = json.dumps(config).encode("utf-8")
    return members


def _message(excinfo: pytest.ExceptionInfo[DataError], tmp_path: Path) -> str:
    message = str(excinfo.value)
    assert str(tmp_path) not in message
    return message


# -- declarations -------------------------------------------------------


def test_declarations_match_requirement_two() -> None:
    plugin = CobrePlugin()
    assert plugin.name == "cobre"
    assert plugin.model_name == "COBRE"
    assert plugin.sanitize_encoding is False
    assert plugin.sanitize_exclude == ()
    assert plugin.output_patterns == (r"(?:[^/]+/)?output/.+",)
    assert plugin.output_patterns == case.OUTPUT_PATTERNS
    assert plugin.log_patterns == ()
    assert plugin.parent_model is None
    assert plugin.parent_artifacts == ()
    assert plugin.always_write_parent_path is False


def test_declarations_executables_has_no_licence_or_normalizer() -> None:
    spec = CobrePlugin().executables
    assert isinstance(spec, ExecutableSpec)
    assert spec.entrypoint == "novomodelo-mpi"
    assert spec.license_files == ()
    assert spec.name_normalizer is None


def test_declarations_package_reexports_only_the_plugin() -> None:
    assert cobre.__all__ == ["CobrePlugin"]
    assert cobre.CobrePlugin is CobrePlugin


def test_synthesis_args_returns_the_dashboard_args(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path).ws
    write_outputs(case.case_root(ws))

    args = CobrePlugin().synthesis_args(ws, 4)

    assert args is not None
    assert args == outputs.dashboard_args(ws)


def test_declarations_abc_defaults_for_unused_hooks(tmp_path: Path) -> None:
    plugin = CobrePlugin()
    ws = Workspace.at(tmp_path)
    plugin.prepare(ws, "name")
    plugin.postprocess(ws)
    parent = ParentRun("s3://b/k", "NEWAVE", "2024-01-01T00:00:00+00:00", ())
    with pytest.raises(UsageError, match="cobre does not support parent runs"):
        plugin.fetch_parent(ws, parent)
    with pytest.raises(UsageError, match="offline ingestion"):
        plugin.ingest_offline(ws, ())


@pytest.mark.parametrize(
    ("member", "purged"),
    [
        ("caso_cobre/output/training/metadata.json", True),
        ("output/training/metadata.json", True),
        ("caso_cobre/OUTPUT/policy/cuts/stage_000.bin", True),
        ("caso_cobre/config.json", False),
        ("caso_cobre/system/output.json", False),
        ("caso_cobre/output", False),
    ],
)
def test_declarations_output_patterns_select_the_output_tree(
    member: str, purged: bool
) -> None:
    hit = any(
        re.fullmatch(pattern, member, re.IGNORECASE)
        for pattern in CobrePlugin.output_patterns
    )
    assert hit is purged


# -- study_info ---------------------------------------------------------


def test_study_info_top_folder_and_earliest_start_date(tmp_path: Path) -> None:
    members = case_members(start_dates=("2024-02-01", "2024-01-01"))
    ws = cobre_workspace(tmp_path, members=members).ws
    info = CobrePlugin().study_info(ws)
    assert info == StudyInfo(
        name="caso_cobre", starting_date="2024-01-01T00:00:00+00:00"
    )


def test_study_info_flat_zip_has_empty_name(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path, top=None).ws
    assert CobrePlugin().study_info(ws) == StudyInfo(
        name="", starting_date="2024-01-01T00:00:00+00:00"
    )


def test_study_info_starting_date_is_timezone_aware_utc(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(tmp_path).ws
    parsed = datetime.fromisoformat(case.study_info(ws).starting_date)
    assert parsed.utcoffset() == timedelta(0)


def _without(name: str) -> dict[str, bytes]:
    members = case_members()
    del members[name]
    return members


@pytest.mark.parametrize(
    ("members", "match"),
    [
        pytest.param(
            _without("stages.json"),
            r"^missing caso_cobre/stages\.json$",
            id="missing-stages",
        ),
        pytest.param(
            case_members(start_dates=("2024-13-01",)),
            r"caso_cobre/stages\.json: stages\[0\]\.start_date is not a valid",
            id="month-13",
        ),
        pytest.param(
            case_members(start_dates=("2024-01-01", "2024-1-01")),
            r"caso_cobre/stages\.json: stages\[1\]\.start_date is not a YYYY",
            id="not-padded",
        ),
        pytest.param(
            case_members(start_dates=()),
            r"caso_cobre/stages\.json: stages is not a non-empty list",
            id="empty-stages",
        ),
        pytest.param(
            {**case_members(), "stages.json": b"{"},
            r"caso_cobre/stages\.json: invalid JSON",
            id="invalid-json",
        ),
        pytest.param(
            {**case_members(), "stages.json": b"[]"},
            r"caso_cobre/stages\.json: not a JSON object",
            id="not-an-object",
        ),
        pytest.param(
            {**case_members(), "stages.json": b"\xff\xfe\x00"},
            r"caso_cobre/stages\.json: cannot be read as UTF-8 text",
            id="not-utf8",
        ),
        pytest.param(
            {**case_members(), "stages.json": b'{"stages": {"a": 1}}'},
            r"stages is not a non-empty list",
            id="stages-not-a-list",
        ),
        pytest.param(
            {**case_members(), "stages.json": b'{"stages": [{"id": 0}]}'},
            r"stages\[0\]\.start_date is not a YYYY",
            id="start-date-missing",
        ),
        pytest.param(
            {**case_members(), "stages.json": b'{"stages": [20240101]}'},
            r"stages\[0\]\.start_date is not a YYYY",
            id="stage-not-an-object",
        ),
    ],
)
def test_study_info_defect_raises_data_error(
    tmp_path: Path, members: dict[str, bytes], match: str
) -> None:
    ws = cobre_workspace(tmp_path, members=members).ws
    with pytest.raises(DataError, match=match) as excinfo:
        CobrePlugin().study_info(ws)
    _message(excinfo, tmp_path)


def test_study_info_flat_zip_defect_names_the_bare_file(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(tmp_path, top=None, members=_without("stages.json")).ws
    with pytest.raises(DataError, match=r"^missing stages\.json$"):
        CobrePlugin().study_info(ws)


def test_study_info_missing_eco_deck_raises_data_error(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path).ws
    ws.eco_deck_path.unlink()
    with pytest.raises(DataError, match=r"^eco_deck\.zip missing: ") as excinfo:
        CobrePlugin().study_info(ws)
    _message(excinfo, tmp_path)


def test_study_info_not_a_zip_raises_data_error(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    ws.eco_deck_path.write_bytes(b"not a zip")
    with pytest.raises(
        DataError, match=r"^not a zip archive: eco_deck\.zip$"
    ) as excinfo:
        CobrePlugin().study_info(ws)
    _message(excinfo, tmp_path)


def test_study_info_unsafe_member_raises_data_error_chained(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    with zipfile.ZipFile(ws.eco_deck_path, "w") as archive:
        archive.writestr("../escape.json", b"{}")
    with pytest.raises(DataError, match="escapes archive root") as excinfo:
        CobrePlugin().study_info(ws)
    assert excinfo.value.__cause__ is not None


def test_study_info_unreadable_eco_deck_raises_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = cobre_workspace(tmp_path).ws

    def denied(archive: Path) -> list[str]:
        raise PermissionError(str(archive))

    monkeypatch.setattr(case, "validated_members", denied)
    with pytest.raises(DataError, match=r"^eco_deck\.zip cannot be read$") as e:
        case.deck_members(ws)
    _message(e, tmp_path)


def test_study_info_output_named_top_folder_fails_after_purge(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(tmp_path, top="output").ws
    purged = purge_stale_outputs(ws, CobrePlugin(), case.deck_members(ws))
    assert len(purged) == len(case.deck_members(ws))
    with pytest.raises(DataError, match=r"^missing output/stages\.json$"):
        CobrePlugin().study_info(ws)


# -- case layout --------------------------------------------------------


def test_study_info_case_layout_with_and_without_top_folder(
    tmp_path: Path,
) -> None:
    nested = cobre_workspace(tmp_path / "nested").ws
    flat = cobre_workspace(tmp_path / "flat", top=None).ws
    assert case.top_folder(nested) == _STUDY
    assert case.case_root(nested) == nested.root / _STUDY
    assert case.case_prefix(nested) == f"{_STUDY}/"
    assert case.top_folder(flat) is None
    assert case.case_root(flat) == flat.root
    assert case.case_prefix(flat) == ""


# -- input_files --------------------------------------------------------


def test_input_files_sorted_prefixed_and_without_stale_output(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(
        tmp_path, stale_outputs=("output/training/metadata.json",)
    ).ws
    names = CobrePlugin().input_files(ws)
    assert list(names) == sorted(names)
    assert names[0] == "caso_cobre/config.json"
    assert "caso_cobre/output/training/metadata.json" not in names
    assert "caso_cobre/system/hydros.json" in names
    assert len(names) == len(set(names)) == len(case_members())


def test_input_files_flat_zip_has_no_prefix(tmp_path: Path) -> None:
    ws = cobre_workspace(
        tmp_path, top=None, stale_outputs=("output/simulation/_SUCCESS",)
    ).ws
    names = CobrePlugin().input_files(ws)
    assert names[0] == "config.json"
    assert all(not name.startswith("output/") for name in names)


def test_input_files_output_directory_match_ignores_case(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(
        tmp_path, stale_outputs=("OUTPUT/training/metadata.json",)
    ).ws
    assert not any(
        "metadata.json" in name for name in CobrePlugin().input_files(ws)
    )


def test_input_files_are_exactly_the_members_the_purge_keeps(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(
        tmp_path,
        stale_outputs=("output/training/metadata.json", "output/x/y.bin"),
    ).ws
    members = case.deck_members(ws)
    purged = purge_stale_outputs(ws, CobrePlugin(), members)
    assert set(purged) == {
        "caso_cobre/output/training/metadata.json",
        "caso_cobre/output/x/y.bin",
    }
    assert set(CobrePlugin().input_files(ws)) == set(members) - set(purged)


def test_input_files_only_output_members_raises_data_error(
    tmp_path: Path,
) -> None:
    ws = cobre_workspace(
        tmp_path, members={"output/training/metadata.json": b"{}"}
    ).ws
    with pytest.raises(
        DataError, match=r"^the cobre case zip has no input members$"
    ):
        CobrePlugin().input_files(ws)


def test_input_files_missing_eco_deck_raises_data_error(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path).ws
    ws.eco_deck_path.unlink()
    with pytest.raises(DataError, match="eco_deck.zip missing") as excinfo:
        CobrePlugin().input_files(ws)
    _message(excinfo, tmp_path)


# -- phases -------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [{}, {"training": {}, "simulation": {}}, {"modeling": {}}],
    ids=["empty", "empty-sections", "other-keys"],
)
def test_phases_defaults_are_training_on_simulation_off(
    tmp_path: Path, config: dict[str, object]
) -> None:
    ws = cobre_workspace(tmp_path, members=_config_members(config)).ws
    assert case.phases(ws) == case.Phases(True, False)


@pytest.mark.parametrize(
    ("training", "simulation"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_phases_explicit_switches_are_read(
    tmp_path: Path, training: bool, simulation: bool
) -> None:
    members = case_members(training=training, simulation=simulation)
    ws = cobre_workspace(tmp_path, members=members).ws
    assert case.phases(ws) == case.Phases(training, simulation)


def test_phases_is_frozen_dataclass() -> None:
    with pytest.raises(AttributeError):
        case.Phases(True, False).training = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param(
            {"training": {"enabled": "yes"}},
            r"caso_cobre/config\.json: training\.enabled is not a boolean",
            id="training-string",
        ),
        pytest.param(
            {"simulation": {"enabled": 1}},
            r"simulation\.enabled is not a boolean",
            id="simulation-int",
        ),
        pytest.param(
            {"training": {"enabled": None}},
            r"training\.enabled is not a boolean",
            id="training-null",
        ),
        pytest.param(
            {"training": []},
            r"caso_cobre/config\.json: training is not an object",
            id="training-list",
        ),
        pytest.param(
            {"simulation": None},
            r"simulation is not an object",
            id="simulation-null",
        ),
        pytest.param(
            [],
            r"caso_cobre/config\.json: not a JSON object",
            id="config-list",
        ),
    ],
)
def test_phases_invalid_config_raises_data_error(
    tmp_path: Path, config: object, match: str
) -> None:
    ws = cobre_workspace(tmp_path, members=_config_members(config)).ws
    with pytest.raises(DataError, match=match) as excinfo:
        case.phases(ws)
    _message(excinfo, tmp_path)


def test_phases_missing_config_is_not_an_empty_config(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path, members=_without("config.json")).ws
    with pytest.raises(
        DataError, match=r"^missing caso_cobre/config\.json$"
    ) as excinfo:
        case.phases(ws)
    _message(excinfo, tmp_path)


def test_phases_invalid_json_config_raises_data_error(tmp_path: Path) -> None:
    members = {**case_members(), "config.json": b"{not json"}
    ws = cobre_workspace(tmp_path, top=None, members=members).ws
    with pytest.raises(DataError, match=r"^config\.json: invalid JSON") as e:
        case.read_config(ws)
    _message(e, tmp_path)


def test_phases_missing_eco_deck_raises_data_error(tmp_path: Path) -> None:
    ws = cobre_workspace(tmp_path).ws
    ws.eco_deck_path.unlink()
    with pytest.raises(DataError, match="eco_deck.zip missing") as excinfo:
        case.phases(ws)
    _message(excinfo, tmp_path)


# -- check_executables (static probe) ----------------------------------

_PASS_LINE = (
    "novomodelo-mpi: regular ELF executable; "
    "comm and solver are verified in the model job"
)


def _assets(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = cobre_workspace(tmp_path).ws
    return ws, ws.assets


def test_static_probe_passes_and_logs_the_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws, assets = _assets(tmp_path)
    write_elf_stub(assets)
    with caplog.at_level(logging.INFO, logger=cobre_plugin_module.__name__):
        CobrePlugin().check_executables(ws)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [_PASS_LINE]
    assert str(tmp_path) not in messages[0]


def test_static_probe_ignores_a_plain_cobre_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws, assets = _assets(tmp_path)
    write_elf_stub(assets)
    (assets / "cobre").write_text("not a binary\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger=cobre_plugin_module.__name__):
        CobrePlugin().check_executables(ws)
    assert [record.getMessage() for record in caplog.records] == [_PASS_LINE]


def test_static_probe_empty_assets_raises_data_error(tmp_path: Path) -> None:
    ws, _ = _assets(tmp_path)
    with pytest.raises(
        DataError,
        match=r"^novomodelo-mpi missing from the fetched versoes files$",
    ) as excinfo:
        CobrePlugin().check_executables(ws)
    _message(excinfo, tmp_path)


def test_static_probe_plain_cobre_alone_does_not_satisfy_the_check(
    tmp_path: Path,
) -> None:
    ws, assets = _assets(tmp_path)
    write_elf_stub(assets, name="cobre")
    with pytest.raises(
        DataError,
        match=r"^novomodelo-mpi missing from the fetched versoes files$",
    ) as excinfo:
        CobrePlugin().check_executables(ws)
    _message(excinfo, tmp_path)


def test_static_probe_legacy_cobre_mpi_alone_names_the_reupload(
    tmp_path: Path,
) -> None:
    ws, assets = _assets(tmp_path)
    write_elf_stub(assets, name="cobre-mpi")
    hint = (
        "novomodelo-mpi missing from the fetched versoes files "
        "(found cobre-mpi: re-upload this version from a novomodelo-mpi build)"
    )
    with pytest.raises(DataError, match=f"^{re.escape(hint)}$") as excinfo:
        CobrePlugin().check_executables(ws)
    _message(excinfo, tmp_path)


def _defect_symlink(assets: Path) -> None:
    real = write_elf_stub(assets, name="cobre-real")
    (assets / "novomodelo-mpi").symlink_to(real)


def _defect_dangling(assets: Path) -> None:
    (assets / "novomodelo-mpi").symlink_to(assets / "missing")


def _defect_directory(assets: Path) -> None:
    (assets / "novomodelo-mpi").mkdir()


def _defect_not_executable(assets: Path) -> None:
    write_elf_stub(assets).chmod(0o644)


def _defect_group_only(assets: Path) -> None:
    write_elf_stub(assets).chmod(0o655)


@pytest.mark.parametrize(
    "defect",
    [
        _defect_symlink,
        _defect_dangling,
        _defect_directory,
        _defect_not_executable,
        _defect_group_only,
    ],
    ids=["symlink", "dangling", "directory", "mode-644", "mode-655"],
)
def test_static_probe_not_a_regular_executable_raises_data_error(
    tmp_path: Path, defect: Callable[[Path], None]
) -> None:
    ws, assets = _assets(tmp_path)
    defect(assets)
    with pytest.raises(
        DataError, match=r"^novomodelo-mpi is not an executable regular file$"
    ) as excinfo:
        CobrePlugin().check_executables(ws)
    _message(excinfo, tmp_path)


@pytest.mark.parametrize(
    "content",
    [b"#!/bin/sh\necho cobre\n", b"", b"\x7fEL"],
    ids=["shell-script", "empty", "truncated-magic"],
)
def test_static_probe_not_an_elf_raises_data_error(
    tmp_path: Path, content: bytes
) -> None:
    ws, assets = _assets(tmp_path)
    binary = assets / "novomodelo-mpi"
    binary.write_bytes(content)
    binary.chmod(0o755)
    with pytest.raises(
        DataError, match=r"^novomodelo-mpi is not an ELF executable$"
    ) as excinfo:
        CobrePlugin().check_executables(ws)
    _message(excinfo, tmp_path)


def test_static_probe_unreadable_file_raises_data_error_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, assets = _assets(tmp_path)
    write_elf_stub(assets)

    def denied(path: Path) -> bytes:
        raise PermissionError(str(path))

    monkeypatch.setattr(cobre_plugin_module, "_read_head", denied)
    with pytest.raises(
        DataError, match=r"^novomodelo-mpi cannot be read$"
    ) as excinfo:
        CobrePlugin().check_executables(ws)
    assert isinstance(excinfo.value.__cause__, PermissionError)
    _message(excinfo, tmp_path)
