"""ADR-003/ADR-051/ADR-011/R20/R28/R54/R93/R100 tests for the DECOMP
plugin package (ticket-050)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hpc_model_utils import models
from hpc_model_utils.core.diagnosis import Diagnosis, RunStatus
from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.launch import (
    Launcher,
    LaunchSpec,
    Resources,
    TelemetryConfig,
    Toolchain,
    render_model_script,
)
from hpc_model_utils.core.lifecycle.fetch import fetch_executables, fetch_inputs
from hpc_model_utils.core.lifecycle.prepare import extract_sanitize_inputs
from hpc_model_utils.core.plugin import ExecutableSpec, ParentRun
from hpc_model_utils.core.state import StateStore, StudyInfo, render_metadata
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.s3 import S3Uri
from hpc_model_utils.models import get_plugin
from hpc_model_utils.models.decomp import DecompPlugin, deck
from hpc_model_utils.models.decomp import diagnosis as decomp_diagnosis
from tests.support.decks import (
    FIXTURES,
    archive_golden,
    binary_cut_files,
    decomp_workspace,
    input_zip,
)
from tests.support.object_store import RecordingObjectStore
from tests.support.reporter import RecordingReporter

_REGEN_ENV = "HPCMU_REGEN_GOLDENS"
_GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "goldens"
    / "v2"
    / "sbatch"
    / "decomp_model.sbatch"
)
_ON_GOLDEN_PATH = _GOLDEN_PATH.with_name("decomp_model-telemetry.sbatch")
_RM_EXIT_GOLDEN_LINE = "rm -f <WS>/.hpcmu/jobs/model.exit"
_SIDECAR_GOLDEN_LINE = (
    'srun --overlap --nodes="${SLURM_JOB_NUM_NODES:-?}" '
    '--ntasks="${SLURM_JOB_NUM_NODES:-?}" --ntasks-per-node=1 '
    "--cpus-per-task=1 --cpu-bind=none --mpi=none --kill-on-bad-exit=0 "
    "--input=none --wait=0 --job-name=hpcmu-telemetry "
    "--output=<WS>/.hpcmu/telemetry/model-%j-r"
    '"${SLURM_RESTART_COUNT:-0}"-%t.log /opt/hpcmu/python -I -m '
    "hpc_model_utils.infra.telemetry.sampler --staging "
    "<WS>/.hpcmu/telemetry --phase model --workspace <WS> --interval 5 "
    "--workspace-interval 60 --tool-version 9.9.9 > "
    '<WS>/.hpcmu/telemetry/model-"${SLURM_JOB_ID:-?}"-r'
    '"${SLURM_RESTART_COUNT:-0}"-srun.log 2>&1 & disown'
)
_PROJECTIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "goldens"
    / "v1_1_2"
    / "projections.json"
)
_STUDY_NAME = (
    "PMO - NOVEMBRO/25 - DEZEMBRO/25 - REV 0 - FCF COM CVAR - 12 REE - "
    "VALOR ESP"
)
_STUDY_STARTING_DATE = "2025-11-01T00:00:00+00:00"

_VERSIONS_BUCKET = "versions-bucket"
_INPUTS_BUCKET = "inputs-bucket"
_OUTPUTS_BUCKET = "outputs-bucket"
_VERSOES_URI = f"s3://{_VERSIONS_BUCKET}/versoes/decomp/32.0/"
_INPUT_URI = f"s3://{_INPUTS_BUCKET}/ingest/deck_decomp.zip"
_PARENT_URI = f"s3://{_OUTPUTS_BUCKET}/artifacts/parenthash02"


def _load_golden_metadata(sequence: str) -> str:
    data = json.loads(_PROJECTIONS_PATH.read_text(encoding="utf-8"))
    return str(data["sequences"][sequence]["metadata.modelops"])


def _rewrite_dadger(
    ws: Workspace, transform: Callable[[list[str]], list[str]]
) -> None:
    path = ws.root / deck.dadger_name(ws)
    lines = path.read_text(encoding="latin-1").splitlines(keepends=True)
    path.write_text("".join(transform(lines)), encoding="latin-1")


# -- deck.py readers ---------------------------------------------------


def test_caso_fixture_returns_object_with_extension_rv0(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    assert deck.caso(dws.ws).arquivos == "rv0"


def test_caso_missing_file_raises_data_error(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(DataError, match="caso.dat"):
        deck.caso(ws)


def test_extension_fixture_returns_rv0(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    assert deck.extension(dws.ws) == "rv0"


def test_extension_empty_value_raises_data_error(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    (ws.root / "caso.dat").write_text("\n\n", encoding="utf-8")
    with pytest.raises(DataError, match="caso.dat"):
        deck.extension(ws)


def test_arquivos_fixture_returns_object_with_dadger_rv0(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    assert deck.arquivos(dws.ws).dadger == "dadger.rv0"


def test_arquivos_missing_index_file_raises_data_error(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    (dws.ws.root / "rv0").unlink()
    with pytest.raises(DataError, match="rv0"):
        deck.arquivos(dws.ws)


def test_dadger_name_fixture_returns_dadger_rv0(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    assert deck.dadger_name(dws.ws) == "dadger.rv0"


def test_dadger_missing_file_raises_data_error(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    (dws.ws.root / "dadger.rv0").unlink()
    with pytest.raises(DataError, match="dadger.rv0"):
        deck.dadger(dws.ws)


def test_study_info_fixture_returns_expected_value(tmp_path: Path) -> None:
    dws = decomp_workspace(tmp_path)
    assert deck.study_info(dws.ws) == StudyInfo(
        name=_STUDY_NAME, starting_date=_STUDY_STARTING_DATE
    )


def test_study_info_empty_workspace_raises_data_error_matching_caso_dat(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(DataError, match="caso.dat"):
        deck.study_info(ws)


def test_study_info_missing_te_register_raises_data_error(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    ws = dws.ws
    _rewrite_dadger(
        ws, lambda lines: [ln for ln in lines if not ln.startswith("TE ")]
    )
    with pytest.raises(DataError, match="missing TE register"):
        deck.study_info(ws)


def test_study_info_missing_dt_register_raises_data_error(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    ws = dws.ws
    _rewrite_dadger(
        ws, lambda lines: [ln for ln in lines if not ln.startswith("DT ")]
    )
    with pytest.raises(DataError, match="missing DT register"):
        deck.study_info(ws)


def test_study_info_incomplete_dt_date_raises_data_error(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    ws = dws.ws
    _rewrite_dadger(
        ws,
        lambda lines: [
            "DT  01   11\n" if ln.startswith("DT ") else ln for ln in lines
        ],
    )
    with pytest.raises(DataError, match="incomplete DT date"):
        deck.study_info(ws)


# -- AC4: deck sets and purge invariants ---------------------------------


def test_input_files_fixture_matches_deck_processado_archive_names(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    expected = set(archive_golden("decomp")["archives"]["deck_processado.zip"])
    assert set(deck.input_files(dws.ws)) == expected


def test_input_files_fa_register_names_missing_file_raises_data_error(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    ws = dws.ws
    (ws.root / "indices.csv").unlink()
    with pytest.raises(DataError, match="indices.csv"):
        deck.input_files(ws)


def test_decompplugin_output_patterns_relato_family_fullmatches() -> None:
    for name in (
        "relato.rv0",
        "relato2.rv0",
        "inviab_unic.rv0",
        "inviab.rv0",
        "sumario.rv0",
        "relgnl.rv0",
    ):
        assert any(
            re.fullmatch(pattern, name, re.IGNORECASE)
            for pattern in DecompPlugin.output_patterns
        )


def test_decompplugin_output_patterns_exclude_input_files_and_cuts(
    tmp_path: Path,
) -> None:
    dws = decomp_workspace(tmp_path)
    candidates = deck.input_files(dws.ws) + ("cortesh.dat", "cortes-012.dat")
    for name in candidates:
        assert not any(
            re.fullmatch(pattern, name, re.IGNORECASE)
            for pattern in DecompPlugin.output_patterns
        )


def test_decompplugin_sanitize_exclude_matches_expected_names() -> None:
    for name in ("hidr.dat", "mlt.dat", "vazoes.rv0", "cortes-012.dat"):
        assert any(
            re.fullmatch(pattern, name)
            for pattern in DecompPlugin.sanitize_exclude
        )


# -- plugin.py ------------------------------------------------------------


def test_decompplugin_declared_attributes_match_requirement_2() -> None:
    plugin = DecompPlugin()
    assert plugin.name == "decomp"
    assert plugin.executables == ExecutableSpec(
        "decomp",
        ("decomp.lic", "ddsDECOMP.cep", "decomp.cep", "decomp_trial.cep"),
        "convertenomesdecomp",
    )
    assert plugin.sanitize_exclude == (
        r"cortes[^/]*\.dat",
        r"[^/]+\.zip",
        r"hidr\.dat",
        r"mlt\.dat",
        r"vazoes\.[^/]+",
    )
    assert plugin.output_patterns == (
        r"(?:relato|relato2|inviab_unic|inviab|sumario|relgnl|custos|"
        r"cortdeco|mapcut)\.[^/]+",
        r"decomp\.tim",
    )
    assert plugin.log_patterns == decomp_diagnosis.LOG_PATTERNS
    assert plugin.parent_model == "NEWAVE"
    assert plugin.parent_artifacts == ("cortes.zip",)
    assert plugin.always_write_parent_path is False


def test_get_plugin_both_cases_return_same_registered_decomp_instance() -> None:
    assert get_plugin("DECOMP") is get_plugin("decomp")
    assert get_plugin("decomp") is models.PLUGINS["decomp"]
    assert isinstance(get_plugin("decomp"), DecompPlugin)


def test_sorted_plugins_returns_cobre_decomp_and_newave() -> None:
    assert sorted(models.PLUGINS) == ["cobre", "decomp", "newave"]


def test_import_models_package_does_not_import_idecomp() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, hpc_model_utils.models\n"
            "assert 'idecomp' not in sys.modules\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_decompplugin_launch_with_max_cores_per_node_sets_ntasks_per_node(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    res = Resources(
        queue="batch", cores=64, max_cores_per_node=32, time_limit_hours=24
    )
    spec = DecompPlugin().launch(ws, res)
    assert spec == LaunchSpec(
        Launcher.MPIEXEC_HYDRA,
        (str(ws.assets / "decomp"),),
        ntasks=64,
        ntasks_per_node=32,
    )


def test_decompplugin_launch_without_max_cores_per_node_omits_ntasks_per_node(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    res = Resources(queue="batch", cores=4)
    spec = DecompPlugin().launch(ws, res)
    assert spec.ntasks_per_node is None
    assert spec.cpus_per_task == 1
    assert spec.prepend_mpich_lib is False
    assert spec.env == {}


def test_decompplugin_synthesis_args_returns_expected_tuple(
    tmp_path: Path,
) -> None:
    assert DecompPlugin().synthesis_args(Workspace.at(tmp_path), 8) == (
        "completa",
        "--processadores",
        "8",
    )


# -- AC2: the regenerable DECOMP-shaped golden ---------------------------


def _decomp_golden_case(
    tmp_path: Path,
) -> tuple[Workspace, LaunchSpec, Resources, Toolchain]:
    root = tmp_path / "study-decomp"
    root.mkdir()
    ws = Workspace.at(root)
    tools = Toolchain(
        mpich_bin=Path("/opt/mpich/bin"),
        slurm_bin=Path("/opt/slurm/bin"),
        cli_bin=Path("/opt/hpcmu/bin"),
    )
    res = Resources(
        queue="batch", cores=64, max_cores_per_node=32, time_limit_hours=24
    )
    return ws, DecompPlugin().launch(ws, res), res, tools


def _assert_matches_golden(
    rendered: str, ws: Workspace, golden_path: Path, tmp_path: Path
) -> None:
    normalized = rendered.replace(str(ws.root), "<WS>")
    if os.environ.get(_REGEN_ENV) == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized, encoding="utf-8")
        pytest.fail(
            f"{_REGEN_ENV}=1: regenerated {golden_path}; "
            "re-run without it to verify"
        )

    assert normalized == golden_path.read_text(encoding="utf-8")

    script_path = tmp_path / "golden-check.sbatch"
    script_path.write_text(rendered, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_render_model_script_decomp_launch_matches_golden(
    tmp_path: Path,
) -> None:
    ws, spec, res, tools = _decomp_golden_case(tmp_path)
    rendered = render_model_script(ws, spec, res, tools)

    _assert_matches_golden(rendered, ws, _GOLDEN_PATH, tmp_path)
    assert "#SBATCH --ntasks=64" in rendered
    assert "#SBATCH --ntasks-per-node=32" in rendered
    assert "#SBATCH --cpus-per-task=1" in rendered
    assert "LD_LIBRARY_PATH" not in rendered


def test_render_model_script_decomp_launch_telemetry_matches_golden(
    tmp_path: Path,
) -> None:
    ws, spec, res, tools = _decomp_golden_case(tmp_path)
    telemetry = TelemetryConfig(
        enabled=True, interval=5, workspace_interval=60, tool_version="9.9.9"
    )
    rendered = render_model_script(ws, spec, res, tools, telemetry=telemetry)

    _assert_matches_golden(rendered, ws, _ON_GOLDEN_PATH, tmp_path)
    off_lines = _GOLDEN_PATH.read_text(encoding="utf-8").split("\n")
    on_lines = _ON_GOLDEN_PATH.read_text(encoding="utf-8").split("\n")
    rm_index = off_lines.index(_RM_EXIT_GOLDEN_LINE)
    assert on_lines[rm_index + 1] == _SIDECAR_GOLDEN_LINE
    assert on_lines[: rm_index + 1] + on_lines[rm_index + 2 :] == off_lines


# -- prepare: writes nothing (R20, D1, R54) --------------------------------


def test_decompplugin_prepare_deck_coupling_keeps_files_and_logs_not_applied(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dws = decomp_workspace(
        tmp_path,
        extra_files=binary_cut_files(("cortesh.dat", "cortes-012.dat")),
    )
    ws = dws.ws
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ws.root.iterdir()
        if p.is_file()
    }

    with caplog.at_level(logging.INFO):
        DecompPlugin().prepare(ws, "any name")

    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ws.root.iterdir()
        if p.is_file()
    }
    assert before == after
    not_applied = [
        record
        for record in caplog.records
        if "not applied" in record.getMessage()
    ]
    assert [record.levelno for record in not_applied] == [logging.INFO]


# -- fetch_parent: validated no-op -----------------------------------------


def test_decompplugin_fetch_parent_writes_nothing_at_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace.at(root)
    archives_dir = tmp_path / "archives"
    cortes = input_zip(
        {"cortdeco.rv0": b"c1", "mapcut.rv0": b"m1"},
        archives_dir / "cortes.zip",
    )
    parent = ParentRun(
        uri="s3://b/p",
        model_name="NEWAVE",
        starting_date=_STUDY_STARTING_DATE,
        archives=(cortes,),
    )

    DecompPlugin().fetch_parent(ws, parent)

    assert list(ws.root.iterdir()) == []


def test_decompplugin_fetch_parent_archive_count_mismatch_raises_value_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    parent = ParentRun(
        uri="s3://b/p",
        model_name="NEWAVE",
        starting_date=_STUDY_STARTING_DATE,
        archives=(),
    )
    with pytest.raises(ValueError, match="shorter than argument"):
        DecompPlugin().fetch_parent(ws, parent)


# -- AC7: golden lifecycle sequences ---------------------------------------


def _run_decomp_sequence(tmp_path: Path, *, parent_path: str) -> str:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace.at(root)
    store = RecordingObjectStore()
    store.seed(
        S3Uri(_VERSIONS_BUCKET, "versoes/decomp/32.0/convertenomesdecomp"),
        b"#!/bin/bash\nexit 0\n",
    )
    store.seed(
        S3Uri(_INPUTS_BUCKET, "ingest/deck_decomp.zip"),
        (FIXTURES / "decks" / "deck_decomp.zip").read_bytes(),
    )
    if parent_path:
        store.seed(
            S3Uri(
                _OUTPUTS_BUCKET,
                "artifacts/parenthash02/saidas/metadata.modelops",
            ),
            json.dumps(
                {
                    "model_name": "NEWAVE",
                    "status": "SUCCESS",
                    "study_starting_date": _STUDY_STARTING_DATE,
                }
            ).encode("utf-8"),
        )
        for name in DecompPlugin.parent_artifacts:
            store.seed(
                S3Uri(_OUTPUTS_BUCKET, f"artifacts/parenthash02/saidas/{name}"),
                b"cut bytes",
            )

    plugin = DecompPlugin()
    state_store = StateStore(ws)
    reporter = RecordingReporter()

    fetch_executables(ws, plugin, store, reporter, state_store, _VERSOES_URI)
    fetch_inputs(
        ws,
        plugin,
        store,
        reporter,
        state_store,
        _INPUT_URI,
        parent_path=parent_path,
        delete=False,
    )
    state = extract_sanitize_inputs(ws, plugin, state_store, reporter)

    final_state = replace(
        state,
        diagnosis=Diagnosis(
            status=RunStatus.SUCCESS, rule_id="decomp.manual", reason="ok"
        ),
        reported_job_id="2001",
    )
    rendered = render_metadata(
        final_state, always_write_parent_path=plugin.always_write_parent_path
    )
    assert rendered is not None
    return rendered


def test_golden_lifecycle_sequence_decomp_no_parent_matches_metadata(
    tmp_path: Path,
) -> None:
    rendered = _run_decomp_sequence(tmp_path, parent_path="")
    assert rendered == _load_golden_metadata("decomp_no_parent")


def test_golden_lifecycle_sequence_decomp_with_parent_matches_metadata(
    tmp_path: Path,
) -> None:
    rendered = _run_decomp_sequence(tmp_path, parent_path=_PARENT_URI)
    assert rendered == _load_golden_metadata("decomp_with_parent")
