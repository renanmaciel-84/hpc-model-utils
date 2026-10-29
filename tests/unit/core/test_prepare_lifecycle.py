"""ADR-017/ADR-018/ADR-028/R21/R26/R50/R123/R130 tests for the
``extract_sanitize_inputs``/``preprocess`` lifecycle steps
(ticket-045)."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from hpc_model_utils.core.errors import DataError, UsageError
from hpc_model_utils.core.lifecycle.prepare import (
    extract_archive,
    extract_sanitize_inputs,
    flatten_execution_name,
    move_licences,
    parent_run,
    preprocess,
    purge_stale_outputs,
    run_name_normalizer,
    sanitize_files,
    sanitize_workspace,
)
from hpc_model_utils.core.plugin import ExecutableSpec, ParentRun
from hpc_model_utils.core.state import ParentInfo, StateStore
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.infra.errors import UnsafeArchiveError
from tests.support.decks import input_zip
from tests.support.fake_plugin import FakePlugin
from tests.support.fake_slurm.cli import write_executable_stub
from tests.support.reporter import RecordingReporter

_LATIN1 = b"NOME ACENTUA\xc7\xc3O\r\n"


class _PurgePlugin(FakePlugin):
    name = "fakepurge"
    # FakePlugin's own one-element literal narrows the inferred
    # ClassVar type to tuple[str]; this override is still a valid
    # tuple[str, ...] at runtime.
    output_patterns = (r"fake\.out", r"pmo\.dat")


class _NoSanitizePlugin(FakePlugin):
    name = "fakenosan"
    sanitize_encoding = False


class _ExclusionPlugin(FakePlugin):
    name = "fakeexclude"
    executables = ExecutableSpec(
        entrypoint="fake-model", license_files=("fake.lic",)
    )
    sanitize_exclude = (r"cortes[^/]*\.dat",)


class _NormalizerPlugin(FakePlugin):
    name = "fakenorm"
    executables = ExecutableSpec(
        entrypoint="fake-model", name_normalizer="normalize"
    )


class _ParentArtifactPlugin(FakePlugin):
    name = "fakeparentart"
    parent_artifacts = ("cortes.zip",)


class _ParentFetchPlugin(FakePlugin):
    name = "fakeparentfetch"
    output_patterns = (r"fake\.out", r"pmo\.dat")
    parent_artifacts = ("cortes.zip",)

    def fetch_parent(self, ws: Workspace, parent: ParentRun) -> None:
        (ws.root / "fake.out").write_bytes(b"PARENT\n")
        (ws.root / "parent.txt").write_bytes(_LATIN1)


class _RecordingPreparePlugin(FakePlugin):
    name = "fakerecordprep"

    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls: list[tuple[Workspace, str]] = []

    def prepare(self, ws: Workspace, execution_name: str) -> None:
        self.prepare_calls.append((ws, execution_name))


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------


def test_extract_archive_valid_zip_extracts_and_returns_sorted_members(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"b.txt": b"B", "a.txt": b"A"}, ws.eco_deck_path)

    members = extract_archive(ws, ws.eco_deck_path)

    assert members == ("a.txt", "b.txt")
    assert (ws.root / "a.txt").read_bytes() == b"A"
    assert (ws.root / "b.txt").read_bytes() == b"B"


def test_extract_archive_assets_member_raises_data_error_before_writing(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"assets/fake-model": b"bin"}, ws.eco_deck_path)

    with pytest.raises(DataError, match="assets"):
        extract_archive(ws, ws.eco_deck_path)

    assert not (ws.root / "assets").exists()


def test_extract_archive_assets_directory_entry_raises_data_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"a.txt": b"A"}, ws.eco_deck_path)
    with zipfile.ZipFile(ws.eco_deck_path, "a") as zf:
        zf.writestr("assets/planted/", "")

    with pytest.raises(DataError, match="assets"):
        extract_archive(ws, ws.eco_deck_path)

    assert not (ws.root / "assets" / "planted").exists()


def test_extract_archive_keeps_empty_case_directory(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"case/boundary/manifest.bin": b"M"}, ws.eco_deck_path)
    with zipfile.ZipFile(ws.eco_deck_path, "a") as zf:
        zf.writestr("case/boundary/basis/", "")

    members = extract_archive(ws, ws.eco_deck_path)

    assert members == ("case/boundary/manifest.bin",)
    assert (ws.root / "case/boundary/basis").is_dir()


def test_extract_archive_self_reference_member_raises_data_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"eco_deck.zip": b"nested"}, ws.eco_deck_path)

    with pytest.raises(DataError, match="eco_deck.zip"):
        extract_archive(ws, ws.eco_deck_path)


def test_extract_archive_unsafe_member_raises_unsafe_archive_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"../escape": b"evil"}, ws.eco_deck_path)

    with pytest.raises(UnsafeArchiveError, match="escapes"):
        extract_archive(ws, ws.eco_deck_path)

    assert {p.name for p in ws.root.iterdir()} == {"eco_deck.zip"}


# ---------------------------------------------------------------------------
# purge_stale_outputs
# ---------------------------------------------------------------------------


def test_purge_stale_outputs_plugin_pattern_case_insensitive_deletes_member(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    (ws.root / "PMO.DAT").write_bytes(b"stale")

    purged = purge_stale_outputs(ws, _PurgePlugin(), ("PMO.DAT",))

    assert purged == ("PMO.DAT",)
    assert not (ws.root / "PMO.DAT").exists()


def test_purge_stale_outputs_core_pattern_at_depth_deletes_nested_modelops(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    (ws.root / "sub").mkdir()
    (ws.root / "sub" / "metadata.modelops").write_bytes(b"stale")

    purged = purge_stale_outputs(ws, FakePlugin(), ("sub/metadata.modelops",))

    assert purged == ("sub/metadata.modelops",)
    assert not (ws.root / "sub" / "metadata.modelops").exists()


def test_purge_stale_outputs_skips_symlink_member(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    target = ws.root / "real.out"
    target.write_bytes(b"real")
    link = ws.root / "fake.out"
    os.symlink(target, link)

    purged = purge_stale_outputs(ws, _PurgePlugin(), ("fake.out",))

    assert purged == ()
    assert link.is_symlink()
    assert target.read_bytes() == b"real"


def test_purge_stale_outputs_skips_member_missing_on_disk(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)

    purged = purge_stale_outputs(ws, _PurgePlugin(), ("fake.out",))

    assert purged == ()


def test_purge_stale_outputs_non_matching_member_is_kept(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    (ws.root / "deck.txt").write_bytes(b"keep")

    purged = purge_stale_outputs(ws, _PurgePlugin(), ("deck.txt",))

    assert purged == ()
    assert (ws.root / "deck.txt").exists()


# ---------------------------------------------------------------------------
# run_name_normalizer
# ---------------------------------------------------------------------------


def test_run_name_normalizer_no_normalizer_configured_is_noop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = Workspace.at(tmp_path)

    with caplog.at_level(logging.WARNING):
        run_name_normalizer(ws, FakePlugin())

    assert caplog.records == []


def test_run_name_normalizer_nonzero_exit_logs_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)
    write_executable_stub(ws.assets, "normalize", "exit 3")

    with caplog.at_level(logging.WARNING):
        run_name_normalizer(ws, _NormalizerPlugin())

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "exited 3" in caplog.records[0].getMessage()


def test_run_name_normalizer_missing_executable_logs_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)

    with caplog.at_level(logging.WARNING):
        run_name_normalizer(ws, _NormalizerPlugin())

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "failed to start" in caplog.records[0].getMessage()


def test_run_name_normalizer_timeout_logs_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)
    write_executable_stub(ws.assets, "normalize", "sleep 5")

    with caplog.at_level(logging.WARNING):
        run_name_normalizer(ws, _NormalizerPlugin(), timeout=0.5)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "timed out" in caplog.records[0].getMessage()


def test_run_name_normalizer_runs_with_cwd_equal_to_workspace_root(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)
    marker = ws.root / "cwd.marker"
    write_executable_stub(ws.assets, "normalize", f"pwd > {marker}")

    run_name_normalizer(ws, _NormalizerPlugin())

    assert marker.read_text().strip() == str(ws.root)


# ---------------------------------------------------------------------------
# sanitize_workspace
# ---------------------------------------------------------------------------


def test_sanitize_workspace_converts_latin1_top_level_file_to_utf8(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    path = ws.root / "notes.txt"
    path.write_bytes(_LATIN1)

    converted = sanitize_workspace(ws, FakePlugin())

    assert converted == ("notes.txt",)
    assert path.read_bytes() == "NOME ACENTUAÇÃO\n".encode("utf-8")


def test_sanitize_workspace_disabled_skips_every_file(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    path = ws.root / "notes.txt"
    path.write_bytes(_LATIN1)

    converted = sanitize_workspace(ws, _NoSanitizePlugin())

    assert converted == ()
    assert path.read_bytes() == _LATIN1


def test_sanitize_workspace_excludes_deck_modelops_license_pattern_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace.at(root)
    plugin = _ExclusionPlugin()

    ws.eco_deck_path.write_bytes(_LATIN1)
    (ws.root / "status.modelops").write_bytes(_LATIN1)
    (ws.root / "FAKE.LIC").write_bytes(_LATIN1)
    (ws.root / "cortes-001.dat").write_bytes(_LATIN1)
    outside = tmp_path / "outside.dat"
    outside.write_bytes(_LATIN1)
    link = ws.root / "link.dat"
    os.symlink(outside, link)

    converted = sanitize_workspace(ws, plugin)

    assert converted == ()
    assert ws.eco_deck_path.read_bytes() == _LATIN1
    assert (ws.root / "status.modelops").read_bytes() == _LATIN1
    assert (ws.root / "FAKE.LIC").read_bytes() == _LATIN1
    assert (ws.root / "cortes-001.dat").read_bytes() == _LATIN1
    assert link.is_symlink()
    assert outside.read_bytes() == _LATIN1


def test_sanitize_files_converts_only_named_files(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    named = ws.root / "dger.dat"
    unnamed = ws.root / "pmo.dat"
    named.write_bytes(_LATIN1)
    unnamed.write_bytes(_LATIN1)

    converted = sanitize_files(ws, FakePlugin(), ("dger.dat", "missing.dat"))

    assert converted == ("dger.dat",)
    assert named.read_bytes() == "NOME ACENTUAÇÃO\n".encode("utf-8")
    assert unnamed.read_bytes() == _LATIN1


@pytest.mark.parametrize("name", ["sub/dger.dat", "../outside.dat", "..", ""])
def test_sanitize_files_skips_names_that_are_not_top_level(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    ws = Workspace.at(root)
    nested = root / "sub" / "dger.dat"
    outside = tmp_path / "outside.dat"
    nested.write_bytes(_LATIN1)
    outside.write_bytes(_LATIN1)

    converted = sanitize_files(ws, FakePlugin(), (name,))

    assert converted == ()
    assert nested.read_bytes() == _LATIN1
    assert outside.read_bytes() == _LATIN1


def test_sanitize_files_applies_the_workspace_exclusions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace.at(root)
    ws.eco_deck_path.write_bytes(_LATIN1)
    (ws.root / "status.modelops").write_bytes(_LATIN1)
    (ws.root / "FAKE.LIC").write_bytes(_LATIN1)
    (ws.root / "cortes-001.dat").write_bytes(_LATIN1)

    converted = sanitize_files(
        ws,
        _ExclusionPlugin(),
        (
            ws.eco_deck_path.name,
            "status.modelops",
            "FAKE.LIC",
            "cortes-001.dat",
        ),
    )

    assert converted == ()
    for name in ("status.modelops", "FAKE.LIC", "cortes-001.dat"):
        assert (ws.root / name).read_bytes() == _LATIN1


# ---------------------------------------------------------------------------
# move_licences
# ---------------------------------------------------------------------------


def test_move_licences_moves_present_file_from_assets_to_root(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)
    (ws.assets / "fake.lic").write_bytes(b"LICENSE")

    moved = move_licences(ws, _ExclusionPlugin())

    assert moved == ("fake.lic",)
    assert (ws.root / "fake.lic").read_bytes() == b"LICENSE"
    assert not (ws.assets / "fake.lic").exists()


def test_move_licences_skips_missing_license_file(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    ws.assets.mkdir(parents=True)

    moved = move_licences(ws, _ExclusionPlugin())

    assert moved == ()


# ---------------------------------------------------------------------------
# parent_run
# ---------------------------------------------------------------------------


def test_parent_run_builds_parent_run_with_resolved_archives(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    ws.parent_dir.mkdir(parents=True)
    (ws.parent_dir / "cortes.zip").write_bytes(b"CUTS")
    info = ParentInfo(
        path="s3://b/p", model_name="NEWAVE", starting_date="2025-01-01"
    )

    result = parent_run(ws, _ParentArtifactPlugin(), info)

    assert result == ParentRun(
        uri="s3://b/p",
        model_name="NEWAVE",
        starting_date="2025-01-01",
        archives=(ws.parent_dir / "cortes.zip",),
    )


def test_parent_run_missing_archive_raises_usage_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    ws.parent_dir.mkdir(parents=True)
    info = ParentInfo(
        path="s3://b/p", model_name="NEWAVE", starting_date="2025-01-01"
    )

    with pytest.raises(UsageError, match="cortes.zip missing"):
        parent_run(ws, _ParentArtifactPlugin(), info)


# ---------------------------------------------------------------------------
# extract_sanitize_inputs
# ---------------------------------------------------------------------------


def test_extract_sanitize_inputs_missing_eco_deck_raises_usage_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    store = StateStore(ws)

    with pytest.raises(UsageError, match="eco_deck.zip missing"):
        extract_sanitize_inputs(ws, FakePlugin(), store, RecordingReporter())


def test_extract_sanitize_inputs_assets_member_raises_data_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"assets/fake-model": b"bin"}, ws.eco_deck_path)
    store = StateStore(ws)

    with pytest.raises(DataError, match="assets"):
        extract_sanitize_inputs(ws, FakePlugin(), store, RecordingReporter())

    assert not (ws.root / "assets" / "fake-model").exists()


def test_extract_sanitize_inputs_self_reference_member_raises_data_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"eco_deck.zip": b"nested"}, ws.eco_deck_path)
    store = StateStore(ws)

    with pytest.raises(DataError, match="eco_deck.zip"):
        extract_sanitize_inputs(ws, FakePlugin(), store, RecordingReporter())


def test_extract_sanitize_inputs_unsafe_member_raises_unsafe_archive_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"../escape": b"evil"}, ws.eco_deck_path)
    store = StateStore(ws)

    with pytest.raises(UnsafeArchiveError, match="escapes"):
        extract_sanitize_inputs(ws, FakePlugin(), store, RecordingReporter())


def test_extract_sanitize_inputs_purge_and_order_matches_ac1(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    plugin = _ParentFetchPlugin()
    store = StateStore(ws)
    initial = store.load_or_create(plugin.name)
    store.save(
        replace(
            initial,
            parent=ParentInfo(
                path="s3://b/parent",
                model_name="NEWAVE",
                starting_date="2025-01-01",
            ),
        )
    )
    ws.parent_dir.mkdir(parents=True)
    (ws.parent_dir / "cortes.zip").write_bytes(b"CUTS")

    input_zip(
        {
            "deck.txt": b"deck content",
            "PMO.DAT": b"stale pmo",
            "fake.out": b"INFEASIBLE\n",
            "metadata.modelops": b"{}",
            "sub/run.json": b"{}",
            "notes.txt": _LATIN1,
        },
        ws.eco_deck_path,
    )

    reporter = RecordingReporter()
    new_state = extract_sanitize_inputs(ws, plugin, store, reporter)

    assert not (ws.root / "PMO.DAT").exists()
    # The deck's own stale metadata.modelops is purged; write_projections
    # (Requirement 7.10) legitimately recreates the file afterward, so
    # its content -- never the stale "{}" the deck shipped -- is what
    # proves the purge ran before the rewrite.
    metadata_payload = json.loads((ws.root / "metadata.modelops").read_bytes())
    assert metadata_payload["study_name"] == "fake"
    assert not (ws.root / "sub" / "run.json").exists()
    assert (ws.root / "fake.out").read_bytes() == b"PARENT\n"
    assert (ws.root / "parent.txt").read_bytes() == _LATIN1
    assert (ws.root / "notes.txt").read_text(encoding="utf-8") == (
        "NOME ACENTUAÇÃO\n"
    )
    assert new_state.input_files == ("deck.txt", "notes.txt")
    assert reporter.metadata_calls == [
        ("study_starting_date", "2026-01-01"),
        ("study_name", "fake"),
    ]


def test_extract_sanitize_inputs_r130_listing_logs_one_line_per_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ws = Workspace.at(tmp_path)
    input_zip({"deck.txt": b"deck"}, ws.eco_deck_path)
    store = StateStore(ws)

    with caplog.at_level(logging.INFO):
        extract_sanitize_inputs(ws, FakePlugin(), store, RecordingReporter())

    deck_size = (ws.root / "deck.txt").stat().st_size
    messages = [r.message for r in caplog.records]
    assert f"{deck_size} deck.txt" in messages
    assert not any(
        m.endswith(".hpcmu") or m.endswith(".hpcmu/") for m in messages
    )


# ---------------------------------------------------------------------------
# flatten_execution_name
# ---------------------------------------------------------------------------


def test_flatten_execution_name_control_and_separator_chars_to_space() -> None:
    raw = "a\r\nb\x85c\u2028d\u2029e\x00f"
    expected = "".join(
        " " if ch in "\r\n\x85\u2028\u2029\x00" else ch for ch in raw
    )

    assert flatten_execution_name(raw) == expected


def test_flatten_execution_name_printable_unicode_preserved() -> None:
    assert flatten_execution_name("café") == "café"


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_no_state_calls_prepare_and_never_creates_hpcmu(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    (ws.root / "pmo.dat").write_bytes(b"stale")
    plugin = _RecordingPreparePlugin()
    store = StateStore(ws)

    preprocess(ws, plugin, "a\r\nb", store=store)

    assert plugin.prepare_calls == [(ws, "a  b")]
    assert not ws.hpcmu_dir.exists()
    assert (ws.root / "pmo.dat").read_bytes() == b"stale"


def test_preprocess_state_present_records_exactly_one_step(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    store = StateStore(ws)
    store.load_or_create("fakerecordprep")
    plugin = _RecordingPreparePlugin()

    preprocess(ws, plugin, "name", store=store)

    state = store.load()
    assert state is not None
    assert [s.command for s in state.steps] == ["preprocess"]
    assert plugin.prepare_calls == [(ws, "name")]


def test_preprocess_plugin_mismatch_raises_usage_error(
    tmp_path: Path,
) -> None:
    ws = Workspace.at(tmp_path)
    store = StateStore(ws)
    store.load_or_create("other-plugin")

    with pytest.raises(UsageError, match="plugin mismatch"):
        preprocess(ws, FakePlugin(), "name", store=store)
