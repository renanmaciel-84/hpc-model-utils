"""ADR-048/R82/R16/R17/R55: v2 ``publish`` over the real NEWAVE and
DECOMP plugins against the frozen v1.1.2 archive and projection goldens
(ticket-055, C5-C7).

The uploaded key set, archive members, root files and projection bytes
must equal the golden, except through a row of the README's
``## Intended differences`` table. ``INTENDED_DIFFERENCES`` encodes each
such row's effect, and the README test keeps the two in lockstep.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

from hpc_model_utils.core.diagnosis import RunStatus
from hpc_model_utils.core.workspace import Phase
from hpc_model_utils.infra.telemetry.schema import (
    EXPORT_FILES,
    HEADLINE_KEY_PREFIX,
)
from hpc_model_utils.platform.modelops import STATUS_HOOKS, HookMethod
from tests.support.decks import (
    FIXTURES,
    GOLDENS,
    archive_golden,
    golden_archive_workspace,
)
from tests.support.published_run import (
    JOB_IDS,
    PREFIX,
    ModelName,
    PublishedRun,
    publish_golden_run,
)

_MODELS: Final[tuple[ModelName, ...]] = ("newave", "decomp")
_README = GOLDENS / "README.md"
_PROJECTIONS = GOLDENS / "projections.json"
_METADATA_KEY = "saidas/metadata.modelops"
_STATUS_KEY = "saidas/status.modelops"
_RUN_JSON_KEY = "saidas/run.json"
_TELEMETRY_PREFIX = "saidas/telemetry/"
_TELEMETRY_KEYS: Final = tuple(
    f"{_TELEMETRY_PREFIX}{name}" for name in EXPORT_FILES
)
_DECK_ARCHIVE_KEY = "entradas/deck_processado.zip"
_ECO_DECK = "eco_deck.zip"

# AC1: raw outputs the consumers read (encadeador-pem, ranqueamento).
_REQUIRED_UPLOADS: Final[Mapping[ModelName, frozenset[str]]] = MappingProxyType(
    {
        "newave": frozenset(
            {
                "saidas/pmo.dat",
                "saidas/cortes.zip",
                "saidas/recursos.zip",
                "saidas/simulacao.zip",
            }
        ),
        "decomp": frozenset(
            {
                "saidas/relato.rv0",
                "saidas/relgnl.rv0",
                "saidas/inviab_unic.rv0",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Difference:
    """One README ``## Intended differences`` row's effect on the golden
    workspace: upload keys added or removed per model, and members added
    per (model, archive name)."""

    added: Mapping[ModelName, frozenset[str]] = field(default_factory=dict)
    removed: Mapping[ModelName, frozenset[str]] = field(default_factory=dict)
    members_added: Mapping[tuple[ModelName, str], frozenset[str]] = field(
        default_factory=dict
    )


def _log_keys(model: ModelName) -> frozenset[str]:
    model_job, finalize_job = JOB_IDS[model]
    return frozenset(
        {
            f"saidas/logs/{Phase.MODEL}-{model_job}.out",
            f"saidas/logs/{Phase.FINALIZE}-{finalize_job}.out",
        }
    )


_NEWAVE_DECK_ONLY_INPUTS = frozenset(
    {"bid.dat", "elnino.dat", "ensoaux.dat", "itaipu.dat"}
)

INTENDED_DIFFERENCES: Final[Mapping[str, Difference]] = MappingProxyType(
    {
        "`archives.decomp.uploads`": Difference(
            added={"decomp": frozenset({"saidas/relgnl.rv0"})}
        ),
        "`archives.newave.uploads` / `archives.decomp.uploads`": Difference(
            added={model: _log_keys(model) for model in _MODELS}
        ),
        "`archives.newave.archives.deck_processado.zip`": Difference(
            members_added={
                ("newave", "deck_processado.zip"): _NEWAVE_DECK_ONLY_INPUTS
            }
        ),
        "`archives.newave.uploads.residual_inputs`": Difference(
            removed={
                "newave": frozenset(
                    f"saidas/{name}" for name in _NEWAVE_DECK_ONLY_INPUTS
                )
            }
        ),
        # Selection semantics only: no name in the golden workspace
        # tells v1's ``search`` from v2's ``.dat`` suffix rule.
        "`archives.newave.uploads.residual_rule`": Difference(),
        "`archives.decomp.uploads.dadger_echo`": Difference(
            removed={"decomp": frozenset({"entradas/dadger.rv0"})}
        ),
        "`archives.*.uploads.status_modelops`": Difference(
            added={model: frozenset({_STATUS_KEY}) for model in _MODELS}
        ),
        "`archives.*.uploads.run_json`": Difference(
            added={model: frozenset({"saidas/run.json"}) for model in _MODELS}
        ),
        # No upload or member effect; the root is asserted by AC3.
        "`archives.*.remaining`": Difference(),
    }
)


def _union(sets: Iterable[frozenset[str]]) -> frozenset[str]:
    result: set[str] = set()
    for item in sets:
        result |= item
    return frozenset(result)


def _added(model: ModelName) -> frozenset[str]:
    return _union(
        diff.added.get(model, frozenset())
        for diff in INTENDED_DIFFERENCES.values()
    )


def _removed(model: ModelName) -> frozenset[str]:
    return _union(
        diff.removed.get(model, frozenset())
        for diff in INTENDED_DIFFERENCES.values()
    )


def _members_added(model: ModelName, archive: str) -> frozenset[str]:
    return _union(
        diff.members_added.get((model, archive), frozenset())
        for diff in INTENDED_DIFFERENCES.values()
    )


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise TypeError(f"{label} is not a string list")
    return value


def _golden_uploads(model: ModelName) -> list[str]:
    return _string_list(
        archive_golden(model)["uploads"], f"archives.{model}.uploads"
    )


def _golden_remaining(model: ModelName) -> list[str]:
    return _string_list(
        archive_golden(model)["remaining"], f"archives.{model}.remaining"
    )


def _golden_archives(model: ModelName) -> dict[str, list[str]]:
    archives = archive_golden(model)["archives"]
    if not isinstance(archives, dict):
        raise TypeError(f"archives.{model}.archives is not an object")
    return {
        str(name): _string_list(members, f"archives.{model}.archives.{name}")
        for name, members in archives.items()
    }


def _projection_golden(model: ModelName) -> tuple[bytes, bytes]:
    """The ``<model>_no_parent`` (metadata, status) golden bytes."""
    data = json.loads(_PROJECTIONS.read_text(encoding="utf-8"))
    sequence = data["sequences"][f"{model}_no_parent"]
    metadata = sequence["metadata.modelops"]
    status = sequence["status.modelops"]
    if not isinstance(metadata, str) or not isinstance(status, str):
        raise TypeError(f"projections.json[{model}_no_parent] is malformed")
    return metadata.encode("ascii"), status.encode("ascii")


def _upload_keys(run: PublishedRun) -> list[str]:
    """Every recorded upload, in order, relative to ``run.prefix``."""
    base = f"{run.prefix}/"
    keys: list[str] = []
    for op, uri in run.store.ops:
        if op != "upload":
            continue
        assert uri.startswith(base), f"upload outside the prefix: {uri}"
        keys.append(uri[len(base) :])
    return keys


def _object(run: PublishedRun, key: str) -> bytes:
    return run.store.get_bytes(run.prefix.join(key))


def _namelist(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return sorted(archive.namelist())


def _is_archive_key(key: str) -> bool:
    if key == _DECK_ARCHIVE_KEY:
        return True
    name = key.removeprefix("saidas/")
    return name != key and "/" not in name and name.endswith(".zip")


def _regular_files(root: Path) -> dict[str, bytes]:
    """Relative path to bytes of every regular file under ``root``,
    ``.hpcmu/`` excluded."""
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] == ".hpcmu" or path.is_symlink():
            continue
        if path.is_file():
            files[rel.as_posix()] = path.read_bytes()
    return files


def _readme_difference_keys() -> list[str]:
    """The stripped first cell of every ``## Intended differences`` row:
    the rows after the ``| Golden |`` header, up to the first blank
    line."""
    lines = _README.read_text(encoding="utf-8").splitlines()
    try:
        section = lines.index("## Intended differences")
    except ValueError:
        raise AssertionError(
            "README has no '## Intended differences' section"
        ) from None
    header = next(
        (
            index
            for index in range(section + 1, len(lines))
            if lines[index].startswith("| Golden |")
        ),
        None,
    )
    assert header is not None, "README has no '| Golden |' table header"
    cells: list[str] = []
    for line in lines[header + 1 :]:
        if not line.strip():
            break
        assert line.startswith("|"), f"malformed table row: {line!r}"
        first = line.split("|")[1].strip()
        if set(first) <= {"-", ":"}:
            continue
        cells.append(first)
    return cells


# -- AC1: upload key set --------------------------------------------------


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_fresh_prefix_uploads_golden_keys_with_intended_differences(
    tmp_path: Path, model: ModelName
) -> None:
    golden = frozenset(_golden_uploads(model))
    removed = _removed(model)
    added = _added(model)
    # Each listed difference must really differ from the golden.
    assert removed <= golden
    assert added.isdisjoint(golden)

    run = publish_golden_run(tmp_path, model)

    keys = _upload_keys(run)
    assert sorted(keys) == sorted((golden - removed) | added)
    assert _REQUIRED_UPLOADS[model] <= set(keys)


# -- AC2: uploaded archive members and bytes ------------------------------


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_fresh_prefix_archive_members_equal_golden_with_additions(
    tmp_path: Path, model: ModelName
) -> None:
    golden = _golden_archives(model)
    run = publish_golden_run(tmp_path, model)

    checked: set[str] = set()
    for key in _upload_keys(run):
        if not _is_archive_key(key):
            continue
        name = key.rsplit("/", 1)[-1]
        additions = _members_added(model, name)
        assert additions.isdisjoint(golden[name])
        assert _namelist(_object(run, key)) == sorted(
            [*golden[name], *additions]
        )
        checked.add(name)
    assert checked == set(golden) - {_ECO_DECK}


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_fresh_prefix_eco_deck_equals_fixture_bytes(
    tmp_path: Path, model: ModelName
) -> None:
    run = publish_golden_run(tmp_path, model)

    fixture = (FIXTURES / "decks" / f"deck_{model}.zip").read_bytes()
    assert _object(run, f"entradas/{_ECO_DECK}") == fixture


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_success_projections_equal_no_parent_golden_bytes(
    tmp_path: Path, model: ModelName
) -> None:
    metadata, status = _projection_golden(model)
    run = publish_golden_run(tmp_path, model)

    assert _object(run, _METADATA_KEY) == metadata
    assert _object(run, _STATUS_KEY) == status == b"SUCCESS"


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_fresh_prefix_last_upload_is_metadata_marker(
    tmp_path: Path, model: ModelName
) -> None:
    run = publish_golden_run(tmp_path, model)

    assert _upload_keys(run)[-1] == _METADATA_KEY


# -- C7: sintese files ----------------------------------------------------


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_sintese_files_uploaded_under_own_names(
    tmp_path: Path, model: ModelName
) -> None:
    run = publish_golden_run(tmp_path, model)

    sintese = sorted((run.ws.root / "sintese").iterdir())
    assert sintese
    keys = set(_upload_keys(run))
    for path in sintese:
        key = f"sintese/{path.name}"
        assert key in keys
        assert _object(run, key) == path.read_bytes()


# -- AC3: no root cleanup -------------------------------------------------


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_reference_snapshot_root_files_unchanged(
    tmp_path: Path, model: ModelName
) -> None:
    reference = _regular_files(
        golden_archive_workspace(tmp_path / "ref", model).root
    )
    assert reference

    run = publish_golden_run(tmp_path / "pub", model)

    published = _regular_files(run.ws.root)
    for rel, data in reference.items():
        assert published.get(rel) == data, rel


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_golden_remaining_archives_only_under_outputs_dir(
    tmp_path: Path, model: ModelName
) -> None:
    archives = [
        name
        for name in _golden_remaining(model)
        if name.endswith(".zip") and name != _ECO_DECK
    ]
    assert archives

    run = publish_golden_run(tmp_path, model)

    for name in archives:
        assert (run.ws.outputs_dir / name).is_file(), name
        assert not (run.ws.root / name).exists(), name


# -- publish_golden_run's status parameter --------------------------------


@pytest.mark.parametrize("status", (RunStatus.SUCCESS, RunStatus.RUNTIME_ERROR))
@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_given_status_projects_status_and_hooks(
    tmp_path: Path, model: ModelName, status: RunStatus
) -> None:
    run = publish_golden_run(tmp_path, model, status=status)

    metadata = json.loads(_object(run, _METADATA_KEY).decode("ascii"))
    assert metadata["status"] == status.value
    assert metadata["job_id"] == JOB_IDS[model][1]
    assert _object(run, _STATUS_KEY) == status.value.encode("ascii")
    artifacts_paths = [
        hook.args
        for hook in run.hooks
        if hook.method == HookMethod.SET_EXECUTION_ARTIFACTS_PATH
    ]
    assert artifacts_paths == [(PREFIX,)]
    status_hooks = set(STATUS_HOOKS.values())
    terminal = [
        hook.method for hook in run.hooks if hook.method in status_hooks
    ]
    assert terminal == [STATUS_HOOKS[status]]


# -- publish_golden_run's telemetry parameter -----------------------------
#
# v1.1.2 has no telemetry run, so the keys are pinned relative to the same
# model's telemetry-off run, not through the goldens or INTENDED_DIFFERENCES.


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_telemetry_keys_inserted_before_status_marker(
    tmp_path: Path, model: ModelName
) -> None:
    off = _upload_keys(publish_golden_run(tmp_path / "off", model))
    on = _upload_keys(
        publish_golden_run(tmp_path / "on", model, telemetry=True)
    )

    index = off.index(_STATUS_KEY)
    assert on == [*off[:index], *_TELEMETRY_KEYS, *off[index:]]
    assert on[index - 1].startswith("sintese/")


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_telemetry_manifest_lists_sorted_entries_with_sizes(
    tmp_path: Path, model: ModelName
) -> None:
    run = publish_golden_run(tmp_path, model, telemetry=True)

    artifacts = json.loads(_object(run, _RUN_JSON_KEY).decode("utf-8"))[
        "artifacts"
    ]
    paths = [entry["path"] for entry in artifacts]
    telemetry = [
        index
        for index, path in enumerate(paths)
        if path.startswith(_TELEMETRY_PREFIX)
    ]
    assert [paths[index] for index in telemetry] == sorted(_TELEMETRY_KEYS)
    assert telemetry == list(range(telemetry[0], telemetry[0] + len(telemetry)))
    assert paths == sorted(paths)
    sintese = [
        index for index, path in enumerate(paths) if path.startswith("sintese/")
    ]
    assert sintese
    assert paths.index(_STATUS_KEY) < telemetry[0] <= telemetry[-1] < sintese[0]
    for index in telemetry:
        assert artifacts[index]["bytes"] == len(_object(run, paths[index]))


@pytest.mark.parametrize("model", _MODELS)
def test_publish_golden_run_telemetry_absent_by_default(
    tmp_path: Path, model: ModelName
) -> None:
    run = publish_golden_run(tmp_path, model)

    assert run.hooks
    assert not [
        key for key in _upload_keys(run) if key.startswith(_TELEMETRY_PREFIX)
    ]
    assert not [
        hook
        for hook in run.hooks
        if hook.args and hook.args[0].startswith(HEADLINE_KEY_PREFIX)
    ]


# -- AC4: README consistency ----------------------------------------------


def test_readme_difference_keys_archives_rows_equal_intended_differences() -> (
    None
):
    archive_rows = [
        cell for cell in _readme_difference_keys() if "archives." in cell
    ]
    assert len(archive_rows) == len(set(archive_rows))
    assert set(archive_rows) == set(INTENDED_DIFFERENCES)


def test_readme_difference_keys_dsvagua_encoding_row_present() -> None:
    assert any("dsvagua.dat" in cell for cell in _readme_difference_keys())
