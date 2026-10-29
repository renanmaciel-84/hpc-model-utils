from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Literal

import pytest

from hpc_model_utils.infra import archive
from hpc_model_utils.infra.errors import UnsafeArchiveError

UNSAFE_MEMBER_NAMES = [
    "../evil",
    "/abs",
    ".hpcmu/state.json",
    "ok/../../x",
    "C:/x",
    "\\abs",
    ".hpcmu",
    "./.hpcmu/x",
    "",
    "C:",
    "C:foo",
]

SAFE_COLON_MEMBER_NAMES = [
    "log_10:00.txt",
    "relatorio:v2.txt",
    "dir/sub:x.txt",
]


def _zip_with_raw_names(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for index, name in enumerate(names):
            zf.writestr(zipfile.ZipInfo(name), f"payload-{index}".encode())
    return path


@pytest.mark.parametrize("name", UNSAFE_MEMBER_NAMES)
def test_validated_members_unsafe_name_raises_unsafe_archive_error(
    tmp_path: Path, name: str
) -> None:
    zip_path = _zip_with_raw_names(tmp_path / "unsafe.zip", [name])

    with pytest.raises(UnsafeArchiveError):
        archive.validated_members(zip_path)


@pytest.mark.parametrize("name", UNSAFE_MEMBER_NAMES)
def test_extract_unsafe_member_raises_and_writes_nothing_outside_dest(
    tmp_path: Path, name: str
) -> None:
    zip_path = _zip_with_raw_names(tmp_path / "unsafe.zip", [name])
    dest = tmp_path / "dest"

    with pytest.raises(UnsafeArchiveError):
        archive.extract(zip_path, dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == [zip_path]


def test_validated_members_non_zip_input_raises_unsafe_archive_error(
    tmp_path: Path,
) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_text("not actually a zip")

    with pytest.raises(UnsafeArchiveError, match="not a zip archive"):
        archive.validated_members(bogus)


@pytest.mark.parametrize("name", SAFE_COLON_MEMBER_NAMES)
def test_validated_members_colon_outside_drive_position_returns_name(
    tmp_path: Path, name: str
) -> None:
    zip_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(name, b"data")

    assert archive.validated_members(zip_path) == [name]


def test_validated_members_directory_entry_excluded_from_result(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "deck.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sub/", "")
        zf.writestr("sub/file.txt", "data")

    names = archive.validated_members(zip_path)

    assert names == ["sub/file.txt"]


def test_extract_selective_members_ignores_missing_members(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "deck.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "A")
        zf.writestr("b.txt", "B")
    dest = tmp_path / "dest"

    extracted = archive.extract(
        zip_path, dest, members=["a.txt", "missing.txt"]
    )

    assert extracted == ["a.txt"]
    assert (dest / "a.txt").read_text() == "A"
    assert not (dest / "b.txt").exists()


def test_extract_all_members_creates_empty_directory_entries(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "deck.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("case/boundary/basis/", "")
        zf.writestr("case/boundary/manifest.bin", "M")
    dest = tmp_path / "dest"

    extracted = archive.extract(zip_path, dest)

    assert extracted == ["case/boundary/manifest.bin"]
    assert (dest / "case/boundary/basis").is_dir()


def test_extract_selective_members_skips_directory_entries(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "deck.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("empty/", "")
        zf.writestr("a.txt", "A")
    dest = tmp_path / "dest"

    archive.extract(zip_path, dest, members={"a.txt"})

    assert not (dest / "empty").exists()


@pytest.mark.parametrize("name", ["../evil/", ".hpcmu/", "/abs/"])
def test_extract_unsafe_directory_entry_raises_and_writes_nothing(
    tmp_path: Path, name: str
) -> None:
    zip_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo(name), "")
    dest = tmp_path / "dest"

    with pytest.raises(UnsafeArchiveError):
        archive.extract(zip_path, dest)

    assert not dest.exists()


def test_extract_all_members_returns_sorted_names_excluding_directories(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "deck.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("b.txt", "B")
        zf.writestr("sub/", "")
        zf.writestr("a.txt", "A")
    dest = tmp_path / "dest"

    extracted = archive.extract(zip_path, dest)

    assert extracted == ["a.txt", "b.txt"]
    assert (dest / "a.txt").read_text() == "A"
    assert (dest / "b.txt").read_text() == "B"


def test_write_flat_duplicate_basename_raises_with_both_paths(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a" / "pmo.dat"
    b = tmp_path / "b" / "pmo.dat"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("A")
    b.write_text("B")
    out = tmp_path / "out.zip"

    with pytest.raises(UnsafeArchiveError, match="pmo.dat") as excinfo:
        archive.write_flat(out, [a, b])

    assert str(a) in str(excinfo.value)
    assert str(b) in str(excinfo.value)


def test_write_flat_leaves_no_partial_archive_on_collision(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a" / "pmo.dat"
    b = tmp_path / "b" / "pmo.dat"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("A")
    b.write_text("B")
    out = tmp_path / "out.zip"

    with pytest.raises(UnsafeArchiveError):
        archive.write_flat(out, [a, b])

    assert not out.exists()
    assert not (tmp_path / ".out.zip.part").exists()


def test_write_flat_writes_flat_names_sorted(tmp_path: Path) -> None:
    a = tmp_path / "x" / "b.txt"
    b = tmp_path / "y" / "a.txt"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("B")
    b.write_text("A")
    out = tmp_path / "out.zip"

    names = archive.write_flat(out, [a, b])

    assert names == ["a.txt", "b.txt"]
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["a.txt", "b.txt"]
        assert zf.read("a.txt") == b"A"
        assert zf.read("b.txt") == b"B"


def test_write_tree_relative_arcname_and_store_for_parquet(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    target = (
        root / "simulation" / "hydros" / "scenario_id=0001" / "data.parquet"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"parquet-bytes")
    out = tmp_path / "out.zip"

    names = archive.write_tree(out, root, [target])

    assert names == ["simulation/hydros/scenario_id=0001/data.parquet"]
    with zipfile.ZipFile(out) as zf:
        info = zf.getinfo("simulation/hydros/scenario_id=0001/data.parquet")
        assert info.compress_type == zipfile.ZIP_STORED


def test_write_tree_file_outside_root_raises_unsafe_archive_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    out = tmp_path / "out.zip"

    with pytest.raises(UnsafeArchiveError):
        archive.write_tree(out, root, [outside])


def test_write_tree_symlink_escaping_root_raises_unsafe_archive_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    link = root / "link.txt"
    link.symlink_to(outside)
    out = tmp_path / "out.zip"

    with pytest.raises(UnsafeArchiveError):
        archive.write_tree(out, root, [link])


def test_write_tree_leaves_no_partial_archive_on_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    out = tmp_path / "out.zip"

    with pytest.raises(UnsafeArchiveError):
        archive.write_tree(out, root, [outside])

    assert not out.exists()
    assert not (tmp_path / ".out.zip.part").exists()


@pytest.mark.parametrize(
    "suffix", [".parquet", ".PARQUET", ".zip", ".gz", ".bz2", ".xz", ".7z"]
)
def test_compression_for_stored_suffix_returns_zip_stored(
    suffix: str,
) -> None:
    assert archive.compression_for(Path(f"file{suffix}")) == (
        zipfile.ZIP_STORED
    )


@pytest.mark.parametrize("suffix", [".txt", ".json", ".csv", ""])
def test_compression_for_other_suffix_returns_zip_deflated(
    suffix: str,
) -> None:
    assert archive.compression_for(Path(f"file{suffix}")) == (
        zipfile.ZIP_DEFLATED
    )


def test_common_root_shared_top_level_directory_returns_directory_name() -> (
    None
):
    names = ["case/config.json", "case/system/hydros.json"]
    assert archive.common_root(names) == "case"


def test_common_root_mixed_top_level_file_returns_none() -> None:
    assert archive.common_root(["config.json", "case/x"]) is None


def test_common_root_empty_sequence_returns_none() -> None:
    assert archive.common_root([]) is None


def test_common_root_single_top_level_file_returns_none() -> None:
    assert archive.common_root(["file.txt"]) is None


def test_common_root_directory_only_entry_returns_none() -> None:
    assert archive.common_root(["case/"]) is None


def test_write_flat_enables_allow_zip64_via_zipfile_spy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, bool] = {}
    real_zip_file = zipfile.ZipFile

    def spy(
        file: Path,
        mode: Literal["w"],
        *,
        compression: int,
        allowZip64: bool,
    ) -> zipfile.ZipFile:
        captured["allowZip64"] = allowZip64
        return real_zip_file(
            file, mode, compression=compression, allowZip64=allowZip64
        )

    monkeypatch.setattr(zipfile, "ZipFile", spy)

    src = tmp_path / "a.txt"
    src.write_text("hi")
    archive.write_flat(tmp_path / "out.zip", [src])

    assert captured["allowZip64"] is True
