"""infra.archive: stdlib-only safe zip adapter.

ADR-028 and the security brief require every member to be validated
before any extraction happens, so an unsafe archive fails loudly instead
of being half-extracted. ADR-040 and ADR-038 require streamed members
(never read whole into memory), Zip64 always on, flat decks with no
basename collisions, and tree archives whose members are POSIX paths
relative to a declared root that can never escape it.

Partial-archive policy: ``write_flat`` and ``write_tree`` validate every
member first -- for ``write_flat`` that includes the whole basename
collision table, for ``write_tree`` the root-escape check -- before
opening any zip file, so a validation failure never creates a file on
disk. Once validation has passed, the archive is built at a sibling
temporary path, ``archive.with_name("." + archive.name + ".part")``,
and only ``os.replace``'d onto ``archive`` after every member has been
written successfully; on any exception the partial file is unlinked.
``archive`` is therefore always either absent/unchanged or a complete,
valid zip, never a half-written one.

ADR-040's parallelism clause, amended by ticket-048b (``design/
amendments.md`` AM-003): members are compressed in parallel, up to
``workers`` threads per archive, each one streaming its member through
raw deflate into a scratch file under the archive's own directory --
never reading a member whole, so memory stays O(workers x 1 MiB). The
calling thread alone appends compressed members to the ``ZipFile``, in
input order, once a member's compression finishes. The scratch
directory is removed, and any ``.part`` file unlinked, on every
``BaseException``.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import warnings
import zipfile
import zlib
from collections import deque
from collections.abc import Collection, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hpc_model_utils.infra.errors import UnsafeArchiveError

_HPCMU_DIR = ".hpcmu"
_STORED_SUFFIXES = frozenset({".parquet", ".zip", ".gz", ".bz2", ".xz", ".7z"})
_DRIVE_LETTER_RE = re.compile(r"[A-Za-z]:.*")
_CHUNK_SIZE = 1 << 20


def _normalized_parts(name: str) -> tuple[str, ...]:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute():
        raise UnsafeArchiveError(f"absolute member path: {name!r}")
    parts = pure.parts
    if not parts:
        raise UnsafeArchiveError(f"empty member name: {name!r}")
    if _DRIVE_LETTER_RE.fullmatch(parts[0]):
        raise UnsafeArchiveError(f"drive-like member path: {name!r}")
    if ".." in parts:
        raise UnsafeArchiveError(f"member path escapes archive root: {name!r}")
    if parts[0] == _HPCMU_DIR:
        raise UnsafeArchiveError(
            f"member path under reserved {_HPCMU_DIR}/: {name!r}"
        )
    return parts


def _validated_entries(archive: Path) -> tuple[list[str], list[str]]:
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as err:
        raise UnsafeArchiveError(f"not a zip archive: {archive.name}") from err
    files: list[str] = []
    directories: list[str] = []
    for info in infos:
        _normalized_parts(info.filename)
        (directories if info.is_dir() else files).append(info.filename)
    return files, directories


def validated_members(archive: Path) -> list[str]:
    return _validated_entries(archive)[0]


def validated_directories(archive: Path) -> list[str]:
    return _validated_entries(archive)[1]


def extract(
    archive: Path, dest: Path, *, members: Collection[str] | None = None
) -> list[str]:
    """Return the extracted file members; a full extraction also
    recreates directory entries, since a deck may need an empty one."""
    valid, directories = _validated_entries(archive)
    selected = valid if members is None else [n for n in valid if n in members]
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        if members is None:
            for name in directories:
                zf.extract(name, dest)
        for name in selected:
            zf.extract(name, dest)
    return sorted(selected)


def common_root(names: Sequence[str]) -> str | None:
    if not names:
        return None
    roots: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) < 2:
            return None
        roots.add(parts[0])
    if len(roots) != 1:
        return None
    return roots.pop()


def compression_for(path: Path) -> int:
    if path.suffix.lower() in _STORED_SUFFIXES:
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


@dataclass(frozen=True, slots=True)
class _CompressedMember:
    info: zipfile.ZipInfo
    payload: Path


def _compress_member(
    path: Path, arcname: str, scratch: Path
) -> _CompressedMember:
    info = zipfile.ZipInfo.from_file(path, arcname)
    info.compress_type = compression_for(path)
    with tempfile.NamedTemporaryFile(
        dir=scratch, suffix=".member", delete=False
    ) as dest:
        payload = Path(dest.name)
        with path.open("rb") as src:
            compressor = (
                zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
                if info.compress_type == zipfile.ZIP_DEFLATED
                else None
            )
            crc = 0
            file_size = 0
            compress_size = 0
            while chunk := src.read(_CHUNK_SIZE):
                crc = zlib.crc32(chunk, crc)
                file_size += len(chunk)
                piece = compressor.compress(chunk) if compressor else chunk
                compress_size += len(piece)
                dest.write(piece)
            if compressor is not None:
                tail = compressor.flush()
                compress_size += len(tail)
                dest.write(tail)
    info.CRC = crc
    info.file_size = file_size
    info.compress_size = compress_size
    return _CompressedMember(info, payload)


def _append_member(zf: zipfile.ZipFile, member: _CompressedMember) -> None:
    # Mirrors zipfile.ZipFile.mkdir (CPython 3.12-3.14): seek to
    # start_dir, stamp header_offset, write the header, then advance
    # start_dir past the written bytes.
    info = member.info
    fp = zf.fp
    assert fp is not None
    fp.seek(zf.start_dir)
    info.header_offset = fp.tell()
    if info.filename in zf.NameToInfo:
        warnings.warn(f"Duplicate name: {info.filename!r}", UserWarning)
    fp.write(info.FileHeader())
    with member.payload.open("rb") as src:
        shutil.copyfileobj(src, fp, _CHUNK_SIZE)
    member.payload.unlink()
    zf.filelist.append(info)
    zf.NameToInfo[info.filename] = info
    zf.start_dir = fp.tell()


def _write_zip(
    archive: Path, entries: Sequence[tuple[Path, str]], *, workers: int
) -> None:
    tmp = archive.with_name("." + archive.name + ".part")
    clamped = max(1, workers)
    window = 2 * clamped
    try:
        with (
            tempfile.TemporaryDirectory(
                prefix="." + archive.name + ".",
                suffix=".members",
                dir=archive.parent,
            ) as scratch,
            ThreadPoolExecutor(max_workers=clamped) as pool,
            zipfile.ZipFile(
                tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zf,
        ):
            scratch_dir = Path(scratch)
            pending: deque[Future[_CompressedMember]] = deque()
            try:
                for path, arcname in entries:
                    pending.append(
                        pool.submit(
                            _compress_member, path, arcname, scratch_dir
                        )
                    )
                    if len(pending) >= window:
                        _append_member(zf, pending.popleft().result())
                while pending:
                    _append_member(zf, pending.popleft().result())
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, archive)


def write_flat(
    archive: Path, files: Sequence[Path], *, workers: int = 1
) -> list[str]:
    seen: dict[str, Path] = {}
    for path in files:
        name = path.name
        if name in seen:
            raise UnsafeArchiveError(
                f"duplicate basename {name!r}: {seen[name]} and {path}"
            )
        seen[name] = path
    _write_zip(
        archive, [(path, name) for name, path in seen.items()], workers=workers
    )
    return sorted(seen)


def write_tree(
    archive: Path, root: Path, files: Sequence[Path], *, workers: int = 1
) -> list[str]:
    root_resolved = root.resolve()
    entries: list[tuple[Path, str]] = []
    for path in files:
        resolved = path.resolve()
        try:
            arcname = resolved.relative_to(root_resolved).as_posix()
        except ValueError as err:
            raise UnsafeArchiveError(
                f"path escapes root {root}: {path}"
            ) from err
        entries.append((path, arcname))
    _write_zip(archive, entries, workers=workers)
    return sorted(arcname for _, arcname in entries)
