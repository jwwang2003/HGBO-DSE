from __future__ import annotations

import base64
import binascii
import io
import os
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Dict, Iterable, List, Union


DEFAULT_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_BYTES_ENV = "HGBO_MCP_MAX_ARCHIVE_BYTES"
MAX_GRAPH_ARCHIVE_FILES = 4096
COPY_CHUNK_SIZE = 1024 * 1024
PPA_KEYS = ("LUT", "FF", "DSP", "BRAM", "CP", "PWR")


PathLike = Union[str, os.PathLike]


def create_graph_archive_base64(prj_path: PathLike) -> str:
    graph_dir = Path(prj_path) / "graph"
    if not graph_dir.is_dir() or graph_dir.is_symlink():
        raise FileNotFoundError("graph directory not found: {0}".format(graph_dir))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in _iter_graph_files(graph_dir):
            archive.write(
                file_path,
                arcname=file_path.relative_to(Path(prj_path)).as_posix(),
            )

    raw_archive = buffer.getvalue()
    max_archive_bytes = _max_archive_bytes()
    if len(raw_archive) > max_archive_bytes:
        raise ValueError(
            "Graph archive too large: {0} bytes exceeds {1} bytes".format(
                len(raw_archive),
                max_archive_bytes,
            )
        )

    return base64.b64encode(raw_archive).decode("ascii")


def extract_graph_archive_base64(payload: str, prj_path: PathLike) -> int:
    max_archive_bytes = _max_archive_bytes()
    raw_archive = _decode_archive_payload(payload, max_archive_bytes)
    _ensure_archive_size(len(raw_archive), max_archive_bytes)

    target_root = Path(prj_path)

    try:
        with zipfile.ZipFile(io.BytesIO(raw_archive), "r") as archive:
            graph_infos = _graph_file_infos(archive, max_archive_bytes)
            copied_bytes = 0
            for info in graph_infos:
                copied_bytes += _extract_graph_file(
                    archive,
                    info,
                    target_root,
                    max_archive_bytes - copied_bytes,
                )
    except zipfile.BadZipFile as exc:
        raise ValueError("Graph archive is not a valid zip file") from exc

    return len(graph_infos)


def validate_ppa_result(result: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in PPA_KEYS if key not in result]
    if missing:
        raise ValueError("PPA result missing required keys: {0}".format(", ".join(missing)))
    return {key: result[key] for key in PPA_KEYS}


def _iter_graph_files(graph_dir: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(str(graph_dir), followlinks=False):
        root_path = Path(root)
        dirs[:] = sorted(
            dirname
            for dirname in dirs
            if not (root_path / dirname).is_symlink()
        )
        for filename in sorted(files):
            file_path = root_path / filename
            if file_path.is_symlink() or not file_path.is_file():
                continue
            yield file_path


def _decode_archive_payload(payload: str, max_archive_bytes: int) -> bytes:
    max_decoded_size = _max_base64_decoded_size(payload)
    if max_decoded_size > max_archive_bytes:
        _raise_archive_too_large(max_decoded_size, max_archive_bytes)

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Graph archive payload is not valid base64") from exc


def _max_base64_decoded_size(payload: str) -> int:
    padding = 0
    if payload.endswith("=="):
        padding = 2
    elif payload.endswith("="):
        padding = 1
    return ((len(payload) + 3) // 4) * 3 - padding


def _max_archive_bytes() -> int:
    raw_limit = os.environ.get(MAX_ARCHIVE_BYTES_ENV)
    if raw_limit is None:
        return DEFAULT_MAX_ARCHIVE_BYTES

    try:
        max_archive_bytes = int(raw_limit)
    except ValueError as exc:
        raise ValueError(
            "{0} must be a positive integer".format(MAX_ARCHIVE_BYTES_ENV)
        ) from exc

    if max_archive_bytes <= 0:
        raise ValueError("{0} must be a positive integer".format(MAX_ARCHIVE_BYTES_ENV))
    return max_archive_bytes


def _validate_archive_path(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or name.startswith("\\")
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).is_absolute()
        or ".." in path.parts
    ):
        raise ValueError("Unsafe archive path: {0}".format(name))


def _graph_file_infos(
    archive: zipfile.ZipFile,
    max_archive_bytes: int,
) -> List[zipfile.ZipInfo]:
    graph_infos: List[zipfile.ZipInfo] = []
    declared_size = 0

    for info in archive.infolist():
        name = info.filename
        _validate_archive_path(name)

        if info.is_dir() or not name.startswith("graph/"):
            continue

        if len(graph_infos) >= MAX_GRAPH_ARCHIVE_FILES:
            raise ValueError(
                "Graph archive contains too many graph files: limit is {0}".format(
                    MAX_GRAPH_ARCHIVE_FILES
                )
            )

        _ensure_archive_size(info.file_size, max_archive_bytes)
        declared_size += info.file_size
        _ensure_archive_size(declared_size, max_archive_bytes)
        graph_infos.append(info)

    if not graph_infos:
        raise ValueError("Graph archive contains no graph files")

    return graph_infos


def _extract_graph_file(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    target_root: Path,
    remaining_bytes: int,
) -> int:
    name = info.filename
    target_path = target_root / name
    _validate_target_path(target_path, target_root, name)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with archive.open(info, "r") as source, target_path.open("wb") as target:
            return _copy_with_limit(source, target, remaining_bytes)
    except ValueError:
        if target_path.exists():
            target_path.unlink()
        raise


def _copy_with_limit(source: BinaryIO, target: BinaryIO, byte_budget: int) -> int:
    copied = 0
    while True:
        read_size = min(COPY_CHUNK_SIZE, byte_budget - copied + 1)
        chunk = source.read(read_size)
        if not chunk:
            return copied
        if copied + len(chunk) > byte_budget:
            _raise_archive_too_large(copied + len(chunk), byte_budget)
        target.write(chunk)
        copied += len(chunk)


def _ensure_archive_size(size: int, max_archive_bytes: int) -> None:
    if size > max_archive_bytes:
        _raise_archive_too_large(size, max_archive_bytes)


def _raise_archive_too_large(size: int, max_archive_bytes: int) -> None:
    raise ValueError(
        "Graph archive too large: {0} bytes exceeds {1} bytes".format(
            size,
            max_archive_bytes,
        )
    )


def _validate_target_path(target_path: Path, target_root: Path, archive_name: str) -> None:
    root = target_root.resolve(strict=False)
    candidate = target_path.resolve(strict=False)
    if not _is_relative_to(candidate, root):
        raise ValueError("Unsafe archive path: {0}".format(archive_name))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
