from __future__ import annotations

import base64
import io
import zipfile

import pytest

import backend.mcp_payload as mcp_payload
from backend.mcp_payload import (
    create_graph_archive_base64,
    extract_graph_archive_base64,
    validate_ppa_result,
)


def test_graph_archive_round_trips_files(tmp_path):
    prj_path = tmp_path / "prj_0"
    graph = prj_path / "graph"
    graph.mkdir(parents=True)
    (graph / "kernel.adb").write_text("adb", encoding="utf-8")
    (graph / "nested").mkdir()
    (graph / "nested" / "kernel.adb.xml").write_text("<xml />", encoding="utf-8")

    archive = create_graph_archive_base64(prj_path)
    restored = tmp_path / "restored"
    count = extract_graph_archive_base64(archive, restored)

    assert count == 2
    assert (restored / "graph" / "kernel.adb").read_text(encoding="utf-8") == "adb"
    assert (
        restored / "graph" / "nested" / "kernel.adb.xml"
    ).read_text(encoding="utf-8") == "<xml />"


def test_graph_archive_skips_symlinks(tmp_path):
    prj_path = tmp_path / "prj_0"
    graph = prj_path / "graph"
    graph.mkdir(parents=True)
    (graph / "kernel.adb").write_text("adb", encoding="utf-8")
    (graph / "linked.adb").symlink_to(graph / "kernel.adb")

    archive = create_graph_archive_base64(prj_path)
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(archive))).namelist()

    assert names == ["graph/kernel.adb"]


def test_graph_archive_requires_graph_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="graph"):
        create_graph_archive_base64(tmp_path / "missing")


def test_extract_rejects_path_traversal(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_extract_rejects_absolute_path(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("/escape.txt", "bad")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_extract_rejects_archive_without_graph_files(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.txt", "ignored")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="Graph archive"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_archive_size_limit_applies_to_encode_and_decode(tmp_path, monkeypatch):
    prj_path = tmp_path / "prj_0"
    graph = prj_path / "graph"
    graph.mkdir(parents=True)
    (graph / "kernel.adb").write_text("abcdef", encoding="utf-8")
    monkeypatch.setenv("HGBO_MCP_MAX_ARCHIVE_BYTES", "10")
    with pytest.raises(ValueError, match="too large"):
        create_graph_archive_base64(prj_path)
    raw = b"x" * 11
    encoded = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError, match="too large"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_extract_rejects_inflated_graph_file_over_limit(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("graph/kernel.adb", b"a" * 4096)
    raw = buffer.getvalue()
    monkeypatch.setenv("HGBO_MCP_MAX_ARCHIVE_BYTES", "512")
    assert len(raw) < 512

    encoded = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError, match="too large"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_extract_rejects_cumulative_graph_size_over_limit(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("graph/a.adb", b"a" * 200)
        archive.writestr("graph/b.adb", b"b" * 200)
    raw = buffer.getvalue()
    monkeypatch.setenv("HGBO_MCP_MAX_ARCHIVE_BYTES", "300")
    assert len(raw) < 300

    encoded = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError, match="too large"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_extract_rejects_too_many_graph_files(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("graph/a.adb", "a")
        archive.writestr("graph/b.adb", "b")
        archive.writestr("graph/c.adb", "c")
    monkeypatch.setattr(mcp_payload, "MAX_GRAPH_ARCHIVE_FILES", 2, raising=False)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="too many"):
        mcp_payload.extract_graph_archive_base64(encoded, tmp_path / "target")


def test_decode_rejects_oversized_base64_before_decoding(tmp_path, monkeypatch):
    encoded = base64.b64encode(b"x" * 11).decode("ascii")
    monkeypatch.setenv("HGBO_MCP_MAX_ARCHIVE_BYTES", "10")

    def fail_decode(*args, **kwargs):
        raise AssertionError("base64 decode should not be called")

    monkeypatch.setattr(mcp_payload.base64, "b64decode", fail_decode)

    with pytest.raises(ValueError, match="too large"):
        extract_graph_archive_base64(encoded, tmp_path / "target")


def test_archive_size_limit_rejects_invalid_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HGBO_MCP_MAX_ARCHIVE_BYTES", "invalid")

    with pytest.raises(ValueError, match="HGBO_MCP_MAX_ARCHIVE_BYTES"):
        extract_graph_archive_base64("", tmp_path / "target")


def test_validate_ppa_result_requires_expected_keys():
    result = validate_ppa_result(
        {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4, "CP": 5, "PWR": 6, "extra": 7}
    )
    assert result == {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4, "CP": 5, "PWR": 6}
    with pytest.raises(ValueError, match="missing"):
        validate_ppa_result({"LUT": 1})
