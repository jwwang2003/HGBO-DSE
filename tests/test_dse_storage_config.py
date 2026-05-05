import pytest

from tests.test_prediction_mode_dispatch import install_hls_dse_import_stubs


install_hls_dse_import_stubs()

from bome.hls_dse import build_storage_url, parse_bool


def test_parse_bool_accepts_extension_false_string():
    assert parse_bool("False") is False
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("True") is True
    assert parse_bool("1") is True


def test_parse_bool_rejects_ambiguous_values():
    with pytest.raises(Exception):
        parse_bool("maybe")


def test_non_parallel_storage_uses_sqlite_db_in_artifact_folder(tmp_path):
    storage = build_storage_url(
        study_name="bfs_motpe_fl_dse",
        parallel=parse_bool("False"),
        isolated_folder_path=str(tmp_path / "artifacts"),
        isolated=True,
    )

    assert storage == f"sqlite:///{tmp_path}/artifacts/bfs_motpe_fl_dse.db"
    assert "mysql" not in storage
