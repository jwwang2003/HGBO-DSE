from pathlib import Path

from bome.hls_basic import HLSBasic


def make_basic(root, isolated=None):
    hls_basic_cls = getattr(HLSBasic, "__wrapped__", HLSBasic)
    basic = hls_basic_cls.__new__(hls_basic_cls)
    basic.root = str(root)
    basic.mode = "hgp"
    basic.bench = "MachSuite"
    basic.case = "bfs"
    basic.ver = "bulk"
    basic.alg = "motpe_fl"
    basic.process = 1
    basic.isolated = isolated
    basic.isolated_folder_path = str(root / isolated) if isolated else None
    return basic


def test_isolated_mode_keeps_static_inputs_under_root(tmp_path):
    basic = make_basic(tmp_path, isolated="context1")

    assert Path(basic.get_config_path()) == tmp_path / "config" / "MachSuite" / "bfs_bulk_config.yaml"
    assert Path(basic.get_params_path()) == tmp_path / "config" / "MachSuite" / "bfs_bulk_params.yaml"
    assert Path(basic.get_ori_prj_path()) == tmp_path / "benchmark" / "MachSuite" / "bfs" / "bulk"


def test_isolated_mode_puts_generated_artifacts_under_isolated_folder(tmp_path):
    basic = make_basic(tmp_path, isolated="context1")

    assert Path(basic.get_dataset_path()) == (
        tmp_path / "context1" / "artifacts" / "MachSuite" / "motpe_fl_ds" / "bfs" / "bulk" / "p1"
    )
