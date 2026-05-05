from pathlib import Path

from bome.hls_basic import HLSBasic


def make_hls_basic(tmp_path):
    hls_cls = getattr(HLSBasic, "__wrapped__", HLSBasic)
    hls = hls_cls.__new__(hls_cls)
    hls.root = str(tmp_path / "hgbo")
    hls.mode = "hgp"
    hls.bench = "MachSuite"
    hls.case = "bfs"
    hls.ver = "bulk"
    hls.alg = "motpe_fl"
    hls.process = 1
    hls.isolated = "run-isolated"
    hls.isolated_folder_path = str(tmp_path / "run-isolated")
    hls.config_path_override = str(tmp_path / ".compass" / "hgbo-package" / "config.yaml")
    hls.params_path_override = str(tmp_path / ".compass" / "hgbo-package" / "params.yaml")
    hls.ori_prj_path_override = str(
        tmp_path / ".compass" / "hgbo-package" / "benchmark" / "MachSuite" / "bfs" / "bulk"
    )
    return hls


def test_compass_package_paths_override_hgbo_static_paths(tmp_path):
    hls = make_hls_basic(tmp_path)

    assert Path(hls.get_config_path()) == tmp_path / ".compass" / "hgbo-package" / "config.yaml"
    assert Path(hls.get_params_path()) == tmp_path / ".compass" / "hgbo-package" / "params.yaml"
    assert Path(hls.get_ori_prj_path()) == (
        tmp_path / ".compass" / "hgbo-package" / "benchmark" / "MachSuite" / "bfs" / "bulk"
    )
