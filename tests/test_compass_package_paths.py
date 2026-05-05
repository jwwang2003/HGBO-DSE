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


def test_hls_temp_script_uses_compass_source_file_override(tmp_path):
    hls = make_hls_basic(tmp_path)
    hls.hls_temp = str(tmp_path / "hls_temp.tcl")
    hls.hls_script_path = str(tmp_path / "script")
    hls.ori_prj_path = str(tmp_path / "benchmark" / "custom" / "bfs" / "bulk")
    hls.top = "edge_detect"
    hls.device = "xc7vx485tffg1761-2"
    hls.clk = "10"
    hls.source_file = "edge_detect.c"

    hls.gen_hls_temp_script()

    script = Path(hls.hls_temp).read_text()
    assert "add_files edge_detect.c\n" in script
    assert "add_files bfs.c\n" not in script


def test_hls_temp_script_defaults_source_file_to_case_name(tmp_path):
    hls = make_hls_basic(tmp_path)
    hls.hls_temp = str(tmp_path / "hls_temp.tcl")
    hls.hls_script_path = str(tmp_path / "script")
    hls.ori_prj_path = str(tmp_path / "benchmark" / "custom" / "bfs" / "bulk")
    hls.top = "bfs"
    hls.device = "xc7vx485tffg1761-2"
    hls.clk = "10"

    hls.gen_hls_temp_script()

    script = Path(hls.hls_temp).read_text()
    assert "add_files bfs.c\n" in script
