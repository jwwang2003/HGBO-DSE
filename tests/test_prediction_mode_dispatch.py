import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def install_hls_dse_import_stubs():
    optuna = types.ModuleType("optuna")
    optuna.study = types.SimpleNamespace(Study=object)
    optuna.trial = types.SimpleNamespace(FrozenTrial=object)
    optuna.samplers = types.SimpleNamespace(
        TPESampler=object,
        NSGAIISampler=object,
        RandomSampler=object,
    )
    optuna.visualization = types.SimpleNamespace(
        plot_optimization_history=lambda *args, **kwargs: None,
        plot_parallel_coordinate=lambda *args, **kwargs: None,
        plot_param_importances=lambda *args, **kwargs: None,
        plot_contour=lambda *args, **kwargs: None,
        plot_slice=lambda *args, **kwargs: None,
    )
    optuna.create_study = lambda *args, **kwargs: None
    sys.modules.setdefault("optuna", optuna)

    pydoe = types.ModuleType("pyDOE")
    pydoe.lhs = lambda *args, **kwargs: []
    sys.modules.setdefault("pyDOE", pydoe)

    sa_sampler = types.ModuleType("bome.alg.sa_sampler")
    sa_sampler.SimulatedAnnealingSampler = object
    sys.modules.setdefault("bome.alg.sa_sampler", sa_sampler)

    hgp_pred = types.ModuleType("bome.hgp_pred")
    hgp_pred.getGNNPred = lambda *args, **kwargs: None
    sys.modules.setdefault("bome.hgp_pred", hgp_pred)

    gen_config = types.ModuleType("bome.tdm.gen_config")
    gen_config.genDirConfig = lambda *args, **kwargs: None
    sys.modules.setdefault("bome.tdm.gen_config", gen_config)

    design_space = types.ModuleType("bome.tdm.design_space")
    design_space.config_tree_space = lambda *args, **kwargs: ({}, {})
    design_space.normalizePCA = lambda *args, **kwargs: (0, 0, 0)
    design_space.normalizePLCA = lambda *args, **kwargs: (0, 0, 0, 0)
    sys.modules.setdefault("bome.tdm.design_space", design_space)

    get_ppa = types.ModuleType("bome.get_ppa")
    get_ppa.getHLS = lambda *args, **kwargs: ({}, False)
    get_ppa.getPPA = lambda *args, **kwargs: ({}, False)
    sys.modules.setdefault("bome.get_ppa", get_ppa)

    save_report = types.ModuleType("bome.save_report")
    save_report.get_adb_rpt_verilog = lambda *args, **kwargs: ([], "")
    sys.modules.setdefault("bome.save_report", save_report)

    vitis_hls = types.ModuleType("bome.vitis_hls")
    vitis_hls.VitisHLSRunner = object
    sys.modules.setdefault("bome.vitis_hls", vitis_hls)

    optuna_helpers = types.ModuleType("helpers.optuna")
    optuna_helpers.get_opt_history_graphs = lambda *args, **kwargs: {}
    optuna_helpers.get_param_importance_graphs = lambda *args, **kwargs: {}
    sys.modules.setdefault("helpers.optuna", optuna_helpers)


install_hls_dse_import_stubs()

import bome.hls_dse as hls_dse


class DummyRunner:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        DummyRunner.calls.append(("init", kwargs))

    def run(self):
        DummyRunner.calls.append(("run", self.kwargs))


class DummyLog:
    def info(self, message):
        pass


def make_basic(tmp_path, mode="hgp", inference_mode=None):
    script_path = tmp_path / "script"
    script_path.mkdir()
    hls_temp = tmp_path / "hls_temp.tcl"
    hls_temp.write_text("source dir_test.tcl\n")

    basic = SimpleNamespace(
        alg="motpe_fl",
        case="bfs",
        context="ctx",
        dataset_path=str(tmp_path / "dataset"),
        encode="float",
        hls_script_path=str(script_path),
        hls_temp=str(hls_temp),
        isolated_folder_path=None,
        log=DummyLog(),
        mode=mode,
        ori_prj_path=str(tmp_path / "project"),
        params={"LUT": 1000, "FF": 1000, "DSP": 100, "BRAM": 100, "CP": 10, "PWR": 10},
        process=1,
        static_config={},
        top="bfs",
    )
    if inference_mode is not None:
        basic.inference_mode = inference_mode
    return basic


@pytest.fixture
def objective_deps(monkeypatch):
    calls = []

    def fake_get_hls(params, rpt_list, log):
        return {
            "HLS": {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4, "URAM": 0, "CP": 5},
            "LATENCY": {"Latency": 10},
        }, False

    monkeypatch.setattr(hls_dse, "VitisHLSRunner", DummyRunner)
    monkeypatch.setattr(hls_dse, "config_tree_space", lambda *args: ({}, {}))
    monkeypatch.setattr(hls_dse, "genDirConfig", lambda *args: None)
    monkeypatch.setattr(hls_dse, "get_adb_rpt_verilog", lambda *args: (["rpt"], "/shared/prj_0"))
    monkeypatch.setattr(hls_dse, "getHLS", fake_get_hls)
    monkeypatch.setattr(hls_dse, "normalizePCA", lambda params, dictPPA: (1, 2, 3))
    monkeypatch.setattr(hls_dse, "normalizePLCA", lambda params, dictPPA: (1, 2, 3, 4))

    return calls


def test_hgp_host_mode_calls_local_gnn_after_host_c_synthesis(tmp_path, monkeypatch, objective_deps):
    calls = objective_deps

    def fake_get_gnn_pred(prj_path, hls_attr, case):
        calls.append(("host", prj_path, hls_attr, case))
        return {"LUT": 11, "FF": 12, "DSP": 13, "BRAM": 14, "CP": 15, "PWR": 16}

    def fake_remote_predict(*args):
        calls.append(("remote", args))
        raise AssertionError("remote dispatcher must not be called in host mode")

    monkeypatch.setattr(hls_dse, "getGNNPred", fake_get_gnn_pred)
    monkeypatch.setattr(hls_dse, "dispatch_remote_inference", fake_remote_predict)

    result = hls_dse.objective(SimpleNamespace(number=0), make_basic(tmp_path, inference_mode="host"))

    assert result == [1, 2, 3]
    assert DummyRunner.calls[-1][0] == "run"
    assert calls == [("host", "/shared/prj_0", [1, 2, 3, 4, 0, 5], "bfs")]
    ppa = json.loads((Path(tmp_path) / "script" / "ppa_0.json").read_text())
    assert ppa["IMPL"]["LUT"] == 11


def test_hgp_remote_mode_dispatches_prediction_after_host_c_synthesis(tmp_path, monkeypatch, objective_deps):
    calls = objective_deps

    def fake_get_gnn_pred(*args):
        calls.append(("host", args))
        raise AssertionError("local getGNNPred must not be called in remote mode")

    def fake_remote_predict(prj_path, hls_attr, case):
        calls.append(("remote", prj_path, hls_attr, case))
        return {"LUT": 21, "FF": 22, "DSP": 23, "BRAM": 24, "CP": 25, "PWR": 26}

    monkeypatch.setattr(hls_dse, "getGNNPred", fake_get_gnn_pred)
    monkeypatch.setattr(hls_dse, "dispatch_remote_inference", fake_remote_predict)

    result = hls_dse.objective(SimpleNamespace(number=0), make_basic(tmp_path, inference_mode="remote"))

    assert result == [1, 2, 3]
    assert DummyRunner.calls[-1][0] == "run"
    assert calls == [("remote", "/shared/prj_0", [1, 2, 3, 4, 0, 5], "bfs")]
    ppa = json.loads((Path(tmp_path) / "script" / "ppa_0.json").read_text())
    assert ppa["IMPL"]["LUT"] == 21


def test_impl_mode_does_not_call_prediction_paths(tmp_path, monkeypatch, objective_deps):
    calls = objective_deps

    monkeypatch.setattr(
        hls_dse,
        "getPPA",
        lambda *args: (
            {
                "LATENCY": {"Latency": 10},
                "IMPL": {"LUT": 31, "FF": 32, "DSP": 33, "BRAM": 34, "CP": 35, "PWR": 36},
            },
            False,
        ),
    )
    monkeypatch.setattr(hls_dse, "getGNNPred", lambda *args: calls.append(("host", args)))
    monkeypatch.setattr(hls_dse, "dispatch_remote_inference", lambda *args: calls.append(("remote", args)))

    result = hls_dse.objective(SimpleNamespace(number=0), make_basic(tmp_path, mode="impl", inference_mode="remote"))

    assert result == [1, 2, 3]
    assert DummyRunner.calls[-1][0] == "run"
    assert calls == []
