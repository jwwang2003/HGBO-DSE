from pathlib import Path


PRED_DIR = Path(__file__).resolve().parents[1] / "bome" / "pred"
PREDICTION_MODULES = [
    "pred_bram.py",
    "pred_cp.py",
    "pred_dsp.py",
    "pred_ff.py",
    "pred_lut.py",
    "pred_pwr.py",
]


def read_module(module_name):
    return (PRED_DIR / module_name).read_text()


def test_prediction_modules_keep_pretrained_state_dict_layout():
    for module_name in PREDICTION_MODULES:
        source = read_module(module_name)

        assert "class HierNet(torch.nn.Module):" in source, module_name
        assert "from .net import HierNet" not in source, module_name
        assert "self.block" not in source, module_name
        assert "self.convs = torch.nn.ModuleList()" in source, module_name
        assert "self.pools = torch.nn.ModuleList()" in source, module_name
        assert "self.mlps = torch.nn.ModuleList()" in source, module_name
        assert "h_list[0] + h_list[1] + h_list[2]" in source, module_name
        assert "strict=False" not in source, module_name


def test_prediction_modules_run_inference_on_selected_device():
    for module_name in PREDICTION_MODULES:
        source = read_module(module_name)

        assert "torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in source, module_name
        assert "data = data.to(device)" in source, module_name
        assert "map_location=device" in source, module_name
        assert "device=device" in source, module_name


def test_architecture_refactor_modules_are_not_kept():
    assert not (PRED_DIR / "net.py").exists()
    assert not (PRED_DIR / "block.py").exists()
