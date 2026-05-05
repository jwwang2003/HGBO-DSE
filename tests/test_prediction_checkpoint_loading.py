import pytest


torch = pytest.importorskip("torch")

from bome.pred.checkpoint import load_pretrained_state_dict
from bome.pred.pred_lut import HierNet


def test_legacy_sagpooling_checkpoint_loads_with_current_torch_geometric():
    model = HierNet(
        in_channels=15,
        hidden_channels=64,
        num_layers=3,
        conv_type="sage",
        hls_dim=6,
        drop_out=0.0,
    )
    checkpoint = torch.load(
        "hgp/model/lut_h64_d0_checkpoint_test.pt",
        map_location="cpu",
    )

    assert "pools.0.select.weight" not in checkpoint["model"]

    load_pretrained_state_dict(model, checkpoint["model"])

    for index in range(3):
        weight = model.state_dict()[f"pools.{index}.select.weight"]
        assert torch.equal(weight, torch.ones_like(weight))
