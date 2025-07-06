import os
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import Data

from .net import HierNet


def bram_pred(data: Data) -> int:
    """
    Predict BRAM usage from a PyG Data object using a pretrained HierNet.

    Args:
        data (Data): Graph data containing `x`, `edge_index`, and `'hls_attr'` fields.

    Returns:
        int: Non-negative, rounded BRAM prediction.
    """
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # locate model checkpoint
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'hgp', 'model')
    model_path = os.path.abspath(os.path.join(model_dir, 'bram_mae_h64_d0_checkpoint_test.pt'))

    # build & load model
    model: HierNet = HierNet(
        in_channels=15,
        hidden_channels=64,
        num_layers=3,
        conv_type='sage',
        hls_dim=6,
        drop_out=0.0
    ).to(device)

    checkpoint: Dict[str, Tensor] = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # inference
    with torch.no_grad():
        hls_attr: Tensor = data['hls_attr']
        num_nodes = data.x.size(0)
        batch: Tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)
        out: Tensor = model(data.x, data.edge_index, batch, hls_attr)

    # post-process
    bram_value: float = out.view(-1).item()
    return max(0, round(bram_value))