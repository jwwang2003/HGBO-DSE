import os
from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import SAGPooling, global_add_pool, global_max_pool

# Flag for enabling Jumping Knowledge
jknFlag: bool = False

class HierNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        conv_type: str,
        hls_dim: int,
        drop_out: float = 0.0,
        pool_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio

        # choose convolution
        if conv_type == 'gcn':
            Conv = GCNConv
        elif conv_type == 'gat':
            Conv = GATConv
        else:  # 'sage' or fallback
            Conv = SAGEConv

        # build conv + pool stacks
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.convs.append(Conv(in_c, hidden_channels))
            self.pools.append(SAGPooling(hidden_channels, pool_ratio))

        # optional Jumping Knowledge
        if jknFlag:
            self.jkn = JumpingKnowledge('lstm', channels=hidden_channels, num_layers=2)

        # final MLP head
        # combine two global pools + hls attributes
        self.channels = [hidden_channels * 2 + hls_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels, self.channels[1:]):
            self.mlps.append(Linear(in_ch, out_ch))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        hls_attr: Tensor,
    ) -> Tensor:
        x = x.to(torch.float32)
        h_list = []

        # downsample and pool at each layer
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch, None)
            pooled = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            h_list.append(pooled)

        # aggregate representations
        if jknFlag:
            x = self.jkn(h_list)
        else:
            # sum last three layers (or all if fewer)
            x = sum(h_list[-3:])

        # concat with HLS attributes
        x = torch.cat([x, hls_attr], dim=-1)

        # final MLP
        for i, mlp in enumerate(self.mlps):
            x = mlp(x)
            if i < len(self.mlps) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_out, training=self.training)
        return x


def dsp_pred(data: Data) -> int:
    """
    Predict DSP usage from a PyG Data object using a pretrained HierNet.

    Args:
        data (Data): Graph data containing `x`, `edge_index`, and `'hls_attr'`.

    Returns:
        int: Non-negative, rounded DSP prediction.
    """
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and build model
    # model_dir = os.path.join(os.path.dirname(__file__), '..', 'hgp', 'model')
    # model_path = os.path.abspath(os.path.join(model_dir, 'dsp_mae_h64_d0_checkpoint_test.pt'))
    model_path = os.path.abspath(os.path.join(os.curdir, 'hgp', 'model', 'dsp_mae_h64_d0_checkpoint_test.pt'))

    model = HierNet(
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
    data = data.to(device)
    with torch.no_grad():
        hls_attr: Tensor = data['hls_attr']
        num_nodes = data.x.size(0)
        batch: Tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)
        out: Tensor = model(data.x, data.edge_index, batch, hls_attr)

    dsp_value: float = out.view(-1).item()
    return max(0, round(dsp_value))
