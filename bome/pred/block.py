import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch_geometric.nn import SAGPooling, global_max_pool, global_mean_pool

class GraphBlock(torch.nn.Module):
    """
    Encapsulates a sequence of conv+pool layers with read-out.
    """
    def __init__(self,
                 convs: List[torch.nn.Module],
                 pools: List[SAGPooling],
                 drop_out: float):
        super(GraphBlock, self).__init__()
        self.convs    = torch.nn.ModuleList(convs)
        self.pools    = torch.nn.ModuleList(pools)
        self.drop_out = drop_out

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor
    ) -> List[torch.Tensor]:
        h_list: List[torch.Tensor] = []
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch, None)
            h = torch.cat([
                global_max_pool(x, batch),
                global_mean_pool(x, batch)
            ], dim=1)
            h_list.append(h)
        return h_list