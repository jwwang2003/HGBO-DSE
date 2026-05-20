import os
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.pool import SAGPooling as PyGSAGPooling
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


def _legacy_topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        scores_max = scatter(x, batch, reduce="max")[batch] - tol
        scores_min = scores_max.clamp(max=min_score)
        return (x > scores_min).nonzero().view(-1)

    if ratio is None:
        raise ValueError("At least one of 'ratio' and 'min_score' must be specified")

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce="sum")

    if ratio >= 1:
        k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
    else:
        k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

    _, x_perm = torch.sort(x.view(-1), descending=True)
    batch = batch[x_perm]
    batch, batch_perm = torch.sort(batch, descending=False, stable=True)

    arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    ptr = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)])
    batched_arange = arange - ptr[batch]
    mask = batched_arange < k[batch]

    return x_perm[batch_perm[mask]]


def _legacy_filter_adj(
    edge_index: Tensor,
    edge_attr: OptTensor,
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, OptTensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    edge_mask = (row >= 0) & (col >= 0)
    row, col = row[edge_mask], col[edge_mask]

    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]

    return torch.stack([row, col], dim=0), edge_attr


class LegacySAGPooling(torch.nn.Module):
    """PyG 2.3-style SAGPooling for reproducing original HGBO-DSE training."""

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, Tensor, Tensor, Tensor]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = _legacy_topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = _legacy_filter_adj(
            edge_index,
            edge_attr,
            perm,
            num_nodes=score.size(0),
        )

        return x, edge_index, edge_attr, batch, perm, score[perm]


def use_legacy_sag_pooling() -> bool:
    return os.environ.get("HGBO_LEGACY_SAGPOOL", "").lower() in {"1", "true", "yes", "on"}


def get_sag_pooling_class():
    return LegacySAGPooling if use_legacy_sag_pooling() else PyGSAGPooling


SAGPooling = get_sag_pooling_class()
