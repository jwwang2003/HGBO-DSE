from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GATConv, GCNConv, GINEConv, SAGEConv, TransformerConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool

from hgp.arch_aware_arch import (
    ARCH_AWARE_LAYOUT_COLS,
    ARCH_AWARE_METADATA_DIM,
    ARCH_AWARE_TILE_SLOTS,
    arch_aware_arch_to_device,
    ensure_arch_aware_arch_cache,
    load_arch_aware_arch_cache,
)
from hgp.board_fabric import board_fabric_to_device, ensure_board_fabric_cache, load_board_fabric_cache
from hgp.board_utils import ARCH_ATTR_FIELDS, DEFAULT_BOARD_DEVICE, normalize_device_name
from hgp.dataset_utils import (
    generate_dataset,
    generate_multi_board_dataset,
    leave_one_board_out_split,
    mae_loss,
    mape_loss,
    split_dataset,
)
from hgp.feature_normalization import (
    apply_feature_stats,
    compute_feature_stats,
    stats_from_jsonable,
    stats_to_jsonable,
)
from hgp.multi_board import (
    board_devices_for_batch,
    dispatch_arch_aware_payload,
    dispatch_fabric_payload,
    prepare_arch_caches,
    prepare_board_fabric_caches,
)
from hgp.pyg_compat import SAGPooling


TARGETS = ["lut", "ff", "dsp", "bram", "uram", "srl", "cp", "power", "dynamic_power"]
ARCH_AWARE_MODE = "arch-aware"
ATAPP_CONV_TYPE = "atapp"
jknFlag = 0


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target_index: int
    dataset_subdir: str
    hls_dim: int
    pool_mode: str
    metric_name: str
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    label_scale: float = 1.0
    checkpoint_stem: str | None = None


TARGET_SPECS = {
    "lut": TargetSpec("lut", 0, "std_arch", 6, "add", "mape", mape_loss),
    "ff": TargetSpec("ff", 1, "std_arch", 6, "add", "mape", mape_loss),
    "dsp": TargetSpec("dsp", 2, "std_arch", 6, "add", "mae", mae_loss, checkpoint_stem="dsp_mae"),
    "bram": TargetSpec("bram", 3, "std_arch", 6, "add", "mae", mae_loss, checkpoint_stem="bram_mae"),
    "cp": TargetSpec("cp", 6, "rdc_arch", 1, "mean", "mape", mape_loss, checkpoint_stem="cp_mean"),
    "power": TargetSpec("power", 7, "std_arch", 6, "mean", "mape", mape_loss, label_scale=100.0, checkpoint_stem="power_mean"),
    "dynamic_power": TargetSpec(
        "dynamic_power",
        8,
        "std_arch",
        6,
        "mean",
        "mape",
        mape_loss,
        label_scale=100.0,
        checkpoint_stem="dynamic_power_mean",
    ),
}


HGBO_ROOT = Path(__file__).resolve().parents[1]


def _default_dataset_dir(dataset_subdir):
    return HGBO_ROOT / "dataset" / dataset_subdir


def _make_mlp(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        Linear(hidden_dim, output_dim),
    )


def _make_design_conv(conv_type, in_channels, out_channels, edge_dim=None):
    if conv_type == "gcn":
        return GCNConv(in_channels, out_channels)
    if conv_type == "gat":
        return GATConv(in_channels, out_channels)
    if conv_type == "sage":
        return SAGEConv(in_channels, out_channels)
    if conv_type == "gine":
        if edge_dim is None:
            raise ValueError("conv_type='gine' requires design_edge_dim")
        return GINEConv(_make_mlp(in_channels, out_channels, out_channels), edge_dim=edge_dim)
    if conv_type == ATAPP_CONV_TYPE:
        if edge_dim is None:
            raise ValueError("conv_type='atapp' requires design_edge_dim")
        return TransformerConv(
            in_channels,
            out_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            beta=True,
        )
    raise ValueError("Unknown conv_type {!r}".format(conv_type))


def _resolve_conv_type(target: str, conv_type: str) -> str:
    if conv_type != "auto":
        return conv_type
    if target == "dynamic_power":
        return ATAPP_CONV_TYPE
    return "gine"


def _apply_design_conv(conv, conv_type, x, edge_index, edge_attr=None):
    if conv_type not in {"gine", ATAPP_CONV_TYPE}:
        return conv(x, edge_index)
    if edge_attr is None:
        raise ValueError("conv_type={!r} requires edge_attr in forward".format(conv_type))
    return conv(x, edge_index, edge_attr.to(torch.float32))


class BoardFabricEncoder(torch.nn.Module):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        graph_input_dim,
        hidden_dim=32,
        num_layers=2,
        output_dim=32,
    ):
        super().__init__()
        self.node_proj = Linear(node_input_dim, hidden_dim)
        self.edge_proj = Linear(edge_input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList(
            GINEConv(_make_mlp(hidden_dim, hidden_dim, hidden_dim), edge_dim=hidden_dim)
            for _ in range(num_layers)
        )
        self.norms = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers))
        self.graph_mlp = _make_mlp(graph_input_dim, hidden_dim, output_dim)
        self.fuse = _make_mlp(output_dim * 2, hidden_dim, output_dim)

    def forward(self, arch_graph):
        if torch.is_tensor(arch_graph):
            return arch_graph

        arch_x = arch_graph["arch_x"].to(torch.float32)
        arch_edge_index = arch_graph["arch_edge_index"]
        arch_edge_attr = arch_graph["arch_edge_attr"].to(torch.float32)
        arch_graph_attr = arch_graph["arch_graph_attr"].to(torch.float32)

        if arch_graph_attr.dim() == 1:
            arch_graph_attr = arch_graph_attr.unsqueeze(0)

        arch_x = self.node_proj(arch_x)
        arch_edge_attr = self.edge_proj(arch_edge_attr)
        for layer, norm in zip(self.layers, self.norms):
            residual = arch_x
            arch_x = layer(arch_x, arch_edge_index, arch_edge_attr)
            arch_x = norm(arch_x)
            arch_x = F.relu(arch_x + residual)

        batch = torch.zeros(arch_x.size(0), dtype=torch.long, device=arch_x.device)
        pooled = global_mean_pool(arch_x, batch)
        graph_embedding = self.graph_mlp(arch_graph_attr)
        return self.fuse(torch.cat([pooled, graph_embedding], dim=-1))


class ArchAwareArchitectureEncoder(torch.nn.Module):
    def __init__(
        self,
        layout_cols=ARCH_AWARE_LAYOUT_COLS,
        tile_slots=ARCH_AWARE_TILE_SLOTS,
        metadata_dim=ARCH_AWARE_METADATA_DIM,
        hidden_dim=32,
        output_dim=32,
    ):
        super().__init__()
        self.layout_cols = layout_cols
        self.tile_slots = tile_slots
        self.metadata_dim = metadata_dim
        self.input_dim = layout_cols * tile_slots + metadata_dim
        self.mlp = torch.nn.Sequential(
            Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, output_dim),
        )

    def _layout_batch(self, layout, metadata_batch_size):
        layout = layout.to(torch.float32)
        if layout.dim() == 3:
            if metadata_batch_size > 1 and layout.size(0) == metadata_batch_size * self.layout_cols:
                return layout.view(metadata_batch_size, self.layout_cols, 1, self.tile_slots)
            return layout.unsqueeze(0)
        if layout.dim() == 4:
            return layout
        raise ValueError("Expected arch-aware layout with 3 or 4 dimensions, got {}".format(tuple(layout.shape)))

    def forward(self, arch_input):
        if torch.is_tensor(arch_input):
            return arch_input

        metadata = arch_input["arch_aware_arch_metadata"].to(torch.float32)
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        layout = self._layout_batch(arch_input["arch_aware_arch_layout"], metadata.size(0))
        if metadata.size(0) == 1 and layout.size(0) > 1:
            metadata = metadata.expand(layout.size(0), -1)
        elif layout.size(0) == 1 and metadata.size(0) > 1:
            layout = layout.expand(metadata.size(0), -1, -1, -1)
        elif layout.size(0) != metadata.size(0):
            raise ValueError(
                "arch-aware layout batch {} does not match metadata batch {}".format(
                    layout.size(0),
                    metadata.size(0),
                )
            )

        flat_layout = layout.reshape(layout.size(0), -1)
        return self.mlp(torch.cat([flat_layout, metadata], dim=-1))


class AtappDesignEncoder(torch.nn.Module):
    """ATAPP-style edge-aware design encoder.

    ATAPP uses UniMP-style edge-aware attention and sums graph embeddings across
    layers. PyG's TransformerConv with edge_dim is the closest available local
    primitive in this codebase.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        edge_dim,
        drop_out=0.0,
    ):
        super().__init__()
        if edge_dim is None:
            raise ValueError("ATAPP design encoder requires edge attributes")
        self.drop_out = drop_out
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for index in range(num_layers):
            layer_in = in_channels if index == 0 else hidden_channels
            self.layers.append(
                TransformerConv(
                    layer_in,
                    hidden_channels,
                    heads=1,
                    concat=False,
                    edge_dim=edge_dim,
                    beta=True,
                )
            )
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.output_dim = hidden_channels

    def forward(self, x, edge_index, batch, edge_attr):
        if edge_attr is None:
            raise ValueError("ATAPP design encoder requires edge_attr in forward")
        h = x
        pooled_layers = []
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, edge_attr.to(torch.float32))
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.drop_out, training=self.training)
            pooled_layers.append(global_add_pool(h, batch))
        return sum(pooled_layers)


class ArchAwareHierNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        conv_type,
        hls_dim,
        arch_dim=len(ARCH_ATTR_FIELDS),
        arch_node_dim=24,
        arch_edge_dim=4,
        arch_graph_dim=32,
        arch_hidden_dim=16,
        fabric_hidden_dim=32,
        design_edge_dim=None,
        arch_mode="fabric",
        arch_aware_hidden_dim=32,
        arch_aware_output_dim=32,
        drop_out=0.0,
        pool_ratio=0.5,
        pool_mode="add",
    ):
        super(ArchAwareHierNet, self).__init__()

        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.pool_mode = pool_mode
        self.conv_type = conv_type
        self.arch_mode = arch_mode

        self.atapp_encoder = None
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        if conv_type == ATAPP_CONV_TYPE:
            self.atapp_encoder = AtappDesignEncoder(
                in_channels,
                hidden_channels,
                num_layers,
                design_edge_dim,
                drop_out=drop_out,
            )
            design_embedding_dim = hidden_channels
        else:
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(_make_design_conv(conv_type, in_channels, hidden_channels, design_edge_dim))
                else:
                    self.convs.append(_make_design_conv(conv_type, hidden_channels, hidden_channels, design_edge_dim))
                self.pools.append(SAGPooling(hidden_channels, self.pool_ratio))
            design_embedding_dim = hidden_channels * 2
        if jknFlag:
            self.jkn = JumpingKnowledge("lstm", channels=hidden_channels, num_layers=2)

        if arch_mode == ARCH_AWARE_MODE:
            self.arch_mlp = None
        else:
            self.arch_mlp = torch.nn.Sequential(
                Linear(arch_dim, arch_hidden_dim),
                torch.nn.ReLU(),
                Linear(arch_hidden_dim, arch_hidden_dim),
            )
        self.arch_graph_dim = arch_graph_dim
        self.board_encoder = BoardFabricEncoder(
            node_input_dim=arch_node_dim,
            edge_input_dim=arch_edge_dim,
            graph_input_dim=arch_graph_dim,
            hidden_dim=fabric_hidden_dim,
            num_layers=2,
            output_dim=fabric_hidden_dim,
        )
        self.arch_aware_encoder = ArchAwareArchitectureEncoder(
            metadata_dim=ARCH_AWARE_METADATA_DIM,
            hidden_dim=arch_aware_hidden_dim,
            output_dim=arch_aware_output_dim,
        )
        if arch_mode == ARCH_AWARE_MODE:
            self.channels = [design_embedding_dim + hls_dim + arch_aware_output_dim, 64, 64, 1]
        else:
            self.channels = [design_embedding_dim + hls_dim + arch_hidden_dim + fabric_hidden_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()

        for i in range(len(self.channels) - 1):
            fc = Linear(self.channels[i], self.channels[i + 1])
            self.mlps.append(fc)

    def _pool(self, x, batch):
        if self.pool_mode == "mean":
            return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

    def _board_embedding(self, arch_graph):
        if self.arch_mode == ARCH_AWARE_MODE:
            return self.arch_aware_encoder(arch_graph)
        return self.board_encoder(arch_graph)

    def forward(self, x, edge_index, batch, hls_attr, arch_graph, arch_attr=None, edge_attr=None):
        x = x.to(torch.float32)
        hls_attr = hls_attr.to(torch.float32)
        if arch_attr is None:
            arch_attr = arch_graph
            if self.arch_mode == ARCH_AWARE_MODE:
                arch_graph = {
                    "arch_aware_arch_layout": torch.zeros(
                        (ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS),
                        dtype=torch.float32,
                        device=x.device,
                    ),
                    "arch_aware_arch_metadata": torch.zeros(
                        (1, ARCH_AWARE_METADATA_DIM),
                        dtype=torch.float32,
                        device=x.device,
                    ),
                }
            else:
                arch_graph = torch.zeros((1, self.arch_graph_dim), dtype=torch.float32, device=x.device)
        arch_attr = arch_attr.to(torch.float32)
        if self.atapp_encoder is not None:
            x = self.atapp_encoder(x, edge_index, batch, edge_attr)
        else:
            h_list = []
            for step in range(len(self.convs)):
                x = _apply_design_conv(self.convs[step], self.conv_type, x, edge_index, edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_out, training=self.training)
                x, edge_index, edge_attr, batch, _, _ = self.pools[step](x, edge_index, edge_attr, batch, None)
                h = self._pool(x, batch)
                h_list.append(h)

            if jknFlag:
                x = self.jkn(h_list)
            x = sum(h_list)
        board_embedding = self._board_embedding(arch_graph)
        if board_embedding.size(0) != x.size(0):
            board_embedding = board_embedding.expand(x.size(0), -1)
        if self.arch_mode == ARCH_AWARE_MODE:
            x = torch.cat([x, hls_attr, board_embedding], dim=-1)
        else:
            arch_embedding = self.arch_mlp(arch_attr)
            x = torch.cat([x, hls_attr, arch_embedding, board_embedding], dim=-1)

        for f in range(len(self.mlps)):
            if f < len(self.mlps) - 1:
                x = F.relu(self.mlps[f](x))
                x = F.dropout(x, p=self.drop_out, training=self.training)
            else:
                x = self.mlps[f](x)

        return x


def _target_values(data, spec):
    return data["y"].t()[spec.target_index] * spec.label_scale


def _runtime_target_scale(args):
    return float(getattr(args, "_target_scale", 1.0) or 1.0)


def _default_runtime_args():
    return argparse.Namespace(
        _target_scale=1.0,
        target_transform="none",
        hls_residual_index=None,
        normalize_features=False,
        warmup_batches=0,
        lr=0.001,
    )


def _transform_target(true_y, args):
    scale = _runtime_target_scale(args)
    if scale != 1.0:
        true_y = true_y / scale
    if args.target_transform == "none":
        return true_y
    if args.target_transform == "log1p":
        return torch.log1p(true_y.clamp_min(0))
    raise ValueError("Unknown target transform: {}".format(args.target_transform))


def _apply_hls_residual(out, hls_attr, args, spec):
    if args.hls_residual_index is None:
        return out
    if args.target_transform != "none":
        raise ValueError("--hls-residual-index is only supported with --target-transform none")
    if getattr(args, "normalize_features", False):
        raise ValueError("--hls-residual-index is incompatible with --normalize-features (residual reads raw hls_attr)")
    if hls_attr is None:
        raise ValueError("--hls-residual-index requires hls_attr")
    if args.hls_residual_index < 0 or args.hls_residual_index >= hls_attr.size(-1):
        raise ValueError(
            "--hls-residual-index {} out of range for hls_attr width {}".format(
                args.hls_residual_index, hls_attr.size(-1)
            )
        )
    base = hls_attr[:, args.hls_residual_index].to(device=out.device, dtype=out.dtype)
    if spec.label_scale != 1.0:
        base = base * spec.label_scale
    return out + base.view_as(out)


def _metric_predictions(out, hls_attr, args, spec):
    out = _apply_hls_residual(out, hls_attr, args, spec)
    if args.target_transform == "log1p":
        out = torch.expm1(out.clamp(max=14.0)).clamp_min(0)
    scale = _runtime_target_scale(args)
    if scale != 1.0:
        out = out * scale
    return out


def _initialize_hls_residual_head(model, args, train_ds=None, spec=None):
    if args.hls_residual_index is None and args.target_transform == "none":
        return
    final_layer = model.mlps[-1]
    torch.nn.init.zeros_(final_layer.weight)
    if final_layer.bias is None:
        return
    if args.target_transform == "log1p" and train_ds is not None and spec is not None:
        ys = torch.stack(
            [_target_values(s, spec).view(-1).float().mean() for s in train_ds]
        )
        bias_value = torch.log1p(ys.mean().clamp_min(0)).item()
        final_layer.bias.data.fill_(bias_value)
        return
    torch.nn.init.zeros_(final_layer.bias)


def _ensure_finite_tensor(value, name, phase, epoch=None, batch_idx=None):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if torch.isfinite(value).all():
        return

    bad_count = int((~torch.isfinite(value)).sum().item())
    parts = ["non-finite {}".format(name), "during {}".format(phase)]
    if epoch is not None:
        parts.append("epoch {}".format(epoch))
    if batch_idx is not None:
        parts.append("batch {}".format(batch_idx))
    parts.append("({} of {} values)".format(bad_count, value.numel()))
    raise RuntimeError(" ".join(parts))


def _architecture_input_from_batch(model, data, fallback):
    """Pick the correct arch payload for ``data``.

    Single-board path (legacy): if the dataset shards already contain
    arch_aware_arch_layout / arch_aware_arch_metadata tensors (attached during
    augmentation), forward those. Otherwise return ``fallback`` (a single-device
    payload preloaded by :func:`_prepare_architecture_cache`).

    Multi-board path: a caller that has populated ``model._arch_caches`` /
    ``model._fabric_caches`` (via :func:`_attach_multi_board_caches`) takes
    precedence over both branches above. The dispatch reads per-sample
    ``board_device`` from the batch and stitches the right payload back in.
    """
    arch_mode = getattr(model, "arch_mode", None)
    arch_caches = getattr(model, "_arch_caches", None)
    fabric_caches = getattr(model, "_fabric_caches", None)
    if arch_mode == ARCH_AWARE_MODE and arch_caches:
        per_sample = board_devices_for_batch(data)
        if per_sample is None:
            return fallback
        return dispatch_arch_aware_payload(arch_caches, per_sample)
    if arch_mode != ARCH_AWARE_MODE and fabric_caches:
        per_sample = board_devices_for_batch(data)
        if per_sample is None:
            return fallback
        return dispatch_fabric_payload(fabric_caches, per_sample)

    if (
        arch_mode == ARCH_AWARE_MODE
        and getattr(data, "arch_aware_arch_layout", None) is not None
        and getattr(data, "arch_aware_arch_metadata", None) is not None
    ):
        return {
            "arch_aware_arch_layout": data["arch_aware_arch_layout"],
            "arch_aware_arch_metadata": data["arch_aware_arch_metadata"],
        }
    return fallback


def _attach_multi_board_caches(
    model,
    *,
    arch_caches: dict[str, dict[str, object]] | None = None,
    fabric_caches: dict[str, dict[str, object]] | None = None,
) -> None:
    """Stash per-device payload dicts on the model so the forward path can dispatch."""
    model._arch_caches = arch_caches or {}
    model._fabric_caches = fabric_caches or {}


def _apply_warmup(optimizer, args, epoch, batch_idx, batches_per_epoch):
    if args.warmup_batches <= 0:
        return
    global_batch = epoch * batches_per_epoch + batch_idx
    if global_batch >= args.warmup_batches:
        return
    scale = (global_batch + 1) / args.warmup_batches
    for group in optimizer.param_groups:
        group["lr"] = args.lr * scale


def train_epoch(model, train_loader, optimizer, device, spec, board_embedding, args=None, epoch=0, grad_clip=None):
    if args is None:
        args = _default_runtime_args()
    model.train()
    total_loss = 0
    total_metric = 0
    try:
        batches_per_epoch = len(train_loader)
    except TypeError:
        batches_per_epoch = 1
    for batch_idx, data in enumerate(train_loader):
        _apply_warmup(optimizer, args, epoch, batch_idx, batches_per_epoch)
        data = data.to(device)
        optimizer.zero_grad()
        arch_input = _architecture_input_from_batch(model, data, board_embedding)
        edge_attr = getattr(data, "edge_attr", None)
        model_kwargs = {"edge_attr": edge_attr} if edge_attr is not None else {}
        out = model(
            data.x,
            data.edge_index,
            data.batch,
            data["hls_attr"],
            arch_input,
            data["arch_attr"],
            **model_kwargs,
        )
        out = out.view(-1)
        true_y = _target_values(data, spec)
        _ensure_finite_tensor(out, "model output", "training", epoch, batch_idx)
        _ensure_finite_tensor(true_y, "target", "training", epoch, batch_idx)
        hls_attr = data["hls_attr"]
        out_residual = _apply_hls_residual(out, hls_attr, args, spec)
        transformed_y = _transform_target(true_y, args)
        loss = F.huber_loss(out_residual, transformed_y).float()
        pred_for_metric = _metric_predictions(out, hls_attr, args, spec)
        metric = spec.metric_fn(pred_for_metric, true_y).float()
        _ensure_finite_tensor(loss, "loss", "training", epoch, batch_idx)
        _ensure_finite_tensor(metric, "{} metric".format(spec.metric_name), "training", epoch, batch_idx)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _ensure_finite_tensor(grad_norm, "gradient norm", "training", epoch, batch_idx)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_metric += metric.item() * data.num_graphs
    ds = train_loader.dataset
    return total_loss / len(ds), total_metric / len(ds)


def evaluate(model, loader, device, spec, board_embedding, args=None, epoch=0, print_predictions=False):
    if args is None:
        args = _default_runtime_args()
    model.eval()
    with torch.no_grad():
        loss = 0
        metric = 0
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            arch_input = _architecture_input_from_batch(model, data, board_embedding)
            edge_attr = getattr(data, "edge_attr", None)
            model_kwargs = {"edge_attr": edge_attr} if edge_attr is not None else {}
            out = model(
                data.x,
                data.edge_index,
                data.batch,
                data["hls_attr"],
                arch_input,
                data["arch_attr"],
                **model_kwargs,
            )
            out = out.view(-1)
            true_y = _target_values(data, spec)
            _ensure_finite_tensor(out, "model output", "evaluation", epoch, batch_idx)
            _ensure_finite_tensor(true_y, "target", "evaluation", epoch, batch_idx)
            hls_attr = data["hls_attr"]
            out_residual = _apply_hls_residual(out, hls_attr, args, spec)
            transformed_y = _transform_target(true_y, args)
            batch_loss = F.huber_loss(out_residual, transformed_y).float()
            pred_for_metric = _metric_predictions(out, hls_attr, args, spec)
            batch_metric = spec.metric_fn(pred_for_metric, true_y).float()
            _ensure_finite_tensor(batch_loss, "loss", "evaluation", epoch, batch_idx)
            _ensure_finite_tensor(batch_metric, "{} metric".format(spec.metric_name), "evaluation", epoch, batch_idx)
            loss += batch_loss.item() * data.num_graphs
            metric += batch_metric.item() * data.num_graphs
            if print_predictions and epoch % 10 == 0:
                print("pred.y:", pred_for_metric / spec.label_scale)
                print("data.y:", true_y / spec.label_scale)
        ds = loader.dataset
        return loss / len(ds), metric / len(ds)


def _checkpoint_name(spec, split):
    stem = spec.checkpoint_stem or spec.name
    return "{}_arch_h64_d0_checkpoint_{}.pt".format(stem, split)


def _select_checkpoint_metric_source(args, has_validation: bool) -> str:
    source = args.checkpoint_metric_source
    if source == "auto":
        return "val" if has_validation else "test"
    if source == "val" and not has_validation:
        raise ValueError("--checkpoint-metric-source=val requires a non-empty validation split")
    return source


def _checkpoint_payload(
    *,
    model,
    optimizer,
    epoch,
    metric_value,
    metric_key,
    spec,
    all_devices,
    args,
    cache_path,
    feature_stats,
    target_scale,
    extra_metrics=None,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": metric_value,
        "metric_name": spec.metric_name,
        "{}_{}".format(metric_key, spec.metric_name): metric_value,
        "board_devices": all_devices,
        "test_board": args.test_board,
        "arch_mode": args.arch_mode,
        "fabric_mode": args.fabric_mode,
        "arch_attr_fields": ARCH_ATTR_FIELDS,
        "architecture_cache": str(cache_path),
        "feature_stats": stats_to_jsonable(feature_stats),
        "target_scale": target_scale,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    return payload


def _prepare_board_training_input(model, board_fabric, fabric_mode):
    if fabric_mode == "trainable":
        return board_fabric
    with torch.no_grad():
        return model._board_embedding(board_fabric).detach()


def _prepare_architecture_cache(args, device, devices: list[str] | None = None):
    """Load the legacy single-device arch payload used as ``fallback`` in dispatch.

    Returns ``(cache_path, payload)`` for the first device in ``devices`` (or
    :data:`DEFAULT_BOARD_DEVICE` when ``devices`` is ``None``). When training
    on a single board this single payload is enough; when training across
    multiple boards the dispatch helper supplied by
    :func:`_attach_multi_board_caches` overrides it per-sample.
    """
    chosen_device = (devices or [DEFAULT_BOARD_DEVICE])[0]
    if args.arch_mode == ARCH_AWARE_MODE:
        cache_path = ensure_arch_aware_arch_cache(chosen_device, cache_dir=args.arch_cache_dir)
        arch_payload = arch_aware_arch_to_device(load_arch_aware_arch_cache(cache_path), device)
        return cache_path, arch_payload

    cache_path = ensure_board_fabric_cache(chosen_device, cache_dir=args.arch_cache_dir)
    arch_payload = board_fabric_to_device(load_board_fabric_cache(cache_path), device)
    return cache_path, arch_payload


def _resolve_device(device_option):
    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_cpu_threads(cpu_threads):
    if cpu_threads is None:
        return torch.get_num_threads()
    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    torch.set_num_threads(cpu_threads)
    return torch.get_num_threads()


def _loader_kwargs(args):
    return {
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }


def _test_loader_options(args):
    if args.deterministic_eval:
        return {"shuffle": False, "drop_last": False}
    return {"shuffle": True, "drop_last": True}


def _checkpoint_state(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _jsonable_settings(args):
    settings = vars(args).copy()
    return {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}


def _write_summary(args, summary):
    if not args.summary_path:
        return
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(args):
    active_threads = _set_cpu_threads(args.cpu_threads)
    spec = TARGET_SPECS[args.target]
    args.conv_type = _resolve_conv_type(args.target, args.conv_type)

    # Resolve devices.
    all_devices = _resolve_all_devices(args)
    multi_board = len(all_devices) > 1 or args.split_mode == "board-out"

    # Load data.
    if multi_board and args.dataset_root:
        dataset_list = generate_multi_board_dataset(
            args.dataset_root,
            all_devices,
            spec.dataset_subdir,
            print_info=True,
        )
    else:
        dataset_dir = os.path.abspath(args.dataset_dir or _default_dataset_dir(spec.dataset_subdir))
        dataset = os.listdir(dataset_dir)
        dataset_list = generate_dataset(dataset_dir, dataset, print_info=False)

    # Split.
    if args.split_mode == "board-out" and args.test_board:
        train_boards = _parse_device_list(args.train_boards) if args.train_boards else None
        train_ds, val_ds, test_ds = leave_one_board_out_split(
            dataset_list,
            train_boards=train_boards,
            test_board=normalize_device_name(args.test_board),
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(
            "Board-out split: train={}, val={}, test={} (held-out {})".format(
                len(train_ds), len(val_ds), len(test_ds), args.test_board
            )
        )
    else:
        train_ds, test_ds = split_dataset(dataset_list, shuffle=True, seed=args.seed)
        val_ds = None
        print("train_ds size = {}, test_ds size = {}".format(len(train_ds), len(test_ds)))

    feature_stats = None
    init_payload_stats = None
    init_payload_target_scale = None
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(init_payload, dict):
            init_payload_stats = stats_from_jsonable(init_payload.get("feature_stats"))
            init_payload_target_scale = init_payload.get("target_scale")
    if args.normalize_features:
        if init_payload_stats is not None:
            feature_stats = init_payload_stats
            print("Reusing feature stats from --init-checkpoint")
        else:
            feature_stats = compute_feature_stats(list(train_ds))
        apply_feature_stats(train_ds, feature_stats)
        if val_ds is not None and len(val_ds) > 0:
            apply_feature_stats(val_ds, feature_stats)
        apply_feature_stats(test_ds, feature_stats)
        print(
            "Feature normalization: x[{}] hls_attr[{}]".format(
                tuple(feature_stats["x_mean"].shape),
                tuple(feature_stats["hls_attr_mean"].shape),
            )
        )

    target_scale = 1.0
    if args.normalize_target:
        if init_payload_target_scale is not None:
            target_scale = float(init_payload_target_scale)
            print("Reusing target scale from --init-checkpoint: {:.4f}".format(target_scale))
        else:
            train_targets = torch.stack(
                [_target_values(s, spec).view(-1).float().mean() for s in train_ds]
            )
            target_scale = max(float(train_targets.mean().item()), 1.0)
            print("Target scale (mean train y): {:.4f}".format(target_scale))
    args._target_scale = target_scale

    loader_kwargs = _loader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    checkpoint_metric_source = _select_checkpoint_metric_source(args, val_loader is not None)
    checkpoint_split = "val" if checkpoint_metric_source == "val" else "test"

    data_ini = None
    for step, data in enumerate(train_loader):
        if step == 0:
            data_ini = data
            break
    if data_ini is None:
        raise RuntimeError("No training data loaded")
    if "arch_attr" not in data_ini:
        raise RuntimeError(
            "Dataset does not contain arch_attr. Generate it with "
            "`python3 hgp/data_process/gen_dataset_board.py` first."
        )

    device = _resolve_device(args.device)
    print("Using device {}".format(device))
    print("Torch CPU threads: {}".format(active_threads))
    print("DataLoader workers: {}".format(args.num_workers))
    cache_path, board_fabric = _prepare_architecture_cache(args, device, all_devices)

    # Multi-board: load all arch caches and stash on model later.
    arch_caches = None
    fabric_caches = None
    if multi_board:
        if args.arch_mode == ARCH_AWARE_MODE:
            arch_caches = prepare_arch_caches(
                all_devices, cache_dir=args.arch_cache_dir, torch_device=device
            )
        else:
            fabric_caches = prepare_board_fabric_caches(
                all_devices, cache_dir=args.arch_cache_dir, torch_device=device
            )

    arch_node_dim = board_fabric["arch_x"].shape[-1] if args.arch_mode == "fabric" else 24
    arch_edge_dim = board_fabric["arch_edge_attr"].shape[-1] if args.arch_mode == "fabric" else 4
    arch_graph_dim = board_fabric["arch_graph_attr"].shape[-1] if args.arch_mode == "fabric" else 32
    model = ArchAwareHierNet(
        in_channels=data_ini.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        hls_dim=spec.hls_dim,
        arch_dim=data_ini["arch_attr"].shape[-1],
        arch_node_dim=arch_node_dim,
        arch_edge_dim=arch_edge_dim,
        arch_graph_dim=arch_graph_dim,
        arch_hidden_dim=args.arch_hidden_dim,
        fabric_hidden_dim=args.fabric_hidden_dim,
        design_edge_dim=data_ini.edge_attr.shape[-1] if getattr(data_ini, "edge_attr", None) is not None else None,
        arch_mode=args.arch_mode,
        arch_aware_hidden_dim=args.arch_aware_hidden_dim,
        arch_aware_output_dim=args.fabric_hidden_dim,
        drop_out=args.drop_out,
        pool_mode=spec.pool_mode,
    )
    model = model.to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(_checkpoint_state(payload))
    else:
        _initialize_hls_residual_head(model, args, train_ds=train_ds, spec=spec)
    if multi_board:
        _attach_multi_board_caches(
            model, arch_caches=arch_caches, fabric_caches=fabric_caches
        )
    board_training_input = _prepare_board_training_input(model, board_fabric, args.fabric_mode)
    print(model)

    board_label = ", ".join(all_devices) if multi_board else all_devices[0]
    print(
        "Training target {} with board(s) {} ({})".format(
            args.target, board_label, args.arch_mode,
        )
    )

    model_dir = os.path.abspath(args.model_dir or "./model")
    os.makedirs(model_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_train_metric = float("inf")
    min_val_metric = float("inf")
    min_test_metric = float("inf")
    min_selection_metric = float("inf")
    selected_checkpoint_epoch = None
    selected_checkpoint_metrics = None
    history = []
    checkpoint_paths = {}
    if args.epochs == 0:
        train_loss, train_metric = evaluate(model, train_loader, device, spec, board_training_input, args, epoch=-1)
        if val_loader is not None:
            val_loss, val_metric = evaluate(model, val_loader, device, spec, board_training_input, args, epoch=-1)
            min_val_metric = val_metric
        else:
            val_loss, val_metric = None, None
        test_loss, test_metric = evaluate(model, test_loader, device, spec, board_training_input, args, epoch=-1)
        min_train_metric = train_metric
        min_test_metric = test_metric
        min_selection_metric = val_metric if checkpoint_metric_source == "val" else test_metric
        selected_checkpoint_epoch = -1
        selected_checkpoint_metrics = {
            "epoch": -1,
            "selection_metric": min_selection_metric,
            "train_metric": train_metric,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }
        history.append(
            {
                "epoch": -1,
                "learning_rate": args.lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "test_metric": test_metric,
                "best_val_metric": min_val_metric if val_loader is not None else None,
                "best_test_metric": min_test_metric,
                "best_selection_metric": min_selection_metric,
                "selected_checkpoint_epoch": selected_checkpoint_epoch,
                "selected_train_metric": selected_checkpoint_metrics["train_metric"],
                "selected_val_metric": selected_checkpoint_metrics["val_metric"],
                "selected_test_metric": selected_checkpoint_metrics["test_metric"],
            }
        )
    for epoch in range(args.epochs):
        train_loss, train_metric = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            spec,
            board_training_input,
            args,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )
        test_loss, test_metric = evaluate(
            model,
            test_loader,
            device,
            spec,
            board_training_input,
            args,
            epoch,
            print_predictions=args.print_predictions,
        )
        if val_loader is not None:
            val_loss, val_metric = evaluate(
                model,
                val_loader,
                device,
                spec,
                board_training_input,
                args,
                epoch,
            )
        else:
            val_loss, val_metric = None, None

        if val_metric is not None and val_metric < min_val_metric:
            min_val_metric = val_metric
        selection_metric = val_metric if checkpoint_metric_source == "val" else test_metric
        print_parts = [
            f"Epoch: {epoch:03d}",
            f"Train Loss: {train_loss:.4f}",
        ]
        if val_loss is not None:
            print_parts.append(f"Val Loss: {val_loss:.4f}")
        print_parts.append(f"Test Loss: {test_loss:.4f}")
        print(", ".join(print_parts))
        metric_parts = [
            "Epoch: {:03d}".format(epoch),
            "Train {}: {:.4f}".format(spec.metric_name.upper(), train_metric),
        ]
        if val_metric is not None:
            metric_parts.append("Val {}: {:.4f}".format(spec.metric_name.upper(), val_metric))
        metric_parts.append("Test {}: {:.4f}".format(spec.metric_name.upper(), test_metric))
        print(", ".join(metric_parts))
        print(
            "Checkpoint selection {} {}: {:.4f}".format(
                checkpoint_metric_source.upper(),
                spec.metric_name.upper(),
                selection_metric,
            )
        )

        if args.lr_decay_factor != 1.0 and args.lr_decay_interval > 0 and epoch % args.lr_decay_interval == 0:
            for p in optimizer.param_groups:
                p["lr"] *= args.lr_decay_factor

        if train_metric < min_train_metric:
            min_train_metric = train_metric
            checkpoint_paths["train"] = os.path.join(model_dir, _checkpoint_name(spec, "train"))
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metric_value=min_train_metric,
                    metric_key="min_train",
                    spec=spec,
                    all_devices=all_devices,
                    args=args,
                    cache_path=cache_path,
                    feature_stats=feature_stats,
                    target_scale=target_scale,
                    extra_metrics={
                        "val_{}".format(spec.metric_name): val_metric,
                        "test_{}".format(spec.metric_name): test_metric,
                        "checkpoint_metric_source": checkpoint_metric_source,
                    },
                ),
                os.path.join(model_dir, _checkpoint_name(spec, "train")),
            )

        if test_metric < min_test_metric:
            min_test_metric = test_metric
        if selection_metric < min_selection_metric:
            min_selection_metric = selection_metric
            selected_checkpoint_epoch = epoch
            selected_checkpoint_metrics = {
                "epoch": epoch,
                "selection_metric": selection_metric,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "test_metric": test_metric,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
            checkpoint_paths[checkpoint_split] = os.path.join(model_dir, _checkpoint_name(spec, checkpoint_split))
            extra_metrics = {
                "selection_{}".format(spec.metric_name): selection_metric,
                "test_{}".format(spec.metric_name): test_metric,
                "checkpoint_metric_source": checkpoint_metric_source,
            }
            if val_metric is not None:
                extra_metrics["val_{}".format(spec.metric_name)] = val_metric
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metric_value=min_selection_metric,
                    metric_key="min_{}".format(checkpoint_metric_source),
                    spec=spec,
                    all_devices=all_devices,
                    args=args,
                    cache_path=cache_path,
                    feature_stats=feature_stats,
                    target_scale=target_scale,
                    extra_metrics=extra_metrics,
                ),
                os.path.join(model_dir, _checkpoint_name(spec, checkpoint_split)),
            )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "test_metric": test_metric,
                "best_val_metric": min_val_metric if val_loader is not None else None,
                "best_test_metric": min_test_metric,
                "best_selection_metric": min_selection_metric,
                "selected_checkpoint_epoch": selected_checkpoint_epoch,
                "selected_train_metric": (
                    selected_checkpoint_metrics["train_metric"] if selected_checkpoint_metrics else None
                ),
                "selected_val_metric": (
                    selected_checkpoint_metrics["val_metric"] if selected_checkpoint_metrics else None
                ),
                "selected_test_metric": (
                    selected_checkpoint_metrics["test_metric"] if selected_checkpoint_metrics else None
                ),
            }
        )

    print("Min Train {}: {}".format(spec.metric_name.upper(), min_train_metric))
    if val_loader is not None:
        print("Min Val {}: {}".format(spec.metric_name.upper(), min_val_metric))
    print("Min Test {}: {}".format(spec.metric_name.upper(), min_test_metric))
    print(
        "Best Selection {} ({}): {}".format(
            spec.metric_name.upper(),
            checkpoint_metric_source,
            min_selection_metric,
        )
    )
    summary = {
        "target": args.target,
        "min_train_metric": min_train_metric,
        "min_val_metric": min_val_metric if val_loader is not None else None,
        "min_test_metric": min_test_metric,
        "checkpoint_metric_source": checkpoint_metric_source,
        "best_selection_metric": min_selection_metric,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "selected_checkpoint_metrics": selected_checkpoint_metrics,
        "metric_name": spec.metric_name,
        "checkpoints": checkpoint_paths,
        "history": history,
        "feature_stats": stats_to_jsonable(feature_stats),
        "target_scale": target_scale,
        "settings": {
            **_jsonable_settings(args),
            "active_cpu_threads": active_threads,
            "architecture_cache": str(cache_path),
            "board_devices": all_devices,
            "test_board": args.test_board,
            "split_mode": args.split_mode,
        },
    }
    _write_summary(args, summary)
    return summary


def _parse_device_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [normalize_device_name(s.strip()) for s in value.split(",") if s.strip()]


def _resolve_all_devices(args) -> list[str]:
    """Build the full list of FPGA devices involved in this run."""
    devices: list[str] = []
    if args.train_boards:
        devices.extend(_parse_device_list(args.train_boards) or [])
    if args.test_board:
        canonical = normalize_device_name(args.test_board)
        if canonical not in devices:
            devices.append(canonical)
    if not devices:
        devices.append(DEFAULT_BOARD_DEVICE)
    return devices


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train architecture-aware HGBO-DSE HGP models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", choices=sorted(TARGET_SPECS), default="lut", help="PPA metric to predict")
    parser.add_argument("--dataset-dir", default=None, help="single-board dataset directory (legacy)")
    parser.add_argument("--dataset-root", default=None, help="multi-board dataset root (<root>/<device>/std_arch/)")
    parser.add_argument("--model-dir", default="./model", help="directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="samples per batch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="torch device")
    parser.add_argument("--cpu-threads", type=int, default=None, help="limit torch CPU threads")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--hidden-channels", type=int, default=64, help="design GNN hidden size")
    parser.add_argument("--num-layers", type=int, default=3, help="number of design GNN layers")
    parser.add_argument("--arch-hidden-dim", type=int, default=16, help="arch_attr MLP hidden size")
    parser.add_argument("--fabric-hidden-dim", type=int, default=32, help="fabric encoder output dim")
    parser.add_argument("--arch-aware-hidden-dim", dest="arch_aware_hidden_dim", type=int, default=32, help="arch-aware encoder hidden size")
    parser.add_argument("--arch-cache-dir", default=None, help="directory for RapidWright-derived arch caches")
    parser.add_argument("--arch-mode", choices=[ARCH_AWARE_MODE, "fabric"], default=ARCH_AWARE_MODE, help="architecture encoding strategy")
    parser.add_argument("--fabric-mode", choices=["cached", "trainable"], default="cached", help="whether the fabric encoder receives gradients")
    parser.add_argument(
        "--conv-type",
        choices=["auto", "gcn", "gat", "sage", "gine", ATAPP_CONV_TYPE],
        default="auto",
        help="design GNN type; auto uses atapp for dynamic_power and gine for other targets",
    )
    parser.add_argument("--drop-out", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--lr-decay-factor", type=float, default=0.9, help="multiplicative LR decay factor")
    parser.add_argument("--lr-decay-interval", type=int, default=10, help="epochs between LR decay steps")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Adam weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient norm clipping (0 to disable)")
    parser.add_argument("--print-predictions", action="store_true", help="print pred/true every 10 epochs")
    parser.add_argument("--deterministic-eval", action="store_true", help="disable shuffle/drop_last in eval loader")
    parser.add_argument("--init-checkpoint", default=None, help="path to a .pt checkpoint to warm-start from")
    parser.add_argument("--summary-path", default=None, help="write a JSON summary of the run to this path")
    parser.add_argument(
        "--checkpoint-metric-source",
        choices=["auto", "test", "val"],
        default="auto",
        help="metric used to select the saved eval checkpoint; auto uses validation when available",
    )
    parser.add_argument("--seed", type=int, default=128, help="random seed for data shuffle + split")
    parser.add_argument(
        "--train-boards",
        default=None,
        help="comma-separated device names for training (multi-board mode)",
    )
    parser.add_argument(
        "--test-board",
        default=None,
        help="held-out device name for leave-one-board-out evaluation",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "board-out"],
        default="random",
        help="dataset split strategy: random 80/20 or leave-one-board-out",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of training data reserved for in-distribution validation (board-out mode)",
    )
    parser.add_argument(
        "--target-transform",
        choices=["none", "log1p"],
        default="none",
        help="optional output transform; log1p compresses LUT/FF dynamic range",
    )
    parser.add_argument(
        "--hls-residual-index",
        type=int,
        default=None,
        help="if set, predict residual on top of hls_attr[:, idx] (e.g. 0 for HLS LUT estimate); not compatible with --target-transform",
    )
    parser.add_argument(
        "--normalize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="z-score data.x and data.hls_attr using train-split statistics",
    )
    parser.add_argument(
        "--normalize-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="divide the regression target by mean(train_y) so the head learns ~unit-scale outputs",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=0,
        help="ramp LR linearly from 0 to --lr over this many batches at training start (0 disables)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
