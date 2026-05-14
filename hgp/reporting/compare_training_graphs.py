from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import SAGPooling, global_add_pool, global_max_pool, global_mean_pool
from torch.utils.data import random_split

from hgp.arch_aware_arch import arch_aware_arch_to_device, ensure_arch_aware_arch_cache, load_arch_aware_arch_cache
from hgp.board_fabric import board_fabric_to_device, ensure_board_fabric_cache, load_board_fabric_cache
from hgp.board_utils import DEFAULT_BOARD_DEVICE
from hgp.dataset_utils import generate_dataset, mae_loss, mape_loss
from hgp.hier_arch_model import (
    ARCH_AWARE_MODE,
    ArchAwareHierNet,
    TARGET_SPECS as ARCH_TARGET_SPECS,
    _apply_design_conv,
    _loader_kwargs,
    _make_design_conv,
    _prepare_board_training_input,
    _resolve_device,
    _set_cpu_threads,
    evaluate as evaluate_arch,
    train_epoch as train_arch_epoch,
)


HGBO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OriginalTargetSpec:
    name: str
    target_index: int
    dataset_subdir: str
    hls_dim: int
    pool_mode: str
    metric_name: str
    label_scale: float = 1.0


ORIGINAL_TARGET_SPECS = {
    "lut": OriginalTargetSpec("lut", 0, "std", 6, "add", "mape"),
    "ff": OriginalTargetSpec("ff", 1, "std", 6, "add", "mape"),
    "dsp": OriginalTargetSpec("dsp", 2, "std", 6, "add", "mae"),
    "bram": OriginalTargetSpec("bram", 3, "std", 6, "add", "mae"),
    "cp": OriginalTargetSpec("cp", 6, "rdc", 1, "mean", "mape"),
    "power": OriginalTargetSpec("power", 7, "std", 6, "mean", "mape", label_scale=100.0),
}


TARGETS = ["lut", "ff", "dsp", "bram", "cp", "power"]
COLOR_ORIGINAL_TRAIN = "#6b7280"
COLOR_ORIGINAL_TEST = "#111827"
COLOR_ARCH_TRAIN = "#2a9d8f"
COLOR_ARCH_TEST = "#d1495b"


def _ordered_targets(target_map):
    unknown = sorted(set(target_map) - set(TARGETS))
    if unknown:
        raise ValueError("Unknown target(s): {}".format(", ".join(unknown)))
    return [target for target in TARGETS if target in target_map]


def _target_metric_note(targets):
    metrics = sorted({ORIGINAL_TARGET_SPECS[target].metric_name.upper() for target in targets})
    return "Lower is better. Metric: {}.".format(metrics[0] if len(metrics) == 1 else "/".join(metrics))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return torch.nn.Sequential(
        Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        Linear(hidden_dim, output_dim),
    )


class OriginalHierNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        conv_type,
        hls_dim,
        design_edge_dim=None,
        pool_mode="add",
        drop_out=0.0,
        pool_ratio=0.5,
    ):
        super().__init__()
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.pool_mode = pool_mode
        self.conv_type = conv_type

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.convs.append(_make_design_conv(conv_type, in_channels, hidden_channels, design_edge_dim))
            else:
                self.convs.append(_make_design_conv(conv_type, hidden_channels, hidden_channels, design_edge_dim))
            self.pools.append(SAGPooling(hidden_channels, self.pool_ratio))

        self.jkn = JumpingKnowledge("lstm", channels=hidden_channels, num_layers=2)
        self.channels = [hidden_channels * 2 + hls_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()
        for idx in range(len(self.channels) - 1):
            self.mlps.append(Linear(self.channels[idx], self.channels[idx + 1]))

    def _pool(self, x, batch):
        if self.pool_mode == "mean":
            return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

    def forward(self, x, edge_index, batch, hls_attr, edge_attr=None):
        x = x.to(torch.float32)
        hls_attr = hls_attr.to(torch.float32)
        h_list = []
        for step in range(len(self.convs)):
            x = _apply_design_conv(self.convs[step], self.conv_type, x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x, edge_index, edge_attr, batch, _, _ = self.pools[step](x, edge_index, edge_attr, batch, None)
            h_list.append(self._pool(x, batch))

        x = h_list[0] + h_list[1] + h_list[2] if len(h_list) >= 3 else sum(h_list)
        x = torch.cat([x, hls_attr], dim=-1)
        for idx, layer in enumerate(self.mlps):
            if idx < len(self.mlps) - 1:
                x = F.relu(layer(x))
                x = F.dropout(x, p=self.drop_out, training=self.training)
            else:
                x = layer(x)
        return x


def _metric_fn(metric_name: str):
    if metric_name == "mae":
        return mae_loss
    return mape_loss


def _target_values(data, spec: OriginalTargetSpec):
    return data["y"].t()[spec.target_index] * spec.label_scale


def _ensure_finite_tensor(value, name, phase, target, epoch, batch_idx):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if torch.isfinite(value).all():
        return
    bad_count = int((~torch.isfinite(value)).sum().item())
    raise RuntimeError(
        "non-finite {} during {} target {} epoch {} batch {} ({} of {} values)".format(
            name,
            phase,
            target,
            epoch,
            batch_idx,
            bad_count,
            value.numel(),
        )
    )


def _load_split(dataset_subdir: str, seed: int):
    dataset_dir = HGBO_ROOT / "dataset" / dataset_subdir
    dataset_files = sorted(path.name for path in dataset_dir.glob("*.pt"))
    dataset = generate_dataset(str(dataset_dir), dataset_files, print_info=False)
    np.random.RandomState(seed=seed).shuffle(dataset)
    train_size = round(0.8 * len(dataset))
    test_size = round(0.2 * len(dataset))
    return random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))


def _get_split(split_cache, dataset_subdir: str, seed: int):
    key = (dataset_subdir, seed)
    if key not in split_cache:
        split_cache[key] = _load_split(dataset_subdir, seed)
    return split_cache[key]


def _first_batch(loader):
    for data in loader:
        return data
    raise RuntimeError("No data loaded")


def train_original_epoch(model, loader, optimizer, device, spec, epoch, grad_clip):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    metric_fn = _metric_fn(spec.metric_name)
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data["hls_attr"], edge_attr=getattr(data, "edge_attr", None)).view(-1)
        true_y = _target_values(data, spec)
        _ensure_finite_tensor(out, "model output", "training", spec.name, epoch, batch_idx)
        _ensure_finite_tensor(true_y, "target", "training", spec.name, epoch, batch_idx)
        loss = F.huber_loss(out, true_y).float()
        metric = metric_fn(out, true_y).float()
        _ensure_finite_tensor(loss, "loss", "training", spec.name, epoch, batch_idx)
        _ensure_finite_tensor(metric, "{} metric".format(spec.metric_name), "training", spec.name, epoch, batch_idx)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _ensure_finite_tensor(grad_norm, "gradient norm", "training", spec.name, epoch, batch_idx)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_metric += metric.item() * data.num_graphs
    return total_loss / len(loader.dataset), total_metric / len(loader.dataset)


def evaluate_original(model, loader, device, spec, epoch):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    metric_fn = _metric_fn(spec.metric_name)
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data["hls_attr"], edge_attr=getattr(data, "edge_attr", None)).view(-1)
            true_y = _target_values(data, spec)
            _ensure_finite_tensor(out, "model output", "evaluation", spec.name, epoch, batch_idx)
            _ensure_finite_tensor(true_y, "target", "evaluation", spec.name, epoch, batch_idx)
            loss = F.huber_loss(out, true_y).float()
            metric = metric_fn(out, true_y).float()
            _ensure_finite_tensor(loss, "loss", "evaluation", spec.name, epoch, batch_idx)
            _ensure_finite_tensor(metric, "{} metric".format(spec.metric_name), "evaluation", spec.name, epoch, batch_idx)
            total_loss += loss.item() * data.num_graphs
            total_metric += metric.item() * data.num_graphs
    return total_loss / len(loader.dataset), total_metric / len(loader.dataset)


def run_original_target(target, args, device, split_cache=None):
    _set_seed(args.seed)
    spec = ORIGINAL_TARGET_SPECS[target]
    if split_cache is None:
        train_ds, test_ds = _load_split(spec.dataset_subdir, args.seed)
    else:
        train_ds, test_ds = _get_split(split_cache, spec.dataset_subdir, args.seed)
    loader_kwargs = _loader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    data_ini = _first_batch(train_loader)
    model = OriginalHierNet(
        in_channels=data_ini.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        hls_dim=spec.hls_dim,
        design_edge_dim=data_ini.edge_attr.shape[-1] if getattr(data_ini, "edge_attr", None) is not None else None,
        pool_mode=spec.pool_mode,
        drop_out=args.drop_out,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_test_metric = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_metric = train_original_epoch(model, train_loader, optimizer, device, spec, epoch, args.grad_clip)
        test_loss, test_metric = evaluate_original(model, test_loader, device, spec, epoch)
        if epoch % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.9
        best_test_metric = min(best_test_metric, test_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "test_metric": test_metric,
                "best_test_metric": best_test_metric,
            }
        )
    return history


def run_arch_target(target, args, device, split_cache=None):
    _set_seed(args.seed)
    spec = ARCH_TARGET_SPECS[target]
    if split_cache is None:
        train_ds, test_ds = _load_split(spec.dataset_subdir, args.seed)
    else:
        train_ds, test_ds = _get_split(split_cache, spec.dataset_subdir, args.seed)
    loader_kwargs = _loader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    data_ini = _first_batch(train_loader)
    if args.arch_mode == ARCH_AWARE_MODE:
        cache_path = ensure_arch_aware_arch_cache(DEFAULT_BOARD_DEVICE, cache_dir=args.arch_cache_dir)
        board_fabric = arch_aware_arch_to_device(load_arch_aware_arch_cache(cache_path), device)
        arch_node_dim = 24
        arch_edge_dim = 4
        arch_graph_dim = 32
    else:
        cache_path = ensure_board_fabric_cache(DEFAULT_BOARD_DEVICE, cache_dir=args.arch_cache_dir)
        board_fabric = board_fabric_to_device(load_board_fabric_cache(cache_path), device)
        arch_node_dim = board_fabric["arch_x"].shape[-1]
        arch_edge_dim = board_fabric["arch_edge_attr"].shape[-1]
        arch_graph_dim = board_fabric["arch_graph_attr"].shape[-1]
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
    ).to(device)
    board_training_input = _prepare_board_training_input(model, board_fabric, args.fabric_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_test_metric = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_metric = train_arch_epoch(
            model,
            train_loader,
            optimizer,
            device,
            spec,
            board_training_input,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )
        test_loss, test_metric = evaluate_arch(
            model,
            test_loader,
            device,
            spec,
            board_training_input,
            epoch=epoch,
            print_predictions=False,
        )
        if epoch % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.9
        best_test_metric = min(best_test_metric, test_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "test_metric": test_metric,
                "best_test_metric": best_test_metric,
            }
        )
    return history


def _svg_escape(value) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _polyline(points):
    return " ".join("{:.2f},{:.2f}".format(x, y) for x, y in points)


def _line_points(values, left, top, width, height):
    finite_values = [value for value in values if math.isfinite(value)]
    y_min = min(finite_values)
    y_max = max(finite_values)
    if y_min == y_max:
        y_min *= 0.95
        y_max *= 1.05
        if y_min == y_max:
            y_min = 0.0
            y_max = 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    points = []
    for idx, value in enumerate(values):
        x = left + (idx / max(1, len(values) - 1)) * width
        y = top + height - ((value - y_min) / (y_max - y_min)) * height
        points.append((x, y))
    return points, y_min, y_max


def write_training_curves_svg(results, output_path):
    targets = _ordered_targets(results["targets"])
    panel_width = 330
    panel_height = 230
    margin_x = 58
    margin_y = 54
    inner_width = 245
    inner_height = 132
    cols = 3
    rows = math.ceil(len(targets) / cols)
    width = cols * panel_width
    height = rows * panel_height + 58
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">Training Metric Curves</text>',
        '<text x="24" y="51" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{}</text>'.format(_target_metric_note(targets)),
    ]

    legend = [
        ("Original train", COLOR_ORIGINAL_TRAIN),
        ("Original test", COLOR_ORIGINAL_TEST),
        ("Arch train", COLOR_ARCH_TRAIN),
        ("Arch test", COLOR_ARCH_TEST),
    ]
    legend_x = width - 455
    for idx, (label, color) in enumerate(legend):
        x = legend_x + idx * 112
        lines.append('<line x1="{0}" y1="30" x2="{1}" y2="30" stroke="{2}" stroke-width="3"/>'.format(x, x + 22, color))
        lines.append('<text x="{0}" y="34" font-family="Arial, sans-serif" font-size="11" fill="#374151">{1}</text>'.format(x + 28, _svg_escape(label)))

    for idx, target in enumerate(targets):
        row = idx // cols
        col = idx % cols
        panel_left = col * panel_width + 20
        panel_top = row * panel_height + 76
        plot_left = panel_left + margin_x
        plot_top = panel_top + margin_y

        target_result = results["targets"][target]
        original = target_result["original"]["history"]
        arch = target_result["architecture_aware"]["history"]
        series = {
            "Original train": [item["train_metric"] for item in original],
            "Original test": [item["test_metric"] for item in original],
            "Arch train": [item["train_metric"] for item in arch],
            "Arch test": [item["test_metric"] for item in arch],
        }
        all_values = [value for values in series.values() for value in values if math.isfinite(value)]
        y_min = min(all_values)
        y_max = max(all_values)
        if y_min == y_max:
            y_min = 0.0
            y_max = y_max if y_max else 1.0
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

        lines.append('<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">{}</text>'.format(panel_left, panel_top + 20, target.upper()))
        lines.append('<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{}</text>'.format(panel_left, panel_top + 38, target_result["metric_name"].upper()))
        lines.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#d1d5db" stroke-width="1"/>'.format(plot_left, plot_top, plot_top + inner_height))
        lines.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#d1d5db" stroke-width="1"/>'.format(plot_left, plot_top + inner_height, plot_left + inner_width))
        for tick in range(3):
            value = y_min + (y_max - y_min) * tick / 2
            y = plot_top + inner_height - tick * inner_height / 2
            lines.append('<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#f3f4f6" stroke-width="1"/>'.format(plot_left, y, plot_left + inner_width))
            lines.append('<text x="{0}" y="{1:.2f}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" fill="#6b7280">{2:.3g}</text>'.format(plot_left - 6, y + 3, value))
        lines.append('<text x="{0}" y="{1}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#6b7280">0</text>'.format(plot_left, plot_top + inner_height + 16))
        lines.append('<text x="{0}" y="{1}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#6b7280">{2}</text>'.format(plot_left + inner_width, plot_top + inner_height + 16, len(original) - 1))

        color_by_label = {
            "Original train": COLOR_ORIGINAL_TRAIN,
            "Original test": COLOR_ORIGINAL_TEST,
            "Arch train": COLOR_ARCH_TRAIN,
            "Arch test": COLOR_ARCH_TEST,
        }
        dash_by_label = {
            "Original train": ' stroke-dasharray="5 4"',
            "Original test": "",
            "Arch train": ' stroke-dasharray="5 4"',
            "Arch test": "",
        }
        for label, values in series.items():
            points = []
            for point_idx, value in enumerate(values):
                x = plot_left + (point_idx / max(1, len(values) - 1)) * inner_width
                y = plot_top + inner_height - ((value - y_min) / (y_max - y_min)) * inner_height
                points.append((x, y))
            lines.append(
                '<polyline fill="none" stroke="{0}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{1} points="{2}"/>'.format(
                    color_by_label[label],
                    dash_by_label[label],
                    _polyline(points),
                )
            )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_delta_svg(results, output_path):
    targets = _ordered_targets(results["targets"])
    width = 900
    height = 390
    left = 76
    top = 72
    chart_width = 770
    chart_height = 210
    values = []
    for target in targets:
        target_result = results["targets"][target]
        original = target_result["original"]["best_test_metric"]
        arch = target_result["architecture_aware"]["best_test_metric"]
        values.append((target, ((original - arch) / original) * 100.0))

    y_min = min(0.0, min(value for _, value in values))
    y_max = max(0.0, max(value for _, value in values))
    pad = max(2.0, (y_max - y_min) * 0.12)
    y_min -= pad
    y_max += pad
    zero_y = top + chart_height - ((0.0 - y_min) / (y_max - y_min)) * chart_height
    bar_width = chart_width / len(values) * 0.58
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">Architecture-Aware Best Test Change</text>',
        '<text x="24" y="53" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">Positive bars mean the architecture-aware model reduced the error metric versus original HGBO-DSE.</text>',
        '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#111827" stroke-width="1.2"/>'.format(left, zero_y, left + chart_width),
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = top + chart_height - ((value - y_min) / (y_max - y_min)) * chart_height
        lines.append('<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#f3f4f6" stroke-width="1"/>'.format(left, y, left + chart_width))
        lines.append('<text x="{0}" y="{1:.2f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end" fill="#6b7280">{2:.0f}%</text>'.format(left - 8, y + 4, value))
    for idx, (target, value) in enumerate(values):
        center = left + chart_width * (idx + 0.5) / len(values)
        y = top + chart_height - ((value - y_min) / (y_max - y_min)) * chart_height
        bar_top = min(y, zero_y)
        bar_height = abs(y - zero_y)
        color = COLOR_ARCH_TRAIN if value >= 0 else COLOR_ARCH_TEST
        lines.append('<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="{4}"/>'.format(center - bar_width / 2, bar_top, bar_width, bar_height, color))
        label_y = bar_top - 8 if value >= 0 else bar_top + bar_height + 16
        lines.append('<text x="{0:.2f}" y="{1:.2f}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#111827">{2:+.1f}%</text>'.format(center, label_y, value))
        lines.append('<text x="{0:.2f}" y="{1}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#374151">{2}</text>'.format(center, top + chart_height + 30, target.upper()))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hgbo_arch_training_history.json"
    csv_path = output_dir / "hgbo_arch_training_history.csv"
    curves_path = output_dir / "hgbo_arch_training_curves.svg"
    delta_path = output_dir / "hgbo_arch_best_test_delta.svg"

    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "variant",
                "metric_name",
                "epoch",
                "train_loss",
                "test_loss",
                "train_metric",
                "test_metric",
                "best_test_metric",
            ],
        )
        writer.writeheader()
        for target in _ordered_targets(results["targets"]):
            for variant_key, variant_name in [
                ("original", "Original"),
                ("architecture_aware", "Architecture-aware"),
            ]:
                target_result = results["targets"][target]
                for item in target_result[variant_key]["history"]:
                    writer.writerow(
                        {
                            "target": target,
                            "variant": variant_name,
                            "metric_name": target_result["metric_name"],
                            **item,
                        }
                    )

    write_training_curves_svg(results, curves_path)
    write_best_delta_svg(results, delta_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "training_curves": str(curves_path),
        "best_test_delta": str(delta_path),
    }


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def build_consistency_summary(runs):
    if not runs:
        raise ValueError("At least one run is required for consistency summary")

    seeds = [run["settings"]["seed"] for run in runs]
    first_settings = runs[0]["settings"]
    targets = _ordered_targets(runs[0]["targets"])
    if any(set(run["targets"]) != set(targets) for run in runs):
        raise ValueError("All runs must contain the same target set")
    summary = {
        "settings": {
            "seeds": seeds,
            "epochs": first_settings["epochs"],
            "lr": first_settings["lr"],
            "grad_clip": first_settings["grad_clip"],
            "arch_mode": first_settings.get("arch_mode", "fabric"),
            "fabric_mode": first_settings["fabric_mode"],
            "board_device": first_settings["board_device"],
        },
        "targets": {},
        "overall": {
            "run_count": len(runs),
            "target_count": len(targets),
            "target_run_count": len(runs) * len(targets),
            "architecture_aware_win_count": 0,
            "original_win_count": 0,
        },
    }
    for key in ["batch_size", "device_request", "device", "cpu_threads", "num_workers", "weight_decay"]:
        if key in first_settings:
            summary["settings"][key] = first_settings[key]

    for target in targets:
        original_values = []
        arch_values = []
        deltas = []
        winners = []
        per_seed = []
        metric_name = runs[0]["targets"][target]["metric_name"]
        for run in runs:
            original = run["targets"][target]["original"]["best_test_metric"]
            arch = run["targets"][target]["architecture_aware"]["best_test_metric"]
            delta_pct = ((original - arch) / original) * 100.0
            winner = "architecture-aware" if arch < original else "original"
            original_values.append(original)
            arch_values.append(arch)
            deltas.append(delta_pct)
            winners.append(winner)
            per_seed.append(
                {
                    "seed": run["settings"]["seed"],
                    "original_best_test_metric": original,
                    "architecture_aware_best_test_metric": arch,
                    "delta_pct": delta_pct,
                    "winner": winner,
                }
            )

        arch_wins = winners.count("architecture-aware")
        original_wins = winners.count("original")
        if arch_wins == len(runs):
            consistent_winner = "architecture-aware"
        elif original_wins == len(runs):
            consistent_winner = "original"
        else:
            consistent_winner = "mixed"

        summary["targets"][target] = {
            "metric_name": metric_name,
            "per_seed": per_seed,
            "original_mean_best_test": _mean(original_values),
            "original_std_best_test": _std(original_values),
            "architecture_aware_mean_best_test": _mean(arch_values),
            "architecture_aware_std_best_test": _std(arch_values),
            "delta_pct_mean": _mean(deltas),
            "delta_pct_std": _std(deltas),
            "architecture_aware_win_count": arch_wins,
            "original_win_count": original_wins,
            "consistent_winner": consistent_winner,
        }
        summary["overall"]["architecture_aware_win_count"] += arch_wins
        summary["overall"]["original_win_count"] += original_wins

    return summary


def write_consistency_summary(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hgbo_arch_consistency_summary.json"
    csv_path = output_dir / "hgbo_arch_consistency_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "metric_name",
                "original_mean_best_test",
                "original_std_best_test",
                "architecture_aware_mean_best_test",
                "architecture_aware_std_best_test",
                "delta_pct_mean",
                "delta_pct_std",
                "architecture_aware_win_count",
                "original_win_count",
                "consistent_winner",
            ],
        )
        writer.writeheader()
        for target in _ordered_targets(summary["targets"]):
            row = summary["targets"][target].copy()
            row.pop("per_seed")
            writer.writerow({"target": target, **row})
    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }


def run_comparison(args):
    active_threads = _set_cpu_threads(args.cpu_threads)
    device = _resolve_device(args.device)
    print("Using device {}".format(device))
    print("Torch CPU threads: {}".format(active_threads))
    print("DataLoader workers: {}".format(args.num_workers))
    split_cache = {}
    results = {
        "settings": {
            "epochs": args.epochs,
            "seed": args.seed,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "device_request": args.device,
            "cpu_threads": active_threads,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "arch_mode": args.arch_mode,
            "fabric_mode": args.fabric_mode,
            "device": str(device),
            "board_device": DEFAULT_BOARD_DEVICE,
        },
        "targets": {},
    }

    for target in args.targets:
        print("Running target {}".format(target))
        original_history = run_original_target(target, args, device, split_cache=split_cache)
        arch_history = run_arch_target(target, args, device, split_cache=split_cache)
        metric_name = ORIGINAL_TARGET_SPECS[target].metric_name
        results["targets"][target] = {
            "metric_name": metric_name,
            "original": {
                "spec": asdict(ORIGINAL_TARGET_SPECS[target]),
                "history": original_history,
                "best_test_metric": min(item["test_metric"] for item in original_history),
            },
            "architecture_aware": {
                "spec": {
                    "name": ARCH_TARGET_SPECS[target].name,
                    "target_index": ARCH_TARGET_SPECS[target].target_index,
                    "dataset_subdir": ARCH_TARGET_SPECS[target].dataset_subdir,
                    "hls_dim": ARCH_TARGET_SPECS[target].hls_dim,
                    "pool_mode": ARCH_TARGET_SPECS[target].pool_mode,
                    "metric_name": ARCH_TARGET_SPECS[target].metric_name,
                    "label_scale": ARCH_TARGET_SPECS[target].label_scale,
                },
                "history": arch_history,
                "best_test_metric": min(item["test_metric"] for item in arch_history),
            },
        }
        original_best = results["targets"][target]["original"]["best_test_metric"]
        arch_best = results["targets"][target]["architecture_aware"]["best_test_metric"]
        winner = "architecture-aware" if arch_best < original_best else "original"
        print(
            "Target {} best {}: original {:.6g}, architecture-aware {:.6g}, winner {}".format(
                target,
                metric_name.upper(),
                original_best,
                arch_best,
                winner,
            )
        )

    output_paths = write_results(results, Path(args.output_dir))
    print(json.dumps(output_paths, indent=2, sort_keys=True))
    return results


def run_seed_comparisons(args):
    output_dir = Path(args.output_dir)
    runs = []
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.output_dir = str(output_dir / "seed_{}".format(seed))
        print("Running comparison seed {}".format(seed))
        runs.append(run_comparison(seed_args))
    summary = build_consistency_summary(runs)
    output_paths = write_consistency_summary(summary, output_dir)
    print(json.dumps(output_paths, indent=2, sort_keys=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Generate HGBO-DSE architecture-aware training comparison graphs.")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--epochs", type=int, default=_env_int("COMPARE_EPOCHS", 30))
    parser.add_argument("--lr", type=float, default=_env_float("COMPARE_LR", 0.001))
    parser.add_argument("--grad-clip", type=float, default=_env_float("COMPARE_GRAD_CLIP", 1.0))
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--arch-hidden-dim", type=int, default=16)
    parser.add_argument("--fabric-hidden-dim", type=int, default=32)
    parser.add_argument("--arch-aware-hidden-dim", dest="arch_aware_hidden_dim", type=int, default=32)
    parser.add_argument("--arch-cache-dir", default=None)
    parser.add_argument("--arch-mode", choices=[ARCH_AWARE_MODE, "fabric"], default=ARCH_AWARE_MODE)
    parser.add_argument("--fabric-mode", choices=["cached", "trainable"], default="cached")
    parser.add_argument("--conv-type", choices=["gcn", "gat", "sage", "gine"], default="gine")
    parser.add_argument("--drop-out", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--output-dir", default=str(HGBO_ROOT / "img" / "training"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seeds:
        run_seed_comparisons(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
