#!/usr/bin/env python3
"""Regenerate prediction SVGs from built-in HGBO-DSE checkpoints.

This script verifies the original HGBO-DSE checkpoint quality without
overwriting the checked-in ``img/*_pred.svg`` files or ``hgp/model/*.pt``
weights. By default it writes into a timestamped directory:

    img/reproduced_builtin_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


HGBO_ROOT = Path(__file__).resolve().parents[1]
if str(HGBO_ROOT) not in sys.path:
    sys.path.insert(0, str(HGBO_ROOT))

from bome.pred import pred_bram, pred_cp, pred_dsp, pred_ff, pred_lut, pred_pwr  # noqa: E402
from bome.pred.checkpoint import load_pretrained_state_dict  # noqa: E402
from hgp.dataset_utils import generate_dataset  # noqa: E402


TARGET_SPECS = {
    "lut": {
        "title": "LUT",
        "model_cls": pred_lut.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "std",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "lut_h64_d0_checkpoint_test.pt",
        "target_index": 0,
        "in_dim": 15,
        "hls_dim": 6,
        "label_scale": 1.0,
        "metric_name": "MAPE",
        "svg_name": "lut_pred.svg",
    },
    "ff": {
        "title": "FF",
        "model_cls": pred_ff.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "std",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "ff_h64_d0_checkpoint_test.pt",
        "target_index": 1,
        "in_dim": 15,
        "hls_dim": 6,
        "label_scale": 1.0,
        "metric_name": "MAPE",
        "svg_name": "ff_pred.svg",
    },
    "dsp": {
        "title": "DSP",
        "model_cls": pred_dsp.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "std",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "dsp_mae_h64_d0_checkpoint_test.pt",
        "target_index": 2,
        "in_dim": 15,
        "hls_dim": 6,
        "label_scale": 1.0,
        "metric_name": "MAE",
        "svg_name": "dsp_pred.svg",
    },
    "bram": {
        "title": "BRAM",
        "model_cls": pred_bram.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "std",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "bram_mae_h64_d0_checkpoint_test.pt",
        "target_index": 3,
        "in_dim": 15,
        "hls_dim": 6,
        "label_scale": 1.0,
        "metric_name": "MAE",
        "svg_name": "bram_pred.svg",
    },
    "cp": {
        "title": "CP",
        "model_cls": pred_cp.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "rdc",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "cp_mean_h64_d0_checkpoint_test.pt",
        "target_index": 6,
        "in_dim": 11,
        "hls_dim": 1,
        "label_scale": 1.0,
        "metric_name": "MAPE",
        "svg_name": "cp_pred.svg",
    },
    "pwr": {
        "title": "Power",
        "model_cls": pred_pwr.HierNet,
        "dataset_dir": HGBO_ROOT / "dataset" / "std",
        "checkpoint_path": HGBO_ROOT / "hgp" / "model" / "power_mean_h64_d0_checkpoint_test.pt",
        "target_index": 7,
        "in_dim": 15,
        "hls_dim": 6,
        "label_scale": 100.0,
        "metric_name": "MAPE",
        "svg_name": "pwr_pred.svg",
    },
}


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return HGBO_ROOT / "img" / f"reproduced_builtin_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate HGBO-DSE prediction SVGs from built-in hgp/model checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for regenerated SVGs and summary.json. Defaults to a timestamped img/reproduced_builtin_* directory.",
    )
    parser.add_argument("--seed", type=int, default=128, help="Dataset shuffle seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="torch random_split seed.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-existing", action="store_true", help="Allow writing into an existing non-empty output directory.")
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGET_SPECS), default=list(TARGET_SPECS))
    return parser.parse_args()


def prepare_output_dir(path: Path, allow_existing: bool) -> Path:
    path = path if path.is_absolute() else HGBO_ROOT / path
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise SystemExit(f"Refusing to write into non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(HGBO_ROOT))
    except ValueError:
        return str(path)


def load_test_split(dataset_dir: Path, seed: int, split_seed: int):
    names = sorted(name for name in os.listdir(dataset_dir) if name.endswith(".pt"))
    dataset = generate_dataset(str(dataset_dir), names, print_info=False)
    np.random.RandomState(seed=seed).shuffle(dataset)
    train_size = round(0.8 * len(dataset))
    test_size = round(0.2 * len(dataset))
    _, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    return test_ds


def write_svg(path: Path, title: str, actual: np.ndarray, pred: np.ndarray, metric_name: str, metric: float) -> None:
    width, height = 760, 760
    left, right, top, bottom = 90, 30, 50, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    def sx(value):
        return left + (float(value) - lo) / (hi - lo) * plot_w

    def sy(value):
        return top + plot_h - (float(value) - lo) / (hi - lo) * plot_h

    metric_text = f"{metric_name}: {metric * 100:.3f}%" if metric_name == "MAPE" else f"{metric_name}: {metric:.4f}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<text x="{width / 2}" y="48" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{metric_text}; n={len(actual)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111"/>',
        f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}" stroke="#444" stroke-width="1" opacity="0.45"/>',
    ]
    for idx in range(6):
        value = lo + (hi - lo) * idx / 5
        x = sx(value)
        y = sy(value)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#111"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3g}</text>')
        lines.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111"/>')
        lines.append(f'<text x="{left - 9}" y="{y + 3:.2f}" text-anchor="end" font-family="Arial" font-size="10">{value:.3g}</text>')
    for actual_value, pred_value in zip(actual, pred):
        lines.append(
            f'<circle cx="{sx(actual_value):.2f}" cy="{sy(pred_value):.2f}" r="1.8" fill="#1f77b4" fill-opacity="0.8"/>'
        )
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="14">Actual {title}</text>')
    lines.append(
        f'<text transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">Predicted {title}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_target(target: str, args: argparse.Namespace, output_dir: Path):
    spec = TARGET_SPECS[target]
    device = torch.device(args.device)
    loader = DataLoader(
        load_test_split(spec["dataset_dir"], args.seed, args.split_seed),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    model = spec["model_cls"](
        in_channels=spec["in_dim"],
        hidden_channels=64,
        num_layers=3,
        conv_type="sage",
        hls_dim=spec["hls_dim"],
        drop_out=0.0,
    ).to(device)
    checkpoint = torch.load(spec["checkpoint_path"], map_location=device, weights_only=False)
    load_pretrained_state_dict(model, checkpoint["model"])
    model.eval()

    actual = []
    pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data["hls_attr"]).view(-1) / spec["label_scale"]
            y = data["y"].t()[spec["target_index"]]
            actual.extend(y.cpu().tolist())
            pred.extend(out.cpu().tolist())

    actual_np = np.asarray(actual, dtype=float)
    pred_np = np.asarray(pred, dtype=float)
    if spec["metric_name"] == "MAE":
        metric = float(np.mean(np.abs(actual_np - pred_np)))
    else:
        metric = float(np.mean(np.abs((actual_np - pred_np) / actual_np)))

    svg_path = output_dir / spec["svg_name"]
    write_svg(svg_path, spec["title"], actual_np, pred_np, spec["metric_name"], metric)
    return {
        "points": len(actual),
        "metric_name": spec["metric_name"],
        "metric": metric,
        "checkpoint": display_path(spec["checkpoint_path"]),
        "svg": display_path(svg_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.allow_existing)
    summary = {target: evaluate_target(target, args, output_dir) for target in args.targets}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for target, item in summary.items():
        display = item["metric"] * 100 if item["metric_name"] == "MAPE" else item["metric"]
        suffix = "%" if item["metric_name"] == "MAPE" else ""
        print(f"{target}: {item['metric_name']}={display:.4g}{suffix}, points={item['points']}, svg={item['svg']}")
    print(f"OUT={display_path(output_dir)}")


if __name__ == "__main__":
    main()
