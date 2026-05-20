from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from hgp.reporting.compare_training_graphs import (
    ORIGINAL_CONV_TYPE,
    ORIGINAL_TARGET_SPECS,
    TARGETS,
    OriginalHierNet,
    _first_batch,
    _get_split,
    _loader_kwargs,
    _resolve_device,
    _set_cpu_threads,
    _set_seed,
    _target_values,
)
from hgp.reporting.original_stable_training import initial_checkpoint_path, load_initial_checkpoint


def summarize_values(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().cpu().to(torch.float64).view(-1)
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "max": float(quantiles[4].item()),
    }


def prediction_rows(model, loader, device, spec):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            pred = model(
                data.x,
                data.edge_index,
                data.batch,
                data["hls_attr"],
                edge_attr=getattr(data, "edge_attr", None),
            ).view(-1)
            true = _target_values(data, spec)
            for item_idx, (true_value, pred_value) in enumerate(zip(true.detach().cpu(), pred.detach().cpu())):
                rows.append(
                    {
                        "batch": batch_idx,
                        "index": item_idx,
                        "true": float(true_value.item()),
                        "pred": float(pred_value.item()),
                        "abs_error": float(abs(true_value.item() - pred_value.item())),
                    }
                )
    return rows


def summarize_error_buckets(true: torch.Tensor, abs_error: torch.Tensor) -> list[dict[str, float | int | str]]:
    buckets = [
        ("true == 0", true == 0),
        ("0 < true <= 1", (true > 0) & (true <= 1)),
        ("1 < true <= 10", (true > 1) & (true <= 10)),
        ("10 < true <= 100", (true > 10) & (true <= 100)),
        ("true > 100", true > 100),
    ]
    summaries = []
    for label, mask in buckets:
        count = int(mask.sum().item())
        if count == 0:
            summaries.append({"bucket": label, "count": 0})
            continue
        bucket_errors = abs_error[mask]
        summaries.append(
            {
                "bucket": label,
                "count": count,
                "mae": float(bucket_errors.mean().item()),
                "median_abs_error": float(bucket_errors.median().item()),
                "max_abs_error": float(bucket_errors.max().item()),
            }
        )
    return summaries


def run(args):
    _set_seed(args.seed)
    active_threads = _set_cpu_threads(args.cpu_threads)
    device = _resolve_device(args.device)
    spec = ORIGINAL_TARGET_SPECS[args.target]
    split_cache = {}
    train_ds, test_ds = _get_split(split_cache, spec.dataset_subdir, args.seed)
    loader_kwargs = _loader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    data_ini = _first_batch(train_loader)
    model = OriginalHierNet(
        in_channels=data_ini.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        conv_type=ORIGINAL_CONV_TYPE,
        hls_dim=spec.hls_dim,
        design_edge_dim=data_ini.edge_attr.shape[-1] if getattr(data_ini, "edge_attr", None) is not None else None,
        pool_mode=spec.pool_mode,
        drop_out=args.drop_out,
    ).to(device)
    checkpoint_path = initial_checkpoint_path(args.target, args)
    if checkpoint_path is None:
        raise ValueError("Pass --init-from-builtins or --init-checkpoint-dir")
    init_metadata = load_initial_checkpoint(model, checkpoint_path, device)

    rows = prediction_rows(model, test_loader, device, spec)
    true = torch.tensor([row["true"] for row in rows], dtype=torch.float64)
    pred = torch.tensor([row["pred"] for row in rows], dtype=torch.float64)
    abs_error = torch.tensor([row["abs_error"] for row in rows], dtype=torch.float64)
    signed_error = pred - true
    result = {
        "settings": {
            "target": args.target,
            "metric": spec.metric_name,
            "seed": args.seed,
            "device": str(device),
            "cpu_threads": active_threads,
            "batch_size": args.batch_size,
            "checkpoint": init_metadata,
        },
        "true": summarize_values(true),
        "pred": summarize_values(pred),
        "abs_error": summarize_values(abs_error),
        "signed_error": summarize_values(signed_error),
        "true_buckets": summarize_error_buckets(true, abs_error),
        "mae": float(abs_error.mean().item()),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Collect deterministic prediction statistics for an HGP checkpoint.")
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--drop-out", type=float, default=0.0)
    parser.add_argument("--init-from-builtins", action="store_true")
    parser.add_argument("--init-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
