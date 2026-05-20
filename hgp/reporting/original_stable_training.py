from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from bome.pred.checkpoint import load_pretrained_state_dict
from hgp.reporting.compare_training_graphs import (
    HGBO_ROOT,
    ORIGINAL_CONV_TYPE,
    ORIGINAL_TARGET_SPECS,
    TARGETS,
    OriginalHierNet,
    _first_batch,
    _get_split,
    _loader_kwargs,
    _metric_fn,
    _resolve_device,
    _set_cpu_threads,
    _set_seed,
    _target_values,
)


MAE_TARGETS = {"dsp", "bram"}
DEFAULT_STABLE_TARGETS = ["dsp", "bram"]
CHECKPOINT_PREFIX = {
    "lut": "lut",
    "ff": "ff",
    "dsp": "dsp_mae",
    "bram": "bram_mae",
    "cp": "cp_mean",
    "power": "power_mean",
}


@dataclass(frozen=True)
class StepResult:
    applied: bool
    reason: str = ""
    grad_norm: float | None = None


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return HGBO_ROOT / "img" / "training" / f"original_stable_seed128_{stamp}"


def snapshot_model_state(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state(model, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def model_state_all_finite(model) -> bool:
    return all(torch.isfinite(value.detach()).all().item() for value in model.state_dict().values())


def gradients_all_finite(model) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all().item():
            return False
    return True


def safe_optimizer_step(*, loss, model, optimizer, grad_clip, fallback_state=None) -> StepResult:
    if not torch.isfinite(loss.detach()).all().item():
        optimizer.zero_grad(set_to_none=True)
        if fallback_state is not None:
            restore_model_state(model, fallback_state)
        return StepResult(False, "non-finite loss")

    loss.backward()
    if not gradients_all_finite(model):
        optimizer.zero_grad(set_to_none=True)
        if fallback_state is not None:
            restore_model_state(model, fallback_state)
        return StepResult(False, "non-finite gradient")

    grad_norm = None
    if grad_clip is not None and grad_clip > 0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            if fallback_state is not None:
                restore_model_state(model, fallback_state)
            return StepResult(False, "non-finite gradient norm", grad_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not model_state_all_finite(model):
        if fallback_state is not None:
            restore_model_state(model, fallback_state)
        return StepResult(False, "non-finite parameters", grad_norm)
    return StepResult(True, grad_norm=grad_norm)


def target_learning_rate(target: str, args) -> float:
    if target in MAE_TARGETS:
        return args.mae_lr
    return args.lr


def reduce_optimizer_lr(optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] *= factor


def should_decay_learning_rate(*, epoch: int, args) -> bool:
    if args.lr_decay_factor == 1.0:
        return False
    return args.lr_decay_interval > 0 and epoch % args.lr_decay_interval == 0


def test_loader_options(args) -> dict:
    if args.deterministic_eval:
        return {"shuffle": False, "drop_last": False}
    return {"shuffle": True, "drop_last": True}


def transform_targets(true_y: torch.Tensor, args) -> torch.Tensor:
    if args.target_transform == "none":
        return true_y
    if args.target_transform == "log1p":
        return torch.log1p(true_y.clamp_min(0))
    raise ValueError(f"Unknown target transform: {args.target_transform}")


def metric_predictions(out: torch.Tensor, args) -> torch.Tensor:
    if args.target_transform == "none":
        return out
    if args.target_transform == "log1p":
        return torch.expm1(out).clamp_min(0)
    raise ValueError(f"Unknown target transform: {args.target_transform}")


def metric_value(out: torch.Tensor, true_y: torch.Tensor, metric_name: str, args) -> torch.Tensor:
    metric_fn = _metric_fn(metric_name)
    return metric_fn(metric_predictions(out, args), true_y).float()


def loss_weights(true_y: torch.Tensor, args) -> torch.Tensor:
    weights = torch.ones_like(true_y, dtype=torch.float32)
    if args.loss_weighting == "none":
        return weights
    if args.loss_weighting != "bram-tail":
        raise ValueError(f"Unknown loss weighting: {args.loss_weighting}")

    weights = weights.to(device=true_y.device)
    weights[(true_y > 0) & (true_y <= 1)] = args.bram_positive_weight
    weights[(true_y > 1) & (true_y <= 10)] = args.bram_mid_weight
    weights[(true_y > 10) & (true_y <= 100)] = args.bram_high_weight
    weights[true_y > 100] = args.bram_extreme_weight
    return weights


def training_loss(out: torch.Tensor, true_y: torch.Tensor, args) -> torch.Tensor:
    transformed_true_y = transform_targets(true_y, args)
    per_sample_loss = F.huber_loss(out, transformed_true_y, reduction="none").float()
    weights = loss_weights(true_y, args).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_sample_weights(dataset, spec, args) -> torch.Tensor | None:
    if args.train_sampler == "none":
        return None
    if args.train_sampler not in {"bram-bucket-balanced", "bram-nonzero-balanced"}:
        raise ValueError(f"Unknown train sampler: {args.train_sampler}")
    if spec.name != "bram":
        return None

    true_y = torch.tensor([float(_target_values(data, spec).view(-1)[0].item()) for data in dataset], dtype=torch.float64)
    weights = torch.zeros_like(true_y)
    if args.train_sampler == "bram-nonzero-balanced":
        bucket_masks = [true_y == 0, true_y > 0]
    else:
        bucket_masks = [
            true_y == 0,
            (true_y > 0) & (true_y <= 1),
            (true_y > 1) & (true_y <= 10),
            (true_y > 10) & (true_y <= 100),
            true_y > 100,
        ]
    for mask in bucket_masks:
        count = int(mask.sum().item())
        if count > 0:
            weights[mask] = 1.0 / count
    return weights


def make_train_loader(train_ds, spec, args, loader_kwargs):
    sample_weights = train_sample_weights(train_ds, spec, args)
    if sample_weights is None:
        return DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True, **loader_kwargs)


def checkpoint_name(target: str, split: str) -> str:
    return f"{CHECKPOINT_PREFIX[target]}_h64_d0_checkpoint_{split}.pt"


def initial_checkpoint_path(target: str, args) -> Path | None:
    if args.init_checkpoint_dir is None and not args.init_from_builtins:
        return None
    checkpoint_dir = Path(args.init_checkpoint_dir) if args.init_checkpoint_dir else HGBO_ROOT / "hgp" / "model"
    return checkpoint_dir / checkpoint_name(target, "test")


def load_initial_checkpoint(model, checkpoint_path: Path, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_pretrained_state_dict(model, state_dict)
    return {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "best_metric": checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None,
        "legacy_metric": checkpoint.get("min_test_mae", checkpoint.get("min_test_mape")) if isinstance(checkpoint, dict) else None,
        "settings": checkpoint.get("settings") if isinstance(checkpoint, dict) else None,
    }


def save_checkpoint(output_dir: Path, *, target, split, model, optimizer, epoch, metric_name, metric_value, settings):
    checkpoint_dir = output_dir / "hgp" / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / checkpoint_name(target, split)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        f"min_{split}_{metric_name}": metric_value,
        "best_metric": metric_value,
        "metric_name": metric_name,
        "target": target,
        "stable_training": True,
        "settings": settings,
    }
    torch.save(payload, path)
    return str(path)


def train_stable_original_epoch(
    *,
    model,
    loader,
    optimizer,
    device,
    spec,
    epoch,
    grad_clip,
    fallback_state,
    recovery_lr_decay,
    args,
):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    counted_graphs = 0
    skipped_batches = 0
    skip_reasons = {}
    last_finite_state = fallback_state

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index, data.batch, data["hls_attr"], edge_attr=getattr(data, "edge_attr", None)).view(-1)
        true_y = _target_values(data, spec)
        if not torch.isfinite(out.detach()).all().item() or not torch.isfinite(true_y.detach()).all().item():
            skipped_batches += 1
            reason = "non-finite forward"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            restore_model_state(model, last_finite_state)
            reduce_optimizer_lr(optimizer, recovery_lr_decay)
            continue

        loss = training_loss(out, true_y, args)
        metric = metric_value(out, true_y, spec.metric_name, args)
        step = safe_optimizer_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            grad_clip=grad_clip,
            fallback_state=last_finite_state,
        )
        if not step.applied:
            skipped_batches += 1
            skip_reasons[step.reason] = skip_reasons.get(step.reason, 0) + 1
            reduce_optimizer_lr(optimizer, recovery_lr_decay)
            continue

        last_finite_state = snapshot_model_state(model)
        total_loss += loss.item() * data.num_graphs
        total_metric += metric.item() * data.num_graphs
        counted_graphs += data.num_graphs

    if counted_graphs == 0:
        return float("inf"), float("inf"), skipped_batches, skip_reasons, last_finite_state
    return (
        total_loss / counted_graphs,
        total_metric / counted_graphs,
        skipped_batches,
        skip_reasons,
        last_finite_state,
    )


def evaluate_stable_original(model, loader, device, spec, epoch, args):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    metric_name = spec.metric_name
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data["hls_attr"], edge_attr=getattr(data, "edge_attr", None)).view(-1)
            true_y = _target_values(data, spec)
            loss = training_loss(out, true_y, args)
            metric = metric_value(out, true_y, metric_name, args)
            total_loss += loss.item() * data.num_graphs
            total_metric += metric.item() * data.num_graphs
    return total_loss / len(loader.dataset), total_metric / len(loader.dataset)


def run_stable_original_target(target: str, args, device, output_dir: Path, split_cache=None):
    _set_seed(args.seed)
    spec = ORIGINAL_TARGET_SPECS[target]
    if split_cache is None:
        split_cache = {}
    train_ds, test_ds = _get_split(split_cache, spec.dataset_subdir, args.seed)
    loader_kwargs = _loader_kwargs(args)
    train_loader = make_train_loader(train_ds, spec, args, loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, **test_loader_options(args), **loader_kwargs)
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
    init_checkpoint = initial_checkpoint_path(target, args)
    init_metadata = None
    if init_checkpoint is not None:
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        init_metadata = load_initial_checkpoint(model, init_checkpoint, device)
    lr = target_learning_rate(target, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    settings = {
        "target": target,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": lr,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "conv_type": ORIGINAL_CONV_TYPE,
        "recovery_lr_decay": args.recovery_lr_decay,
        "lr_decay_factor": args.lr_decay_factor,
        "lr_decay_interval": args.lr_decay_interval,
        "init_checkpoint": init_metadata,
        "deterministic_eval": args.deterministic_eval,
        "target_transform": args.target_transform,
        "train_sampler": args.train_sampler,
        "loss_weighting": args.loss_weighting,
        "bram_positive_weight": args.bram_positive_weight,
        "bram_mid_weight": args.bram_mid_weight,
        "bram_high_weight": args.bram_high_weight,
        "bram_extreme_weight": args.bram_extreme_weight,
    }
    history = []
    checkpoints = {}
    best_train_metric = float("inf")
    best_test_metric = float("inf")
    last_finite_state = snapshot_model_state(model)

    if init_metadata is not None:
        train_loss, train_metric = evaluate_stable_original(model, train_loader, device, spec, -1, args)
        test_loss, test_metric = evaluate_stable_original(model, test_loader, device, spec, -1, args)
        if math.isfinite(train_metric):
            best_train_metric = train_metric
            checkpoints["train"] = save_checkpoint(
                output_dir,
                target=target,
                split="train",
                model=model,
                optimizer=optimizer,
                epoch=-1,
                metric_name=spec.metric_name,
                metric_value=best_train_metric,
                settings=settings,
            )
        if math.isfinite(test_metric):
            best_test_metric = test_metric
            checkpoints["test"] = save_checkpoint(
                output_dir,
                target=target,
                split="test",
                model=model,
                optimizer=optimizer,
                epoch=-1,
                metric_name=spec.metric_name,
                metric_value=best_test_metric,
                settings=settings,
            )
        history.append(
            {
                "epoch": -1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "test_metric": test_metric,
                "best_test_metric": best_test_metric,
                "skipped_batches": 0,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "Epoch: init, {0} train={1:.6g}, test={2:.6g}, best_test={3:.6g}".format(
                spec.metric_name.upper(),
                train_metric,
                test_metric,
                best_test_metric,
            )
        )

    for epoch in range(args.epochs):
        train_loss, train_metric, skipped_batches, skip_reasons, last_finite_state = train_stable_original_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            spec=spec,
            epoch=epoch,
            grad_clip=args.grad_clip,
            fallback_state=last_finite_state,
            recovery_lr_decay=args.recovery_lr_decay,
            args=args,
        )
        if skipped_batches:
            print(f"{target} epoch {epoch}: skipped {skipped_batches} unstable batch(es): {skip_reasons}")
        test_loss, test_metric = evaluate_stable_original(model, test_loader, device, spec, epoch, args)
        if should_decay_learning_rate(epoch=epoch, args=args):
            reduce_optimizer_lr(optimizer, args.lr_decay_factor)
        if math.isfinite(train_metric) and train_metric < best_train_metric:
            best_train_metric = train_metric
            checkpoints["train"] = save_checkpoint(
                output_dir,
                target=target,
                split="train",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=spec.metric_name,
                metric_value=best_train_metric,
                settings=settings,
            )
        if math.isfinite(test_metric) and test_metric < best_test_metric:
            best_test_metric = test_metric
            checkpoints["test"] = save_checkpoint(
                output_dir,
                target=target,
                split="test",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=spec.metric_name,
                metric_value=best_test_metric,
                settings=settings,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_metric": train_metric,
                "test_metric": test_metric,
                "best_test_metric": best_test_metric,
                "skipped_batches": skipped_batches,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "Epoch: {0:03d}, {1} train={2:.6g}, test={3:.6g}, best_test={4:.6g}".format(
                epoch,
                spec.metric_name.upper(),
                train_metric,
                test_metric,
                best_test_metric,
            )
        )

    return {
        "metric_name": spec.metric_name,
        "best_train_metric": best_train_metric,
        "best_test_metric": best_test_metric,
        "checkpoints": checkpoints,
        "history": history,
        "settings": settings,
    }


def write_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "history.csv"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "epoch",
                "train_loss",
                "test_loss",
                "train_metric",
                "test_metric",
                "best_test_metric",
                "skipped_batches",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for target, target_result in results["targets"].items():
            for row in target_result["history"]:
                writer.writerow({"target": target, **row})
    return {"summary": str(summary_path), "history": str(csv_path)}


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_threads = _set_cpu_threads(args.cpu_threads)
    device = _resolve_device(args.device)
    split_cache = {}
    results = {
        "settings": {
            "targets": args.targets,
            "epochs": args.epochs,
            "seed": args.seed,
            "lr": args.lr,
            "mae_lr": args.mae_lr,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "device": str(device),
            "cpu_threads": active_threads,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "conv_type": ORIGINAL_CONV_TYPE,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_decay_interval": args.lr_decay_interval,
            "init_from_builtins": args.init_from_builtins,
            "init_checkpoint_dir": str(args.init_checkpoint_dir) if args.init_checkpoint_dir else None,
            "deterministic_eval": args.deterministic_eval,
            "target_transform": args.target_transform,
            "train_sampler": args.train_sampler,
            "loss_weighting": args.loss_weighting,
            "bram_positive_weight": args.bram_positive_weight,
            "bram_mid_weight": args.bram_mid_weight,
            "bram_high_weight": args.bram_high_weight,
            "bram_extreme_weight": args.bram_extreme_weight,
        },
        "targets": {},
    }
    print(f"OUT={output_dir}")
    print(f"device={device} cpu_threads={active_threads}")
    for target in args.targets:
        print(f"Training stable original target {target}")
        results["targets"][target] = run_stable_original_target(target, args, device, output_dir, split_cache=split_cache)
    paths = write_results(results, output_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    for target, target_result in results["targets"].items():
        metric = target_result["metric_name"].upper()
        print(f"{target}: best test {metric} = {target_result['best_test_metric']:.6g}")
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Train original HGBO-DSE HGP with finite-value safeguards.")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=DEFAULT_STABLE_TARGETS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--mae-lr", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--recovery-lr-decay", type=float, default=0.5)
    parser.add_argument("--lr-decay-factor", type=float, default=0.9)
    parser.add_argument("--lr-decay-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--drop-out", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")
    parser.add_argument("--train-sampler", choices=["none", "bram-bucket-balanced", "bram-nonzero-balanced"], default="none")
    parser.add_argument("--loss-weighting", choices=["none", "bram-tail"], default="none")
    parser.add_argument("--bram-positive-weight", type=float, default=2.0)
    parser.add_argument("--bram-mid-weight", type=float, default=4.0)
    parser.add_argument("--bram-high-weight", type=float, default=16.0)
    parser.add_argument("--bram-extreme-weight", type=float, default=32.0)
    parser.add_argument("--init-from-builtins", action="store_true")
    parser.add_argument("--init-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use shuffle=False and drop_last=False for the test loader instead of the paper-style shuffled/drop-last test loader.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
