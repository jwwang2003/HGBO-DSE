#!/usr/bin/env python3
"""Audit raw and PyG power labels used by HGBO-DSE training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POWER_KEYS = ("PWR", "PWR_DYNAMIC", "PWR_STATIC")
IMPL_METRIC_KEYS = ["LUT", "FF", "DSP", "BRAM", "URAM", "SRL", "CP", "PWR", "PWR_DYNAMIC"]
TOTAL_POWER_INDEX = IMPL_METRIC_KEYS.index("PWR")
DYNAMIC_POWER_INDEX = IMPL_METRIC_KEYS.index("PWR_DYNAMIC")


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _audit_raw_ppa(path: Path, *, tolerance: float) -> list[str]:
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"json_read:{exc}"]

    impl = payload.get("IMPL", {})
    values: dict[str, float] = {}
    for key in POWER_KEYS:
        value = _finite_number(impl.get(key))
        if value is None:
            issues.append(f"missing_or_nonfinite:{key}")
            continue
        values[key] = value
        if value < 0.0:
            issues.append(f"negative:{key}")
        if key in {"PWR", "PWR_STATIC"} and value == 0.0:
            issues.append(f"zero:{key}")

    if all(key in values for key in POWER_KEYS):
        total = values["PWR"]
        subtotal = values["PWR_DYNAMIC"] + values["PWR_STATIC"]
        if abs(total - subtotal) > tolerance:
            issues.append(
                "total_mismatch:PWR={:.12g},dynamic+static={:.12g}".format(
                    total,
                    subtotal,
                )
            )
        if values["PWR_DYNAMIC"] > total + tolerance:
            issues.append("dynamic_exceeds_total")
        if total > 100.0:
            issues.append("implausible_total_power_gt_100w")

    return issues


def audit_raw_power_labels(raw_root: Path, *, tolerance: float = 1e-3) -> dict:
    stats = {
        "seen": 0,
        "ok": 0,
        "failed": 0,
        "issues": {},
    }
    for ppa_path in sorted(raw_root.glob("*/*/script/ppa_*.json")):
        stats["seen"] += 1
        issues = _audit_raw_ppa(ppa_path, tolerance=tolerance)
        if issues:
            stats["failed"] += 1
            stats["issues"][str(ppa_path)] = issues
        else:
            stats["ok"] += 1
    return stats


def _audit_sample_y(sample, *, path: Path, sample_index: int) -> list[str]:
    issues: list[str] = []
    y = getattr(sample, "y", None)
    if y is None:
        return [f"{path}:{sample_index}:missing_y"]
    try:
        flat = y.view(-1).detach().cpu().tolist()
    except Exception as exc:
        return [f"{path}:{sample_index}:bad_y:{exc}"]
    needed = max(TOTAL_POWER_INDEX, DYNAMIC_POWER_INDEX)
    if len(flat) <= needed:
        return [f"{path}:{sample_index}:short_y:{len(flat)}"]
    total = _finite_number(flat[TOTAL_POWER_INDEX])
    dynamic = _finite_number(flat[DYNAMIC_POWER_INDEX])
    if total is None:
        issues.append(f"{path}:{sample_index}:nonfinite_total_index_{TOTAL_POWER_INDEX}")
    if dynamic is None:
        issues.append(f"{path}:{sample_index}:nonfinite_dynamic_index_{DYNAMIC_POWER_INDEX}")
    if total is not None and total < 0.0:
        issues.append(f"{path}:{sample_index}:negative_total")
    if dynamic is not None and dynamic < 0.0:
        issues.append(f"{path}:{sample_index}:negative_dynamic")
    if total is not None and dynamic is not None and dynamic > total + 1e-6:
        issues.append(f"{path}:{sample_index}:dynamic_exceeds_total")
    return issues


def _std_shard_paths(dataset_root: Path) -> list[Path]:
    paths = set(dataset_root.glob("**/std/*.pt"))
    paths.update(dataset_root.glob("**/std_arch/*.pt"))
    paths.update(dataset_root.glob("**/std*.pt"))
    return sorted(paths)


def audit_pyg_power_labels(dataset_root: Path) -> dict:
    try:
        import torch
    except ImportError:
        return {
            "seen": 0,
            "ok": 0,
            "failed": 0,
            "issues": {"torch": ["torch_not_installed"]},
        }

    stats = {
        "seen": 0,
        "ok": 0,
        "failed": 0,
        "issues": {},
    }
    for shard_path in _std_shard_paths(dataset_root):
        try:
            samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            stats["failed"] += 1
            stats["issues"][str(shard_path)] = [f"torch_load:{exc}"]
            continue
        if not isinstance(samples, list):
            samples = [samples]
        for idx, sample in enumerate(samples):
            stats["seen"] += 1
            issues = _audit_sample_y(sample, path=shard_path, sample_index=idx)
            if issues:
                stats["failed"] += 1
                stats["issues"].setdefault(str(shard_path), []).extend(issues)
            else:
                stats["ok"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit PWR/PWR_DYNAMIC/PWR_STATIC labels in raw JSON and optional PyG shards."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="absolute watts tolerance for PWR ~= PWR_DYNAMIC + PWR_STATIC",
    )
    args = parser.parse_args(argv)

    report = {
        "raw": audit_raw_power_labels(args.raw_root, tolerance=args.tolerance),
        "label_indices": {
            "PWR": TOTAL_POWER_INDEX,
            "PWR_DYNAMIC": DYNAMIC_POWER_INDEX,
        },
    }
    if args.dataset_root is not None:
        report["pyg"] = audit_pyg_power_labels(args.dataset_root)

    print(json.dumps(report, indent=2, sort_keys=True))
    failed = report["raw"]["failed"]
    if "pyg" in report:
        failed += report["pyg"]["failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
