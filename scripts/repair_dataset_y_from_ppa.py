#!/usr/bin/env python3
"""Refresh PyG sample ``y`` labels from repaired raw ppa_*.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HGBO_ROOT = Path(__file__).resolve().parents[1]
if str(HGBO_ROOT) not in sys.path:
    sys.path.insert(0, str(HGBO_ROOT))

from hgp.board_utils import equivalent_device_names, normalize_device_name
from hgp.data_process.gen_dataset_std import IMPL_METRIC_KEYS


def _candidate_ppa_paths(raw_root: Path, bench_name: str, board_device: str, prj_name: str) -> list[Path]:
    try:
        idx = int(str(prj_name).split("_", 1)[1])
    except (IndexError, ValueError):
        return []
    return [
        raw_root / bench_name / board_name / "script" / f"ppa_{idx}.json"
        for board_name in equivalent_device_names(board_device)
    ]


def _label_from_ppa(ppa_path: Path) -> torch.Tensor:
    payload = json.loads(ppa_path.read_text(encoding="utf-8"))
    impl = payload.get("IMPL", {})
    values = [float(impl.get(key, 0.0)) for key in IMPL_METRIC_KEYS]
    return torch.tensor([values], dtype=torch.float32)


def repair_dataset_y(dataset_root: Path, raw_root: Path, *, dry_run: bool = False) -> dict[str, int]:
    stats = {
        "shards_seen": 0,
        "shards_updated": 0,
        "samples_seen": 0,
        "samples_updated": 0,
        "missing_metadata": 0,
        "missing_ppa": 0,
    }
    shard_roots = [
        dataset_root / "std",
        dataset_root / "std_arch",
    ]
    shard_roots.extend(dataset_root.glob("*/std"))
    shard_roots.extend(dataset_root.glob("*/std_arch"))

    seen_paths: set[Path] = set()
    for shard_root in sorted({path for path in shard_roots if path.is_dir()}):
        for shard_path in sorted(shard_root.glob("*.pt")):
            if shard_path in seen_paths:
                continue
            seen_paths.add(shard_path)
            stats["shards_seen"] += 1
            samples = torch.load(shard_path, map_location="cpu", weights_only=False)
            changed = False
            for sample in samples:
                stats["samples_seen"] += 1
                bench_name = getattr(sample, "bench_name", None)
                prj_name = getattr(sample, "prj_name", None)
                board_device = getattr(sample, "board_device", None)
                if not bench_name or not prj_name or not board_device:
                    stats["missing_metadata"] += 1
                    continue
                candidates = _candidate_ppa_paths(
                    raw_root,
                    str(bench_name),
                    normalize_device_name(str(board_device)),
                    str(prj_name),
                )
                ppa_path = next((path for path in candidates if path.is_file()), None)
                if ppa_path is None:
                    stats["missing_ppa"] += 1
                    continue
                new_y = _label_from_ppa(ppa_path)
                old_y = sample.y.detach().cpu().to(dtype=torch.float32)
                if old_y.shape != new_y.shape or not torch.equal(old_y, new_y):
                    sample.y = new_y
                    changed = True
                    stats["samples_updated"] += 1
            if changed:
                stats["shards_updated"] += 1
                if not dry_run:
                    torch.save(samples, shard_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair PyG y labels from raw ppa_*.json files.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = repair_dataset_y(args.dataset_root, args.raw_root, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
