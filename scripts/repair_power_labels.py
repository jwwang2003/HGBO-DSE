#!/usr/bin/env python3
"""Backfill power fields in raw HGBO-DSE ppa_*.json files from Vivado reports.

The dataset keeps both paper conventions:

* ``PWR`` stores Vivado total on-chip power for HGBO-DSE compatibility.
* ``PWR_DYNAMIC`` stores Vivado dynamic power for ATAPP-style experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HGBO_ROOT = Path(__file__).resolve().parents[1]
if str(HGBO_ROOT) not in sys.path:
    sys.path.insert(0, str(HGBO_ROOT))

from bome.get_ppa import _power_summary_watts


def _power_report_for_ppa(ppa_path: Path) -> Path | None:
    try:
        idx = int(ppa_path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    bench_root = ppa_path.parent.parent
    report_dir = bench_root / f"prj_{idx}" / "report"
    candidates = [
        report_dir / "power.rpt",
        report_dir / "bd_0_wrapper_power_routed.rpt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def repair_power_labels(raw_root: Path, *, dry_run: bool = False) -> dict[str, int]:
    stats = {
        "seen": 0,
        "updated": 0,
        "already_ok": 0,
        "missing_report": 0,
        "unparsed_report": 0,
    }
    for ppa_path in sorted(raw_root.glob("*/*/script/ppa_*.json")):
        stats["seen"] += 1
        report_path = _power_report_for_ppa(ppa_path)
        if report_path is None:
            stats["missing_report"] += 1
            continue
        summary = _power_summary_watts(str(report_path))
        required = {
            "PWR": "Total On-Chip Power",
            "PWR_DYNAMIC": "Dynamic",
            "PWR_STATIC": "Device Static",
        }
        if any(field not in summary for field in required.values()):
            stats["unparsed_report"] += 1
            continue

        payload = json.loads(ppa_path.read_text(encoding="utf-8"))
        impl = payload.setdefault("IMPL", {})
        changed = False
        for key, field in required.items():
            old_power = float(impl.get(key, 0.0) or 0.0)
            new_power = summary[field]
            if abs(old_power - new_power) > 1e-12:
                impl[key] = new_power
                changed = True
        if not changed:
            stats["already_ok"] += 1
            continue
        if not dry_run:
            ppa_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
        stats["updated"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair raw ppa_*.json power labels from Vivado power reports."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = repair_power_labels(args.raw_root, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
