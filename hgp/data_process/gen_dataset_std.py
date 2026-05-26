"""Build std/ and rdc/ PyG dataset shards from raw HGBO-DSE exports.

Dynamically discovers all ``<raw_root>/<bench>/<ver>/`` directories, infers
the top function name from the shortest ``.adb`` filename in each project's
``graph/`` directory, and produces per-board shards under
``<output_root>/<device>/std/`` and ``<output_root>/<device>/rdc/``.

Usage (from the HGBO-DSE repo root):

    uv run python hgp/data_process/gen_dataset_std.py

Or with explicit paths:

    uv run python hgp/data_process/gen_dataset_std.py \\
        --raw-root dataset/raw \\
        --output-root dataset \\
        --device xc7vx485tffg1761-2

When ``--device`` is omitted, the script processes ALL bench/ver combinations
found under ``--raw-root`` and infers the device from the ``ver`` directory
name (which matches the canonical device key used by the HLS export pipeline).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Support running from hgp/data_process/ directly.
_THIS_DIR = Path(__file__).resolve().parent
_HGBO_ROOT = _THIS_DIR.parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_HGBO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HGBO_ROOT))

import networkx as nx
import torch

from graph_construct import CDFG
from feature_encode import generate_pyg_dot
from gen_dataframe import generate_dataframe

# Node feature sets matching the original HGBO-DSE convention.
N_NUM_ITEMS_STD = ["m_delay", "latency", "bitwidth", "lut", "ff", "dsp"]
N_NUM_ITEMS_RDC = ["m_delay", "latency"]

DEFAULT_RAW_ROOT = _HGBO_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT_ROOT = _HGBO_ROOT / "dataset"


def _infer_top_name(graph_dir: Path) -> str | None:
    """Infer the HLS top function name from the shortest .adb filename.

    Convention: Vitis HLS names the top-level .adb after the top function
    (e.g. ``atax.adb``). Sub-module and pipeline .adb files have longer names
    with suffixes like ``_Pipeline_lp1.adb`` or ``atax.bind.adb``. The shortest
    stem (excluding ``.bind.`` and ``.sched.`` variants) is the top function.
    """
    adb_files = sorted(graph_dir.glob("*.adb"))
    candidates: list[str] = []
    for adb in adb_files:
        stem = adb.stem
        if ".bind" in adb.name or ".sched" in adb.name:
            continue
        if "_Pipeline_" in stem:
            continue
        candidates.append(stem)
    if not candidates:
        return None
    return min(candidates, key=len)


def _discover_bench_vers(raw_root: Path) -> list[tuple[str, str, Path]]:
    """Walk raw_root and return (bench, ver, path) for each bench/ver directory."""
    results: list[tuple[str, str, Path]] = []
    if not raw_root.is_dir():
        return results
    for bench_dir in sorted(raw_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for ver_dir in sorted(bench_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            ver = ver_dir.name
            results.append((bench, ver, ver_dir))
    return results


def process_bench_ver(
    bench: str,
    ver: str,
    bench_path: Path,
    *,
    output_root: Path,
    board_device: str | None = None,
    layout: str = "per_device",
    emit_dot: bool = False,
) -> dict[str, int]:
    """Process one bench/ver directory into std and rdc shards.

    Returns counts of successfully processed samples per split.
    """
    script_dir = bench_path / "script"
    ppa_files = sorted(glob.glob(str(script_dir / "ppa_*.json")))
    sample_count = len(ppa_files)
    if sample_count == 0:
        print(f"  [{bench}/{ver}] no ppa_*.json found, skipping")
        return {"std": 0, "rdc": 0}

    std_samples: list = []
    rdc_samples: list = []
    skipped = 0

    for idx in range(sample_count):
        prj_name = f"prj_{idx}"
        ppa_path = script_dir / f"ppa_{idx}.json"
        graph_dir = bench_path / prj_name / "graph"

        if not graph_dir.is_dir():
            skipped += 1
            continue
        if not ppa_path.is_file():
            skipped += 1
            continue

        top_name = _infer_top_name(graph_dir)
        if top_name is None:
            skipped += 1
            continue

        cdfg_dir = bench_path / prj_name / "cdfg"
        cdfg_dir.mkdir(parents=True, exist_ok=True)

        try:
            graph = CDFG(str(graph_dir), str(cdfg_dir), top_name)
        except Exception as exc:
            print(f"  [{bench}/{ver}/{prj_name}] CDFG failed: {exc}")
            skipped += 1
            continue

        DG = graph.G
        if DG is None or DG.number_of_nodes() == 0:
            skipped += 1
            continue

        with open(ppa_path, "r") as f:
            ppa_info = json.load(f)

        dict_metric = ppa_info.get("IMPL", {})
        dict_hls = ppa_info.get("HLS", {})
        metric_list = list(dict_metric.values())
        hls_attr_std = list(dict_hls.values())
        hls_attr_rdc = [hls_attr_std[-1]] if hls_attr_std else [0.0]

        # Standard features (6 node features)
        std_dot_path = str(cdfg_dir / "std_pyg_G.dot") if emit_dot else None
        std_df_path = str(cdfg_dir / "std_pyg_G.pt")
        try:
            std_pyg = generate_pyg_dot(DG, std_dot_path, N_NUM_ITEMS_STD)
            std_pyg = nx.convert_node_labels_to_integers(std_pyg)
            std_df = generate_dataframe(
                std_pyg, metric_list, hls_attr_std, bench, prj_name, std_df_path,
                board_device=board_device,
            )
            std_samples.append(std_df)
        except Exception as exc:
            print(f"  [{bench}/{ver}/{prj_name}] std failed: {exc}")
            skipped += 1
            continue

        # Reduced features (2 node features, for CP prediction)
        rdc_dot_path = str(cdfg_dir / "rdc_pyg_G.dot") if emit_dot else None
        rdc_df_path = str(cdfg_dir / "rdc_pyg_G.pt")
        try:
            rdc_pyg = generate_pyg_dot(DG, rdc_dot_path, N_NUM_ITEMS_RDC)
            rdc_pyg = nx.convert_node_labels_to_integers(rdc_pyg)
            rdc_df = generate_dataframe(
                rdc_pyg, metric_list, hls_attr_rdc, bench, prj_name, rdc_df_path,
                board_device=board_device,
            )
            rdc_samples.append(rdc_df)
        except Exception as exc:
            print(f"  [{bench}/{ver}/{prj_name}] rdc failed: {exc}")

    # Write shards.
    device_key = board_device or ver
    if layout == "per_device":
        std_dir = output_root / device_key / "std"
        rdc_dir = output_root / device_key / "rdc"
    else:
        std_dir = output_root / "std"
        rdc_dir = output_root / "rdc"

    std_dir.mkdir(parents=True, exist_ok=True)
    rdc_dir.mkdir(parents=True, exist_ok=True)

    shard_name = f"{bench}.pt"
    if std_samples:
        torch.save(std_samples, std_dir / shard_name)
    if rdc_samples:
        torch.save(rdc_samples, rdc_dir / shard_name)

    print(
        f"  [{bench}/{ver}] {len(std_samples)} std, {len(rdc_samples)} rdc "
        f"({skipped} skipped) → {device_key}/"
    )
    return {"std": len(std_samples), "rdc": len(rdc_samples)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build std/rdc PyG dataset shards from raw HGBO-DSE exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="root directory containing <bench>/<ver>/ subdirs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory where std/ and rdc/ shards are written",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override board_device tag on all samples (default: infer from ver name)",
    )
    parser.add_argument(
        "--bench",
        default=None,
        help="process only this benchmark (default: all found under raw-root)",
    )
    parser.add_argument(
        "--ver",
        default=None,
        help="process only this version/device (default: all found under each bench)",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="write shards to flat std/ and rdc/ (no per-device subdirs)",
    )
    parser.add_argument(
        "--emit-dot",
        action="store_true",
        help="write graphviz .dot files for each sample (slow, for debugging only)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    layout = "legacy" if args.legacy_layout else "per_device"

    discoveries = _discover_bench_vers(args.raw_root)
    if not discoveries:
        print(f"No bench/ver directories found under {args.raw_root}")
        sys.exit(1)

    # Filter if requested.
    if args.bench:
        discoveries = [(b, v, p) for b, v, p in discoveries if b == args.bench]
    if args.ver:
        discoveries = [(b, v, p) for b, v, p in discoveries if v == args.ver]

    print(f"Found {len(discoveries)} bench/ver combinations under {args.raw_root}")
    total_std = 0
    total_rdc = 0

    for bench, ver, bench_path in discoveries:
        board_device = args.device or ver
        counts = process_bench_ver(
            bench,
            ver,
            bench_path,
            output_root=args.output_root,
            board_device=board_device,
            layout=layout,
            emit_dot=args.emit_dot,
        )
        total_std += counts["std"]
        total_rdc += counts["rdc"]

    print(f"\nDone. Total: {total_std} std samples, {total_rdc} rdc samples.")


if __name__ == "__main__":
    main()
