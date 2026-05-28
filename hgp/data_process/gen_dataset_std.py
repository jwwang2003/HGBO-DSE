"""Build std/ and rdc/ PyG dataset shards from raw HGBO-DSE exports.

Dynamically discovers all ``<raw_root>/<bench>/<ver>/`` directories, infers
the top function name from the shortest ``.adb`` filename in each project's
``graph/`` directory, and produces per-board shards under
``<output_root>/<device>/std/`` and ``<output_root>/<device>/rdc/``.

Usage (from the HGBO-DSE repo root):

    uv run python hgp/data_process/gen_dataset_std.py

Or with explicit paths and parallelism:

    uv run python hgp/data_process/gen_dataset_std.py \\
        --raw-root dataset/raw \\
        --output-root dataset \\
        --device xc7vx485tffg1761-2 \\
        --n-jobs 32

When ``--device`` is omitted, the script processes ALL bench/ver combinations
found under ``--raw-root`` and infers the device from the ``ver`` directory
name (which matches the canonical device key used by the HLS export pipeline).

Parallelism: each ``prj_<idx>/`` is independent (own input/output dirs), so
the work scales near-linearly until memory bandwidth saturates. Default
``--n-jobs`` is ``os.cpu_count()``; pass ``--n-jobs 1`` for serial execution
(useful for debugging tracebacks).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Support running from hgp/data_process/ directly.
_THIS_DIR = Path(__file__).resolve().parent
_HGBO_ROOT = _THIS_DIR.parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_HGBO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HGBO_ROOT))

from hgp.data_process.operator_activity import (
    clean_cell as _clean_cell,
    hardware_operator_key as _hardware_operator_key,
    mean_metric as _mean_metric,
    metric_pair as _metric_pair,
    node_activity_from_cdfg_csv as _node_activity_from_cdfg_csv,
    node_activity_from_operator_merge as _node_activity_from_operator_merge,
)

# Node feature sets matching the original HGBO-DSE convention, extended with
# ATAPP-style operator sharing and node switching features.
N_NUM_ITEMS_STD = [
    "m_delay",
    "latency",
    "bitwidth",
    "lut",
    "ff",
    "dsp",
    "merged_node_count",
    "activity_sa",
    "activity_ar",
]
N_NUM_ITEMS_RDC = ["m_delay", "latency", "merged_node_count", "activity_sa", "activity_ar"]
IMPL_METRIC_KEYS = ["LUT", "FF", "DSP", "BRAM", "URAM", "SRL", "CP", "PWR", "PWR_DYNAMIC"]
HLS_ATTR_KEYS = ["LUT", "FF", "DSP", "BRAM", "URAM", "CP"]

DEFAULT_RAW_ROOT = _HGBO_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT_ROOT = _HGBO_ROOT / "dataset"
ACTIVITY_MODES = ("auto", "none", "node", "opcode", "shuffled")
OPERATOR_MERGE_MODES = ("none", "activity", "graph")


def _ensure_paths() -> None:
    """Make hgp/data_process and hgp/ importable (called in spawn workers)."""
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    if str(_HGBO_ROOT) not in sys.path:
        sys.path.insert(0, str(_HGBO_ROOT))


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


def _stable_activity_seed(seed_parts: tuple) -> int:
    payload = "\0".join(str(part) for part in seed_parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _shuffle_activity_by_opcode(sa_by_opcode: dict, *, seed_parts: tuple) -> dict:
    """Shuffle SA/AR metric dictionaries across opcodes deterministically."""
    keys = sorted(sa_by_opcode)
    values = [dict(sa_by_opcode[key]) for key in keys]
    random.Random(_stable_activity_seed(seed_parts)).shuffle(values)
    return {key: value for key, value in zip(keys, values)}


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_numeric(nodes: list[dict], key: str, mode: str = "max") -> str:
    values = [_to_float(node.get(key), 0.0) for node in nodes]
    if not values:
        return "0"
    if mode == "min":
        return str(min(values))
    if mode == "mean":
        return str(sum(values) / len(values))
    return str(max(values))


def _merge_latency(nodes: list[dict]) -> list[str]:
    starts: list[float] = []
    ends: list[float] = []
    for node in nodes:
        latency = node.get("latency")
        if isinstance(latency, (list, tuple)) and len(latency) >= 2:
            starts.append(_to_float(latency[0], 0.0))
            ends.append(_to_float(latency[1], 0.0))
    if not starts:
        return []
    return [str(min(starts)), str(max(ends))]


def _common_or_first(nodes: list[dict], key: str, default: str = "not_exist") -> str:
    values = [_clean_cell(node.get(key)) for node in nodes if _clean_cell(node.get(key))]
    if not values:
        return default
    first = values[0]
    if all(value == first for value in values):
        return first
    return first


def _merged_operator_attrs(rep_id: str, nodes: list[dict], operator_key: str) -> dict:
    attrs = dict(nodes[0])
    attrs["node_name"] = _common_or_first(nodes, "node_name", default=rep_id)
    attrs["node_type"] = "0"
    attrs["line_num"] = _merge_numeric(nodes, "line_num", mode="min")
    attrs["rtl_name"] = operator_key.split(":", 1)[1] if operator_key.startswith("rtl:") else attrs.get("rtl_name", "not_exist")
    attrs["op_type"] = _common_or_first(nodes, "op_type")
    attrs["core_name"] = _common_or_first(nodes, "core_name")
    attrs["bitwidth"] = _merge_numeric(nodes, "bitwidth", mode="max")
    attrs["opcode"] = _common_or_first(nodes, "opcode", default="none")
    attrs["m_delay"] = _merge_numeric(nodes, "m_delay", mode="max")
    attrs["topo_index"] = _merge_numeric(nodes, "topo_index", mode="min")
    attrs["oprand_edges"] = []
    attrs["latency"] = _merge_latency(nodes)
    for resource_key in ("lut", "ff", "dsp", "bram", "uram"):
        attrs[resource_key] = _merge_numeric(nodes, resource_key, mode="max")
    attrs["operator_key"] = operator_key
    attrs["merged_node_count"] = len(nodes)
    return attrs


def _merge_graph_by_hardware_operator(DG):
    """Collapse CDFG nodes sharing the same RTL operator into one graph node."""
    import networkx as nx

    operator_to_nodes: dict[str, list[str]] = defaultdict(list)
    node_to_rep: dict[str, str] = {}
    for node_id, node in DG.nodes(data=True):
        node_type = _clean_cell(node.get("node_type"))
        if node_type == "0":
            operator_key = _hardware_operator_key(node, node_id=str(node_id))
        else:
            operator_key = "node:{}".format(node_id)
        operator_to_nodes[operator_key].append(node_id)

    for operator_key, node_ids in operator_to_nodes.items():
        node_to_rep.update({node_id: node_ids[0] for node_id in node_ids})

    merged = nx.DiGraph()
    for operator_key, node_ids in operator_to_nodes.items():
        rep_id = node_ids[0]
        nodes = [DG.nodes[node_id] for node_id in node_ids]
        if operator_key.startswith("rtl:") and len(node_ids) > 1:
            merged.add_node(rep_id, **_merged_operator_attrs(rep_id, nodes, operator_key))
        else:
            attrs = dict(nodes[0])
            attrs["operator_key"] = operator_key
            attrs["merged_node_count"] = len(node_ids)
            merged.add_node(rep_id, **attrs)

    for src, dst, edge in DG.edges(data=True):
        new_src = node_to_rep.get(src, src)
        new_dst = node_to_rep.get(dst, dst)
        if new_src == new_dst:
            continue
        attrs = {
            "edge_id": edge.get("edge_id", "0"),
            "edge_type": edge.get("edge_type", "0"),
            "is_back_edge": edge.get("is_back_edge", "0"),
        }
        if merged.has_edge(new_src, new_dst):
            old = merged.edges[new_src, new_dst]
            old["edge_type"] = str(max(_to_float(old.get("edge_type")), _to_float(attrs["edge_type"])))
            old["is_back_edge"] = str(max(_to_float(old.get("is_back_edge")), _to_float(attrs["is_back_edge"])))
        else:
            merged.add_edge(new_src, new_dst, **attrs)
    return merged


def _remap_activity_to_merged_graph(activity_payload: dict | None, original_DG, merged_DG) -> dict | None:
    if not activity_payload or "by_node_id" not in activity_payload:
        return activity_payload
    by_node_id = activity_payload.get("by_node_id") or {}
    if not by_node_id:
        return activity_payload

    by_operator: dict[str, list[dict[str, float]]] = defaultdict(list)
    for node_id, metrics in by_node_id.items():
        if node_id not in original_DG.nodes:
            continue
        operator_key = _hardware_operator_key(original_DG.nodes[node_id], node_id=str(node_id))
        metric_pair = _metric_pair(metrics)
        if metric_pair is not None:
            by_operator[operator_key].append(metric_pair)

    merged_by_node: dict[str, dict[str, float]] = {}
    for node_id, node in merged_DG.nodes(data=True):
        operator_key = node.get("operator_key") or _hardware_operator_key(node, node_id=str(node_id))
        if operator_key in by_operator:
            merged_by_node[str(node_id)] = _mean_metric(by_operator[operator_key])

    return {**activity_payload, "by_node_id": merged_by_node}


def _load_activity_payload(
    top_name: str,
    *,
    activity_mode: str,
    seed_parts: tuple = (),
    cache_dir: Path | None = None,
) -> dict | None:
    """Load optional ATAPP-style switching activity according to ablation mode."""
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError("unknown activity_mode {!r}".format(activity_mode))
    if activity_mode == "none":
        return None

    cache_root = cache_dir or (_HGBO_ROOT / "dataset" / "switching_activity")
    cache_file = cache_root / f"{top_name}_switching_activity.json"
    if not cache_file.exists():
        return None
    try:
        sa_data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    sa_by_opcode = sa_data.get("by_opcode", {}) or {}
    sa_by_node = sa_data.get("by_node_id", {}) or {}
    if activity_mode == "shuffled":
        if not sa_by_opcode:
            return None
        return {
            "by_opcode": _shuffle_activity_by_opcode(
                sa_by_opcode,
                seed_parts=(top_name, *seed_parts),
            )
        }

    if activity_mode == "opcode":
        return {"by_opcode": sa_by_opcode} if sa_by_opcode else None
    if activity_mode == "node":
        return {"by_node_id": sa_by_node} if sa_by_node else None
    if not sa_by_node and not sa_by_opcode:
        return None
    return {
        "by_node_id": sa_by_node,
        "by_opcode": sa_by_opcode,
        "by_op_id": sa_data.get("by_op_id", {}) or {},
    }


def _load_activity_by_opcode(
    top_name: str,
    *,
    activity_mode: str,
    seed_parts: tuple = (),
    cache_dir: Path | None = None,
) -> dict | None:
    """Backward-compatible loader returning opcode-level SA/AR only."""
    payload = _load_activity_payload(
        top_name,
        activity_mode=activity_mode,
        seed_parts=seed_parts,
        cache_dir=cache_dir,
    )
    if payload is None:
        return None
    if "by_opcode" in payload:
        return payload.get("by_opcode") or None
    return None


def _process_one_sample(task: tuple) -> tuple:
    """Worker: process a single prj_<idx>/ directory.

    Each project's std/rdc dataframes are written to disk inside ``cdfg/``
    by ``generate_dataframe``; the worker only returns lightweight path
    strings and a status, so no torch tensors cross the IPC boundary
    (avoiding ``received 0 items of ancdata`` FD-sharing issues on Linux).

    Args:
        task: ``(bench, ver, idx, bench_path_str, board_device, emit_dot,
        timeout_s, activity_mode, operator_merge, activity_cache_dir_str)``.

    Returns:
        ``(bench, ver, idx, std_path, rdc_path, status)`` where ``std_path``
        / ``rdc_path`` are absolute paths to ``.pt`` files (or None) and
        ``status`` is ``"ok"``, ``"skip:<why>"``, or ``"<stage>_fail:<exc>"``.
    """
    _ensure_paths()
    import signal
    import networkx as nx
    from graph_construct import CDFG
    from feature_encode import generate_pyg_dot
    from gen_dataframe import generate_dataframe

    (
        bench,
        ver,
        idx,
        bench_path_str,
        board_device,
        emit_dot,
        timeout_s,
        activity_mode,
        operator_merge,
        activity_cache_dir_str,
    ) = task
    if operator_merge not in OPERATOR_MERGE_MODES:
        return (bench, ver, idx, None, None, "config_fail:unknown_operator_merge")

    # Per-sample SIGALRM watchdog. Pathological .adb files can wedge the
    # XML parser or CDFG construction; without this, one hang stalls a
    # worker forever and the chunk never completes.
    def _on_alarm(signum, frame):
        raise TimeoutError(f"sample exceeded {timeout_s}s")

    if timeout_s and timeout_s > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(timeout_s))
    bench_path = Path(bench_path_str)

    prj_name = f"prj_{idx}"
    ppa_path = bench_path / "script" / f"ppa_{idx}.json"
    graph_dir = bench_path / prj_name / "graph"
    cdfg_dir = bench_path / prj_name / "cdfg"

    std_path: str | None = None
    rdc_path: str | None = None

    try:
        if not graph_dir.is_dir():
            return (bench, ver, idx, None, None, "skip:no_graph_dir")
        if not ppa_path.is_file():
            return (bench, ver, idx, None, None, "skip:no_ppa")

        top_name = _infer_top_name(graph_dir)
        if top_name is None:
            return (bench, ver, idx, None, None, "skip:no_top")

        activity_payload = _load_activity_payload(
            top_name,
            activity_mode=activity_mode,
            seed_parts=(bench, ver, idx),
            cache_dir=Path(activity_cache_dir_str) if activity_cache_dir_str else None,
        )

        cdfg_dir.mkdir(parents=True, exist_ok=True)

        try:
            graph = CDFG(str(graph_dir), str(cdfg_dir), top_name)
        except TimeoutError:
            raise
        except Exception as exc:
            return (bench, ver, idx, None, None, f"cdfg_fail:{exc}")

        if activity_payload and activity_mode == "auto":
            if operator_merge in {"activity", "graph"}:
                sample_node_activity = _node_activity_from_operator_merge(
                    by_op_id=activity_payload.get("by_op_id", {}),
                    by_opcode=activity_payload.get("by_opcode", {}),
                    cdfg_node_csv=cdfg_dir / "cdfg_node_dict.csv",
                )
            else:
                sample_node_activity = _node_activity_from_cdfg_csv(
                    activity_payload.get("by_op_id", {}),
                    cdfg_dir / "cdfg_node_dict.csv",
                )
            if sample_node_activity:
                activity_payload = {
                    **activity_payload,
                    "by_node_id": sample_node_activity,
                }

        DG = graph.G
        if DG is None or DG.number_of_nodes() == 0:
            return (bench, ver, idx, None, None, "skip:empty_graph")
        if operator_merge == "graph":
            original_DG = DG
            DG = _merge_graph_by_hardware_operator(DG)
            activity_payload = _remap_activity_to_merged_graph(activity_payload, original_DG, DG)

        try:
            with open(ppa_path, "r") as f:
                ppa_info = json.load(f)
        except Exception as exc:
            return (bench, ver, idx, None, None, f"ppa_read_fail:{exc}")

        dict_metric = ppa_info.get("IMPL", {})
        dict_hls = ppa_info.get("HLS", {})
        metric_list = [float(dict_metric.get(key, 0.0)) for key in IMPL_METRIC_KEYS]
        hls_attr_std = [float(dict_hls.get(key, 0.0)) for key in HLS_ATTR_KEYS]
        hls_attr_rdc = [hls_attr_std[-1]] if hls_attr_std else [0.0]

        std_dot_path = str(cdfg_dir / "std_pyg_G.dot") if emit_dot else None
        std_df_path = str(cdfg_dir / "std_pyg_G.pt")
        try:
            std_pyg = generate_pyg_dot(DG, std_dot_path, N_NUM_ITEMS_STD, sa_by_opcode=activity_payload)
            std_pyg = nx.convert_node_labels_to_integers(std_pyg)
            generate_dataframe(
                std_pyg, metric_list, hls_attr_std, bench, prj_name, std_df_path,
                board_device=board_device,
            )
            std_path = std_df_path
        except TimeoutError:
            raise
        except Exception as exc:
            return (bench, ver, idx, None, None, f"std_fail:{exc}")

        rdc_dot_path = str(cdfg_dir / "rdc_pyg_G.dot") if emit_dot else None
        rdc_df_path = str(cdfg_dir / "rdc_pyg_G.pt")
        try:
            rdc_pyg = generate_pyg_dot(DG, rdc_dot_path, N_NUM_ITEMS_RDC, sa_by_opcode=activity_payload)
            rdc_pyg = nx.convert_node_labels_to_integers(rdc_pyg)
            generate_dataframe(
                rdc_pyg, metric_list, hls_attr_rdc, bench, prj_name, rdc_df_path,
                board_device=board_device,
            )
            rdc_path = rdc_df_path
        except TimeoutError:
            raise
        except Exception as exc:
            return (bench, ver, idx, std_path, None, f"rdc_fail:{exc}")

        return (bench, ver, idx, std_path, rdc_path, "ok")
    except TimeoutError as exc:
        return (bench, ver, idx, std_path, rdc_path, f"timeout:{exc}")
    finally:
        if timeout_s and timeout_s > 0:
            signal.alarm(0)


def _build_tasks(
    discoveries: list[tuple[str, str, Path]],
    *,
    device_override: str | None,
    emit_dot: bool,
    timeout_s: int,
    activity_mode: str = "auto",
    operator_merge: str = "activity",
    activity_cache_dir: Path | None = None,
) -> list[tuple]:
    """Flatten bench/ver/prj_<idx> into a worker task list."""
    tasks: list[tuple] = []
    for bench, ver, bench_path in discoveries:
        board_device = device_override or ver
        script_dir = bench_path / "script"
        ppa_files = sorted(glob.glob(str(script_dir / "ppa_*.json")))
        for ppa_file in ppa_files:
            stem = Path(ppa_file).stem  # ppa_<idx>
            try:
                idx = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            tasks.append(
                (
                    bench,
                    ver,
                    idx,
                    str(bench_path),
                    board_device,
                    emit_dot,
                    timeout_s,
                    activity_mode,
                    operator_merge,
                    str(activity_cache_dir) if activity_cache_dir else None,
                )
            )
    return tasks


def _write_shards(
    std_by_key: dict[tuple[str, str], dict[int, str]],
    rdc_by_key: dict[tuple[str, str], dict[int, str]],
    *,
    output_root: Path,
    device_override: str | None,
    layout: str,
) -> tuple[int, int]:
    """Write per-bench .pt shards in deterministic idx order.

    ``std_by_key`` / ``rdc_by_key`` map ``(bench, ver) -> {idx: pt_path}``;
    this function loads each per-sample ``.pt`` lazily on the main process
    and concatenates into a list before writing the shard. Avoids round-
    tripping torch tensors through worker IPC.
    """
    import torch

    total_std = 0
    total_rdc = 0
    all_keys = set(std_by_key.keys()) | set(rdc_by_key.keys())
    from hgp.board_utils import normalize_device_name

    for bench, ver in sorted(all_keys):
        device_key = normalize_device_name(device_override or ver)
        if layout == "per_device":
            std_dir = output_root / device_key / "std"
            rdc_dir = output_root / device_key / "rdc"
        else:
            std_dir = output_root / "std"
            rdc_dir = output_root / "rdc"
        std_dir.mkdir(parents=True, exist_ok=True)
        rdc_dir.mkdir(parents=True, exist_ok=True)

        shard_name = f"{bench}.pt"
        std_map = std_by_key.get((bench, ver), {})
        rdc_map = rdc_by_key.get((bench, ver), {})
        std_samples = [torch.load(std_map[i], weights_only=False) for i in sorted(std_map)]
        rdc_samples = [torch.load(rdc_map[i], weights_only=False) for i in sorted(rdc_map)]
        if std_samples:
            torch.save(std_samples, std_dir / shard_name)
        if rdc_samples:
            torch.save(rdc_samples, rdc_dir / shard_name)
        print(
            f"  [{bench}/{ver}] {len(std_samples)} std, {len(rdc_samples)} rdc "
            f"→ {device_key}/"
        )
        total_std += len(std_samples)
        total_rdc += len(rdc_samples)
    return total_std, total_rdc


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
        nargs="*",
        default=None,
        help="process only these benchmarks (default: all found under raw-root)",
    )
    parser.add_argument(
        "--exclude-bench",
        nargs="*",
        default=None,
        help="skip these benchmarks (e.g. polybench_xczu9eg legacy export)",
    )
    parser.add_argument(
        "--ver",
        nargs="*",
        default=None,
        help="process only these versions/devices (default: all found)",
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="parallel workers (1 = serial; default = min(32, cpu_count))",
    )
    parser.add_argument(
        "--sample-timeout",
        type=int,
        default=120,
        help="kill any single sample taking longer than this (seconds, 0 = unlimited)",
    )
    parser.add_argument(
        "--activity-mode",
        choices=ACTIVITY_MODES,
        default="auto",
        help=(
            "switching-activity edge features: auto maps trace op ids to sample CDFG nodes "
            "with opcode fallback, node uses cached node ids only, opcode uses opcode averages, "
            "none disables SA/AR, shuffled preserves the opcode distribution but breaks mapping"
        ),
    )
    parser.add_argument(
        "--operator-merge",
        choices=OPERATOR_MERGE_MODES,
        default="activity",
        help=(
            "ATAPP-style IR/FSMD hardware-operator merge: none keeps the existing CDFG, "
            "activity merges SA/AR across shared RTL operators, graph also collapses "
            "shared RTL operators into one graph node"
        ),
    )
    parser.add_argument(
        "--activity-cache-dir",
        type=Path,
        default=None,
        help=(
            "directory containing <kernel>_switching_activity.json caches "
            "(default: <repo>/dataset/switching_activity)"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print a progress line every N completed samples",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    layout = "legacy" if args.legacy_layout else "per_device"

    discoveries = _discover_bench_vers(args.raw_root)
    if not discoveries:
        print(f"No bench/ver directories found under {args.raw_root}")
        sys.exit(1)

    if args.bench:
        bench_set = set(args.bench)
        discoveries = [(b, v, p) for b, v, p in discoveries if b in bench_set]
    if args.exclude_bench:
        excl = set(args.exclude_bench)
        discoveries = [(b, v, p) for b, v, p in discoveries if b not in excl]
    if args.ver:
        ver_set = set(args.ver)
        discoveries = [(b, v, p) for b, v, p in discoveries if v in ver_set]

    print(f"Found {len(discoveries)} bench/ver combinations under {args.raw_root}")

    tasks = _build_tasks(
        discoveries,
        device_override=args.device,
        emit_dot=args.emit_dot,
        timeout_s=args.sample_timeout,
        activity_mode=args.activity_mode,
        operator_merge=args.operator_merge,
        activity_cache_dir=args.activity_cache_dir,
    )
    if not tasks:
        print("No ppa_*.json samples found, nothing to do.")
        return

    n_jobs = max(1, args.n_jobs)
    print(f"Processing {len(tasks)} samples with {n_jobs} worker(s)...")

    std_by_key: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    rdc_by_key: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    fail_count = 0
    skip_count = 0
    timeout_count = 0
    done = 0

    def _accumulate(result: tuple) -> None:
        nonlocal fail_count, skip_count, timeout_count
        bench, ver, idx, std_path, rdc_path, status = result
        if std_path is not None:
            std_by_key[(bench, ver)][idx] = std_path
        if rdc_path is not None:
            rdc_by_key[(bench, ver)][idx] = rdc_path
        if status.startswith("skip"):
            skip_count += 1
        elif status.startswith("timeout"):
            timeout_count += 1
            print(f"  [{bench}/{ver}/prj_{idx}] {status}")
        elif status != "ok":
            fail_count += 1
            print(f"  [{bench}/{ver}/prj_{idx}] {status}")

    if n_jobs == 1:
        for task in tasks:
            _accumulate(_process_one_sample(task))
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"  progress: {done}/{len(tasks)}")
    else:
        ctx = mp.get_context("spawn")
        # Chunked execution: rebuild the pool every CHUNK_SIZE samples so a
        # hard worker crash (segfault, OOM, abort) loses at most one chunk
        # instead of breaking the entire run. Each task itself is short
        # (~6s), so the pool startup cost is amortized fine over a chunk.
        chunk_size = max(n_jobs * 4, 64)
        for start in range(0, len(tasks), chunk_size):
            chunk = tasks[start : start + chunk_size]
            try:
                with ctx.Pool(processes=n_jobs) as pool:
                    for result in pool.imap_unordered(
                        _process_one_sample, chunk, chunksize=1
                    ):
                        _accumulate(result)
                        done += 1
                        if args.progress_every and done % args.progress_every == 0:
                            print(f"  progress: {done}/{len(tasks)}")
                    pool.close()
                    pool.join()
            except Exception as exc:
                # One chunk's pool died — record losses and continue.
                lost = len(chunk) - (done - start)
                print(f"  pool failed mid-chunk ({exc}); {lost} sample(s) lost, restarting")
                fail_count += max(0, lost)
                done = start + len(chunk)

    total_std, total_rdc = _write_shards(
        std_by_key,
        rdc_by_key,
        output_root=args.output_root,
        device_override=args.device,
        layout=layout,
    )

    print(
        f"\nDone. Total: {total_std} std samples, {total_rdc} rdc samples "
        f"({skip_count} skipped, {timeout_count} timed out, {fail_count} failed)."
    )


if __name__ == "__main__":
    main()
