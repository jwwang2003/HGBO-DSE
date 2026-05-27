"""ATAPP-style switching activity extraction for HGBO-DSE kernels.

SA/AR are computed by running hand-instrumented versions of each kernel
with representative input data. Each arithmetic/load result is logged via
printf so we can compute Hamming-distance switching activity and activation
ratio per operation.

SA_{i} = sum_k HD(bits(v_i(k)), bits(v_i(k-1))) / L
AR_{i} = N_active_cycles / L

where L = max invocation count across all ops (proxy for latency).

Results are cached as JSON under <output_dir>/<kernel>_switching_activity.json.
The JSON maps cdfg node_id -> {sa, ar} so gen_dataset_std can attach these
to each PyG sample.

Usage:
    python -m hgp.data_process.switching_activity \\
        --raw-root dataset/raw \\
        --output-dir dataset/switching_activity \\
        [--kernels atax bicg ...] \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_HGBO_ROOT = Path(__file__).resolve().parents[2]
_TB_DIR = Path(__file__).resolve().parent / "testbenches"

# Mapping: kernel top-function name -> instrumented source file (self-contained)
_INSTR_SOURCES: dict[str, str] = {
    "atax":     "instrumented/polybench/atax_instr.c",
    "bicg":     "instrumented/polybench/bicg_instr.c",
    "gemm":     "instrumented/polybench/gemm_instr.c",
    "gesummv":  "instrumented/polybench/gesummv_instr.c",
    "k2mm":     "instrumented/polybench/k2mm_instr.c",
    "k3mm":     "instrumented/polybench/k3mm_instr.c",
    "mvt":      "instrumented/polybench/mvt_instr.c",
    "syr2k":    "instrumented/polybench/syr2k_instr.c",
    "syrk":     "instrumented/polybench/syrk_instr.c",
    "bfs":      "instrumented/machsuite/bfs_instr.c",
    "spmv":     "instrumented/machsuite/spmv_instr.c",
    "stencil":  "instrumented/machsuite/stencil_instr.c",
    "stencil3d":"instrumented/machsuite/stencil3d_instr.c",
}


# Opcode-to-op_id mapping: which instrumented op_ids correspond to each CDFG opcode.
# Structured as {kernel: {cdfg_opcode: [op_id, ...]}}
# Derived from the instrumented source files.
_KERNEL_OPCODE_MAP: dict[str, dict[str, list[int]]] = {
    "atax": {
        "load": [0, 3], "store": [8],
        "fmul": [4, 6], "mul": [4, 6],
        "fadd": [5, 7], "add": [5, 7],
    },
    "bicg": {
        "load": [0, 1, 4], "store": [9, 10],
        "fmul": [6, 8], "mul": [6, 8],
        "fadd": [5, 7], "add": [5, 7],
    },
    "gemm": {
        "load": [0, 1, 2], "store": [8],
        "fmul": [7], "mul": [7],
        "fadd": [5, 6], "add": [5, 6],
    },
    "gesummv": {
        "load": [0, 3, 4], "store": [9],
        "fmul": [6, 8], "mul": [6, 8],
        "fadd": [5, 7], "add": [5, 7],
    },
    "k2mm": {
        "load": [0, 1, 2, 3], "store": [12],
        "fmul": [7, 11], "mul": [7, 11],
        "fadd": [6, 10], "add": [6, 10],
    },
    "k3mm": {
        "load": [0, 1, 2], "store": [9],
        "fmul": [6, 8], "mul": [6, 8],
        "fadd": [5, 7], "add": [5, 7],
    },
    "mvt": {
        "load": [0, 1, 2, 3, 6], "store": [13, 14],
        "fmul": [9, 12], "mul": [9, 12],
        "fadd": [8, 11], "add": [8, 11],
    },
    "syr2k": {
        "load": [0, 1, 2], "store": [7],
        "fmul": [5, 6], "mul": [5, 6],
        "fadd": [4], "add": [4],
    },
    "syrk": {
        "load": [0], "store": [4],
        "fmul": [3], "mul": [3],
        "fadd": [2], "add": [2],
    },
    "bfs": {
        "load": [2, 3, 4], "store": [5, 6],
        "add": [7], "icmp": [0],
    },
    "spmv": {
        "load": [1, 2], "store": [5],
        "mul": [4], "add": [3],
    },
    "stencil": {
        "load": [0], "store": [3],
        "mul": [2], "add": [1],
    },
    "stencil3d": {
        "load": [0], "store": [3],
        "mul": [2], "add": [1],
    },
}


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_opcode_sa_map(
    kernel_name: str,
    by_op_id: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Build {cdfg_opcode: {sa, ar}} by averaging over matching op_ids."""
    opcode_map = _KERNEL_OPCODE_MAP.get(kernel_name, {})
    result: dict[str, dict[str, float]] = {}
    for opcode, op_ids in opcode_map.items():
        metrics = [by_op_id[str(i)] for i in op_ids if str(i) in by_op_id]
        if metrics:
            result[opcode] = {
                "sa": sum(m["sa"] for m in metrics) / len(metrics),
                "ar": sum(m["ar"] for m in metrics) / len(metrics),
            }
    return result


def compute_sa_ar(trace_lines: list[str]) -> dict[str, dict[str, float]]:
    """Parse trace output and return {op_id: {sa, ar, n}} per operation.

    Trace format: ``SA <op_id> <bits> <uint_value>``
    """
    per_op: dict[str, list[int]] = {}
    for line in trace_lines:
        if not line.startswith("SA "):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        op_id = parts[1]
        val = int(parts[3])
        per_op.setdefault(op_id, []).append(val)

    if not per_op:
        return {}

    L = max(len(v) for v in per_op.values())
    result: dict[str, dict[str, float]] = {}
    for op_id, values in per_op.items():
        n = len(values)
        hd = sum(_hamming(values[k], values[k - 1]) for k in range(1, n))
        result[op_id] = {"sa": hd / max(L, 1), "ar": n / max(L, 1), "n": n}
    return result


def run_instrumented_kernel(
    kernel_name: str,
    *,
    clang: str = "clang",
    timeout: int = 120,
) -> list[str]:
    """Compile and run the pre-instrumented kernel source; return trace lines."""
    instr_src = _TB_DIR / _INSTR_SOURCES[kernel_name]
    if not instr_src.exists():
        raise FileNotFoundError("Instrumented source not found: {}".format(instr_src))

    with tempfile.TemporaryDirectory() as tmp:
        out_bin = os.path.join(tmp, "sim_bin")
        cmd = [clang, "-O0", "-o", out_bin, str(instr_src), "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Compile failed for {}: {}".format(kernel_name, e.stderr.decode()[-2000:])
            )
        result = subprocess.run([out_bin], capture_output=True, timeout=timeout)
        return result.stdout.decode(errors="replace").splitlines()


def extract_switching_activity(
    kernel_name: str,
    cdfg_node_csv: Path | None = None,
    *,
    clang: str = "clang",
    timeout: int = 120,
) -> dict[str, Any]:
    """Simulate kernel and compute SA/AR dict.

    Returns:
        {
          "kernel": str,
          "L": int,
          "by_op_id": {op_id: {sa, ar, n}},
          "by_node_id": {node_id: {sa, ar}},  # if cdfg_node_csv given
        }
    """
    trace = run_instrumented_kernel(kernel_name, clang=clang, timeout=timeout)
    by_op_id = compute_sa_ar(trace)
    L = max((v["n"] for v in by_op_id.values()), default=1)

    by_node_id: dict[str, dict[str, float]] = {}
    if cdfg_node_csv is not None and cdfg_node_csv.exists():
        import csv
        with open(cdfg_node_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            op_id = str(i)
            if op_id in by_op_id:
                node_id = row.get("node_id", "")
                if node_id:
                    by_node_id[node_id] = {
                        "sa": by_op_id[op_id]["sa"],
                        "ar": by_op_id[op_id]["ar"],
                    }

    result = {"kernel": kernel_name, "L": L, "by_op_id": by_op_id, "by_node_id": by_node_id}
    # Add opcode-level SA/AR for mapping to all CDFG nodes by opcode
    result["by_opcode"] = build_opcode_sa_map(kernel_name, by_op_id)
    return result


def cache_path(kernel_name: str, output_dir: Path) -> Path:
    return Path(output_dir) / "{}_switching_activity.json".format(kernel_name)


def load_or_compute(
    kernel_name: str,
    output_dir: Path,
    cdfg_node_csv: Path | None = None,
    *,
    force: bool = False,
    clang: str = "clang",
) -> dict[str, Any]:
    p = cache_path(kernel_name, output_dir)
    if p.exists() and not force:
        with open(p) as f:
            return json.load(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = extract_switching_activity(kernel_name, cdfg_node_csv=cdfg_node_csv, clang=clang)
    p.write_text(json.dumps(data, indent=2))
    return data


def compute_all(
    raw_root: Path,
    output_dir: Path,
    kernels: list[str] | None = None,
    *,
    force: bool = False,
    clang: str = "clang",
    n_jobs: int = 1,
) -> dict[str, Path]:
    """Compute SA/AR for all (or specified) kernels. Returns {kernel: cache_path}."""
    import concurrent.futures

    target_kernels = kernels or list(_INSTR_SOURCES.keys())
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)

    def _find_cdfg_csv(kernel_name: str) -> Path | None:
        for family in ("polybench", "machsuite"):
            family_root = raw_root / family
            if not family_root.exists():
                continue
            for board_dir in sorted(family_root.iterdir()):
                for prj in sorted(board_dir.glob("prj_*")):
                    adb = prj / "graph" / "{}.adb".format(kernel_name)
                    csv_cand = prj / "cdfg" / "cdfg_node_dict.csv"
                    if adb.exists() and csv_cand.exists():
                        return csv_cand
        return None

    def _one(kernel_name: str) -> tuple[str, Path | None, str | None]:
        cdfg_csv = _find_cdfg_csv(kernel_name)
        try:
            load_or_compute(kernel_name, output_dir, cdfg_csv, force=force, clang=clang)
            return kernel_name, cache_path(kernel_name, output_dir), None
        except Exception as e:
            return kernel_name, None, str(e)

    results: dict[str, Path] = {}
    if n_jobs == 1:
        for k in target_kernels:
            name, p, err = _one(k)
            if err:
                print("WARNING {}: {}".format(name, err[:200]))
            elif p:
                results[name] = p
                print("OK {}".format(name))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for name, p, err in pool.map(_one, target_kernels):
                if err:
                    print("WARNING {}: {}".format(name, err[:200]))
                elif p:
                    results[name] = p
                    print("OK {}".format(name))
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute ATAPP-style switching activity")
    p.add_argument("--raw-root", type=Path, default=_HGBO_ROOT / "dataset" / "raw")
    p.add_argument("--output-dir", type=Path, default=_HGBO_ROOT / "dataset" / "switching_activity")
    p.add_argument("--kernels", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--clang", default="clang")
    p.add_argument("--n-jobs", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    results = compute_all(
        args.raw_root, args.output_dir, args.kernels,
        force=args.force, clang=args.clang, n_jobs=args.n_jobs,
    )
    print("Computed SA/AR for {} kernels".format(len(results)))


if __name__ == "__main__":
    main()
