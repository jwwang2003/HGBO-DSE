#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${RUN_DIR:=$HGBO_ROOT/img/training/original_paper_seed128_$STAMP}"
: "${TARGETS:=lut ff dsp bram cp power}"
: "${UV_CACHE_DIR:=/tmp/hgbo-dse-uv-cache}"
: "${PYTHON_BIN:=}"
: "${ALLOW_COMPAT_DEPS:=0}"
: "${CHECK_DEPS_ONLY:=0}"
: "${HGP_TORCH_SEED:=128}"

export UV_CACHE_DIR

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif [[ -x "$HGBO_ROOT/.venv113/bin/python" ]]; then
  PYTHON_CMD=("$HGBO_ROOT/.venv113/bin/python")
elif [[ -x "$HGBO_ROOT/.venv/bin/python" ]]; then
  PYTHON_CMD=("$HGBO_ROOT/.venv/bin/python")
else
  PYTHON_CMD=(uv run python)
fi

check_original_deps() {
  "${PYTHON_CMD[@]}" - <<'PY'
import sys

expected = {
    "torch": "1.13.1",
    "torch_geometric": "2.3.1",
    "torch_scatter": "2.1.1",
    "torch_sparse": "0.6.17",
}

loaded = {}
for module_name in expected:
    module = __import__(module_name)
    loaded[module_name] = getattr(module, "__version__", "")

bad = {
    module_name: (expected_version, loaded[module_name])
    for module_name, expected_version in expected.items()
    if not loaded[module_name].startswith(expected_version)
}

for module_name, version in loaded.items():
    print(f"{module_name}={version}")

if bad:
    for module_name, (expected_version, actual_version) in bad.items():
        print(
            f"Expected {module_name} {expected_version}, got {actual_version}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY
}

if [[ "$ALLOW_COMPAT_DEPS" != "1" ]]; then
  if ! check_original_deps; then
    cat >&2 <<EOF
Original HGBO-DSE HGP retraining is sensitive to Torch/PyG versions.
Run:
  scripts/setup_original_hgp_env.sh

Then rerun this script, or set ALLOW_COMPAT_DEPS=1 to bypass this guard.
EOF
    exit 1
  fi
fi

if [[ "$CHECK_DEPS_ONLY" == "1" ]]; then
  echo "Dependency check passed for: ${PYTHON_CMD[*]}"
  exit 0
fi

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to write into non-empty RUN_DIR: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$RUN_DIR/hgp/model"
mkdir -p "$RUN_DIR/dataset/std" "$RUN_DIR/dataset/rdc"

for split in std rdc; do
  for pt in "$HGBO_ROOT/dataset/$split"/*.pt; do
    ln -s "$pt" "$RUN_DIR/dataset/$split/$(basename "$pt")"
  done
done

cat > "$RUN_DIR/settings.txt" <<EOF
HGBO-DSE original paper-style baseline
Model: HGP+SAGE+GF
Epochs: 500
Batch size: 32
Hidden channels: 64
Layers: 3
Dropout: 0.0
Optimizer: Adam
Initial LR: 0.005
Weight decay: 0.001
Dataset split: 80/20
Shuffle seed: 128
torch random_split seed: 42
Torch/model seed: $HGP_TORCH_SEED
Output: $RUN_DIR
Targets: $TARGETS
UV cache: $UV_CACHE_DIR
Python: ${PYTHON_CMD[*]}
Dependency guard: ALLOW_COMPAT_DEPS=$ALLOW_COMPAT_DEPS
Dataset view: run-local std/rdc symlinks to .pt files only
EOF

echo "RUN_DIR=$RUN_DIR"
echo "TARGETS=$TARGETS"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "PYTHON=${PYTHON_CMD[*]}"

for target in $TARGETS; do
  case "$target" in
    lut) script="$HGBO_ROOT/hgp/hier_lut_model.py" ;;
    ff) script="$HGBO_ROOT/hgp/hier_ff_model.py" ;;
    dsp) script="$HGBO_ROOT/hgp/hier_dsp_model.py" ;;
    bram) script="$HGBO_ROOT/hgp/hier_bram_model.py" ;;
    cp) script="$HGBO_ROOT/hgp/hier_cp_model.py" ;;
    power|pwr) script="$HGBO_ROOT/hgp/hier_pwr_model.py" ;;
    *)
      echo "Unknown target: $target" >&2
      exit 2
      ;;
  esac

  log="$RUN_DIR/${target}.log"
  echo "[$(date +%H:%M:%S)] Training $target with $script"
  (
    cd "$RUN_DIR/hgp"
    HGP_TORCH_SEED="$HGP_TORCH_SEED" "${PYTHON_CMD[@]}" - "$script" <<'PY'
import os
import random
import runpy
import sys

import numpy as np
import torch

script = sys.argv[1]
seed = int(os.environ.get("HGP_TORCH_SEED", "128"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

runpy.run_path(script, run_name="__main__")
PY
  ) > "$log" 2>&1

  echo "[$(date +%H:%M:%S)] Finished $target"
  grep "Min Test" "$log" | tail -1 || true
done

grep -h "Min Test" "$RUN_DIR"/*.log > "$RUN_DIR/min_test_summary.txt" || true
find "$RUN_DIR/hgp/model" -maxdepth 1 -name "*checkpoint_test.pt" -print | sort > "$RUN_DIR/test_checkpoints.txt"

echo "Summary:"
cat "$RUN_DIR/min_test_summary.txt"
echo "Test checkpoints:"
cat "$RUN_DIR/test_checkpoints.txt"
