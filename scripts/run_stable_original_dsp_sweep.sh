#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SWEEP_ROOT:=$HGBO_ROOT/img/training/original_stable_dsp_sweep_20260518}"
: "${EPOCHS:=150}"
: "${CPU_THREADS:=16}"
: "${PYTHON_BIN:=$HGBO_ROOT/.venv/bin/python}"

mkdir -p "$SWEEP_ROOT"

SUMMARY_CSV="$SWEEP_ROOT/sweep_summary.csv"
printf 'run,target,seed,mae_lr,grad_clip,lr_decay_factor,lr_decay_interval,epochs,best_test_metric,output_dir\n' > "$SUMMARY_CSV"

run_case() {
  local name="$1"
  local seed="$2"
  local mae_lr="$3"
  local grad_clip="$4"
  local lr_decay_factor="$5"
  local lr_decay_interval="$6"
  local out_dir="$SWEEP_ROOT/$name"
  local log="$out_dir/run.log"

  if [[ -e "$out_dir" && -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty output: $out_dir"
    return
  fi

  mkdir -p "$out_dir"
  echo "[$(date +%H:%M:%S)] Running $name"
  (
    cd "$HGBO_ROOT"
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m hgp.reporting.original_stable_training \
      --targets dsp \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --mae-lr "$mae_lr" \
      --grad-clip "$grad_clip" \
      --lr-decay-factor "$lr_decay_factor" \
      --lr-decay-interval "$lr_decay_interval" \
      --cpu-threads "$CPU_THREADS" \
      --output-dir "$out_dir"
  ) > "$log" 2>&1

  "$PYTHON_BIN" - "$name" "$out_dir" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

name = sys.argv[1]
out_dir = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
target = summary["targets"]["dsp"]
settings = target["settings"]
with summary_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            name,
            "dsp",
            settings["seed"],
            settings["lr"],
            settings["grad_clip"],
            settings["lr_decay_factor"],
            settings["lr_decay_interval"],
            settings["epochs"],
            target["best_test_metric"],
            out_dir,
        ]
    )
print(f"{name}: DSP best test MAE = {target['best_test_metric']:.6g}")
PY
}

run_case "seed128_lr003_decay09" 128 0.003 1.0 0.9 10
run_case "seed128_lr005_decay09" 128 0.005 1.0 0.9 10
run_case "seed128_lr005_nodecay" 128 0.005 1.0 1.0 10
run_case "seed42_lr005_decay09" 42 0.005 1.0 0.9 10
run_case "seed7_lr005_decay09" 7 0.005 1.0 0.9 10

echo "Summary written to $SUMMARY_CSV"
cat "$SUMMARY_CSV"
