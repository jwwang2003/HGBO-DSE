#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${FLOW_ROOT:=$HGBO_ROOT/img/training/new_stack_hgbo_dse_flow_$STAMP}"
: "${PYTHON_BIN:=$HGBO_ROOT/.venv/bin/python}"
: "${CPU_THREADS:=16}"
: "${NUM_WORKERS:=0}"
: "${SEED:=128}"
: "${DRY_RUN:=0}"
: "${DETERMINISTIC_EVAL:=1}"

: "${RUN_REFERENCE_EVAL:=1}"
: "${RUN_ORIGINAL_TARGETS:=0}"
: "${RUN_ORIGINAL_MAPE_TARGETS:=$RUN_ORIGINAL_TARGETS}"
: "${RUN_ORIGINAL_DSP:=$RUN_ORIGINAL_TARGETS}"
: "${RUN_ORIGINAL_BRAM:=$RUN_ORIGINAL_TARGETS}"
: "${RUN_BRAM_STATS:=0}"
: "${RUN_ARCH_VERIFY:=0}"

: "${ORIGINAL_EPOCHS:=500}"
: "${MAPE_EPOCHS:=$ORIGINAL_EPOCHS}"
: "${DSP_EPOCHS:=$ORIGINAL_EPOCHS}"
: "${BRAM_EPOCHS:=$ORIGINAL_EPOCHS}"
: "${MAPE_TARGETS:=lut ff cp power}"

: "${BRAM_LOSS_WEIGHTING:=none}"
: "${BRAM_TARGET_TRANSFORM:=none}"
: "${BRAM_TRAIN_SAMPLER:=none}"
: "${BRAM_POSITIVE_WEIGHT:=2}"
: "${BRAM_MID_WEIGHT:=4}"
: "${BRAM_HIGH_WEIGHT:=16}"
: "${BRAM_EXTREME_WEIGHT:=32}"

: "${ARCH_TARGETS:=lut ff dsp bram cp power}"
: "${ARCH_EPOCHS:=30}"
: "${ARCH_LR:=0.001}"
: "${ARCH_WEIGHT_DECAY:=0.001}"
: "${ARCH_GRAD_CLIP:=1.0}"
: "${ARCH_MODE:=arch-aware}"
: "${ARCH_FABRIC_MODE:=cached}"
: "${ARCH_CONV_TYPE:=gine}"

mkdir -p "$FLOW_ROOT"

SUMMARY_CSV="$FLOW_ROOT/flow_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  printf 'stage,name,status,metric_summary,output_dir,log\n' > "$SUMMARY_CSV"
fi

append_summary_row() {
  local stage="$1"
  local name="$2"
  local status="$3"
  local metric_summary="$4"
  local out_dir="$5"
  local log="$6"

  "$PYTHON_BIN" - "$SUMMARY_CSV" "$stage" "$name" "$status" "$metric_summary" "$out_dir" "$log" <<'PY'
import csv
import sys

summary_csv, stage, name, status, metric_summary, out_dir, log = sys.argv[1:]
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([stage, name, status, metric_summary, out_dir, log])
PY
}

append_original_summary() {
  local stage="$1"
  local name="$2"
  local out_dir="$3"
  local log="$4"

  "$PYTHON_BIN" - "$SUMMARY_CSV" "$stage" "$name" "$out_dir" "$log" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv, stage, name, out_dir, log = sys.argv[1:]
summary = json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))
metrics = {
    target: {
        "metric": result["metric_name"],
        "best_test": result["best_test_metric"],
    }
    for target, result in summary["targets"].items()
}
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([stage, name, "completed", json.dumps(metrics, sort_keys=True), out_dir, log])
print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
PY
}

append_bram_stats_summary() {
  local name="$1"
  local out_dir="$2"
  local log="$3"
  local json_path="$4"

  "$PYTHON_BIN" - "$SUMMARY_CSV" "$name" "$out_dir" "$log" "$json_path" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv, name, out_dir, log, json_path = sys.argv[1:]
stats = json.loads(Path(json_path).read_text(encoding="utf-8"))
metrics = {
    "target": stats["settings"]["target"],
    "metric": stats["settings"]["metric"],
    "metric_value": stats["mae"],
    "median_abs_error": stats["abs_error"]["median"],
    "max_abs_error": stats["abs_error"]["max"],
}
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["bram_stats", name, "completed", json.dumps(metrics, sort_keys=True), out_dir, log])
print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
PY
}

append_arch_summary() {
  local name="$1"
  local out_dir="$2"
  local log="$3"

  "$PYTHON_BIN" - "$SUMMARY_CSV" "$name" "$out_dir" "$log" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv, name, out_dir, log = sys.argv[1:]
summary = json.loads((Path(out_dir) / "hgbo_arch_training_history.json").read_text(encoding="utf-8"))
metrics = {}
for target, result in summary["targets"].items():
    metrics[target] = {
        "metric": result["metric_name"],
        "original_best_test": result["original"]["best_test_metric"],
        "architecture_aware_best_test": result["architecture_aware"]["best_test_metric"],
    }
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["arch_verify", name, "completed", json.dumps(metrics, sort_keys=True), out_dir, log])
print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
PY
}

run_logged() {
  local stage="$1"
  local name="$2"
  local out_dir="$3"
  shift 3
  local log="$out_dir/run.log"

  if [[ -e "$out_dir" && -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty output: $out_dir"
    append_summary_row "$stage" "$name" "skipped" "non-empty output" "$out_dir" "$log"
    return 1
  fi

  mkdir -p "$out_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf 'DRY RUN:'
      printf ' %q' "$@"
      printf '\n'
    } | tee "$log"
    append_summary_row "$stage" "$name" "dry-run" "command not executed" "$out_dir" "$log"
    return 1
  fi

  echo "[$(date +%H:%M:%S)] Running $name"
  (
    cd "$HGBO_ROOT"
    "$@"
  ) > "$log" 2>&1
  return 0
}

run_original_training() {
  local stage="$1"
  local name="$2"
  local targets="$3"
  local epochs="$4"
  local lr="$5"
  local mae_lr="$6"
  local grad_clip="$7"
  local lr_decay_factor="$8"
  local lr_decay_interval="$9"
  local weight_decay="${10}"
  local init_flag="${11}"
  shift 11

  local out_dir="$FLOW_ROOT/$name"
  local log="$out_dir/run.log"
  local -a target_args
  local -a cmd
  read -r -a target_args <<< "$targets"
  cmd=(
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.reporting.original_stable_training
    --targets "${target_args[@]}"
    --epochs "$epochs"
    --seed "$SEED"
    --lr "$lr"
    --mae-lr "$mae_lr"
    --grad-clip "$grad_clip"
    --lr-decay-factor "$lr_decay_factor"
    --lr-decay-interval "$lr_decay_interval"
    --weight-decay "$weight_decay"
    --device cpu
    --cpu-threads "$CPU_THREADS"
    --num-workers "$NUM_WORKERS"
    --output-dir "$out_dir"
  )
  if [[ "$DETERMINISTIC_EVAL" == "1" ]]; then
    cmd+=(--deterministic-eval)
  fi
  if [[ -n "$init_flag" ]]; then
    cmd+=("$init_flag")
  fi
  cmd+=("$@")

  if run_logged "$stage" "$name" "$out_dir" "${cmd[@]}"; then
    append_original_summary "$stage" "$name" "$out_dir" "$log"
  fi
}

run_bram_stats() {
  local name="bram_prediction_stats"
  local out_dir="$FLOW_ROOT/$name"
  local log="$out_dir/run.log"
  local json_path="$out_dir/bram_prediction_stats.json"
  : "${BRAM_CHECKPOINT_DIR:=$FLOW_ROOT/original_bram_lr001_wd1e3_decay09_${BRAM_EPOCHS}ep/hgp/model}"

  if [[ "$DRY_RUN" != "1" && ! -d "$BRAM_CHECKPOINT_DIR" ]]; then
    echo "BRAM checkpoint directory not found: $BRAM_CHECKPOINT_DIR" >&2
    echo "Set BRAM_CHECKPOINT_DIR or run with RUN_ORIGINAL_BRAM=1." >&2
    exit 1
  fi

  local -a cmd=(
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.reporting.checkpoint_prediction_stats
    --target bram
    --seed "$SEED"
    --cpu-threads "$CPU_THREADS"
    --init-checkpoint-dir "$BRAM_CHECKPOINT_DIR"
    --output-json "$json_path"
  )
  if run_logged "bram_stats" "$name" "$out_dir" "${cmd[@]}"; then
    append_bram_stats_summary "$name" "$out_dir" "$log" "$json_path"
  fi
}

run_arch_verify() {
  local safe_arch_mode="${ARCH_MODE//[^A-Za-z0-9_]/_}"
  local name="arch_verify_${safe_arch_mode}_${ARCH_EPOCHS}ep"
  local out_dir="$FLOW_ROOT/$name"
  local log="$out_dir/run.log"
  local -a target_args
  local -a cmd
  read -r -a target_args <<< "$ARCH_TARGETS"
  cmd=(
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.reporting.compare_training_graphs
    --targets "${target_args[@]}"
    --epochs "$ARCH_EPOCHS"
    --seed "$SEED"
    --lr "$ARCH_LR"
    --grad-clip "$ARCH_GRAD_CLIP"
    --weight-decay "$ARCH_WEIGHT_DECAY"
    --arch-mode "$ARCH_MODE"
    --fabric-mode "$ARCH_FABRIC_MODE"
    --conv-type "$ARCH_CONV_TYPE"
    --device cpu
    --cpu-threads "$CPU_THREADS"
    --num-workers "$NUM_WORKERS"
    --output-dir "$out_dir"
  )
  if run_logged "arch_verify" "$name" "$out_dir" "${cmd[@]}"; then
    append_arch_summary "$name" "$out_dir" "$log"
  fi
}

if [[ "$RUN_REFERENCE_EVAL" == "1" ]]; then
  run_original_training \
    "reference_eval" \
    "reference_builtin_all_targets" \
    "lut ff dsp bram cp power" \
    0 \
    0.005 \
    0.00001 \
    1.0 \
    0.9 \
    10 \
    0.001 \
    "--init-from-builtins"
fi

if [[ "$RUN_ORIGINAL_MAPE_TARGETS" == "1" ]]; then
  run_original_training \
    "original_train" \
    "original_mape_targets_lr005_wd0_decay095_${MAPE_EPOCHS}ep" \
    "$MAPE_TARGETS" \
    "$MAPE_EPOCHS" \
    0.005 \
    0.002 \
    1.0 \
    0.95 \
    10 \
    0.0 \
    ""
fi

if [[ "$RUN_ORIGINAL_DSP" == "1" ]]; then
  run_original_training \
    "original_train" \
    "original_dsp_lr002_wd0_decay095_${DSP_EPOCHS}ep" \
    "dsp" \
    "$DSP_EPOCHS" \
    0.005 \
    0.002 \
    1.0 \
    0.95 \
    10 \
    0.0 \
    ""
fi

if [[ "$RUN_ORIGINAL_BRAM" == "1" ]]; then
  run_original_training \
    "original_train" \
    "original_bram_lr001_wd1e3_decay09_${BRAM_EPOCHS}ep" \
    "bram" \
    "$BRAM_EPOCHS" \
    0.005 \
    0.001 \
    1.0 \
    0.9 \
    10 \
    0.001 \
    "" \
    --target-transform "$BRAM_TARGET_TRANSFORM" \
    --train-sampler "$BRAM_TRAIN_SAMPLER" \
    --loss-weighting "$BRAM_LOSS_WEIGHTING" \
    --bram-positive-weight "$BRAM_POSITIVE_WEIGHT" \
    --bram-mid-weight "$BRAM_MID_WEIGHT" \
    --bram-high-weight "$BRAM_HIGH_WEIGHT" \
    --bram-extreme-weight "$BRAM_EXTREME_WEIGHT"
fi

if [[ "$RUN_BRAM_STATS" == "1" ]]; then
  run_bram_stats
fi

if [[ "$RUN_ARCH_VERIFY" == "1" ]]; then
  run_arch_verify
fi

echo "Summary written to $SUMMARY_CSV"
cat "$SUMMARY_CSV"
