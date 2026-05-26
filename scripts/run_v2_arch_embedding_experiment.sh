#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${FLOW_ROOT:=$HGBO_ROOT/img/training/v2_arch_embedding_experiment_$STAMP}"
: "${PYTHON_BIN:=$HGBO_ROOT/.venv/bin/python}"
: "${CPU_THREADS:=16}"
: "${NUM_WORKERS:=0}"
: "${SEED:=128}"
: "${MAPE_EPOCHS:=500}"
: "${DSP_EPOCHS:=500}"
: "${BRAM_EPOCHS:=500}"
: "${MAPE_TARGETS:=lut ff cp power}"
: "${RUN_ARCH_MAPE_TARGETS:=1}"
: "${RUN_ARCH_DSP:=1}"
: "${RUN_ARCH_BRAM:=1}"
: "${RUN_DETERMINISTIC_ARCH_EVAL:=1}"
: "${RUN_BRAM_RESIDUAL_CALIBRATOR:=1}"
: "${BRAM_CALIBRATOR_ESTIMATORS:=500}"
: "${BRAM_CALIBRATOR_JOBS:=$CPU_THREADS}"
: "${BRAM_HLS_RESIDUAL_INDEX:=3}"
: "${ARCH_MODE:=arch-aware}"
: "${ARCH_FABRIC_MODE:=cached}"
: "${ARCH_CONV_TYPE:=sage}"
: "${ARCH_WEIGHT_DECAY:=0.0}"
: "${ARCH_GRAD_CLIP:=1.0}"
: "${ARCH_LR_DECAY_FACTOR:=0.95}"
: "${ARCH_LR_DECAY_INTERVAL:=10}"
: "${MAPE_LR:=0.005}"
: "${MAE_LR:=0.002}"

SUMMARY_CSV="$FLOW_ROOT/flow_summary.csv"
mkdir -p "$FLOW_ROOT"
printf 'stage,name,status,metric_summary,output_dir,log\n' > "$SUMMARY_CSV"

checkpoint_name() {
  local target="$1"
  case "$target" in
    lut) echo "lut_arch_h64_d0_checkpoint_test.pt" ;;
    ff) echo "ff_arch_h64_d0_checkpoint_test.pt" ;;
    dsp) echo "dsp_mae_arch_h64_d0_checkpoint_test.pt" ;;
    bram) echo "bram_mae_arch_h64_d0_checkpoint_test.pt" ;;
    cp) echo "cp_mean_arch_h64_d0_checkpoint_test.pt" ;;
    power) echo "power_mean_arch_h64_d0_checkpoint_test.pt" ;;
    *) echo "Unknown target: $target" >&2; return 2 ;;
  esac
}

append_summary() {
  local stage="$1"
  local name="$2"
  local out_dir="$3"
  local log="$4"
  "$PYTHON_BIN" -c '
import csv, json, sys
stage, name, out_dir, log, csv_path = sys.argv[1:]
summary_path = f"{out_dir}/summary.json"
with open(summary_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
metric = {data["target"]: {"best_test": data["min_test_metric"], "metric": data["metric_name"]}}
with open(csv_path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([stage, name, "completed", json.dumps(metric, sort_keys=True), out_dir, log])
' "$stage" "$name" "$out_dir" "$log" "$SUMMARY_CSV"
}

append_bram_calibrator_summary() {
  local name="$1"
  local out_dir="$2"
  local log="$3"
  "$PYTHON_BIN" -c '
import csv, json, sys
name, out_dir, log, csv_path = sys.argv[1:]
with open(f"{out_dir}/summary.json", "r", encoding="utf-8") as handle:
    data = json.load(handle)
metric = {
    "metric": "mae",
    "test_mae": data["test"]["mae"],
    "train_mae": data["train"]["mae"],
    "residual_accuracy": data["test"]["residual_accuracy"],
}
with open(csv_path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["bram_residual_calibrator", name, "completed", json.dumps(metric, sort_keys=True), out_dir, log])
' "$name" "$out_dir" "$log" "$SUMMARY_CSV"
}

run_logged() {
  local stage="$1"
  local name="$2"
  local out_dir="$3"
  shift 3
  local log="$out_dir/run.log"

  if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty output: $out_dir"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] Running $name"
  mkdir -p "$out_dir"
  (
    cd "$HGBO_ROOT"
    "$@"
  ) > "$log" 2>&1
  append_summary "$stage" "$name" "$out_dir" "$log"
  echo "$name: $("$PYTHON_BIN" -c 'import json, sys; data=json.load(open(sys.argv[1])); print(json.dumps({data["target"]: {"best_test": data["min_test_metric"], "metric": data["metric_name"]}}, sort_keys=True))' "$out_dir/summary.json")"
}

run_arch_target() {
  local stage="$1"
  local target="$2"
  local lr="$3"
  local epochs="$4"
  local name="arch_${target}_${ARCH_MODE//[^A-Za-z0-9_]/_}_${epochs}ep_lr${lr//./}"
  local out_dir="$FLOW_ROOT/$name"
  run_logged \
    "$stage" \
    "$name" \
    "$out_dir" \
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.hier_arch_model \
      --target "$target" \
      --epochs "$epochs" \
      --seed "$SEED" \
      --lr "$lr" \
      --lr-decay-factor "$ARCH_LR_DECAY_FACTOR" \
      --lr-decay-interval "$ARCH_LR_DECAY_INTERVAL" \
      --weight-decay "$ARCH_WEIGHT_DECAY" \
      --grad-clip "$ARCH_GRAD_CLIP" \
      --arch-mode "$ARCH_MODE" \
      --fabric-mode "$ARCH_FABRIC_MODE" \
      --conv-type "$ARCH_CONV_TYPE" \
      --device cpu \
      --cpu-threads "$CPU_THREADS" \
      --num-workers "$NUM_WORKERS" \
      --model-dir "$out_dir/model" \
      --summary-path "$out_dir/summary.json"
}

run_arch_eval() {
  local target="$1"
  local lr="$2"
  local train_dir="$3"
  local ckpt="$train_dir/model/$(checkpoint_name "$target")"
  local name="deterministic_eval_arch_${target}"
  local out_dir="$FLOW_ROOT/$name"

  if [[ ! -f "$ckpt" ]]; then
    echo "Skipping deterministic arch eval; checkpoint not found: $ckpt" >&2
    return 0
  fi

  run_logged \
    "deterministic_arch_eval" \
    "$name" \
    "$out_dir" \
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.hier_arch_model \
      --target "$target" \
      --epochs 0 \
      --seed "$SEED" \
      --lr "$lr" \
      --lr-decay-factor "$ARCH_LR_DECAY_FACTOR" \
      --lr-decay-interval "$ARCH_LR_DECAY_INTERVAL" \
      --weight-decay "$ARCH_WEIGHT_DECAY" \
      --grad-clip "$ARCH_GRAD_CLIP" \
      --arch-mode "$ARCH_MODE" \
      --fabric-mode "$ARCH_FABRIC_MODE" \
      --conv-type "$ARCH_CONV_TYPE" \
      --device cpu \
      --cpu-threads "$CPU_THREADS" \
      --num-workers "$NUM_WORKERS" \
      --init-checkpoint "$ckpt" \
      --deterministic-eval \
      --model-dir "$out_dir/model" \
      --summary-path "$out_dir/summary.json"
}

run_bram_residual_calibrator() {
  local name="bram_residual_calibrator_${BRAM_CALIBRATOR_ESTIMATORS}trees"
  local out_dir="$FLOW_ROOT/$name"
  local log="$out_dir/run.log"

  if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty output: $out_dir"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] Running $name"
  mkdir -p "$out_dir"
  (
    cd "$HGBO_ROOT"
    env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m hgp.reporting.bram_residual_calibrator \
      --seed "$SEED" \
      --hls-index "$BRAM_HLS_RESIDUAL_INDEX" \
      --n-estimators "$BRAM_CALIBRATOR_ESTIMATORS" \
      --n-jobs "$BRAM_CALIBRATOR_JOBS" \
      --output-dir "$out_dir"
  ) > "$log" 2>&1
  append_bram_calibrator_summary "$name" "$out_dir" "$log"
  echo "$name: $("$PYTHON_BIN" -c 'import json, sys; data=json.load(open(sys.argv[1])); print(json.dumps({"metric": "mae", "test_mae": data["test"]["mae"], "residual_accuracy": data["test"]["residual_accuracy"]}, sort_keys=True))' "$out_dir/summary.json")"
}

declare -a trained_targets=()
declare -A train_dirs=()
declare -A target_lrs=()

if [[ "$RUN_ARCH_MAPE_TARGETS" == "1" ]]; then
  for target in $MAPE_TARGETS; do
    run_arch_target "arch_train" "$target" "$MAPE_LR" "$MAPE_EPOCHS"
    name="arch_${target}_${ARCH_MODE//[^A-Za-z0-9_]/_}_${MAPE_EPOCHS}ep_lr${MAPE_LR//./}"
    trained_targets+=("$target")
    train_dirs["$target"]="$FLOW_ROOT/$name"
    target_lrs["$target"]="$MAPE_LR"
  done
fi

if [[ "$RUN_ARCH_DSP" == "1" ]]; then
  run_arch_target "arch_train" "dsp" "$MAE_LR" "$DSP_EPOCHS"
  name="arch_dsp_${ARCH_MODE//[^A-Za-z0-9_]/_}_${DSP_EPOCHS}ep_lr${MAE_LR//./}"
  trained_targets+=("dsp")
  train_dirs["dsp"]="$FLOW_ROOT/$name"
  target_lrs["dsp"]="$MAE_LR"
fi

if [[ "$RUN_ARCH_BRAM" == "1" ]]; then
  run_arch_target "arch_train" "bram" "$MAE_LR" "$BRAM_EPOCHS"
  name="arch_bram_${ARCH_MODE//[^A-Za-z0-9_]/_}_${BRAM_EPOCHS}ep_lr${MAE_LR//./}"
  trained_targets+=("bram")
  train_dirs["bram"]="$FLOW_ROOT/$name"
  target_lrs["bram"]="$MAE_LR"
fi

if [[ "$RUN_DETERMINISTIC_ARCH_EVAL" == "1" ]]; then
  for target in "${trained_targets[@]}"; do
    run_arch_eval "$target" "${target_lrs[$target]}" "${train_dirs[$target]}"
  done
fi

if [[ "$RUN_BRAM_RESIDUAL_CALIBRATOR" == "1" ]]; then
  run_bram_residual_calibrator
fi

echo "v2 architecture-embedding experiment outputs: $FLOW_ROOT"
