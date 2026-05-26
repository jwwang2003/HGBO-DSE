#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${FLOW_ROOT:=$HGBO_ROOT/img/training/v2_production_reproduction_$STAMP}"
: "${PYTHON_BIN:=$HGBO_ROOT/.venv/bin/python}"
: "${CPU_THREADS:=16}"
: "${NUM_WORKERS:=0}"
: "${SEED:=128}"
: "${MAPE_EPOCHS:=500}"
: "${DSP_EPOCHS:=500}"
: "${BRAM_CALIBRATOR_ESTIMATORS:=500}"
: "${RUN_DETERMINISTIC_CHECKPOINT_EVAL:=1}"
: "${RUN_REFERENCE_EVAL:=1}"
: "${RUN_ORIGINAL_MAPE_TARGETS:=1}"
: "${RUN_ORIGINAL_DSP:=1}"
: "${RUN_ORIGINAL_BRAM:=0}"
: "${RUN_BRAM_RESIDUAL_CALIBRATOR:=1}"
: "${RUN_ARCH_VERIFY:=0}"
: "${LOADER_RNG_MODE:=isolated}"
: "${ORIGINAL_TRAIN_DETERMINISTIC_EVAL:=0}"
: "${REFERENCE_DETERMINISTIC_EVAL:=1}"

run_checkpoint_eval() {
  local name="$1"
  local targets="$2"
  local checkpoint_dir="$3"
  local out_dir="$FLOW_ROOT/$name"
  local log="$out_dir/run.log"
  local -a target_args
  read -r -a target_args <<< "$targets"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Skipping deterministic eval; checkpoint directory not found: $checkpoint_dir" >&2
    return 0
  fi
  if [[ -e "$out_dir" && -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty deterministic eval output: $out_dir"
    return 0
  fi

  mkdir -p "$out_dir"
  (
    cd "$HGBO_ROOT"
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.reporting.original_stable_training \
      --targets "${target_args[@]}" \
      --epochs 0 \
      --seed "$SEED" \
      --lr 0.005 \
      --mae-lr 0.002 \
      --grad-clip 1.0 \
      --lr-decay-factor 0.95 \
      --lr-decay-interval 10 \
      --weight-decay 0.0 \
      --device cpu \
      --cpu-threads "$CPU_THREADS" \
      --num-workers "$NUM_WORKERS" \
      --loader-rng-mode "$LOADER_RNG_MODE" \
      --init-checkpoint-dir "$checkpoint_dir" \
      --deterministic-eval \
      --output-dir "$out_dir"
  ) > "$log" 2>&1
}

env \
  FLOW_ROOT="$FLOW_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  CPU_THREADS="$CPU_THREADS" \
  NUM_WORKERS="$NUM_WORKERS" \
  SEED="$SEED" \
  MAPE_EPOCHS="$MAPE_EPOCHS" \
  DSP_EPOCHS="$DSP_EPOCHS" \
  BRAM_CALIBRATOR_ESTIMATORS="$BRAM_CALIBRATOR_ESTIMATORS" \
  RUN_REFERENCE_EVAL="$RUN_REFERENCE_EVAL" \
  RUN_ORIGINAL_TARGETS=0 \
  RUN_ORIGINAL_MAPE_TARGETS="$RUN_ORIGINAL_MAPE_TARGETS" \
  RUN_ORIGINAL_DSP="$RUN_ORIGINAL_DSP" \
  RUN_ORIGINAL_BRAM="$RUN_ORIGINAL_BRAM" \
  RUN_BRAM_RESIDUAL_CALIBRATOR="$RUN_BRAM_RESIDUAL_CALIBRATOR" \
  RUN_ARCH_VERIFY="$RUN_ARCH_VERIFY" \
  LOADER_RNG_MODE="$LOADER_RNG_MODE" \
  ORIGINAL_TRAIN_DETERMINISTIC_EVAL="$ORIGINAL_TRAIN_DETERMINISTIC_EVAL" \
  REFERENCE_DETERMINISTIC_EVAL="$REFERENCE_DETERMINISTIC_EVAL" \
  "$HGBO_ROOT/scripts/run_new_stack_hgbo_dse_flow.sh"

if [[ "$RUN_DETERMINISTIC_CHECKPOINT_EVAL" == "1" ]]; then
  run_checkpoint_eval \
    "deterministic_eval_mape_targets" \
    "lut ff cp power" \
    "$FLOW_ROOT/original_mape_targets_lr005_wd0_decay095_${MAPE_EPOCHS}ep/hgp/model"
  run_checkpoint_eval \
    "deterministic_eval_dsp" \
    "dsp" \
    "$FLOW_ROOT/original_dsp_lr002_wd0_decay095_${DSP_EPOCHS}ep/hgp/model"
fi

echo "v2 production reproduction outputs: $FLOW_ROOT"
