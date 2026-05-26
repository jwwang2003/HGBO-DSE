#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGBO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

: "${RUN_ROOT:=$HGBO_ROOT/img/training/new_stack_reference_tuning_$STAMP}"
: "${PYTHON_BIN:=$HGBO_ROOT/.venv/bin/python}"
: "${CPU_THREADS:=16}"
: "${DSP_PILOT_EPOCHS:=200}"
: "${DSP_FULL_EPOCHS:=500}"
: "${MAPE_TARGET_EPOCHS:=500}"
: "${BRAM_EPOCHS:=500}"
: "${ALL_TARGET_EPOCHS:=500}"
: "${BRAM_HLS_RESIDUAL_INDEX:=3}"
: "${LOADER_RNG_MODE:=isolated}"
: "${RUN_REFERENCE:=1}"
: "${RUN_DSP_PILOTS:=1}"
: "${RUN_DSP_FULL:=0}"
: "${RUN_MAPE_TARGETS:=0}"
: "${RUN_BRAM_DEFAULT:=0}"
: "${RUN_ALL_TARGETS:=0}"

mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/tuning_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  printf 'run,targets,epochs,lr,mae_lr,grad_clip,lr_decay_factor,lr_decay_interval,weight_decay,init_from_builtins,best_metrics,output_dir\n' > "$SUMMARY_CSV"
fi

append_summary() {
  local name="$1"
  local out_dir="$2"
  local init_from_builtins="$3"

  "$PYTHON_BIN" - "$name" "$out_dir" "$SUMMARY_CSV" "$init_from_builtins" <<'PY'
import csv
import json
import sys
from pathlib import Path

name = sys.argv[1]
out_dir = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
init_from_builtins = sys.argv[4]
summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
settings = summary["settings"]
best_metrics = {
    target: {
        "metric": target_result["metric_name"],
        "best_test": target_result["best_test_metric"],
    }
    for target, target_result in summary["targets"].items()
}
with summary_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            name,
            " ".join(settings["targets"]),
            settings["epochs"],
            settings["lr"],
            settings["mae_lr"],
            settings["grad_clip"],
            settings["lr_decay_factor"],
            settings["lr_decay_interval"],
            settings["weight_decay"],
            init_from_builtins,
            json.dumps(best_metrics, sort_keys=True),
            out_dir,
        ]
    )
print(f"{name}: {json.dumps(best_metrics, sort_keys=True)}")
PY
}

run_training() {
  local name="$1"
  local targets="$2"
  local epochs="$3"
  local lr="$4"
  local mae_lr="$5"
  local grad_clip="$6"
  local lr_decay_factor="$7"
  local lr_decay_interval="$8"
  local weight_decay="$9"
  local init_flag="${10}"
  shift 10
  local out_dir="$RUN_ROOT/$name"
  local log="$out_dir/run.log"

  if [[ -e "$out_dir" && -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Skipping non-empty output: $out_dir"
    return
  fi

  mkdir -p "$out_dir"
  echo "[$(date +%H:%M:%S)] Running $name"
  (
    cd "$HGBO_ROOT"
    # shellcheck disable=SC2086
    env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 "$PYTHON_BIN" -m hgp.reporting.original_stable_training \
      --targets $targets \
      --epochs "$epochs" \
      --seed 128 \
      --lr "$lr" \
      --mae-lr "$mae_lr" \
      --grad-clip "$grad_clip" \
      --lr-decay-factor "$lr_decay_factor" \
      --lr-decay-interval "$lr_decay_interval" \
      --weight-decay "$weight_decay" \
      --cpu-threads "$CPU_THREADS" \
      --loader-rng-mode "$LOADER_RNG_MODE" \
      --output-dir "$out_dir" \
      $init_flag \
      "$@"
  ) > "$log" 2>&1

  append_summary "$name" "$out_dir" "$init_flag"
}

if [[ "$RUN_REFERENCE" == "1" ]]; then
  run_training "reference_builtin_all_targets" "lut ff dsp bram cp power" 0 0.005 0.00001 1.0 0.9 10 0.001 "--init-from-builtins"
fi

if [[ "$RUN_DSP_PILOTS" == "1" ]]; then
  run_training "dsp_lr003_wd0_decay09_clip1_${DSP_PILOT_EPOCHS}ep" "dsp" "$DSP_PILOT_EPOCHS" 0.005 0.003 1.0 0.9 10 0.0 ""
  run_training "dsp_lr002_wd0_decay095_clip1_${DSP_PILOT_EPOCHS}ep" "dsp" "$DSP_PILOT_EPOCHS" 0.005 0.002 1.0 0.95 10 0.0 ""
  run_training "dsp_lr002_wd0_decay09_clip5_${DSP_PILOT_EPOCHS}ep" "dsp" "$DSP_PILOT_EPOCHS" 0.005 0.002 5.0 0.9 10 0.0 ""
  run_training "dsp_lr003_wd0_decay095_clip1_${DSP_PILOT_EPOCHS}ep" "dsp" "$DSP_PILOT_EPOCHS" 0.005 0.003 1.0 0.95 10 0.0 ""
  run_training "dsp_lr004_wd0_decay095_clip1_${DSP_PILOT_EPOCHS}ep" "dsp" "$DSP_PILOT_EPOCHS" 0.005 0.004 1.0 0.95 10 0.0 ""
fi

if [[ "$RUN_DSP_FULL" == "1" ]]; then
  run_training "dsp_lr002_wd0_decay095_clip1_${DSP_FULL_EPOCHS}ep" "dsp" "$DSP_FULL_EPOCHS" 0.005 0.002 1.0 0.95 10 0.0 ""
fi

if [[ "$RUN_MAPE_TARGETS" == "1" ]]; then
  run_training "mape_targets_lr005_wd0_decay095_clip1_${MAPE_TARGET_EPOCHS}ep" "lut ff cp power" "$MAPE_TARGET_EPOCHS" 0.005 0.002 1.0 0.95 10 0.0 ""
fi

if [[ "$RUN_BRAM_DEFAULT" == "1" ]]; then
  run_training "bram_hls_residual_idx${BRAM_HLS_RESIDUAL_INDEX}_lr001_wd1e3_decay09_clip1_${BRAM_EPOCHS}ep" "bram" "$BRAM_EPOCHS" 0.005 0.001 1.0 0.9 10 0.001 "" \
    --hls-residual-index "$BRAM_HLS_RESIDUAL_INDEX"
fi

if [[ "$RUN_ALL_TARGETS" == "1" ]]; then
  run_training "all_targets_lr005_mae002_wd0_decay095_clip1_${ALL_TARGET_EPOCHS}ep" "lut ff dsp bram cp power" "$ALL_TARGET_EPOCHS" 0.005 0.002 1.0 0.95 10 0.0 ""
fi

echo "Summary written to $SUMMARY_CSV"
cat "$SUMMARY_CSV"
