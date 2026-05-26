# V2 Production Reproduction Runbook

This runbook tracks the production-candidate v2 stack. The goal is not a
one-to-one replay of the paper's exact training mechanics; the goal is to train
or evaluate models whose accuracy is close to or better than the Table IV paper
levels using the maintained v2 Python stack.

## Current Position

Use this interpretation when reporting the current state:

| Target | Paper level | Current v2 result | Status |
| --- | ---: | ---: | --- |
| LUT | `7.72%` MAPE | `9.706%` deterministic MAPE | usable, still the main non-BRAM gap |
| FF | `4.21%` MAPE | `5.125%` deterministic MAPE | close |
| CP | `5.39%` MAPE | `5.349%` deterministic MAPE | paper-level with architecture embeddings |
| Power | `7.39%` MAPE | `7.996%` deterministic MAPE | close |
| DSP | `0.57` MAE | `0.522` deterministic MAE | exceeds paper |
| BRAM | `0.09` MAE | `0.023` residual-calibrator MAE | exceeds paper, target-specific model |

The BRAM number is from the v2 residual calibrator, not from a pure
HGP+SAGE+GF BRAM retrain. That is acceptable for production accuracy, but it
should be stated explicitly in reports.

The current best v2 production interpretation is a hybrid: use the stable
non-architecture v2 retrain for LUT, FF, Power, and DSP; use the architecture
embedding model for CP; and use the BRAM residual calibrator for BRAM.

## Production Defaults

Use these defaults for new v2 runs:

- Python environment: `.venv`
- SAGPooling compatibility: `HGBO_LEGACY_SAGPOOL=1`
- Seed: `128`
- CPU threads: `16`
- Training eval mode: paper-style eval during training
  (`ORIGINAL_TRAIN_DETERMINISTIC_EVAL=0`)
- Loader RNG mode: isolated (`LOADER_RNG_MODE=isolated`)
- Final comparison mode: deterministic checkpoint evaluation
  (`--deterministic-eval`, `shuffle=False`, `drop_last=False`)
- BRAM production path: `hgp.reporting.bram_residual_calibrator`

`LOADER_RNG_MODE=isolated` is important. It gives train and eval loaders
separate `torch.Generator` streams so evaluation cannot perturb future train
shuffle order. Use `LOADER_RNG_MODE=legacy-shared` only to replay historical
coupled-RNG curves from earlier diagnostic runs.

## Full Reproduction Command

Run this from `3rdParty/HGBO-DSE`:

```bash
scripts/run_v2_production_reproduction.sh
```

The wrapper runs:

- built-in reference checkpoint evaluation
- from-scratch MAPE target training for `lut ff cp power`
- from-scratch DSP training
- from-scratch BRAM residual calibrator
- deterministic checkpoint evaluation for the trained MAPE and DSP checkpoints

The default output root is:

```text
img/training/v2_production_reproduction_<timestamp>
```

Override knobs as needed:

```bash
FLOW_ROOT=img/training/v2_production_reproduction_20260521 \
CPU_THREADS=16 \
MAPE_EPOCHS=500 \
DSP_EPOCHS=500 \
BRAM_CALIBRATOR_ESTIMATORS=500 \
scripts/run_v2_production_reproduction.sh
```

## Fast Smoke Command

Use this to verify the v2 pipeline wiring without waiting for a full training
run:

```bash
FLOW_ROOT=img/training/v2_production_smoke \
MAPE_EPOCHS=1 \
DSP_EPOCHS=1 \
BRAM_CALIBRATOR_ESTIMATORS=10 \
scripts/run_v2_production_reproduction.sh
```

This smoke run is not an accuracy run. It only verifies that the wrapper,
training entrypoints, checkpoint evaluation, and BRAM calibrator can execute.

## Latest Full Run: 2026-05-21 Overnight

Command:

```bash
FLOW_ROOT=img/training/v2_production_reproduction_20260521_overnight \
CPU_THREADS=16 \
MAPE_EPOCHS=500 \
DSP_EPOCHS=500 \
BRAM_CALIBRATOR_ESTIMATORS=500 \
scripts/run_v2_production_reproduction.sh
```

Output root:

```text
img/training/v2_production_reproduction_20260521_overnight
```

Training best metrics:

| Target | Metric | Best test | Production gate | Result |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | `0.092338` | `0.105` | pass |
| FF | MAPE | `0.047965` | `0.055` | pass |
| CP | MAPE | `0.054700` | `0.065` | pass |
| Power | MAPE | `0.077481` | `0.085` | pass |
| DSP | MAE | `0.515149` | `0.60` | pass |
| BRAM residual calibrator | MAE | `0.022958` | `0.09` | pass |

Deterministic checkpoint re-evaluation:

| Target | Metric | Deterministic test | Production gate | Result |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | `0.097056` | `0.105` | pass |
| FF | MAPE | `0.051250` | `0.055` | pass |
| CP | MAPE | `0.055517` | `0.065` | pass |
| Power | MAPE | `0.079956` | `0.085` | pass |
| DSP | MAE | `0.521959` | `0.60` | pass |

Consistency versus the previous recorded v2 deterministic results:

| Target | Previous deterministic | Overnight deterministic | Change |
| --- | ---: | ---: | ---: |
| LUT | `0.099050` | `0.097056` | better by `0.001994` |
| FF | `0.050353` | `0.051250` | worse by `0.000897` |
| CP | `0.061363` | `0.055517` | better by `0.005847` |
| Power | `0.079801` | `0.079956` | worse by `0.000155` |
| DSP | `0.549132` | `0.521959` | better by `0.027174` |
| BRAM residual calibrator | `0.022958` | `0.022958` | unchanged at shown precision |

Interpretation: the overnight run is consistent with the previous v2 result and
stronger overall. The only target still materially behind the paper table is
LUT, but it is inside the production gate. DSP now exceeds the paper-level
threshold from scratch under the maintained v2 stack. BRAM remains production
quality through the residual-calibrator path rather than pure HGP regression.

## Architecture-Embedding Experiment: 2026-05-22

Command:

```bash
FLOW_ROOT=img/training/v2_arch_embedding_experiment_20260522 \
CPU_THREADS=16 \
MAPE_EPOCHS=500 \
DSP_EPOCHS=500 \
BRAM_EPOCHS=500 \
BRAM_CALIBRATOR_ESTIMATORS=500 \
scripts/run_v2_arch_embedding_experiment.sh
```

Output root:

```text
img/training/v2_arch_embedding_experiment_20260522
```

The wrapper trains architecture-aware HGP models for all six targets, then
reloads each saved checkpoint with deterministic evaluation. It also reruns the
BRAM residual calibrator as the production BRAM comparator.

Training best metrics:

| Target | Metric | Architecture best test | Paper level | Result |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | `0.098046` | `0.0772` | inside production gate, behind paper |
| FF | MAPE | `0.052199` | `0.0421` | inside production gate, behind paper |
| CP | MAPE | `0.052797` | `0.0539` | better than paper |
| Power | MAPE | `0.076213` | `0.0739` | close to paper |
| DSP | MAE | `0.616334` | `0.57` | behind paper and prior v2 |
| BRAM raw HGP | MAE | `0.397993` | `0.09` | improved, still not enough |
| BRAM residual calibrator | MAE | `0.022958` | `0.09` | better than paper |

Deterministic checkpoint re-evaluation:

| Target | Metric | Architecture deterministic test | Production gate | Result |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | `0.105753` | `0.105` | just misses gate |
| FF | MAPE | `0.052988` | `0.055` | pass |
| CP | MAPE | `0.053493` | `0.065` | pass, paper-level |
| Power | MAPE | `0.080039` | `0.085` | pass |
| DSP | MAE | `0.625710` | `0.60` | miss |
| BRAM raw HGP | MAE | `0.404357` | `0.09` | miss |
| BRAM residual calibrator | MAE | `0.022958` | `0.09` | pass, better than paper |

Comparison to the previous non-architecture v2 deterministic run:

| Target | Non-arch deterministic | Arch deterministic | Change |
| --- | ---: | ---: | ---: |
| LUT | `0.097056` | `0.105753` | worse by `0.008697` |
| FF | `0.051250` | `0.052988` | worse by `0.001738` |
| CP | `0.055517` | `0.053493` | better by `0.002023` |
| Power | `0.079956` | `0.080039` | worse by `0.000083` |
| DSP | `0.521959` | `0.625710` | worse by `0.103751` |
| BRAM raw HGP | raw HGP failure history near `0.972` | `0.404357` | materially better but still not paper-level |
| BRAM residual calibrator | `0.022958` | `0.022958` | unchanged |

Interpretation: architecture embeddings are useful but not a global replacement
for the current single-device v2 stack. They make CP paper-level and materially
reduce the raw BRAM HGP failure, but LUT and DSP regress under the deterministic
reload comparison. Since all current labels target one FPGA, the architecture
embedding is mostly a constant device descriptor; that limits how much it can
learn beyond acting as target-specific regularization.

Recommended production mix after this experiment:

| Target | Recommended v2 path |
| --- | --- |
| LUT | non-architecture v2 HGP |
| FF | non-architecture v2 HGP |
| CP | architecture-embedding HGP |
| Power | non-architecture v2 HGP, with architecture result acceptable as a close fallback |
| DSP | non-architecture v2 HGP |
| BRAM | residual calibrator |

## Manual Deterministic Checkpoint Evaluation

For a trained MAPE checkpoint directory:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff cp power \
  --epochs 0 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.95 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --device cpu \
  --cpu-threads 16 \
  --loader-rng-mode isolated \
  --init-checkpoint-dir <checkpoint-dir>/hgp/model \
  --deterministic-eval \
  --output-dir <eval-output-dir>
```

For DSP, use the same command with `--targets dsp`.

## Acceptance Criteria

For production-candidate status, use deterministic checkpoint evaluation where
possible and compare against these target bands:

| Target | Production gate |
| --- | ---: |
| LUT | at or below `10.5%` MAPE for current v2, with follow-up tuning toward `7.72%` |
| FF | at or below `5.5%` MAPE |
| CP | at or below `6.5%` MAPE |
| Power | at or below `8.5%` MAPE |
| DSP | at or below `0.60` MAE |
| BRAM residual calibrator | at or below `0.09` MAE |

These gates reflect the current evidence and are intentionally practical. LUT
is the only major remaining accuracy gap outside pure neural BRAM.

## Verification Commands

Before treating a run or patch as reproducible, run:

```bash
.venv/bin/python -m pytest \
  tests/test_arch_aware_model.py \
  tests/test_bram_residual_calibrator.py \
  tests/test_stable_original_training.py \
  tests/test_original_reproduction_scripts.py \
  tests/test_prediction_checkpoint_loading.py \
  tests/test_pyg_compat.py \
  -q

git diff --check
```

Expected current result: all focused tests pass. In sandboxed Codex runs, pytest
may warn that it cannot write cache files outside the writable root; those
warnings do not indicate model or script failures.

## Current Review Notes

- The v2 stack can meet or exceed paper-level accuracy for DSP and BRAM when
  BRAM uses the residual-calibrator production path.
- Architecture embeddings improve CP to paper-level in deterministic reload
  evaluation, but they are not currently the best path for LUT or DSP.
- Built-in checkpoints remain useful as a reference oracle and should be kept
  in the release package.
- The main next accuracy work is LUT tuning, followed by multi-seed checks for
  the CP architecture-embedding result and small FF/Power sweeps if needed.
- Before production deployment, run multi-seed and leave-design-family-out
  validation. The paper split alone is not enough to prove generalization.
