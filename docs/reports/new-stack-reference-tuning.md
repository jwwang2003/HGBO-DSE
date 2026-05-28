# New Stack Reference Tuning Log

Goal: train HGBO-DSE HGP from scratch under the new Python/Torch/PyG stack while keeping the required legacy SAGPooling shim enabled. The target is the built-in reference checkpoint quality, not checkpoint initialization.

All runs use:

- `HGBO_LEGACY_SAGPOOL=1`
- seed `128`
- original HGP graph model shape: hidden `64`, layers `3`, dropout `0`
- stable trainer: `python -m hgp.reporting.original_stable_training`
- isolated output directories under `img/training`

## Production Refresh: 2026-05-21 Overnight

The older sections below are retained as tuning history. The current
production-candidate v2 result is documented in
[`v2-production-reproduction.md`](v2-production-reproduction.md).

Full refresh command:

```bash
FLOW_ROOT=img/training/v2_production_reproduction_20260521_overnight \
CPU_THREADS=16 \
MAPE_EPOCHS=500 \
DSP_EPOCHS=500 \
BRAM_CALIBRATOR_ESTIMATORS=500 \
scripts/run_v2_production_reproduction.sh
```

Refreshed deterministic checkpoint metrics:

| Target | Metric | Deterministic test | Status |
| --- | ---: | ---: | --- |
| LUT | MAPE | `0.097056` | inside production gate, still behind paper |
| FF | MAPE | `0.051250` | inside production gate |
| CP | MAPE | `0.055517` | inside production gate, near paper |
| Power | MAPE | `0.079956` | inside production gate |
| DSP | MAE | `0.521959` | inside production gate, better than paper |
| BRAM residual calibrator | MAE | `0.022958` | inside production gate, better than paper |

This supersedes the earlier conclusion that BRAM is the remaining production
blocker. Pure HGP BRAM regression is still a negative diagnostic result, but
the v2 production path now uses the residual calibrator and reaches paper-level
accuracy from scratch.

## Architecture-Embedding Follow-Up: 2026-05-22

The architecture-embedding follow-up is documented in
[`v2-production-reproduction.md`](v2-production-reproduction.md) and
[`arch-aware-hgbo-dse-report.md`](arch-aware-hgbo-dse-report.md).

Run command:

```bash
FLOW_ROOT=img/training/v2_arch_embedding_experiment_20260522 \
CPU_THREADS=16 \
MAPE_EPOCHS=500 \
DSP_EPOCHS=500 \
BRAM_EPOCHS=500 \
BRAM_CALIBRATOR_ESTIMATORS=500 \
scripts/run_v2_arch_embedding_experiment.sh
```

Deterministic checkpoint metrics:

| Target | Metric | Architecture deterministic test | Interpretation |
| --- | ---: | ---: | --- |
| LUT | MAPE | `0.105753` | worse than stable non-arch v2 |
| FF | MAPE | `0.052988` | close, slightly worse than stable non-arch v2 |
| CP | MAPE | `0.053493` | paper-level and better than stable non-arch v2 |
| Power | MAPE | `0.080039` | close, roughly tied with stable non-arch v2 |
| DSP | MAE | `0.625710` | worse than stable non-arch v2 |
| BRAM raw HGP | MAE | `0.404357` | much better than the old raw failure, still not paper-level |
| BRAM residual calibrator | MAE | `0.022958` | production BRAM path, better than paper |

Conclusion: architecture embeddings should be used selectively in the current
v2 production recipe. They are the best observed path for CP, but not for LUT
or DSP. BRAM remains production-quality only through the residual calibrator.

## Reference-Weight Targets

Measured with:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff dsp bram cp power \
  --epochs 0 \
  --seed 128 \
  --mae-lr 0.00001 \
  --cpu-threads 16 \
  --init-from-builtins \
  --output-dir img/training/new_stack_legacy_builtin_reference_all_targets_20260519
```

| Target | Metric | Reference best test |
| --- | ---: | ---: |
| LUT | MAPE | 0.0865198 |
| FF | MAPE | 0.0415146 |
| DSP | MAE | 0.404010 |
| BRAM | MAE | 0.0770227 |
| CP | MAPE | 0.0465708 |
| Power | MAPE | 0.0736198 |

## Final New-Stack From-Scratch Status

These are the best confirmed from-scratch results from this tuning pass. They are not checkpoint-initialized runs.

| Target | Metric | Reference-weight best test | New-stack from-scratch best test | Status |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | 0.086520 | 0.104494 | usable, behind reference |
| FF | MAPE | 0.041515 | 0.052637 | usable, behind reference |
| DSP | MAE | 0.404010 | 0.551700 | improved substantially, still behind reference |
| BRAM | MAE | 0.077023 | 0.972186 | failed to reproduce reference or older run |
| CP | MAPE | 0.046571 | 0.059249 | usable, behind reference |
| Power | MAPE | 0.073620 | 0.075657 | near reference |

Deterministic full-test comparison against the built-in checkpoints, using
`shuffle=False` and `drop_last=False` for saved-checkpoint evaluation:

| Target | Metric | Built-in deterministic test | Best new-stack deterministic test | Absolute gap | Relative gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUT | MAPE | 0.085090 | 0.104430 | +0.019340 | 22.7% worse |
| FF | MAPE | 0.041322 | 0.052320 | +0.010998 | 26.6% worse |
| DSP | MAE | 0.408498 | 0.549132 | +0.140634 | 34.4% worse |
| BRAM | MAE | 0.078283 | 0.971820 | +0.893537 | 12.41x worse |
| CP | MAPE | 0.046914 | 0.062653 | +0.015739 | 33.5% worse |
| Power | MAPE | 0.075617 | 0.080316 | +0.004699 | 6.2% worse |

Interpretation: excluding BRAM, the best new-stack checkpoints are roughly
25% worse than the built-in checkpoints on average. Power is close; LUT/FF are
usable but behind; DSP/CP remain meaningfully behind. BRAM dominates the
remaining gap and is not close to built-in quality. The best BRAM number here
comes from a low-LR nonzero-balanced fine-tune of the raw BRAM checkpoint; it is
an improvement over the raw deterministic BRAM checkpoint (`1.040144`) but still
far from the built-in reference (`0.078283`).

Output directories:

- reference-weight evaluation: `img/training/new_stack_legacy_builtin_reference_all_targets_20260519`
- DSP tuned 500-epoch run: `img/training/new_stack_legacy_from_scratch_dsp_seed128_lr002_wd0_decay095_500ep_20260519`
- CP/power 500-epoch run: `img/training/new_stack_legacy_from_scratch_cp_power_seed128_lr005_wd0_decay095_500ep_20260519`
- BRAM 500-epoch run: `img/training/new_stack_legacy_from_scratch_bram_seed128_lr001_wd1e3_decay09_500ep_20260519`
- BRAM nonzero-balanced fine-tune: `img/training/new_stack_bram_finetune_nonzero_balanced_from_raw_seed128_lr0001_det_eval_100ep_20260520`
- LUT/FF checkpoint re-evaluation: `img/training/new_stack_legacy_from_scratch_lut_ff_recovered_eval_20260519`

Important evaluation caveat: the original HGBO-DSE loaders use `shuffle=True, drop_last=True` for the test loader. That paper-style behavior was preserved for comparison, but it means a saved best-test checkpoint can re-evaluate to a different number after reload because a different partial test subset may be dropped. In the interrupted LUT/FF run, checkpoint payloads recorded LUT `0.097270` and FF `0.050839`; the fresh paper-style re-evaluation reported LUT `0.104494` and FF `0.052637`. The table above uses the fresh re-evaluation numbers.

## DSP Tuning Results

The initial improved run was:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets dsp \
  --epochs 500 \
  --seed 128 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.9 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_dsp_seed128_lr002_wd0_500ep_20260519
```

Result: DSP best test MAE `0.862715`.

Pilot sweep command:

```bash
env RUN_ROOT="$PWD/img/training/new_stack_reference_tuning_dsp_pilots_20260519" \
  RUN_REFERENCE=0 \
  RUN_DSP_PILOTS=1 \
  RUN_DSP_FULL=0 \
  RUN_ALL_TARGETS=0 \
  DSP_PILOT_EPOCHS=200 \
  CPU_THREADS=16 \
  scripts/run_new_stack_reference_tuning.sh
```

| Run | Epochs | MAE LR | Weight decay | LR decay | Grad clip | DSP best test MAE | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dsp_lr003_wd0_decay09_clip1_200ep` | 200 | 0.003 | 0 | 0.9 / 10 | 1 | 0.842195 | Better than 500-epoch `0.002/0.9` pilot at lower budget |
| `dsp_lr002_wd0_decay095_clip1_200ep` | 200 | 0.002 | 0 | 0.95 / 10 | 1 | 0.801528 | Best confirmed pilot |
| `dsp_lr002_wd0_decay09_clip5_200ep` | 200 | 0.002 | 0 | 0.9 / 10 | 5 | 0.968966 | Looser clipping is worse |
| `dsp_lr003_wd0_decay095_clip1_200ep` | 200 | 0.003 | 0 | 0.95 / 10 | 1 | 1.842381 | Higher LR held longer is unstable |
| `dsp_lr004_wd0_decay095_clip1_200ep` | aborted | 0.004 | 0 | 0.95 / 10 | 1 | log only | Aborted after poor early trajectory |

Current follow-up:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets dsp \
  --epochs 500 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.95 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_dsp_seed128_lr002_wd0_decay095_500ep_20260519
```

Result: DSP best test MAE `0.551700`, best train MAE `0.466435`.

Reproduction note: do not pass `--deterministic-eval` during from-scratch DSP
training when trying to match this run. The flag changes the test `DataLoader`
from shuffled/drop-last to ordered/full. Because the training and evaluation
loaders share the process RNG stream, that changes the next epoch's shuffled
training order after epoch 0. The corrected full-flow reproduction trains DSP
with paper-style eval mechanics, then runs a separate deterministic checkpoint
evaluation. In the 2026-05-21 scratch run this recovered the same paper-style
best MAE `0.551700` and deterministic checkpoint MAE `0.549132`.

New training runs now default to `--loader-rng-mode isolated` so eval iteration
does not perturb future train shuffles. Use `--loader-rng-mode legacy-shared`
only when trying to replay the historical coupled-RNG curve exactly.

This is the closest new-stack from-scratch result so far:

- reference DSP test MAE: `0.404010`
- tuned new-stack DSP test MAE: `0.551700`
- previous `lr=0.002`, decay `0.9`, 500-epoch DSP test MAE: `0.862715`

## Current Interpretation

For DSP, the new stack can train from scratch, but the original stable defaults were too conservative or over-regularized. The best observed direction is:

- no weight decay for DSP/BRAM-style MAE targets
- `mae_lr=0.002`
- keep gradient clip at `1.0`
- slower LR decay (`0.95` every 10 epochs)

The remaining gap to reference DSP MAE `0.404010` is now smaller but still real. Since tuned train MAE `0.466435` is close to the reference train MAE observed from the built-in checkpoint (`0.4553`), the remaining issue is not only underfitting. The next likely experiments are seed sensitivity, target normalization/log-space training, or a closer reproduction of the legacy optimizer/data-loader behavior.

## Other Target Runs

After selecting the DSP schedule, the remaining targets were first trained with:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff bram cp power \
  --epochs 500 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.95 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_other_targets_seed128_lr005_mae002_wd0_decay095_500ep_20260519
```

The combined non-DSP run was stopped during BRAM because BRAM failed badly under the DSP-tuned schedule.

Recovered test-checkpoint payload metrics from `img/training/new_stack_legacy_from_scratch_other_targets_seed128_lr005_mae002_wd0_decay095_500ep_20260519/hgp/model`:

| Target | Metric | Best test | Reference | Status |
| --- | ---: | ---: | ---: | --- |
| LUT | MAPE | 0.097270 | 0.086520 | close, still worse than reference |
| FF | MAPE | 0.050839 | 0.041515 | close, still worse than reference |
| BRAM | MAE | 7.454030 | 0.077023 | failed |

Conclusion: do not use the DSP-tuned MAE schedule for every target. A quick original-style MAPE run was attempted:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff cp power \
  --epochs 500 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.001 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.9 \
  --lr-decay-interval 10 \
  --weight-decay 0.001 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_mape_targets_seed128_lr005_wd1e3_decay09_500ep_20260519
```

It was stopped early because LUT was clearly worse than the no-weight-decay, slower-decay run at the same early epoch range.

CP and power were then trained separately with the better MAPE schedule:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets cp power \
  --epochs 500 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.95 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_cp_power_seed128_lr005_wd0_decay095_500ep_20260519
```

Result: CP best test MAPE `0.059249`; power best test MAPE `0.075657`.

LUT and FF were re-evaluated from the interrupted run's checkpoints:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff \
  --epochs 0 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.002 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.95 \
  --lr-decay-interval 10 \
  --weight-decay 0.0 \
  --cpu-threads 16 \
  --init-checkpoint-dir img/training/new_stack_legacy_from_scratch_other_targets_seed128_lr005_mae002_wd0_decay095_500ep_20260519/hgp/model \
  --output-dir img/training/new_stack_legacy_from_scratch_lut_ff_recovered_eval_20260519
```

Result: LUT best test MAPE `0.104494`; FF best test MAPE `0.052637`.

BRAM was rerun with the older default-style MAE schedule because the DSP schedule failed:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets bram \
  --epochs 500 \
  --seed 128 \
  --lr 0.005 \
  --mae-lr 0.001 \
  --grad-clip 1.0 \
  --lr-decay-factor 0.9 \
  --lr-decay-interval 10 \
  --weight-decay 0.001 \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_from_scratch_bram_seed128_lr001_wd1e3_decay09_500ep_20260519
```

Result: BRAM best test MAE `0.972186`. The earlier run `img/training/original_stable_dsp_bram_seed128_full` reached BRAM `0.457353` with the same visible hyperparameters, so the current new-stack rerun is a regression for BRAM.

## Reproduction Script

The reproducible tuning wrapper is:

```bash
scripts/run_new_stack_reference_tuning.sh
```

Useful modes:

```bash
# Reference-weight evaluation.
RUN_REFERENCE=1 RUN_DSP_PILOTS=0 RUN_DSP_FULL=0 RUN_MAPE_TARGETS=0 RUN_BRAM_DEFAULT=0 scripts/run_new_stack_reference_tuning.sh

# DSP pilot sweep.
RUN_REFERENCE=0 RUN_DSP_PILOTS=1 RUN_DSP_FULL=0 RUN_MAPE_TARGETS=0 RUN_BRAM_DEFAULT=0 scripts/run_new_stack_reference_tuning.sh

# Best observed DSP full run.
RUN_REFERENCE=0 RUN_DSP_PILOTS=0 RUN_DSP_FULL=1 RUN_MAPE_TARGETS=0 RUN_BRAM_DEFAULT=0 scripts/run_new_stack_reference_tuning.sh

# MAPE targets with the best observed no-weight-decay schedule.
RUN_REFERENCE=0 RUN_DSP_PILOTS=0 RUN_DSP_FULL=0 RUN_MAPE_TARGETS=1 RUN_BRAM_DEFAULT=0 scripts/run_new_stack_reference_tuning.sh

# BRAM default-style run.
RUN_REFERENCE=0 RUN_DSP_PILOTS=0 RUN_DSP_FULL=0 RUN_MAPE_TARGETS=0 RUN_BRAM_DEFAULT=1 scripts/run_new_stack_reference_tuning.sh
```

## Conclusion

The new stack with the required legacy SAGPooling shim can train usable models for most targets, but this tuning pass did not reach reference-weight quality across the board. Power is near reference, DSP improved from `0.862715` to `0.551700`, and CP/LUT/FF are usable but still worse than reference. BRAM is the blocker: the best new-stack BRAM run here is `0.972186`, much worse than both the built-in reference `0.077023` and the earlier same-hyperparameter run `0.457353`.

For thesis-quality reporting, use the current runs as negative/diagnostic evidence rather than claiming full paper-level reproduction. The next technical step should be either deterministic full-test evaluation plus retuned checkpoint selection, or target-specific BRAM treatment such as label normalization/log-space training. If the immediate requirement is to reproduce the original paper-level checkpoints, `.venv113` remains the cleanest training path.

## BRAM Root-Cause Update: 2026-05-20

The released BRAM checkpoint is not failing because of Torch checkpoint
compatibility. It evaluates under the new stack at deterministic test MAE
`0.078283` when loaded with the SAGPooling compatibility path.

The worse fresh BRAM runs are training failures. The deterministic test split
shows that `hls_attr[3]` is the HLS BRAM estimate and is already a strong
baseline:

| Predictor | Deterministic BRAM MAE |
| --- | ---: |
| Zero BRAM | `7.510817` |
| Raw `hls_attr[3]` | `0.268433` |
| Current legacy-SAGPool HGP checkpoint | `1.040144` |
| HLS residual HGP early checkpoint | `0.305510` |
| Built-in HGP checkpoint | `0.078283` |
| ExtraTrees classifier on `true_bram - hls_attr[3]` | `0.022958` |

The residual distribution is highly discrete and imbalanced:

```text
residual = true_bram - hls_attr[3]
0: 2028 test samples
-3: 81
-1: 75
4: 54
-2: 17
-4: 10
```

This explains the observed BRAM tail behavior. Regression HGP training mostly
learns the dominant zero-residual/near-zero-BRAM cases and misses rare residual
classes, while the released checkpoint somehow learned or memorized the offset
classes. A BRAM-specific residual classifier over the existing graph/global
features corrects the issue on the paper split, but it is a target-specific
postprocessor rather than a pure HGP+SAGE+GF reproduction.

Implementation updates:

- `hgp.reporting.original_stable_training` now supports
  `--hls-residual-index 3` so BRAM experiments can train/evaluate
  `model_output + hls_attr[3]`.
- `hgp.reporting.checkpoint_prediction_stats` reads this setting from
  checkpoint metadata.
- The new-stack flow scripts pass the BRAM residual option for BRAM-only
  training stages.
- `hgp.reporting.bram_residual_calibrator` trains a from-scratch
  `ExtraTreesClassifier` on the discrete residual
  `true_bram - hls_attr[3]`. With seed `128` and 500 trees, the verified
  from-scratch flow run reached deterministic BRAM test MAE `0.022958` and
  residual-class accuracy `0.988962`.

Reproduce the BRAM from-scratch calibrator:

```bash
env RUN_REFERENCE_EVAL=0 RUN_ORIGINAL_TARGETS=0 \
  RUN_BRAM_RESIDUAL_CALIBRATOR=1 RUN_ARCH_VERIFY=0 \
  FLOW_ROOT=img/training/flow_bram_residual_calibrator_from_scratch_20260521 \
  CPU_THREADS=16 scripts/run_new_stack_hgbo_dse_flow.sh
```

## Commit-Prep Update: 2026-05-20

The reusable new-stack flow is now:

```bash
scripts/run_new_stack_hgbo_dse_flow.sh
```

It runs built-in reference evaluation by default and keeps the expensive
from-scratch original HGP and architecture-aware verification stages opt-in.
Generated datasets, architecture caches, reproduced figures, training outputs,
and root-level generated model checkpoints are ignored in `.gitignore` so the
commit surface stays focused on source, scripts, tests, and experiment logs.
