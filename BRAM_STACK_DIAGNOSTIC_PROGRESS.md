# BRAM Stack Diagnostic Progress

Goal: identify why the new-stack HGBO-DSE HGP training does not reach reference-weight or paper-level MAE, with BRAM as the main blocker.

Rules for this diagnostic pass:

- Keep `HGBO_LEGACY_SAGPOOL=1` for all new-stack tests.
- Preserve paper-style training behavior unless explicitly evaluating deterministic checkpoints.
- Record each command/result here before moving to the next expensive run.
- Prefer deterministic checkpoint evaluation (`--deterministic-eval`) for comparing saved weights.

## Baseline From Previous Tuning Pass

| Target | Metric | Reference checkpoint, paper-style eval | Best new-stack run, paper-style eval |
| --- | ---: | ---: | ---: |
| LUT | MAPE | 0.086520 | 0.104494 |
| FF | MAPE | 0.041515 | 0.052637 |
| DSP | MAE | 0.404010 | 0.551700 |
| BRAM | MAE | 0.077023 | 0.972186 |
| CP | MAPE | 0.046571 | 0.059249 |
| Power | MAPE | 0.073620 | 0.075657 |

Important caveat found earlier: original HGBO-DSE uses `shuffle=True, drop_last=True` for the test loader. That makes checkpoint best-test numbers noisy because different test samples can be dropped after reload.

## Step 1: Deterministic Evaluation Patch

Added `--deterministic-eval` to `hgp.reporting.original_stable_training`.

Default behavior remains paper-style:

- test loader `shuffle=True`
- test loader `drop_last=True`

With `--deterministic-eval`:

- test loader `shuffle=False`
- test loader `drop_last=False`

This is only an evaluation-control switch. It does not change the architecture, SAGPooling shim, optimizer, or train loader behavior.

## Step 2: Deterministic Saved-Checkpoint Evaluations

Status: completed.

Planned evaluations:

| Case | Checkpoint source | Output directory | Result |
| --- | --- | --- | --- |
| Built-in reference all targets | `hgp/model` via `--init-from-builtins` | `img/training/deterministic_eval_builtin_reference_all_targets_20260520` | LUT MAPE `0.0850901`; FF MAPE `0.0413223`; DSP MAE `0.408498`; BRAM MAE `0.0782831`; CP MAPE `0.0469139`; power MAPE `0.0756167` |
| New-stack DSP tuned | `img/training/new_stack_legacy_from_scratch_dsp_seed128_lr002_wd0_decay095_500ep_20260519/hgp/model` | `img/training/deterministic_eval_new_stack_dsp_tuned_20260520` | DSP MAE `0.549132` |
| New-stack LUT/FF | `img/training/new_stack_legacy_from_scratch_other_targets_seed128_lr005_mae002_wd0_decay095_500ep_20260519/hgp/model` | `img/training/deterministic_eval_new_stack_lut_ff_20260520` | LUT MAPE `0.10443`; FF MAPE `0.0523203` |
| New-stack CP/power | `img/training/new_stack_legacy_from_scratch_cp_power_seed128_lr005_wd0_decay095_500ep_20260519/hgp/model` | `img/training/deterministic_eval_new_stack_cp_power_20260520` | CP MAPE `0.0626526`; power MAPE `0.0803161` |
| New-stack BRAM latest | `img/training/new_stack_legacy_from_scratch_bram_seed128_lr001_wd1e3_decay09_500ep_20260519/hgp/model` | `img/training/deterministic_eval_new_stack_bram_latest_20260520` | BRAM MAE `1.04014` |
| Earlier BRAM run | `img/training/original_stable_dsp_bram_seed128_full/hgp/model` | `img/training/deterministic_eval_original_stable_bram_prior_20260520` | failed to load: checkpoint contains `jkn.*` and `pools.*.select.weight`, so it is not the current main-branch `OriginalHierNet` state shape |

Command:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets lut ff dsp bram cp power \
  --epochs 0 \
  --seed 128 \
  --mae-lr 0.00001 \
  --cpu-threads 16 \
  --init-from-builtins \
  --deterministic-eval \
  --output-dir img/training/deterministic_eval_builtin_reference_all_targets_20260520
```

## Step 3: Controlled BRAM A/B

Status: completed. The earlier `original_stable_dsp_bram_seed128_full` run cannot be used as the clean old/reference comparison because its checkpoint includes JumpingKnowledge parameters that are not part of the current main-branch original model.

The A/B should compare:

- old stack training via `.venv113` (`torch 1.13.1+cpu`, `torch-geometric 2.3.1`), BRAM only
- new stack training via `.venv` (`torch 2.6.0+cpu`, `torch-geometric 2.6.0`), BRAM only

Both should use the same visible settings:

- target: `bram`
- seed: `128`
- epochs: `500`
- MAE LR: `0.001`
- weight decay: `0.001`
- grad clip: `1.0`
- LR decay: `0.9` every `10` epochs
- CPU threads: `16`

New-stack side is already available from:

```text
img/training/new_stack_legacy_from_scratch_bram_seed128_lr001_wd1e3_decay09_500ep_20260519
```

Old-stack side started with:

```bash
env PYTHONUNBUFFERED=1 HGBO_LEGACY_SAGPOOL=1 .venv113/bin/python -m hgp.reporting.original_stable_training \
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
  --output-dir img/training/old_stack_venv113_bram_seed128_lr001_wd1e3_decay09_500ep_20260520
```

Result:

| Stack | Output directory | Paper-style best test MAE | Deterministic full-test MAE |
| --- | --- | ---: | ---: |
| New stack `.venv` | `img/training/new_stack_legacy_from_scratch_bram_seed128_lr001_wd1e3_decay09_500ep_20260519` | 0.972186 | 1.04014 |
| Old stack `.venv113` | `img/training/old_stack_venv113_bram_seed128_lr001_wd1e3_decay09_500ep_20260520` | 0.864796 | 0.939161 |

The old-stack checkpoint was also deterministically evaluated under `.venv113`:

```text
img/training/deterministic_eval_old_stack_venv113_bram_under_venv113_20260520
```

That result was BRAM MAE `0.939011`, essentially identical to the new-stack evaluation of the same checkpoint (`0.939161`). Therefore, the deterministic BRAM measurement is not an evaluation-runtime artifact.

Interpretation: the old stack trains BRAM slightly better and crosses below MAE `1.0` earlier, but it still does not reproduce the reference checkpoint (`0.0782831`) or paper-level behavior. This weakens the hypothesis that the BRAM failure is caused only by the new Torch/PyG stack.

## Step 4: BRAM Residual Diagnostics

Status: completed.

Prediction-stat JSON outputs:

```text
img/training/bram_prediction_stats_buckets_20260520/reference_builtin_bram.json
img/training/bram_prediction_stats_buckets_20260520/new_stack_bram.json
img/training/bram_prediction_stats_buckets_20260520/old_stack_venv113_bram.json
```

Full deterministic BRAM test split:

- count: `2265`
- true BRAM median: `0.0`
- true BRAM p75: `0.0`
- true BRAM max: `384.0`

This means at least 75% of test examples have zero BRAM, while a small tail has large BRAM values.

Overall deterministic residuals:

| Checkpoint | MAE | Median abs error | Max abs error | Signed error mean |
| --- | ---: | ---: | ---: | ---: |
| Built-in reference | 0.078283 | 0.004279 | 3.652275 | +0.022865 |
| New stack BRAM | 1.040144 | 0.011977 | 148.552094 | -0.264535 |
| Old stack `.venv113` BRAM | 0.939161 | 0.005818 | 161.721825 | -0.199827 |

Bucketed MAE by true BRAM range:

| True BRAM bucket | Count | Reference MAE | New-stack MAE | Old-stack `.venv113` MAE |
| --- | ---: | ---: | ---: | ---: |
| `true == 0` | 1791 | 0.008963 | 0.031307 | 0.031073 |
| `0 < true <= 1` | 41 | 0.352217 | 0.820836 | 0.877164 |
| `1 < true <= 10` | 190 | 0.336897 | 0.453160 | 0.502296 |
| `10 < true <= 100` | 238 | 0.343194 | 7.281705 | 6.155638 |
| `true > 100` | 5 | 0.225299 | 89.411124 | 95.021324 |

Interpretation:

- The failed trained checkpoints are not uniformly bad. Their median absolute errors are small because most BRAM labels are zero.
- The MAE regression is dominated by the high-BRAM tail.
- The reference checkpoint models the high-BRAM tail well, with `0.225299` MAE on the `true > 100` bucket.
- Both from-scratch trained checkpoints fail badly on the tail: new stack `89.411124`, old stack `95.021324`.
- Therefore, the next useful fix is not another global LR tweak. The likely issue is tail learning/checkpoint selection for a highly imbalanced BRAM target distribution.

Next candidate tests:

1. Train BRAM with a target transform or weighted loss that gives nonzero/high-BRAM samples more influence.
2. Preserve original architecture exactly and check whether any hidden architectural difference explains why the built-in checkpoint captures the tail.
3. Use deterministic full-test evaluation for checkpoint selection to avoid selecting a checkpoint that only looks good on a shuffled/drop-last subset.

## Step 5: Tail-Aware BRAM Loss Experiment

Status: completed.

Trainer change:

- Added `--loss-weighting none|bram-tail`.
- Default remains `none`; existing training behavior is unchanged unless explicitly enabled.
- `bram-tail` uses normalized per-sample Huber loss weights:
  - `true == 0`: weight `1`
  - `0 < true <= 1`: `--bram-positive-weight`, default `2`
  - `1 < true <= 10`: `--bram-mid-weight`, default `4`
  - `10 < true <= 100`: `--bram-high-weight`, default `16`
  - `true > 100`: `--bram-extreme-weight`, default `32`

First run:

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
  --loss-weighting bram-tail \
  --bram-positive-weight 2 \
  --bram-mid-weight 4 \
  --bram-high-weight 16 \
  --bram-extreme-weight 32 \
  --deterministic-eval \
  --cpu-threads 16 \
  --output-dir img/training/new_stack_bram_tail_weighted_seed128_w2_4_16_32_det_eval_500ep_20260520
```

Success criterion:

- Primary: deterministic full-test BRAM MAE below the previous new-stack deterministic result `1.04014`.
- Strong signal: tail bucket MAE improves meaningfully, especially `10 < true <= 100` and `true > 100`.
- Reference remains BRAM MAE `0.0782831`, so this experiment is diagnostic unless it improves by an order of magnitude.

Result:

| Run | Deterministic BRAM MAE | Median abs error | `true == 0` MAE | `10 < true <= 100` MAE | `true > 100` MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference checkpoint | 0.078283 | 0.004279 | 0.008963 | 0.343194 | 0.225299 |
| Previous new-stack BRAM | 1.040144 | 0.011977 | 0.031307 | 7.281705 | 89.411124 |
| Tail-weighted new-stack BRAM | 1.035972 | 0.059055 | 0.146472 | 6.281361 | 78.535291 |

Output paths:

```text
img/training/new_stack_bram_tail_weighted_seed128_w2_4_16_32_det_eval_500ep_20260520
img/training/bram_prediction_stats_buckets_20260520/new_stack_bram_tail_weighted_w2_4_16_32.json
```

Conclusion:

- This weighted loss is a weak positive diagnostic, not a usable fix.
- It improves the high-BRAM tail somewhat: `true > 100` MAE improves from `89.411124` to `78.535291`, and `10 < true <= 100` improves from `7.281705` to `6.281361`.
- It damages the zero-heavy majority: `true == 0` MAE worsens from `0.031307` to `0.146472`, and median absolute error worsens from `0.011977` to `0.059055`.
- Overall deterministic MAE only improves from `1.040144` to `1.035972`, which is not meaningful relative to the reference `0.078283`.

Next implication: simple static loss weighting is not enough. If continuing BRAM-specific optimization, prefer a target transform or a two-stage/headed method that can learn the zero/nonzero split and high-BRAM magnitude separately.

## Step 6: New-Stack HGBO-DSE Flow Scaffold

Status: started and smoke-tested.

Added:

```text
scripts/run_new_stack_hgbo_dse_flow.sh
```

Purpose:

- Keep the new-stack HGBO-DSE flow reproducible under `.venv`.
- Always pin `HGBO_LEGACY_SAGPOOL=1` for original HGP training in the new stack.
- Keep output directories isolated and skip non-empty outputs instead of overwriting them.
- Use deterministic full-test evaluation by default for comparable saved-checkpoint metrics.
- Separate the stages:
  - built-in reference checkpoint evaluation
  - original HGP from-scratch training by target group
  - BRAM prediction/residual statistics
  - single-board architecture-aware verification through `hgp.reporting.compare_training_graphs`

Default behavior:

```bash
scripts/run_new_stack_hgbo_dse_flow.sh
```

This runs only the built-in reference evaluation into a timestamped directory under
`img/training/new_stack_hgbo_dse_flow_*`. Long training stages are opt-in.

Full dry-run command used to validate command construction:

```bash
env DRY_RUN=1 \
  RUN_REFERENCE_EVAL=1 \
  RUN_ORIGINAL_TARGETS=1 \
  RUN_BRAM_STATS=1 \
  RUN_ARCH_VERIFY=1 \
  FLOW_ROOT=/tmp/hgbo_dse_flow_dryrun_20260520 \
  scripts/run_new_stack_hgbo_dse_flow.sh
```

Smoke-test command actually executed:

```bash
timeout 1800 env RUN_REFERENCE_EVAL=1 \
  RUN_ORIGINAL_TARGETS=0 \
  RUN_BRAM_STATS=0 \
  RUN_ARCH_VERIFY=0 \
  FLOW_ROOT=/tmp/hgbo_dse_flow_reference_smoke_20260520 \
  scripts/run_new_stack_hgbo_dse_flow.sh
```

Smoke-test result matched the deterministic built-in reference evaluation:

| Target | Metric | Built-in checkpoint deterministic test |
| --- | ---: | ---: |
| LUT | MAPE | 0.0850901 |
| FF | MAPE | 0.0413223 |
| DSP | MAE | 0.408498 |
| BRAM | MAE | 0.0782831 |
| CP | MAPE | 0.0469139 |
| Power | MAPE | 0.0756167 |

Additional BRAM-stat smoke test for the flow parser:

```bash
timeout 900 env RUN_REFERENCE_EVAL=0 \
  RUN_ORIGINAL_TARGETS=0 \
  RUN_BRAM_STATS=1 \
  RUN_ARCH_VERIFY=0 \
  BRAM_CHECKPOINT_DIR=hgp/model \
  FLOW_ROOT=/tmp/hgbo_dse_flow_bram_stats_smoke_20260520 \
  scripts/run_new_stack_hgbo_dse_flow.sh
```

Result: BRAM MAE `0.0782831473`, median abs error `0.0042791143`,
max abs error `3.6522750854`.

Useful next commands:

```bash
# Full original HGP training in the new stack using the best known per-target schedules.
RUN_REFERENCE_EVAL=1 RUN_ORIGINAL_TARGETS=1 RUN_BRAM_STATS=1 scripts/run_new_stack_hgbo_dse_flow.sh

# Single-board architecture-aware verification only.
RUN_REFERENCE_EVAL=0 RUN_ORIGINAL_TARGETS=0 RUN_ARCH_VERIFY=1 scripts/run_new_stack_hgbo_dse_flow.sh
```

Interpretation:

- The flow is ready for controlled full original-HGP retraining under the new stack.
- The architecture-aware stage is intentionally a verification gate: it checks whether adding the embedded architecture representation degrades the network on the single built-in board.
- It is not yet multi-board architecture-aware training. The next architecture-aware implementation step should extend the dataset/cache/model flow to train against multiple board embeddings after the single-board verification gate is stable.

## Step 7: Repository Cleanup For Progress Push

Status: completed for commit preparation.

Cleanup actions:

- Added `.gitignore` entries for generated datasets, architecture caches,
  reproduced built-in figures, `img/training/`, root-level generated `model/`,
  `.pytest_cache/`, `.venv/`, and `.venv113/`.
- Removed local Python/test cache directories from the working tree.
- Preserved generated datasets, training outputs, checkpoints, and reproduced
  figures on disk; they are only hidden from the commit surface.

Commit surface after cleanup should be source, scripts, tests, requirements,
and experiment notes rather than multi-GB generated artifacts.

Verification:

```bash
git diff --check
bash -n scripts/run_new_stack_hgbo_dse_flow.sh scripts/run_new_stack_reference_tuning.sh scripts/train_original_paper_baseline.sh scripts/setup_original_hgp_env.sh scripts/run_stable_original_dsp_sweep.sh
.venv/bin/python -m pytest tests -q
```

Result: `179 passed`. Pytest emitted cache-write warnings because the parent
repository `.pytest_cache` path is read-only in the sandbox; source tests still
passed.

Recommended commit grouping:

1. Architecture-aware dataset/model/reporting support and tests.
2. Original-HGBO reproduction compatibility: dataset filtering, legacy
   SAGPooling shim, original-stack requirements, reproduction scripts, stable
   training/evaluation utilities, and tests.
3. Experiment documentation and new-stack flow scripts.

## Step 8: BRAM Target-Transform And Sampler Experiments

Status: completed first pass.

Trainer changes:

- Added `--target-transform none|log1p`.
  - Default `none` preserves previous behavior.
  - `log1p` trains against `log1p(true_y)` but evaluates/checkpoints using
    inverse-transformed predictions in original BRAM units.
- Added `--train-sampler none|bram-bucket-balanced|bram-nonzero-balanced`.
  - Default `none` preserves previous behavior.
  - `bram-bucket-balanced` gives the five BRAM buckets equal total sampling
    weight.
  - `bram-nonzero-balanced` gives zero and nonzero BRAM examples equal total
    sampling weight.
- `checkpoint_prediction_stats.py` now honors checkpoint `target_transform`
  metadata before computing residuals.

Distribution check:

| Split | Count | Zero | Nonzero | `10 < true <= 100` | `true > 100` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 9062 | 7169 | 1893 | 995 | 23 |
| Test | 2265 | 1791 | 474 | 238 | 5 |

This confirms that the BRAM target is extremely zero-heavy, with a tiny high
tail.

Experiments:

| Run | Command shape | Outcome |
| --- | --- | --- |
| Log1p target from scratch | 500 requested; stopped at epoch 17 | Bad trajectory. Best deterministic MAE `7.61339`; inverse transform produced early extreme MAE spikes. |
| Bucket-balanced sampler from scratch | 200 requested; stopped at epoch 31 | Too aggressive. Best deterministic MAE `4.09534`; much worse than raw baseline. |
| Nonzero-balanced sampler from scratch | 200 epochs completed | Better than bucket-balanced but still behind baseline. Best deterministic MAE `1.18267`. |
| Nonzero-balanced fine-tune from raw BRAM checkpoint, `mae_lr=0.0002` | 100 requested; stopped after improvement was observed | Improved deterministic MAE from `1.04014` to `0.987521`. |
| Nonzero-balanced fine-tune from raw BRAM checkpoint, `mae_lr=0.0001` | 100 epochs completed | Best so far: deterministic MAE `0.971820`. |

Best new-stack BRAM comparison:

| Checkpoint | Deterministic BRAM MAE | Median abs error | `true == 0` MAE | `10 < true <= 100` MAE | `true > 100` MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Built-in reference | 0.078283 | 0.004279 | 0.008963 | 0.343194 | 0.225299 |
| Previous new-stack raw BRAM | 1.040144 | 0.011977 | 0.031307 | 7.281705 | 89.411124 |
| Tail-weighted from scratch | 1.035972 | 0.059055 | 0.146472 | 6.281361 | 78.535291 |
| Nonzero-balanced fine-tune, `mae_lr=0.0001` | 0.971820 | 0.020276 | 0.088340 | 6.379794 | 83.385342 |

Interpretation:

- The first real improvement is a two-stage recipe: train raw BRAM normally,
  then fine-tune gently with nonzero-balanced sampling.
- The improvement is modest: deterministic MAE improves by about `6.6%`
  relative to the previous new-stack deterministic BRAM checkpoint.
- It helps the important `10 < true <= 100` bucket but still damages zero-BRAM
  predictions and does not solve the `true > 100` tail.
- This is still far from built-in reference quality, so it should be reported as
  progress on the new-stack reproduction problem, not as a paper-level fix.

Next BRAM direction:

The remaining gap likely needs a model/objective change that separates the
zero/nonzero decision from positive-BRAM magnitude. The sampler results show
that more nonzero exposure helps only when applied gently after a good baseline,
but aggregate MAE is still too sensitive to harming the zero-heavy majority.
