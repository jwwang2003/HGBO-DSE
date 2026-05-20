# Original HGBO-DSE Reproduction Notes

The original HGBO-DSE HGP training code is sensitive to its Python dependency
stack and to dataset file ordering. The thesis fork uses a newer Torch/PyG stack
for backend and architecture-aware work, but original paper-style retraining
should use the legacy HGP environment.

## Root Causes

1. Dependency drift:
   - Original README stack: Python 3.9, Torch 1.13.1, PyG 2.3.1,
     torch-scatter 2.1.1, torch-sparse 0.6.17.
   - Thesis `.venv` stack: Torch 2.6.0, PyG 2.6.0.
   - The newer stack changes `SAGPooling` checkpoint layout and training
     behavior. The built-in checkpoints can be loaded with compatibility code,
     but fresh original retraining under Torch 2.6 was unstable for DSP/BRAM.
   - PyG 2.6 adds a trainable `SAGPooling.select.weight` parameter that does
     not exist in PyG 2.3. For scalar pooling scores, a negative selector flips
     the score sign before top-k selection. With seed `128`, two of the three
     selectors initialize negative in the default PyG 2.6 model.
   - The extra selector initialization also consumes RNG during model
     construction, so later convolution/pooling/MLP weights no longer match the
     old stack even if `select.weight` is later overwritten.

2. Dataset loading:
   - `dataset/std` and `dataset/rdc` contain tracked `README.md` files.
   - The original scripts pass `os.listdir(dataset_dir)` into
     `generate_dataset()`.
   - Without filtering, direct original-script runs can try to `torch.load()` a
     README file. Without sorting, the train/test split can depend on filesystem
     directory order.

3. Random initialization:
   - The original scripts seed the NumPy dataset shuffle and the
     `random_split()` generator, but they do not seed Torch model
     initialization.
   - Their `DataLoader(..., shuffle=True)` calls also consume Torch RNG state.
   - Fresh retraining is therefore not expected to exactly reproduce the bundled
     `.pt` checkpoint in one run unless the Torch seed is controlled.

4. Bundled checkpoint provenance:
   - The released DSP/BRAM checkpoints are named as MAE checkpoints, but their
     stored metadata key is `min_test_mape`, while the current DSP/BRAM scripts
     write `min_test_mae`.
   - This suggests the bundled weights were produced by a slightly different or
     earlier training script revision, even though the model architecture is
     compatible.

## Fixes In This Fork

- `hgp/dataset_utils.py` now ignores non-`.pt` files and sorts dataset file
  names before loading.
- `scripts/train_original_paper_baseline.sh` now prefers `.venv113/bin/python`
  and refuses to run paper-style retraining unless the original Torch/PyG
  versions are present, unless `ALLOW_COMPAT_DEPS=1` is set.
- `scripts/train_original_paper_baseline.sh` runs each legacy `hier_*_model.py`
  through a small seed wrapper. The default model/DataLoader seed is `128`, and
  it can be changed with `HGP_TORCH_SEED`.
- `requirements-original-hgp.txt` and `scripts/setup_original_hgp_env.sh`
  define the legacy HGP environment separately from the thesis `.venv`.
- `hgp/pyg_compat.py` provides an opt-in PyG 2.3-style `LegacySAGPooling`.
  Set `HGBO_LEGACY_SAGPOOL=1` to use it in HGP training models under the newer
  stack. This removes the extra PyG 2.6 selector parameter and preserves the
  old-stack initial model weights for the same Torch seed.

## Commands

Set up the original HGP environment:

```bash
scripts/setup_original_hgp_env.sh
```

Verify the released checkpoints without overwriting checked-in SVGs:

```bash
.venv113/bin/python scripts/regenerate_builtin_pred_svgs.py \
  --output-dir img/reproduced_builtin_original_env
```

Check that the paper-style training script will use the original HGP
dependencies without starting a training run:

```bash
CHECK_DEPS_ONLY=1 scripts/train_original_paper_baseline.sh
```

Run paper-style fresh retraining into a new non-overwriting run directory:

```bash
TARGETS="dsp bram" scripts/train_original_paper_baseline.sh
```

Run the same command with a different deterministic model seed:

```bash
HGP_TORCH_SEED=42 TARGETS="dsp" scripts/train_original_paper_baseline.sh
```

Run a new-stack diagnostic with old-style pooling enabled:

```bash
HGBO_LEGACY_SAGPOOL=1 .venv/bin/python -m hgp.reporting.original_stable_training \
  --targets dsp --epochs 150 --cpu-threads 16 \
  --output-dir img/training/new_stack_legacy_sagpool_dsp_150ep_seed128_20260519
```

Use the thesis `.venv` for architecture-aware experiments and backend work. Use
`.venv113` for original paper-style HGP retraining.

## Reproduction Interpretation

The released main-branch checkpoints reproduce paper-level HGP prediction
behavior. In the standalone main branch under the original dependency stack, the
built-in DSP checkpoint evaluated at about `0.4057` MAE, while one fresh
500-epoch DSP retrain reached `0.8775` MAE. This means the bundled checkpoint is
the reliable reproduction baseline, and fresh retraining should be described as
stable but initialization-sensitive. The seed wrapper fixes repeatability of new
fresh runs; it does not guarantee that seed `128` is the seed used for the
released checkpoint.

The new-stack compatibility probe confirms that `LegacySAGPooling` fixes the
model-construction mismatch: with `HGBO_LEGACY_SAGPOOL=1`, the HGP state dict
has no `select.weight` keys and the initial weights match the old stack exactly
for seed `128`. The compatibility layer also matches native PyG 2.3
`SAGPooling` under `.venv113`; under PyG 2.6 it matches the native operator when
`select.weight` is forced to `1.0`.

This does not fully recover old-stack training quality by itself. In
stable-trainer probes, default PyG 2.6 reached DSP best test MAE `6.375` after
80 epochs. New-stack training with `HGBO_LEGACY_SAGPOOL=1` reached DSP best
test MAE `2.809` after 150 epochs, with the best checkpoint at epoch `135`.
The old stack full 500-epoch fresh retrain still reached `0.8775`, and the
built-in checkpoint evaluates around `0.4057`. So the `SAGPooling` change is a
real compatibility issue and the shim helps, but it is not the only source of
the fresh-training gap. Torch/PyG runtime differences and the long,
seed-sensitive original training schedule still matter.
