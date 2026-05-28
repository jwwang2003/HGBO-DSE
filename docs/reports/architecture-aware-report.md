# RapidWright ATAPP-Style Architecture-Aware HGBO-DSE Report

## Scope

This document records the HGBO-DSE architecture-aware implementation after aligning the FPGA representation with `ref/notes/ATAPP.md`.

The supported default device for the existing HGBO-DSE raw labels is:

```text
xc7vx485tffg1761-2
```

The raw `dataset/std` and `dataset/rdc` samples are reused. Architecture-aware copies are written to `dataset/std_arch` and `dataset/rdc_arch`.

## Architecture

HGBO-DSE now supports two RapidWright architecture modes:

| Mode | Purpose | Default |
|---|---|---:|
| `atapp` | ATAPP-style compact FSR/clock-region layout representation | yes |
| `fabric` | Previous full-device fabric graph encoder | no |

The ATAPP-style path mirrors the representation described in the notes:

1. Load the FPGA device through RapidWright from `3rdParty/RapidWright`.
2. Select a representative clock region. The default policy chooses the clock region closest to the center of the available RapidWright clock-region grid. `--clock-region` can override it.
3. Reduce tile names/types into fixed categories:

```text
PAD=0, CLEL=1, CLEM=2, INT=3, INT_INTERFACE=4, DSP=5, BRAM=6, BRK=7
```

`PAD=0` is used as padding/dummy zero, matching the ATAPP example where empty slots are zero-filled.

4. Build the padded raw layout:

```text
atapp_arch_layout_raw: [360, 80, 4] int64
```

5. Compress the columnar layout to:

```text
atapp_arch_layout: [360, 1, 4] float32
```

The current compression rule takes the first non-empty row per column, then adds sinusoidal positional encoding directly to the compressed tensor.

6. Attach device/resource/technology/layout metadata:

```text
atapp_arch_metadata: [1, 21] float32
```

7. Encode architecture with a two-hidden-layer MLP over:

```text
flatten([360, 1, 4]) || metadata[21]
```

This is a 1440-dimensional flattened layout plus 21 metadata fields.

## Default Extraction Result

Command:

```bash
cd 3rdParty/HGBO-DSE
uv run python -m hgp.data_process.gen_dataset_board \
  --input-root dataset \
  --output-root dataset \
  --device xc7vx485tffg1761-2 \
  --arch-mode atapp \
  --force-cache
```

Observed output:

```text
Wrote 20 architecture-aware dataset files for xc7vx485tffg1761-2
```

Cache/sample verification:

```bash
uv run python -c "import torch; from pathlib import Path; cache=torch.load('dataset/board_arch/xc7vx485tffg1761-2_atapp_arch.pt', map_location='cpu', weights_only=False); sample=torch.load('dataset/std_arch/bfs.pt', map_location='cpu', weights_only=False)[0]; print('cache layout_raw', tuple(cache['atapp_arch_layout_raw'].shape), cache['atapp_arch_layout_raw'].dtype); print('cache layout', tuple(cache['atapp_arch_layout'].shape), cache['atapp_arch_layout'].dtype); print('cache metadata', tuple(cache['atapp_arch_metadata'].shape), cache['atapp_arch_metadata'].dtype); print('clock_region', cache['atapp_clock_region']); print('valid rows/cols', cache['atapp_valid_rows'], cache['atapp_valid_cols']); print('std_arch files', len(list(Path('dataset/std_arch').glob('*.pt')))); print('rdc_arch files', len(list(Path('dataset/rdc_arch').glob('*.pt')))); print('sample arch_attr', tuple(sample.arch_attr.shape)); print('sample atapp layout', tuple(sample.atapp_arch_layout.shape)); print('sample atapp metadata', tuple(sample.atapp_arch_metadata.shape));"
```

Observed output:

```text
cache layout_raw (360, 80, 4) torch.int64
cache layout (360, 1, 4) torch.float32
cache metadata (1, 21) torch.float32
clock_region X0Y3
valid rows/cols 51 191
std_arch files 10
rdc_arch files 10
sample arch_attr (1, 13)
sample atapp layout (360, 1, 4)
sample atapp metadata (1, 21)
```

## Code Changes

### RapidWright Environment

Files:

```text
pyproject.toml
hgp/rapidwright_env.py
```

`rapidwright` is installed from the sibling local checkout:

```toml
rapidwright = { path = "../RapidWright/python", editable = true }
```

`hgp/rapidwright_env.py` configures `RAPIDWRIGHT_PATH` and `CLASSPATH`, then imports `com.xilinx.rapidwright.device.Device`.

### ATAPP Architecture Extraction

File:

```text
hgp/atapp_arch.py
```

Main API:

```text
classify_atapp_tile()
sinusoidal_positional_encoding()
build_atapp_metadata()
extract_atapp_architecture()
ensure_atapp_arch_cache()
load_atapp_arch_cache()
atapp_arch_to_device()
```

The cache is written to:

```text
dataset/board_arch/xc7vx485tffg1761-2_atapp_arch.pt
```

### Dataset Augmentation

Files:

```text
hgp/board_utils.py
hgp/data_process/gen_dataset_board.py
```

Each augmented PyG sample now keeps the legacy scalar profile and also receives ATAPP tensors:

```text
sample.arch_attr
sample.board_device
sample.board_family
sample.board_arch_device
sample.atapp_arch_layout
sample.atapp_arch_metadata
sample.atapp_arch_device
sample.atapp_clock_region
```

`gen_dataset_board.py` defaults to `--arch-mode atapp`. Use `--arch-mode fabric` to regenerate only the previous full fabric cache path, or `--arch-mode both` to build both caches.

### Model And Training

File:

```text
hgp/hier_arch_model.py
```

New model pieces:

```text
AtappArchitectureEncoder
ArchAwareHierNet(..., arch_mode="atapp")
```

Important behavior:

- `--arch-mode atapp` is the default.
- ATAPP mode uses only the design embedding, HLS attributes, and ATAPP architecture MLP output.
- ATAPP mode does not concatenate the legacy scalar `arch_attr` branch into the final predictor.
- Fabric mode is still available with `--arch-mode fabric`.
- Fabric mode ignores attached ATAPP tensors and uses the legacy `BoardFabricEncoder`.
- `--fabric-mode cached/trainable` is preserved for the legacy fabric graph path.

### Reporting

File:

```text
hgp/reporting/compare_training_graphs.py
```

The comparison runner now defaults to original HGBO-DSE vs ATAPP-aware HGBO-DSE:

```bash
uv run python -m hgp.reporting.compare_training_graphs --arch-mode atapp
```

Use the previous full-device fabric representation with:

```bash
uv run python -m hgp.reporting.compare_training_graphs --arch-mode fabric
```

## Tests

Focused tests cover:

- tile classification,
- positional encoding values,
- metadata shape and normalized defaults,
- fake RapidWright device extraction,
- nested RapidWright clock-region arrays,
- cache round-trip,
- dataset augmentation with ATAPP tensors,
- ATAPP encoder sensitivity to layout and metadata,
- ATAPP mode ignoring legacy scalar `arch_attr`,
- fabric mode ignoring attached ATAPP tensors,
- reporting outputs.

Command:

```bash
uv run pytest \
  tests/test_atapp_architecture.py \
  tests/test_arch_aware_model.py \
  tests/test_board_architecture.py \
  tests/test_training_graph_report.py \
  tests/test_uv_project_config.py \
  -q
```

## Training Commands

Single target, default ATAPP mode:

```bash
uv run python -m hgp.hier_arch_model \
  --target lut \
  --epochs 30 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --arch-mode atapp \
  --model-dir /tmp/hgbo_atapp_lut_30
```

Legacy full-fabric mode:

```bash
uv run python -m hgp.hier_arch_model \
  --target lut \
  --epochs 30 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --arch-mode fabric \
  --fabric-mode cached \
  --model-dir /tmp/hgbo_fabric_lut_30
```

Generate fresh ATAPP multi-seed comparison results on forced CPU:

```bash
PYTHONUNBUFFERED=1 uv run python -m hgp.reporting.compare_training_graphs \
  --epochs 30 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --arch-mode atapp \
  --device cpu \
  --cpu-threads 16 \
  --num-workers 0 \
  --seeds 128 256 512 \
  --output-dir img/training/atapp_multi_seed_cpu_threads16_workers0
```

An attempted forced-CPU run with `--num-workers 4` was stopped because worker startup/IPC overhead made it slower for this small dataset. The results below use the restarted `--cpu-threads 16 --num-workers 0` run.

Generated files:

```text
img/training/atapp_multi_seed_cpu_threads16_workers0/hgbo_arch_consistency_summary.json
img/training/atapp_multi_seed_cpu_threads16_workers0/hgbo_arch_consistency_summary.csv
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_128/hgbo_arch_training_history.json
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_128/hgbo_arch_training_curves.svg
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_128/hgbo_arch_best_test_delta.svg
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_256/hgbo_arch_training_history.json
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_256/hgbo_arch_training_curves.svg
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_256/hgbo_arch_best_test_delta.svg
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_512/hgbo_arch_training_history.json
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_512/hgbo_arch_training_curves.svg
img/training/atapp_multi_seed_cpu_threads16_workers0/seed_512/hgbo_arch_best_test_delta.svg
```

## Comparison Results

Fresh 30-epoch multi-seed run:

| Setting | Value |
|---|---:|
| Architecture mode | `atapp` |
| Epochs | `30` |
| Seeds | `128, 256, 512` |
| Learning rate | `0.001` |
| Gradient clipping | `1.0` |
| Device | `cpu` |
| CPU threads | `16` |
| DataLoader workers | `0` |
| Original datasets | `dataset/std`, `dataset/rdc` |
| Architecture-aware datasets | `dataset/std_arch`, `dataset/rdc_arch` |

Lower is better for every metric.

| Target | Metric | Original Mean Best Test | ATAPP-Aware Mean Best Test | Mean Delta | ATAPP Wins | Original Wins | Consistent Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| LUT | MAPE | `0.2114234192 +/- 0.0054088760` | `0.1979884440 +/- 0.0120475476` | `+6.24%` | `2/3` | `1/3` | Mixed |
| FF | MAPE | `0.1320277464 +/- 0.0142166958` | `0.1308327150 +/- 0.0028155622` | `+0.29%` | `1/3` | `2/3` | Mixed |
| DSP | MAE | `8.7309668523 +/- 0.4203914145` | `8.9491745635 +/- 0.6826879621` | `-2.40%` | `1/3` | `2/3` | Mixed |
| BRAM | MAE | `3.7232751709 +/- 0.2243003030` | `3.7774525950 +/- 0.2040518206` | `-1.66%` | `2/3` | `1/3` | Mixed |
| CP | MAPE | `0.0870551961 +/- 0.0029256996` | `0.0822820346 +/- 0.0049821323` | `+5.35%` | `2/3` | `1/3` | Mixed |
| Power | MAPE | `0.2910346085 +/- 0.0164474547` | `0.2790279675 +/- 0.0161437257` | `+4.13%` | `3/3` | `0/3` | ATAPP-aware |

Across 18 target-runs, ATAPP-aware won 11 and original won 7.

Only Power was consistent across all three seeds. LUT and CP had ATAPP-favorable means but mixed per-seed winners. FF was nearly tied on mean but favored original in two seeds. DSP and BRAM were worse on mean in this run, even though BRAM favored ATAPP in two seeds.

Per-seed best-test values:

| Seed | Target | Metric | Original Best Test | ATAPP-Aware Best Test | Delta | Winner |
|---:|---|---:|---:|---:|---:|---|
| 128 | LUT | MAPE | `0.2085216644` | `0.1959976846` | `+6.01%` | ATAPP-aware |
| 128 | FF | MAPE | `0.1195856407` | `0.1292510578` | `-8.08%` | Original |
| 128 | DSP | MAE | `9.0498689361` | `9.4126081926` | `-4.01%` | Original |
| 128 | BRAM | MAE | `3.7015263960` | `3.5438446525` | `+4.26%` | ATAPP-aware |
| 128 | CP | MAPE | `0.0858019018` | `0.0879911305` | `-2.55%` | Original |
| 128 | Power | MAPE | `0.2864870379` | `0.2764578712` | `+3.50%` | ATAPP-aware |
| 256 | LUT | MAPE | `0.2176639525` | `0.1870602731` | `+14.06%` | ATAPP-aware |
| 256 | FF | MAPE | `0.1475229539` | `0.1340834543` | `+9.11%` | ATAPP-aware |
| 256 | DSP | MAE | `8.2545693842` | `8.1652034532` | `+1.08%` | ATAPP-aware |
| 256 | BRAM | MAE | `3.5106414622` | `3.8676550177` | `-10.17%` | Original |
| 256 | CP | MAPE | `0.0849649226` | `0.0800409279` | `+5.80%` | ATAPP-aware |
| 256 | Power | MAPE | `0.3092773783` | `0.2963025695` | `+4.20%` | ATAPP-aware |
| 512 | LUT | MAPE | `0.2080846408` | `0.2109073744` | `-1.36%` | Original |
| 512 | FF | MAPE | `0.1289746447` | `0.1291636329` | `-0.15%` | Original |
| 512 | DSP | MAE | `8.8884622368` | `9.2697120448` | `-4.29%` | Original |
| 512 | BRAM | MAE | `3.9576576544` | `3.9208581150` | `+0.93%` | ATAPP-aware |
| 512 | CP | MAPE | `0.0903987639` | `0.0788140455` | `+12.82%` | ATAPP-aware |
| 512 | Power | MAPE | `0.2773394092` | `0.2643234617` | `+4.69%` | ATAPP-aware |

## Training Graphs

The following graphs are generated from the fresh ATAPP multi-seed comparison.

Seed 128:

![ATAPP seed 128 training metric curves](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_128/hgbo_arch_training_curves.svg)

![ATAPP seed 128 best test metric delta](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_128/hgbo_arch_best_test_delta.svg)

Seed 256:

![ATAPP seed 256 training metric curves](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_256/hgbo_arch_training_curves.svg)

![ATAPP seed 256 best test metric delta](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_256/hgbo_arch_best_test_delta.svg)

Seed 512:

![ATAPP seed 512 training metric curves](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_512/hgbo_arch_training_curves.svg)

![ATAPP seed 512 best test metric delta](img/training/atapp_multi_seed_cpu_threads16_workers0/seed_512/hgbo_arch_best_test_delta.svg)

## Prior Legacy Fabric Results

Earlier runs under `img/training/` used the legacy full-device fabric graph representation. Those artifacts remain useful for comparison, but they are not the ATAPP-style representation.

The prior three-seed legacy fabric summary is still available at:

```text
img/training/multi_seed/hgbo_arch_consistency_summary.json
img/training/multi_seed/hgbo_arch_consistency_summary.csv
```

That legacy fabric run showed architecture-aware wins for LUT, FF, DSP, BRAM, and Power across all three seeds, while CP favored the original model across all three seeds. The fresh ATAPP forced-CPU run differs: Power is consistently improved, LUT/CP improve on average but are mixed across seeds, FF is nearly tied, and DSP/BRAM are worse on average.

## Interpretation

The current HGBO-DSE implementation now mirrors ATAPP at the architecture-representation level: one representative FSR/clock region, fixed tile categories, padded `[360,80,4]` layout, compressed `[360,1,4]` layout with sinusoidal positional encoding, 21 metadata fields, and a two-hidden-layer architecture MLP.

The fresh multi-seed result is more conservative than the earlier legacy-fabric result. ATAPP-aware modeling wins more target-runs overall (`11/18`) and consistently wins Power, but the compact FSR representation is not uniformly better for every target on this single-device dataset.

The existing dataset is still single-device. Since every sample targets the same FPGA, architecture tensors are mostly constant across samples. This validates the training path and provides a drop-in foundation for future multi-device data, but it should not be interpreted as a complete unseen-FPGA generalization experiment until multi-board labels are available.

## Stability Notes

The training path keeps the previous safety settings:

```text
--lr 0.001
--grad-clip 1.0
```

The model checks outputs, targets, losses, metrics, and gradient norms for finite values. The legacy full-fabric trainable mode remains heavier and should be reserved for bounded experiments; use cached mode for full-device fabric runs.
