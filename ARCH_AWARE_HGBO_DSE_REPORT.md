# HGBO-DSE And Architecture-Aware Extension Report

## Status

This report describes the original HGBO-DSE workflow, the architecture-aware extension implemented in this branch, and the training results available in this workspace as of May 16, 2026.

The current branch is `arch-aware`. The implementation and source changes have been pushed through commit `4504ea5` (`Avoid RapidWright init for cached architecture loads`). The generated dataset and training artifacts are local generated outputs and are not part of the committed source tree.

The completed comparison results currently cover `lut`, `ff`, `dsp`, `bram`, and `cp`. The `power` run created an output directory but did not produce a completed JSON/CSV/SVG artifact before the long-running process exited, so power is excluded from aggregate conclusions below.

## Executive Summary

HGBO-DSE is a design-space exploration framework for HLS designs. Its core flow is:

1. Generate HLS design variants with pragma/configuration changes.
2. Run Vitis HLS and Vivado implementation to obtain post-implementation labels.
3. Convert each design into graph data.
4. Train hierarchical graph neural network predictors for resource, timing, and power metrics.
5. Use Bayesian multi-objective optimization to search the design space with model predictions instead of repeatedly paying full implementation cost.

The original HGBO-DSE setup is effectively fixed-device. It targets the Xilinx Virtex-7 VC707 part `xc7vx485tffg1761-2`, so FPGA architecture information is implicit in the labels and tool setup instead of explicitly represented as a model input.

The architecture-aware extension adds an explicit FPGA representation inspired by the ATAPP paper's architecture and technology aware structure. It extracts a compact RapidWright-based layout profile, attaches that representation to each PyG sample, encodes it with a small MLP, and fuses the resulting architecture embedding with the original HGBO-DSE graph/HLS embedding.

For the five completed targets on the current single-device dataset, architecture-aware training improved best held-out error on four targets and regressed on one:

| Target | Metric | Original best | Original epoch | Arch-aware best | Arch-aware epoch | Relative error reduction | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| LUT | MAPE | 0.181714 | 393 | 0.172564 | 381 | +5.04% | Arch-aware |
| FF | MAPE | 0.114141 | 446 | 0.117674 | 489 | -3.10% | Original |
| DSP | MAE | 2.230035 | 488 | 2.208137 | 484 | +0.98% | Arch-aware |
| BRAM | MAE | 0.185100 | 474 | 0.179140 | 455 | +3.22% | Arch-aware |
| CP | MAPE | 0.072463 | 332 | 0.069859 | 463 | +3.59% | Arch-aware |

Across these completed targets, the mean relative change is about 1.95% lower best test error for the architecture-aware model. This should be treated as an initial single-seed result, not a final paper-quality conclusion. Different seeds can move the result by several percent on this dataset.

## Original HGBO-DSE

### Goal

HGBO-DSE is designed to reduce the cost of high-level synthesis design-space exploration. The expensive operation is not training itself, but generating labels: each candidate design can require HLS, synthesis, placement, and routing before accurate post-implementation PPA values are known.

The framework addresses this by learning predictors from implemented designs, then using those predictors during optimization. The predictor must be accurate enough to rank design choices without rerunning the full backend for every candidate.

### Target And Toolchain

The reference setup described in the local notes uses:

| Item | Value |
|---|---|
| HLS tool | Vitis HLS 2022.1 |
| Implementation tool | Vivado 2022.1 |
| FPGA board/device | Xilinx Virtex-7 VC707 |
| Part | `xc7vx485tffg1761-2` |
| Clock | 100 MHz / 10 ns |
| ML stack | PyTorch, PyTorch Geometric |
| Optimization stack | Optuna-style Bayesian search |

The current local dataset follows that fixed target part. That matters for interpretation: all current samples come from one FPGA, so the model sees only one architecture distribution.

### Dataset

The local generated dataset has 11,327 samples per split directory:

| Benchmark | Samples |
|---|---:|
| aes | 1,106 |
| bfs | 1,209 |
| fft | 1,121 |
| gemm | 1,068 |
| md | 1,261 |
| nw | 1,013 |
| sort | 1,083 |
| spmv | 1,090 |
| stencil | 1,115 |
| viterbi | 1,261 |
| Total | 11,327 |

Local disk usage:

| Directory | Size |
|---|---:|
| `dataset/raw` | 4.5 GB |
| `dataset/std` | 1.1 GB |
| `dataset/rdc` | 864 MB |
| `dataset/std_arch` | 1.2 GB |
| `dataset/rdc_arch` | 935 MB |
| `dataset/board_arch` | 31 MB |
| `dataset` total | 8.5 GB |

The `std` split is used for LUT, FF, DSP, BRAM, and power prediction. The `rdc` split is used for critical path prediction.

### Data Generation Flow

The original data flow is:

1. Start from MachSuite-style kernels.
2. Generate design variants through HLS directives and configurations.
3. Run HLS and implementation to obtain labels.
4. Parse implementation reports for LUT, FF, DSP, BRAM, critical path, and power.
5. Build graph samples for PyTorch Geometric.

Each graph sample represents a design's computation/dataflow structure. The model receives graph features plus HLS-level attributes, then predicts implementation metrics.

### Predictor Structure

The original HGBO-DSE predictor is a hierarchical graph predictor. In this branch, the comparison baseline is implemented as `OriginalHierNet` in `hgp/reporting/compare_training_graphs.py`, mirroring the original HGBO-DSE behavior for controlled comparison.

At a high level:

1. Node features are embedded into hidden channels.
2. Several graph convolution layers process the design graph.
3. Hierarchical pooling reduces the graph to a design-level embedding.
4. The graph embedding is concatenated with HLS attributes.
5. Fully connected layers predict the selected target.

The architecture-aware comparison command keeps the original model and the new model in the same training harness so the curves, best test values, and SVG/CSV/JSON artifacts are generated consistently.

### Target Metrics

The reporting/training harness uses:

| Target | Split | Metric |
|---|---|---|
| LUT | `std` / `std_arch` | MAPE |
| FF | `std` / `std_arch` | MAPE |
| DSP | `std` / `std_arch` | MAE |
| BRAM | `std` / `std_arch` | MAE |
| CP | `rdc` / `rdc_arch` | MAPE |
| Power | `std` / `std_arch` | MAPE |

DSP and BRAM use MAE because many labels are small or zero, where percentage error becomes unstable. Power labels are scaled in the model path.

### Role In DSE

Once trained, the predictor feeds the design-space optimizer. Instead of running implementation for every candidate, the optimizer can score many candidates cheaply with the learned model and reserve full implementation for selected designs. This is where HGBO-DSE gets its runtime advantage: it shifts most evaluation cost from EDA tool execution to model inference.

### Original Limitation

The original model does not explicitly encode FPGA architecture. It can learn what `xc7vx485tffg1761-2` looks like only because every label comes from that part. That is acceptable for single-FPGA optimization, but it limits portability:

- There is no direct input describing another FPGA's layout or resource mix.
- The model cannot learn how architecture changes affect implementation results unless trained on multiple architectures.
- A held-out FPGA test is not meaningful unless the model has seen enough other FPGA architectures during training to learn cross-device behavior.

## ATAPP Architecture-Aware Structure

The ATAPP paper motivates explicit architecture and technology awareness for FPGA prediction. Instead of treating the FPGA as an invisible constant, the model receives a representation of the target architecture.

The relevant structure from `ref/notes/ATAPP.md` is:

1. Use RapidWright to inspect the FPGA layout.
2. Select a representative FPGA subregion, described as an FSR/clock-region style view.
3. Reduce tile types into a small fixed category set:
   - `CLEL`
   - `CLEM`
   - `INT`
   - `INT_INTERFACE`
   - `DSP`
   - `BRAM`
   - `BRK`
4. Build a padded raw layout tensor with shape `[360, 80, 4]`.
5. Compress the mostly columnar FPGA structure to `[360, 1, 4]`.
6. Add sinusoidal positional encoding.
7. Add a metadata vector of length 21 for resource/layout/technology-level properties.
8. Encode this architecture tensor and metadata with an MLP.
9. Fuse the architecture embedding with the design embedding before prediction.

ATAPP's full design branch is power-oriented and uses activity-aware features. The current HGBO-DSE implementation does not reproduce ATAPP's switching activity flow or replace HGBO-DSE with ATAPP's full model. The adaptation here uses ATAPP's architecture representation idea while keeping HGBO-DSE's graph/HLS predictor structure.

## Architecture-Aware HGBO-DSE Implementation

### Current Naming

The public code path uses `arch-aware` naming:

```bash
--arch-mode arch-aware
```

The implementation intentionally avoids `atapp` names in code and commit messages. This keeps the HGBO-DSE branch framed as an architecture-aware extension, while this report can still explain the ATAPP paper connection.

### RapidWright Integration

RapidWright is used only when extracting architecture caches. The latest implementation lazily imports RapidWright so loading an existing architecture cache does not initialize the JVM. This matters for CPU training stability because training should not need RapidWright at all once cached tensors exist.

Relevant files:

| File | Role |
|---|---|
| `hgp/rapidwright_env.py` | Configures the local RapidWright environment when extraction is required. |
| `hgp/arch_aware_arch.py` | Extracts, caches, and loads compact architecture tensors. |
| `hgp/board_fabric.py` | Keeps the older full-fabric graph cache path available. |

The current cache file for the default board is:

```text
dataset/board_arch/xc7vx485tffg1761-2_arch_aware_arch.pt
```

### Compact Architecture Representation

The implementation defines:

```text
ARCH_AWARE_LAYOUT_COLS = 360
ARCH_AWARE_LAYOUT_ROWS = 80
ARCH_AWARE_TILE_SLOTS = 4
ARCH_AWARE_METADATA_DIM = 21
```

Tile categories:

| Name | ID |
|---|---:|
| PAD | 0 |
| CLEL | 1 |
| CLEM | 2 |
| INT | 3 |
| INT_INTERFACE | 4 |
| DSP | 5 |
| BRAM | 6 |
| BRK | 7 |

The cache stores:

| Tensor/value | Meaning |
|---|---|
| `arch_aware_arch_layout_raw` | Padded raw layout, `[360, 80, 4]`. |
| `arch_aware_arch_layout` | Compressed float layout, `[360, 1, 4]`. |
| `arch_aware_arch_metadata` | Device/resource/layout metadata, `[1, 21]`. |
| `arch_aware_arch_positional_encoding` | Sinusoidal positional encoding for layout columns. |
| `arch_aware_valid_rows` / `arch_aware_valid_cols` | Real populated dimensions before padding/truncation. |
| `arch_aware_clock_region` | Selected representative clock region. |
| `arch_aware_tile_type_to_id` | Tile category mapping. |

The architecture encoder flattens the compressed layout and concatenates metadata:

```text
flatten([360, 1, 4]) || metadata[21] = 1440 + 21 = 1461 inputs
```

That vector is passed through a two-hidden-layer MLP to produce an architecture embedding.

### Dataset Augmentation

The dataset augmentation entry point is:

```bash
uv run python -m hgp.data_process.gen_dataset_board \
  --input-root dataset \
  --output-root dataset \
  --device xc7vx485tffg1761-2 \
  --arch-mode arch-aware
```

The current implementation resolves default dataset paths relative to the repository instead of the caller's current working directory. This fixed a review issue where documented invocation from the repo root could accidentally write/read outside HGBO-DSE.

Generated augmented splits:

```text
dataset/std_arch
dataset/rdc_arch
```

Each augmented sample retains the original graph and HLS fields and adds architecture-aware fields such as:

```text
arch_aware_arch_layout
arch_aware_arch_metadata
arch_aware_arch_device
arch_aware_clock_region
```

The older scalar board profile path remains available through `arch_attr`, and the full fabric graph path remains available with `--arch-mode fabric`, but the paper-aligned experiment used the compact `arch-aware` path.

### Model Fusion

The main model is `ArchAwareHierNet` in `hgp/hier_arch_model.py`.

The architecture-aware forward path is:

1. Encode design graph using the HGBO-DSE graph branch.
2. Pool graph features into a design embedding.
3. Concatenate HLS attributes.
4. Encode compact architecture layout and metadata with `ArchAwareArchitectureEncoder`.
5. Concatenate design embedding, HLS attributes, and architecture embedding.
6. Predict the requested target through an MLP head.

For the completed paper-aligned run, the model used:

| Parameter | Value |
|---|---|
| Hidden channels | 64 |
| Graph layers | 3 |
| Convolution | GraphSAGE |
| Dropout | 0.0 |
| Weight decay | 0.001 |
| Learning rate | 0.005 |
| Batch size | 32 |
| Epochs | 500 |
| Device | CPU |
| CPU threads | 16 |
| DataLoader workers | 0 |
| Seed | 128 |
| Gradient clip | 1.0 |

GraphSAGE was used to align with the original-paper training parameters tracked in the local notes. The architecture-aware implementation also supports other convolution choices in code, but those are not the basis of the results reported here.

### Reporting

The comparison driver is:

```bash
uv run python -u -m hgp.reporting.compare_training_graphs \
  --targets <target> \
  --epochs 500 \
  --lr 0.005 \
  --grad-clip 1.0 \
  --seed 128 \
  --batch-size 32 \
  --device cpu \
  --cpu-threads 16 \
  --num-workers 0 \
  --hidden-channels 64 \
  --num-layers 3 \
  --conv-type sage \
  --drop-out 0.0 \
  --weight-decay 0.001 \
  --arch-mode arch-aware \
  --fabric-mode cached \
  --output-dir img/training/arch_aware_paper_cpu_seed128/<target>
```

The report writers were fixed to honor target subsets. For example, `--targets lut` now writes only LUT artifacts instead of indexing every global target and failing on missing entries.

Each completed target writes:

```text
hgbo_arch_training_history.json
hgbo_arch_training_history.csv
hgbo_arch_training_curves.svg
hgbo_arch_best_test_delta.svg
```

## Training Results

### Run Context

Output root:

```text
img/training/arch_aware_paper_cpu_seed128
```

The run was executed sequentially per target on CPU. This was deliberate: previous GPU execution was not clearly faster for this workload, and the CPU path avoids GPU memory pressure while allowing controlled thread usage.

Gradient clipping at `1.0` was used because an unclipped original-baseline run produced non-finite model output during LUT training. This is a practical stability fix rather than a claimed original HGBO-DSE default.

### Completed Results

| Target | Metric | Original best | Original epoch | Arch-aware best | Arch-aware epoch | Relative error reduction | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| LUT | MAPE | 0.181714 | 393 | 0.172564 | 381 | +5.04% | Arch-aware |
| FF | MAPE | 0.114141 | 446 | 0.117674 | 489 | -3.10% | Original |
| DSP | MAE | 2.230035 | 488 | 2.208137 | 484 | +0.98% | Arch-aware |
| BRAM | MAE | 0.185100 | 474 | 0.179140 | 455 | +3.22% | Arch-aware |
| CP | MAPE | 0.072463 | 332 | 0.069859 | 463 | +3.59% | Arch-aware |
| Power | MAPE | Not complete | N/A | Not complete | N/A | N/A | N/A |

### Interpretation

The completed results do not show evidence that the architecture-aware path broadly deteriorates the model. Four of five completed targets improved best held-out error, while FF regressed by 3.10%.

The observed changes are within the range that can plausibly occur from seed variance on this dataset. A single run is useful for validating that the implementation trains and produces reasonable curves, but it is not enough for a final statistical claim.

The strongest result is LUT, with a 5.04% lower best MAPE. CP and BRAM also improved by more than 3%. DSP improved slightly. FF is the target that needs more seed checks before deciding whether the architecture-aware branch is neutral or mildly harmful for that label.

### Why Results Can Move By Several Percent

Several factors make 3% to 5% movement plausible:

1. The dataset has about 11k samples, not hundreds of thousands.
2. Train/test split order and random initialization matter.
3. HLS implementation labels can be noisy because small directive changes may produce discontinuous backend behavior.
4. The architecture-aware branch adds parameters, which can help or hurt depending on target and seed.
5. The current dataset has only one FPGA architecture, so the architecture tensor is mostly a constant feature across samples.

The current result supports the statement that the architecture-aware extension did not obviously damage HGBO-DSE on the completed metrics. It does not prove cross-FPGA generalization yet.

## Multi-FPGA Implications

The architecture-aware branch is most valuable when training includes multiple FPGA devices. With only `xc7vx485tffg1761-2`, the architecture embedding is effectively a constant device descriptor. It can still regularize or shift the predictor, but the model cannot learn how predictions should change across FPGA architectures.

For a real multi-FPGA study, the dataset should include labels from several devices and should reserve at least one device for held-out testing. A reasonable evaluation design is:

1. Generate labels for 8 FPGA devices.
2. Include shared anchor designs across all devices.
3. Train on 7 devices.
4. Test on the held-out 8th device.
5. Rotate the held-out device.
6. Compare original HGBO-DSE, scalar-board features, full-fabric graph features, and compact architecture-aware features.

Using 2,000 to 3,500 designs per FPGA can be reasonable if the designs cover the important pragma and benchmark diversity. The key requirement is not just sample count per FPGA, but overlap and coverage: the model needs enough shared design patterns across devices to separate "this changed because the design changed" from "this changed because the FPGA changed."

## Validation Performed

Implementation validation completed earlier on this branch:

```text
uv run pytest
```

Result:

```text
153 passed
```

The test coverage includes:

- architecture tile classification,
- positional encoding,
- metadata shape and normalization,
- cache round trip,
- dataset augmentation,
- cached architecture loading without RapidWright/JVM initialization,
- architecture-aware model behavior,
- full-fabric fallback behavior,
- reporting output generation,
- target-subset report writing.

## Known Limitations

1. Power is not complete in the current result set. It should be rerun or resumed before making claims about all six targets.
2. The results are single-seed. A cleaner result should run at least three seeds, preferably five.
3. The architecture-aware branch is evaluated on a single FPGA, so it validates implementation but not unseen-FPGA prediction.
4. The ATAPP-inspired adaptation does not include switching activity traces or ATAPP's full power-specific design branch.
5. Full post-implementation dataset generation remains the bottleneck. Multi-core execution helps when jobs are independent, but EDA tool throughput is limited by memory, license availability, and per-run Vivado behavior.

## Recommended Next Steps

1. Rerun or resume the `power` target with the same settings.
2. Run multi-seed comparisons for all completed targets.
3. Add a resumable per-target training wrapper so a failed target does not lose run state.
4. For the multi-FPGA study, generate architecture caches and labeled samples per FPGA.
5. Keep shared anchor designs across FPGA devices to support held-out-device evaluation.
6. Report mean and standard deviation by target across seeds and devices.
