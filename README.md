# HGBO-DSE Framework

<figure>
<img src=img/hgbo.svg alt="Trulli" style="width:100%">
<figcaption align="left">
The proposed HGBO-DSE is an automatic framework composed of HGP, BOME and TDM 
to support fast and efficient multi-objective design space exploration for HLS.
</figcaption>
</figure>

## Prerequisites
### Python Environmant
- python 3.9
- torch 1.13.1
- torch-geometric 2.3.1
- torch-scatter 2.1.1
- torch-sparse 0.6.17
- optuna 3.2.0
- pyDOE 0.3.8
### Vitis Environment
- Vitis HLS 2022.1
- Vivado 2022.1

## Dataset Generator
### Public HLS Dataset
The raw HLS dataset is already generated, which is stored in path "./dataset/raw/". 
Users can access this dataset conveniently to extract features of their interests to fit their own ML models.
### Collect New Samples
If you want to collect new samples (take benchmark 'bfs' as an example), run the following commands:
1. cd bome
2. python3 hls_dse.py --case bfs --ver bulk --alg random

The newly generated samples will be stored in path "./dse_ds/MachSuite/random_ds/".

## Graph Constructor
To generate 'pt' files which store CDFGs for HGP training and testing, run the following commands:
1. cd hgp/data_process
2. python3 gen_dataset_std.py

We already generate 'pt' files in path "./dataset/std/" and "./dataset/rdc/".

### RapidWright ATAPP-Style Architecture Features
HGBO-DSE can now reuse the existing `dataset/std/` and `dataset/rdc/`
training samples while adding a RapidWright-derived ATAPP-style FPGA
architecture representation for the default raw-dataset device,
`xc7vx485tffg1761-2`.

The RapidWright Python package is installed from the sibling local checkout:

```bash
uv sync
```

Build the ATAPP architecture cache once and write architecture-aware dataset
copies:

```bash
cd hgp/data_process
python3 gen_dataset_board.py --device xc7vx485tffg1761-2 --arch-mode atapp
```

This extracts the default board architecture into `dataset/board_arch/` and
then reuses the existing CDFG samples to write `dataset/std_arch/` and
`dataset/rdc_arch/`. The ATAPP tensor uses raw layout `[360,80,4]`,
compressed layout `[360,1,4]`, and 21 metadata fields. The original `std` and
`rdc` datasets are left unchanged.

## HGP
### Training
HGP is trained for LUT/FF/DSP/BRAM/CP/Power prediction.
The well-trained models are in path "./hgp/model/".

If you want to retrain HGP, run commands:
1. cd hgp
2. python3 hier_lut_model.py

For RapidWright ATAPP-aware training, use:

```bash
cd hgp
python3 hier_arch_model.py --target lut --arch-mode atapp
```

The ATAPP-aware trainer defaults to an edge-attribute-aware GINE design encoder,
which is closer to the edge-aware UniMP design branch described by ATAPP than
the legacy SAGE/GCN/GAT options.

The previous full fabric graph path remains available with `--arch-mode fabric`.

HGBO-DSE is configured with CUDA PyTorch/PyG wheels, but the training CLIs
default to CPU so GPU use is explicit. Use `--cpu-threads` for PyTorch CPU
compute threads, and keep DataLoader workers at zero to avoid multiprocessing
overhead on the small HGBO-DSE datasets:

```bash
uv run python -m hgp.hier_arch_model \
  --target lut \
  --epochs 30 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --arch-mode atapp \
  --device cpu \
  --cpu-threads 16 \
  --num-workers 0 \
  --model-dir /tmp/hgbo_atapp_lut_cpu
```

For the full original-vs-ATAPP comparison on CPU:

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

The trainer prints `Using device cpu`, `Torch CPU threads: ...`, and
`DataLoader workers: ...` at startup.

To opt into GPU training on a host with a working NVIDIA driver, pass
`--device cuda`. If CUDA is unavailable, the trainer fails loudly instead of
silently falling back.

The latest forced-CPU 30-epoch comparison uses 16 PyTorch CPU threads and
`num_workers=0`; its summary and SVG training graphs are under
`img/training/atapp_multi_seed_cpu_threads16_workers0/`.

### Inference
The DSE `--mode hgp` inference path currently uses the original HGP prediction
modules in `bome/pred/`. Those modules load fixed best-test checkpoint names
from `./hgp/model/`:

```text
lut_h64_d0_checkpoint_test.pt
ff_h64_d0_checkpoint_test.pt
dsp_mae_h64_d0_checkpoint_test.pt
bram_mae_h64_d0_checkpoint_test.pt
cp_mean_h64_d0_checkpoint_test.pt
power_mean_h64_d0_checkpoint_test.pt
```

Use the `*_checkpoint_test.pt` files for DSE. The
`*_checkpoint_train.pt` files are selected by best training metric and are not
the default inference weights.

ATAPP-aware training writes `*_arch_h64_d0_checkpoint_*.pt` checkpoints with a
different model structure and architecture inputs. Those weights are used for
training/evaluation comparisons, but DSE will not select them until a dedicated
architecture-aware inference path is added.

### HGP Prediction Visualization
The following figures visualize the predicted values of HGP in terms of LUT, FF, DSP, BRAM, CP and Power.

<figure class="half">
    <img src="img/lut_pred.svg" width="47%">
    <img src="img/ff_pred.svg" width="47%">
</figure>

<figure class="half">
    <img src="img/dsp_pred.svg" width="47%">
    <img src="img/bram_pred.svg" width="47%">
</figure>

<figure class="half">
    <img src="img/cp_pred.svg" width="47%">
    <img src="img/pwr_pred.svg" width="47%">
</figure>

## TDM
### Design Space Configuration
- config.yaml (specify which directives to explore)
- params.yaml (specify the options of directives)

We provide the corresponding yaml files for MachSuite benchmarks used in our paper.
For new benchmarks, users can write yaml files based on the rules described in our paper.

### TDM Modeling
TDM is integrated in BOME (in path "./bome/tdm/"). It reads the above yaml files to construct the design space in tree-structure and stores it 
in a dictionary which is then passed to BOME.

## BOME
### Directive Encoding
- Float encoding
- Discrete encoding

BOME supports both of the encoding styles, which can be specified by "--encode [options: float, discrete]"

### PPA Evaluation
- HGP inference flow
- FPGA implementation flow

Specify "--mode [options: hgp, impl]" to choose the PPA evaluation flow.

### DSE Algorithms
Our algorithms:
- MOTPE-D
- MOTPE-F
- MOTPE-FL

Meta-Heuristics for comparison:
- Simulated Annealing (SA)
- Multi-Objective Genetic Algorithm (NSGA-II)

Specify "--alg [options: motpe_d, motpe_f, motpe_fl, nsga, sa]" to choose the DSE algorithm.

### Other Arguments
- --bench [options: MachSuite] (The benchmark set)
- --case [options: aes, bfs, fft, ...] (The specific benchmark)
- --ver [options: aes, bulk, strided] (The version of the specified benchmark)
- --num [options: integer values (e.g., 100, 200)] (The number of optimization steps)
- --device [options: the specific FPGA device (e.g., 'xc7vx485tffg1761-2')]
- --clk [options: e.g., 5, 10] (The frequency running on FPGA board)
- --space [options: tree, uniform] (The design space configuration mode)
- --parallel [options: True, False] (Whether run DSE in parallel or not, MySQL is needed if in parallel)
- --process [options: 1, 2, 3, ...] (The current process number)

### Run DSE
<!-- 1. cd bome -->
1. source /mnt/sda1/Xilinx/Vitis/2022.1/settings64.sh
2. python -m bome.hls_dse

When running the above commands, the default settings are adopted.
You can add the arguments according to your needs.

## Baseline Models
The baseline models for comparison are in path "./baseline/":
- ironman-pro
- pna-r
- powergear
