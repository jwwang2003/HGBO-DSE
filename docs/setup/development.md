# HGBO-DSE Backend Development Notes

## Running the backend of HGBO-DSE

Start the Docker container for running the:
```
docker run --rm -it -v /home/wjw/tools/xilinx:/home/wjw/tools/xilinx:ro -p 8000:8000 <image-name>
```

### Building the Docker container for MCP remote inference

Run the following command in the root of the project directory to build the container:
```
docker build -t <image-name> .
```

## Python environment with uv

Create the host environment with `uv`:

```
uv venv --python 3.9
uv sync
```

Verify the runtime imports:

```
uv run python -c "import optuna, torch, torchvision, torchaudio, torch_geometric, torch_scatter, torch_sparse; print('HGBO deps OK')"
```

HGBO-DSE is configured for CUDA PyTorch/PyG wheels, while the training code
defaults to CPU unless `--device cuda` is passed. Verify
the active runtime with:

```
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.get_num_threads())"
```

Training defaults to CPU even if the environment is later changed back to CUDA
wheels. Use `--cpu-threads` to set PyTorch CPU compute threads and
`--num-workers` to control background DataLoader workers. For the bundled
HGBO-DSE datasets, `--num-workers 0` avoids worker startup overhead and was
faster than the attempted `--num-workers 4` CPU run:

```
uv run python -m hgp.hier_arch_model --target lut --epochs 1 --arch-mode arch-aware --device cpu --cpu-threads 16 --num-workers 0
```

The default architecture-aware design encoder is `--conv-type gine`, so design
graph edge attributes participate in message passing. Pass `--conv-type sage`
only when comparing against the legacy HGBO-style design encoder.

RapidWright is resolved from the sibling checkout at `../RapidWright/python`.
After `uv sync`, verify the local wrapper can be imported:

```
uv run python -c "import rapidwright; from com.xilinx.rapidwright.device import Device; print(Device)"
```

Build the reusable ATAPP architecture cache for the default dataset board:

```
uv run python -m hgp.data_process.gen_dataset_board --device xc7vx485tffg1761-2 --arch-mode atapp
```

The command keeps `dataset/std` and `dataset/rdc` intact, writes `std_arch` and
`rdc_arch`, and stores the extracted RapidWright ATAPP layout under
`dataset/board_arch`. The legacy full fabric graph cache can still be generated
with `--arch-mode fabric`.

### Host DSE + host inference

Run Vitis HLS and HGP model inference on the host:

```
uv run python -m bome.hls_dse --mode hgp --inference-mode host --case viterbi --ver viterbi --num 10 --isolated context1
```

Host DSE inference currently uses the original fixed checkpoint names under
`hgp/model/`:

```
lut_h64_d0_checkpoint_test.pt
ff_h64_d0_checkpoint_test.pt
dsp_mae_h64_d0_checkpoint_test.pt
bram_mae_h64_d0_checkpoint_test.pt
cp_mean_h64_d0_checkpoint_test.pt
power_mean_h64_d0_checkpoint_test.pt
```

The ATAPP-aware training checkpoints use a different model input path and are
not selected by DSE unless a dedicated architecture-aware inference path is
added.

### Host DSE + MCP remote inference

Run Vitis HLS on the host, then dispatch only HGP model inference to the Docker MCP service. The request contains a zip payload of `prj_*/graph`, so the container does not need a host `.compass` mount.

Start the MCP inference server:

```
docker compose up --build mcp-inference
```

Show the persistent API key:

```
docker compose exec mcp-inference uv run python -m backend.mcp_server --show-api-key
```

Then run the host-side DSE process:

```
HGBO_MCP_URL=http://localhost:8000/mcp \
HGBO_MCP_API_KEY=<key> \
uv run python -m bome.hls_dse --mode hgp --inference-mode remote --case bfs --ver bulk --num 10 --isolated context1
```

In `impl` mode, the HGP prediction model is not used; the flow runs HLS/Vivado and reads implementation reports.
