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

### Host DSE + host inference

Run Vitis HLS and HGP model inference on the host:

```
uv run python -m bome.hls_dse --mode hgp --inference-mode host --case viterbi --ver viterbi --num 10 --isolated context1
```

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
