# HGBO-DSE Backend Development Notes

## Running the backend of HGBO-DSE

Start the Docker container for running the:
```
docker run --rm -it -v /home/wjw/tools/xilinx:/home/wjw/tools/xilinx:ro -p 8000:8000 <image-name>
```

### Bulding the Docker container for Celery Backend

Run the following command in the root of the project directory to build the container:
```
docker build -t <image-name> .
```

## Python environment with uv

Create the host environment with `uv`:

```
uv venv --python 3.9
uv pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu
```

### Host DSE + host inference

Run Vitis HLS and HGP model inference on the host:

```
uv run python -m bome.hls_dse --mode hgp --inference-mode host --case viterbi --ver viterbi --num 10 --isolated context1
```

### Host DSE + remote inference

Run Vitis HLS on the host, then dispatch only HGP model inference to the Celery worker. The host and worker must see the same generated project paths through a shared filesystem.

Start Redis and the inference worker:

```
export HGBO_SHARED_HOST_ROOT="$(pwd)"
export HGBO_SHARED_CONTAINER_ROOT="$(pwd)"
docker compose up redis celery
```

Then run the host-side DSE process:

```
HGBO_INFERENCE_MODE=remote \
CELERY_BROKER_URL=redis://localhost:6379/0 \
CELERY_RESULT_BACKEND=redis://localhost:6379/0 \
uv run python -m bome.hls_dse --mode hgp --inference-mode remote --case viterbi --ver viterbi --num 10 --isolated context1
```

In `impl` mode, the HGP prediction model is not used; the flow runs HLS/Vivado and reads implementation reports.
