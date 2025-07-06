# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y graphviz \
    && rm -rf /var/lib/apt/lists/*

# Mount your Vivado/Vitis installation into container at /tools/xilinx
# (e.g., with docker-compose: - /home/wjw/tools/xilinx:/tools/xilinx:ro)
ENV XILINX_INSTALL=/tools/xilinx
ENV PATH="${XILINX_INSTALL}/Vitis/2021.2/bin:${PATH}"

# Install PyTorch CPU wheels
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Dummy entrypoint for verification\ENTRYPOINT ["echo", "Dummy entrypoint: container is up"]

CMD ["echo", "Dummy entrypoint: container is up"]