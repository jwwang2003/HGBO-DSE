# Dockerfile
FROM python:3.9-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /usr/local/bin/
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    graphviz \
    build-essential \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    \
    # Create /etc/locale.gen and generate the locale
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen en_US.UTF-8

# Installing some pre-requisites for the Vivado Suite
RUN apt-get -y update; apt-get -y install curl
RUN apt-get install -y libstdc++6 dpkg-dev \
    && curl -O http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && rm libtinfo5_6.3-2ubuntu0.1_amd64.deb

# Mount our Vivado/Vitis installation into container
# (e.g., with docker-compose: - /home/{username}/tools/xilinx:/home/{username}/tools/xilinx:ro)
ENV XILINX_INSTALL=/home/wjw/tools/xilinx
# ENV PATH="${XILINX_INSTALL}/Vitis/2022.1/bin:${PATH}"

# Install Python dependencies with uv.
COPY pyproject.toml requirements.txt ./
RUN uv pip install --system --requirement requirements.txt \
    --index-url http://192.168.139.1:5000/index/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links https://download.pytorch.org/whl/cpu \
    --allow-insecure-host 192.168.139.1

# Copy application code
COPY . .

# Copy and install our entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose your port
EXPOSE 8000

# Example 1:
# CMD ["echo", "Dummy entrypoint: container is up"]
# ENTRYPOINT ["vitis_hls"]

# Example 2:
# Use our script as entrypoint, and default to an interactive login shell
# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["gunicorn -k uvicorn.workers.UvicornWorker backend.entry:app"]

ENTRYPOINT ["/entrypoint.sh"]
# ENTRYPOINT ["/bin/bash", "-c", "./entrypoint.sh"]
