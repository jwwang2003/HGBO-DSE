# Dockerfile
FROM python:3.9-slim
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

# Install PyTorch CPU wheels
RUN pip install --upgrade pip
RUN pip install --trusted-host 192.168.139.1 --index-url http://192.168.139.1:5000/index/ \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache --trusted-host 192.168.139.1 --index-url http://192.168.139.1:5000/index/ \
    -r requirements.txt -f https://download.pytorch.org/whl/cpu

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