# Dockerfile Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Base Image & Shell](#base-image--shell)
3. [Working Directory](#working-directory)
4. [System Dependencies & Locale](#system-dependencies--locale)
5. [Vivado Suite Prerequisites](#vivado-suite-prerequisites)
6. [Environment Variables](#environment-variables)
7. [PyTorch Installation](#pytorch-installation)
8. [Python Dependencies](#python-dependencies)
9. [Application Code](#application-code)
10. [Entrypoint Script](#entrypoint-script)
11. [Ports & Execution](#ports--execution)
12. [Building & Running the Image](#building--running-the-image)
13. [Mounting Vivado Installation](#mounting-vivado-installation)
14. [Customization Tips](#customization-tips)

---

## Overview

This Dockerfile sets up a Python 3.9 environment tailored for applications that require:

* System libraries (Graphviz, build tools, locales).
* Integration with Xilinx Vivado/Vitis tool suite.
* CPU-only PyTorch and associated Python dependencies.
* A custom entrypoint script for container initialization.

---

## Base Image & Shell

```dockerfile
FROM python:3.9-slim
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
```

* **python:3.9-slim**: A lightweight Debian-based image with Python 3.9 preinstalled.
* **Switch to Bash**: Replaces `/bin/sh` symlink to use Bash for consistency with scripts that expect Bash features.

---

## Working Directory

```dockerfile
WORKDIR /app
```

Sets the working directory inside the container to `/app`. All subsequent commands execute relative to this path.

---

## System Dependencies & Locale

```dockerfile
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
```

1. **apt-get update**: Refreshes package lists.
2. **Install packages**:

   * `graphviz`: Graph generation tools.
   * `build-essential`: Compiler and build tools (gcc, make, etc.).
   * `locales`: Support for generating custom locales.
3. **Clean up**: Removes cached package lists to reduce image size.
4. **Locale setup**: Enables and generates the `en_US.UTF-8` locale for proper Unicode support.

---

## Vivado Suite Prerequisites

```dockerfile
RUN apt-get -y update; apt-get -y install curl
RUN apt-get install -y libstdc++6 dpkg-dev \
    && curl -O http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && rm libtinfo5_6.3-2ubuntu0.1_amd64.deb

# Mount our Vivado/Vitis installation into container
# (e.g., with docker-compose: - /home/{username}/tools/xilinx:/home/{username}/tools/xilinx:ro)
ENV XILINX_INSTALL=/home/wjw/tools/xilinx
```

* **curl**: Tool to download files from URLs.
* **libstdc++6 & dpkg-dev**: Required by Vivado for C++ runtime and package handling.
* **libtinfo5**: Specific ncurses library version needed by some Xilinx tools.
* **XILINX\_INSTALL**: Environment variable pointing to the host-mounted path of the Vivado/Vitis installation. Mount read-only in your `docker-compose.yml`.

---

## Environment Variables

```dockerfile
ENV XILINX_INSTALL=/home/wjw/tools/xilinx
```

* **XILINX\_INSTALL**: Default path inside the container where your Vivado/Vitis tools are mounted.
* Uncomment and adjust `PATH` if you want to include Vitis binaries in the container’s `PATH`:

  ```dockerfile
  # ENV PATH="${XILINX_INSTALL}/Vitis/2022.1/bin:${PATH}"
  ```

---

## Python Dependencies

```dockerfile
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project
```

* **UV_PROJECT_ENVIRONMENT**: Installs the project environment into the container Python prefix instead of a hidden `.venv`.
* **COPY pyproject.toml uv.lock**: Uses the uv project metadata and lockfile as the single dependency source.
* **uv sync --no-install-project**: Installs pinned runtime dependencies, including PyTorch CPU wheels and PyG native wheels routed by `tool.uv.sources`.

---

## Application Code

```dockerfile
COPY . .
```

Copies all files from the host directory (where the Docker build is executed) into `/app` in the container.

---

## Entrypoint Script

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash", "-l"]
```

1. **COPY entrypoint.sh**: Brings your custom initialization script into the container.
2. **Permissions**: Makes the script executable.
3. **ENTRYPOINT**: Ensures `entrypoint.sh` runs on container start.
4. **CMD**: Provides a default command (`bash -l`) if no other command is specified. The entrypoint can override or wrap the CMD.

---

## Ports & Execution

```dockerfile
EXPOSE 8000
```

* **EXPOSE 8000**: Documents that the container listens on port `8000` (e.g., for a web server or API).

---

## Building & Running the Image

1. **Build**:

   ```bash
   docker build -t my-python-vivado-app .
   ```

2. **Run** (with Vivado tools mounted):

   ```bash
   docker run --rm -it \
     -v /home/username/tools/xilinx:/home/username/tools/xilinx:ro \
     -p 8000:8000 \
     -e XILINX_INSTALL=/home/username/tools/xilinx \
     my-python-vivado-app
   ```

* `--rm`: Automatically remove the container when it exits.
* `-it`: Interactive terminal.
* `-v`: Mount Vivado tools directory as read-only.
* `-p`: Map host port `8000` to container port `8000`.
* `-e`: Override environment variables if needed.

---

## Mounting Vivado Installation

In your `docker-compose.yml`:

```yaml
version: '3.8'
services:
  vivado-app:
    build: .
    volumes:
      - /home/${USER}/tools/xilinx:/home/${USER}/tools/xilinx:ro
    ports:
      - '8000:8000'
    environment:
      - XILINX_INSTALL=/home/${USER}/tools/xilinx
```

---

## Customization Tips

* **Python Version**: Change the `FROM` line to another Python version (e.g., `python:3.10-slim`) if required.
* **Additional Tools**: Add more `apt-get install` lines for other system packages.
* **GPU Support**: Swap the base image to a GPU-enabled one (e.g., `pytorch/pytorch:2.6.0-cuda11.7-cudnn8-runtime`) and adjust PyTorch installation.
* **Multi-Stage Builds**: Use multi-stage builds to compile and copy only artifacts to a smaller final image.

> Provided to you courtesy of __*ChatGPT*__
