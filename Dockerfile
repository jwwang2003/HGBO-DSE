# Use an official Python runtime as a parent image
FROM python:3.9-bookworm

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install required system dependencies for shell access and any other needed tools
# RUN apt-get update && apt-get install -y \
#     bash \
#     curl \
#     vim \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (if any required by your project)
# ENV VAR_NAME=value

# Expose the port the app runs on
EXPOSE 5000

# Make sure the shell features (e.g., bash) are available
CMD ["bash"]

ENV PATH=/opt/Xilinx/Vitis_HLS/2022.1/bin:$PATH

ENTRYPOINT ["python", "-m", "backend.server"]