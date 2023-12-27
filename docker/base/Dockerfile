# Use the official Ubuntu image as a parent image
FROM ubuntu:20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and pip
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y python3-pip
RUN apt-get install -y libopenblas-base
RUN apt-get install -y libomp-dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Create a new user named 'faiss-gpu-txt-to-vector'
RUN useradd -ms /bin/bash faiss-gpu-txt-to-vector

# Switch to the new user
USER faiss-gpu-txt-to-vector

# Set the home directory for the user
ENV HOME /home/faiss-gpu-txt-to-vector
WORKDIR $HOME

# Install the required Python packages as the new user
RUN pip3 install --user jinja2
RUN pip3 install --user networkx
RUN pip3 install --user numpy

# Install nvidia libraries separately for unstable connections
RUN pip3 install --user nvidia-cublas-cu12
RUN pip3 install --user nvidia-cuda-cupti-cu12
RUN pip3 install --user nvidia-cuda-nvrtc-cu12
RUN pip3 install --user nvidia-cuda-runtime-cu12
RUN pip3 install --user nvidia-cudnn-cu12
RUN pip3 install --user nvidia-cufft-cu12
RUN pip3 install --user nvidia-curand-cu12
RUN pip3 install --user nvidia-cusolver-cu12
RUN pip3 install --user nvidia-cusparse-cu12
RUN pip3 install --user nvidia-nccl-cu12
RUN pip3 install --user nvidia-nvtx-cu12

# Install machine learning libraries
RUN pip3 install --user torch
RUN pip3 install --user transformers
RUN pip3 install --user tokenizers

# Install faiss
COPY ./deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /deps/
RUN pip3 install --user /deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
