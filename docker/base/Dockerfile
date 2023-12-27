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

# Install the required Python packages
RUN pip3 install jinja2
RUN pip3 install networkx
RUN pip3 install numpy

# Install nvidia libraries seperately for unstable connections
RUN pip3 install nvidia-cublas-cu12
RUN pip3 install nvidia-cuda-cupti-cu12
RUN pip3 install nvidia-cuda-nvrtc-cu12
RUN pip3 install nvidia-cuda-runtime-cu12
RUN pip3 install nvidia-cudnn-cu12
RUN pip3 install nvidia-cufft-cu12
RUN pip3 install nvidia-curand-cu12
RUN pip3 install nvidia-cusolver-cu12
RUN pip3 install nvidia-cusparse-cu12
RUN pip3 install nvidia-nccl-cu12
RUN pip3 install nvidia-nvtx-cu12


# Install ml libraries
RUN pip3 install torch
RUN pip3 install transformers
RUN pip3 install tokenizers


# Install faiss
COPY ./deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install /deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl