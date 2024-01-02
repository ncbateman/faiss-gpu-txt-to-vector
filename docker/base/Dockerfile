# Use the official Ubuntu image as a parent image
FROM ubuntu:20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and pip
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y python3-pip
RUN apt-get install -y libopenblas-base
RUN apt-get install -y libomp-dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Install the required Python packages
RUN pip3 install --default-timeout=1000 jinja2
RUN pip3 install --default-timeout=1000 networkx
RUN pip3 install --default-timeout=1000 numpy
RUN pip3 install --default-timeout=1000 nvidia-cublas-cu12==12.1.3.1
RUN pip3 install --default-timeout=1000 nvidia-cuda-cupti-cu12==12.1.105
RUN pip3 install --default-timeout=1000 nvidia-cuda-nvrtc-cu12==12.1.105
RUN pip3 install --default-timeout=1000 nvidia-cuda-runtime-cu12==12.1.105
RUN pip3 install --default-timeout=1000 nvidia-cudnn-cu12==8.9.2.26
RUN pip3 install --default-timeout=1000 nvidia-cufft-cu12==11.0.2.54
RUN pip3 install --default-timeout=1000 nvidia-curand-cu12==10.3.2.106
RUN pip3 install --default-timeout=1000 nvidia-cusolver-cu12==11.4.5.107
RUN pip3 install --default-timeout=1000 nvidia-cusparse-cu12==12.1.0.106
RUN pip3 install --default-timeout=1000 nvidia-nccl-cu12==2.18.1
RUN pip3 install --default-timeout=1000 nvidia-nvtx-cu12==12.1.105
RUN pip3 install --default-timeout=1000 torch==2.1.0
RUN pip3 install --default-timeout=1000 transformers
RUN pip3 install --default-timeout=1000 tokenizers

COPY ./deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install /deps/faiss_gpu-1.7.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl