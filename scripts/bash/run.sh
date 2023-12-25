#!/bin/bash

# Read values from the YAML file
tokenizer_model=$(yq eval '.tokenizer_model' ./config/config.yaml)
index_output=$(yq eval '.index_output' ./config/config.yaml)
txt_input_dir=$(yq eval '.txt_input_dir' ./config/config.yaml)

# Build Docker images
docker build -f ./docker/base/Dockerfile -t faiss-gpu-txt-to-vector:latest .
docker build -f ./docker/rag-index-builder/Dockerfile -t faiss-gpu-txt-to-vector:latest .

# Stop and remove existing container
docker stop faiss-gpu-txt-to-vector
docker rm faiss-gpu-txt-to-vector

# Run the Docker container with the volumes from the YAML configuration
docker run \
    --name faiss-gpu-txt-to-vector \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    --volume $tokenizer_model:/app/models/tokenizer \
    --volume $index_output:/app/output/index \
    --volume $txt_input_dir:/app/input/txt \
    faiss-gpu-txt-to-vector:latest