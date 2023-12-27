#!/bin/bash

# Default values
default_tokenizer_model="./models/"
default_index_output="./indexes/"
default_txt_input_dir="./input/"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tokenizer-model) tokenizer_model="$2"; shift ;;
        -i|--index-output) index_output="$2"; shift ;;
        -d|--txt-input-dir) txt_input_dir="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set default values if not provided
tokenizer_model=${tokenizer_model:-$default_tokenizer_model}
index_output=${index_output:-$default_index_output}
txt_input_dir=${txt_input_dir:-$default_txt_input_dir}

# Build Docker images
docker build -f ./docker/base/Dockerfile -t faiss-gpu-txt-to-vector:latest .
docker build -f ./docker/rag-index-builder/Dockerfile -t faiss-gpu-txt-to-vector:latest .

# Stop and remove existing container
docker stop faiss-gpu-txt-to-vector 2>/dev/null
docker rm faiss-gpu-txt-to-vector 2>/dev/null

# Run the Docker container with the volumes
docker run \
    --name faiss-gpu-txt-to-vector \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    --volume $tokenizer_model:/app/models/tokenizer \
    --volume $index_output:/app/output/index \
    --volume $txt_input_dir:/app/input/txt \
    faiss-gpu-txt-to-vector:latest
