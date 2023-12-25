# RAG Index Builder Application

![Ubuntu 20.04](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.7.4-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.4.3-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.34.0-yellow.svg)

## Overview
The RAG Index Builder is a sophisticated application designed for building Retrieval-Augmented Generation (RAG) indices. Utilizing PyTorch, Transformers, and FAISS, it processes and indexes large text datasets for efficient retrieval and machine learning applications.

## System Requirements
Ensure your system meets the following requirements:
* Docker environment with CUDA support for GPU acceleration.
* Python 3.10.
* NVIDIA CUDA libraries access.

## Installation and Setup
Follow these steps to set up the RAG Index Builder:

1. **Build Docker Images**
   Use the `run.sh` script in `./scripts/bash/` to build the Docker images.
   ```bash
   cd path/to/your/application
   sh ./scripts/bash/run.sh
   ```

   This script builds two Docker images: `ragbase:latest` and `rag-index-builder:latest`.

2. **Run the Application**
   The `run.sh` script also runs the `rag-index-builder` container, setting it up with the necessary GPU support and mounting volumes for models, input data, and output directories.

   The application will process the input data stored in `/mnt/nvme2/input/cakes/` and create an optimized FAISS index in `/mnt/nvme2/output/index/`.

## Usage
The RAG Index Builder is designed to process text files stored in the specified input directory and generate an index for efficient data retrieval tasks. Ensure that your text data is properly formatted and placed in the input directory before running the application.
