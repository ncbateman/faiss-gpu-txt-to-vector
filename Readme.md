# faiss-gpu-txt-to-vector Application

![Ubuntu 20.04](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)
![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.7.2-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1.105-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-latest-yellow.svg)

## Overview
The "faiss-gpu-txt-to-vector" is an advanced tool designed for converting text documents into vector representations using GPU acceleration. It leverages PyTorch, FAISS, and Transformer models to efficiently process large text datasets for various machine learning applications.

## System Requirements
Before setting up, ensure your system meets these requirements:
* Docker environment with GPU support enabled.
* Python 3.9 or higher.
* Access to NVIDIA CUDA libraries for GPU acceleration.

## Installation and Setup
To set up the "faiss-gpu-txt-to-vector" application, follow these steps:

1. **Build Docker Images:**
   Execute the `run.sh` script in the `./scripts/bash/` directory to construct the required Docker images. Navigate to your application directory and run:
   ```bash
   sh ./scripts/bash/run.sh
   ```
   This will build the Docker images: `faiss-gpu-txt-to-vector-base:latest` and `faiss-gpu-txt-to-vector:latest`.

2. **Running the Application:**
   The `run.sh` script will also initiate the `faiss-gpu-txt-to-vector` Docker container, ensuring it has appropriate GPU support and the necessary directories for models, input data, and output are properly mounted.
   
   The application processes input text data and generates an efficient FAISS index for further use.

## Usage
The "faiss-gpu-txt-to-vector" application is designed to handle text files in the designated input directory and create a vectorized index. Ensure your text data is correctly formatted and placed in the input directory prior to executing the application.
