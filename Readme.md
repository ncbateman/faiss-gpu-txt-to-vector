# faiss-gpu-txt-to-vector Application

![Ubuntu 20.04](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)
![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.7.2-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1.105-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-latest-yellow.svg)

## Overview
The "faiss-gpu-txt-to-vector" is an advanced tool designed for converting text documents into vector representations using GPU acceleration. It leverages PyTorch, FAISS, and Transformer models to efficiently process large text datasets for various machine learning applications.

## System Requirements
Before setting up, ensure your system meets these requirements:
* Docker environment with GPU support enabled.
* Python 3.8 or higher.
* Access to NVIDIA CUDA libraries for GPU acceleration.

## Installation and Running the Application
To install and run the "faiss-gpu-txt-to-vector" application, follow these steps:

1. **Execute the run.sh Script:**
   Navigate to your application directory and run the `run.sh` script. This script builds the required Docker images and initiates the Docker container with appropriate GPU support and mounted directories.

   ```bash
   sh ./scripts/bash/run.sh --tokenizer-model /path/to/models --index-output /path/to/indexes --txt-input-dir /path/to/input
   ```

   ### Script Flags:
   - `-t` or `--tokenizer-model`: Path to the tokenizer model. Default: `./models/`
   - `-i` or `--index-output`: Directory for index output. Default: `./indexes/`
   - `-d` or `--txt-input-dir`: Directory for input text files. Default: `./input/`

   Replace `/path/to/models`, `/path/to/indexes`, and `/path/to/input` with your specific paths to use non-default settings.

## Usage
The "faiss-gpu-txt-to-vector" application is designed to handle text files in the designated input directory and create a vectorized index. Ensure your text data is correctly formatted and placed in the input directory prior to executing the application.
