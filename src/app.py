# __init__.py

import logging
import torch
from transformers import AutoModel, AutoTokenizer
from index.builder import IndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():

    model_path = '/app/models/tokenizer'
    documents_dir = '/app/input/txt'

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        index_builder = IndexBuilder(tokenizer, model, device, documents_dir)
        index_builder.create_index()
        logging.info("Index building process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

