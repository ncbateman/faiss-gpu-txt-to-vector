# index_builder.py

import logging
import os

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

from config.config import Config  # Import Config class from config module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class IndexBuilder:
    """Class to build an index using transformer models and FAISS."""

    def __init__(self, config):
        """Initializes the IndexBuilder with a given configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing IndexBuilder")

        self.config = config
        self._setup_device()
        self._load_model()
        self.documents_dir = config.get('documents_dir')

    def _setup_device(self):
        """Sets up the device for computation (GPU/CPU)."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            self._log_gpu_info()

    def _log_gpu_info(self):
        """Logs information about available GPUs."""
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    def _load_model(self):
        """Loads the tokenizer and model for embeddings."""
        model_path = self.config.get('transformer_model')
        self.logger.info(f"Loading tokenizer and model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

    def _embed_text(self, text):
        """Embeds the text using the loaded transformer model."""
        self.logger.debug("Embedding text")
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=512, padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().detach().numpy()

    def create_index(self, index_path):
        """Creates an index and saves it to the specified path."""
        self.logger.info("Creating index")
        d = 768  # Dimension of embeddings (for BERT-base)
        index = faiss.IndexFlatL2(d)

        document_count = 0
        for filename in sorted(os.listdir(self.documents_dir)):
            file_path = os.path.join(self.documents_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                embedding = self._embed_text(text)
                index.add(embedding)
                document_count += 1
                self.logger.debug(f"Added {filename} to index")

        self.logger.info(f"Index created with {document_count} documents")
        self.logger.info(f"Saving index to {index_path}")
        faiss.write_index(index, index_path)
