import logging
import os
import time
import numpy as np

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class IndexBuilder:
    """Class to build an index using transformer models and FAISS."""

    def __init__(self):
        """Initializes the IndexBuilder with a given configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing IndexBuilder")
        self._setup_device()
        self._load_model()
        self.documents_dir = '/app/input/txt'

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
        model_path = '/app/models/tokenizer'
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

    def _create_gpu_index(self, d):
        """Creates a FAISS index and replicates it over all GPUs."""
        cpu_index = faiss.IndexFlatL2(d)
        cloner_options = faiss.GpuMultipleClonerOptions()
        cloner_options.shard = True
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, cloner_options)
        return gpu_index

    def create_index(self):
        """Creates an index and saves it to /app/output/index/<timestamp>.index"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        index_path = f"/app/output/index/{timestamp}.index"

        self.logger.info("Creating index")
        d = 768  # Dimension of embeddings (for BERT-base)

        if self.device == 'cuda':
            index = self._create_gpu_index(d)
        else:
            index = faiss.IndexFlatL2(d)

        embeddings = []
        for filename in sorted(os.listdir(self.documents_dir)):
            file_path = os.path.join(self.documents_dir, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                embedding = self._embed_text(text)
                embeddings.append(embedding)

        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            index.add(embeddings_matrix)

        document_count = len(embeddings)
        self.logger.info(f"Index created with {document_count} documents")

        # Convert GPU index to CPU index before saving
        if self.device == 'cuda':
            index = faiss.index_gpu_to_cpu(index)

        self.logger.info(f"Saving index to {index_path}")
        faiss.write_index(index, index_path)


if __name__ == "__main__":
    index_builder = IndexBuilder()
    index_builder.create_index()
