import logging
import os
import time
import numpy as np
import faiss

class IndexBuilder:
    """Class to build an index using transformer models and FAISS."""

    def __init__(self, tokenizer, model, device, documents_dir):
        """Initializes the IndexBuilder with given dependencies."""
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.documents_dir = documents_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("IndexBuilder initialized")

    def _embed_text(self, text):
        """Embeds the text using the loaded transformer model."""
        try:
            self.logger.debug("Embedding text")
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True, max_length=512, padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().detach().numpy()
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise

    def _create_gpu_index(self, d):
        """Creates a FAISS index and replicates it over all GPUs."""
        try:
            cpu_index = faiss.IndexFlatL2(d)
            cloner_options = faiss.GpuMultipleClonerOptions()
            cloner_options.shard = True
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, cloner_options)
            return gpu_index
        except Exception as e:
            self.logger.error(f"Error creating GPU index: {e}")
            raise

    def create_index(self):
        """Creates an index and saves it to a specified path."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            index_path = f"/app/output/index/{timestamp}.index"

            self.logger.info("Creating index")
            d = 768  # Dimension of embeddings (for BERT-base)

            index = self._create_gpu_index(d) if self.device == 'cuda' else faiss.IndexFlatL2(d)

            embeddings = []
            for filename in sorted(os.listdir(self.documents_dir)):
                try:
                    file_path = os.path.join(self.documents_dir, filename)
                    with open(file_path, 'r') as file:
                        text = file.read()
                        embedding = self._embed_text(text)
                        embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"Error processing file {filename}: {e}")

            if embeddings:
                embeddings_matrix = np.vstack(embeddings)
                index.add(embeddings_matrix)

            document_count = len(embeddings)
            self.logger.info(f"Index created with {document_count} documents")

            if self.device == 'cuda':
                index = faiss.index_gpu_to_cpu(index)

            self.logger.info(f"Saving index to {index_path}")
            faiss.write_index(index, index_path)
        except Exception as e:
            self.logger.error(f"Error creating or saving index: {e}")
            raise
