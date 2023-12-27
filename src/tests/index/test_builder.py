import pytest
from unittest.mock import Mock, patch, mock_open
import numpy as np

# Assuming your IndexBuilder class is in a file named 'index_builder.py'
from index.builder import IndexBuilder

class TestIndexBuilder:
    @pytest.fixture
    def mock_tokenizer(self):
        return Mock()

    @pytest.fixture
    def mock_model(self):
        mock_model = Mock()
        mock_model.return_value = Mock()
        return mock_model

    @pytest.fixture
    def index_builder(self, mock_tokenizer, mock_model):
        return IndexBuilder(mock_tokenizer, mock_model, 'cpu', '/path/to/documents')

    @pytest.fixture
    def mock_embedding(self):
        return np.random.rand(1, 768)

    def test_initialization(self, index_builder):
        assert index_builder.tokenizer is not None
        assert index_builder.model is not None
        assert index_builder.device == 'cpu'
        assert index_builder.documents_dir == '/path/to/documents'

    @patch('index_builder.logging')
    def test_embed_text_success(self, mock_logging, index_builder, mock_embedding):
        index_builder.model.return_value.last_hidden_state.mean.return_value.cpu.return_value.detach.return_value.numpy.return_value = mock_embedding
        result = index_builder._embed_text("test text")
        assert result.shape == (1, 768)

    @patch('index_builder.logging')
    def test_embed_text_exception(self, mock_logging, index_builder):
        index_builder.model.side_effect = Exception("Test Exception")
        with pytest.raises(Exception):
            index_builder._embed_text("test text")

    @patch('index_builder.faiss')
    @patch('index_builder.logging')
    def test_create_gpu_index_success(self, mock_logging, mock_faiss, index_builder):
        index_builder.device = 'cuda'
        result = index_builder._create_gpu_index(768)
        assert result is not None

    @patch('index_builder.faiss')
    @patch('index_builder.logging')
    def test_create_gpu_index_exception(self, mock_logging, mock_faiss, index_builder):
        index_builder.device = 'cuda'
        mock_faiss.IndexFlatL2.side_effect = Exception("Test Exception")
        with pytest.raises(Exception):
            index_builder._create_gpu_index(768)

    @patch('index_builder.faiss')
    @patch('index_builder.os.listdir', return_value=['doc1.txt', 'doc2.txt'])
    @patch('index_builder.open', new_callable=mock_open, read_data="mock file content")
    @patch('index_builder.logging')
    def test_create_index_success(self, mock_logging, mock_file_open, mock_listdir, mock_faiss, index_builder, mock_embedding):
        index_builder.device = 'cpu'
        index_builder._embed_text = Mock(return_value=mock_embedding)
        index_builder.create_index()
        assert mock_faiss.write_index.called

    @patch('index_builder.faiss')
    @patch('index_builder.os.listdir', side_effect=Exception("Test Exception"))
    @patch('index_builder.logging')
    def test_create_index_exception(self, mock_logging, mock_listdir, mock_faiss, index_builder):
        index_builder.device = 'cpu'
        with pytest.raises(Exception):
            index_builder.create_index()

# Additional tests can be added similarly for other aspects of the IndexBuilder class.
