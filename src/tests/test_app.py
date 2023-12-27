import pytest
from unittest.mock import patch, MagicMock
import app
from index.builder import IndexBuilder

@pytest.fixture
def mock_index_builder_create_index():
    with patch.object(IndexBuilder, 'create_index') as mock:
        yield mock

@pytest.fixture
def mock_tokenizer():
    with patch('transformers.AutoTokenizer.from_pretrained') as mock:
        yield mock

@pytest.fixture
def mock_model():
    with patch('transformers.AutoModel.from_pretrained') as mock:
        yield mock

@pytest.fixture
def mock_torch_cuda():
    with patch('torch.cuda.is_available', return_value=True):
        yield

def test_main_success(mock_index_builder_create_index, mock_tokenizer, mock_model, mock_torch_cuda):
    with patch('logging.info') as mock_logging_info:
        app.main()
        # Check if create_index was called
        mock_index_builder_create_index.assert_called_once()
        # Check if success log was written
        mock_logging_info.assert_called_with("Index building process completed successfully.")

def test_main_exception(mock_index_builder_create_index, mock_tokenizer, mock_model, mock_torch_cuda):
    mock_index_builder_create_index.side_effect = Exception("Test Error")

    with patch('logging.error') as mock_logging_error, pytest.raises(Exception):
        app.main()
        # Check if error log was written
        mock_logging_error.assert_called_with("An error occurred: Test Error")
