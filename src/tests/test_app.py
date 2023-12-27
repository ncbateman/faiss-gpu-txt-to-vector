import pytest
from unittest.mock import patch, MagicMock
import app
from index.builder import IndexBuilder

@pytest.fixture
def mock_index_builder():
    with patch('index.builder.IndexBuilder', autospec=True) as mock_builder:
        yield mock_builder

@pytest.fixture
def mock_tokenizer():
    with patch('transformers.AutoTokenizer.from_pretrained') as mock:
        yield mock

@pytest.fixture
def mock_model():
    with patch('transformers.AutoModel.from_pretrained') as mock:
        yield mock

@pytest.fixture
def mock_torch():
    with patch('torch.cuda.is_available', return_value=True):
        yield

def test_main_success(mock_index_builder, mock_tokenizer, mock_model, mock_torch):
    mock_builder_instance = mock_index_builder.return_value
    mock_builder_instance.create_index.return_value = None

    with patch('logging.Logger.info') as mock_logging_info:
        app.main()
        # Check if IndexBuilder was initialized
        mock_index_builder.assert_called_once()
        # Check if create_index was called
        mock_builder_instance.create_index.assert_called_once()
        # Check if success log was written
        mock_logging_info.assert_called_with("Index building process completed successfully.")

def test_main_exception(mock_index_builder, mock_tokenizer, mock_model, mock_torch):
    mock_builder_instance = mock_index_builder.return_value
    mock_builder_instance.create_index.side_effect = Exception("Test Error")

    with patch('logging.Logger.error') as mock_logging_error, pytest.raises(Exception):
        app.main()
        # Check if error log was written
        mock_logging_error.assert_called_with("An error occurred: Test Error")
