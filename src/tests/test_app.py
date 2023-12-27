import pytest
from unittest.mock import Mock, patch
from app import main  # Importing main from your app module

# Import the IndexBuilder class for testing
from index.builder import IndexBuilder

@pytest.fixture
def mock_index_builder():
    with patch('index.builder.IndexBuilder') as mock_builder:
        yield mock_builder

@pytest.fixture
def mock_tokenizer():
    with patch('index.builder.AutoTokenizer.from_pretrained') as mock:
        yield mock

@pytest.fixture
def mock_model():
    with patch('index.builder.AutoModel.from_pretrained') as mock:
        yield mock

def test_main_success(mock_index_builder, mock_tokenizer, mock_model):
    mock_index_builder.return_value.create_index.return_value = None

    with patch('app.logging') as mock_logging:
        main()
        # Check if IndexBuilder was initialized
        mock_index_builder.assert_called_once()
        # Check if create_index was called
        mock_index_builder.return_value.create_index.assert_called_once()
        # Check if success log was written
        mock_logging.info.assert_called_with("Index building process completed successfully.")

def test_main_exception(mock_index_builder, mock_tokenizer, mock_model):
    # Mocking create_index to raise an exception
    mock_index_builder.return_value.create_index.side_effect = Exception("Test Error")

    with patch('app.logging') as mock_logging, pytest.raises(Exception):
        main()
        # Check if error log was written
        mock_logging.error.assert_any_call("An error occurred: Test Error")
