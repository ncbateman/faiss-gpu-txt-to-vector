import pytest
from unittest.mock import Mock, patch
from app import main  # Importing main from your app module

# Mock the IndexBuilder class
class MockIndexBuilder:
    def create_index(self):
        pass

@pytest.fixture
def mock_index_builder():
    with patch('app.IndexBuilder', return_value=MockIndexBuilder()):
        yield MockIndexBuilder()

def test_main_success(mock_index_builder):
    with patch('app.logging') as mock_logging:
        main()
        # Check if IndexBuilder was initialized
        assert mock_index_builder.called
        # Check if create_index was called
        mock_index_builder.return_value.create_index.assert_called_once()
        # Check if success log was written
        mock_logging.info.assert_called_with("Index building process completed successfully.")

def test_main_exception(mock_index_builder):
    # Mocking create_index to raise an exception
    mock_index_builder.return_value.create_index.side_effect = Exception("Test Error")

    with patch('app.logging') as mock_logging, pytest.raises(Exception):
        main()
        # Check if error log was written
        mock_logging.error.assert_called_with("An error occurred: Test Error")
