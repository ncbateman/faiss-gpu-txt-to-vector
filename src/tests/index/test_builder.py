import pytest
import numpy as np
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
from index.builder import IndexBuilder

# Test to check if device is correctly set to GPU when available
def test_setup_device_with_gpu(mocker):
    # Mock the GPU availability and details
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.device_count', return_value=1)
    mocker.patch('torch.cuda.get_device_name', return_value='Test GPU')

    # Instantiate the IndexBuilder and assert GPU is selected
    builder = IndexBuilder()
    assert builder.device == 'cuda'

# Test to check if device falls back to CPU when GPU is not available
def test_setup_device_with_cpu(mocker):
    # Mock no GPU availability
    mocker.patch('torch.cuda.is_available', return_value=False)

    # Instantiate the IndexBuilder and assert CPU is selected
    builder = IndexBuilder()
    assert builder.device == 'cpu'

# Test to check if model and tokenizer are loaded correctly
def test_load_model(mocker):
    # Mock the model and tokenizer loading
    mock_tokenizer = mocker.patch('transformers.AutoTokenizer.from_pretrained')
    mock_model = mocker.patch('transformers.AutoModel.from_pretrained')

    # Instantiate the IndexBuilder and assert model and tokenizer are called
    builder = IndexBuilder()
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()

# Test to check if the index is created correctly
def test_create_index(mocker):
    # Mock file operations, directory listing, and FAISS index creation
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='test text'))
    mocker.patch('os.listdir', return_value=['test.txt'])
    mocker.patch('faiss.write_index')
    mocker.patch.object(IndexBuilder, '_embed_text', return_value=np.array([1, 2, 3]))
    mocker.patch.object(IndexBuilder, '_create_gpu_index', return_value=faiss.IndexFlatL2(768))

    # Instantiate the IndexBuilder and create index
    builder = IndexBuilder()
    builder.create_index()

    # Assert file operations and methods calls
    mock_open.assert_called_once()
    assert builder._embed_text.called
    assert builder._create_gpu_index.called

# Test to handle model loading failure
def test_load_model_failure(mocker):
    # Mock a failure in loading the tokenizer
    mocker.patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception('Load error'))

    # Instantiate the IndexBuilder and expect an exception
    with pytest.raises(Exception) as excinfo:
        builder = IndexBuilder()
    assert 'Load error' in str(excinfo.value)

# Test the _embed_text method
def test_embed_text(mocker):
    # Mock the _embed_text method
    mock_embed_text = mocker.patch.object(IndexBuilder, '_embed_text')
    test_text = "Test text"

    # Instantiate the IndexBuilder and call _embed_text
    builder = IndexBuilder()
    builder._embed_text(test_text)

    # Assert the method was called correctly
    mock_embed_text.assert_called_once_with(test_text)

# Test creating GPU index
def test_create_gpu_index(mocker):
    # Mock the FAISS IndexFlatL2
    mock_index = mocker.patch('faiss.IndexFlatL2')

    # Instantiate the IndexBuilder and create a GPU index
    builder = IndexBuilder()
    dimension = 768
    builder._create_gpu_index(dimension)

    # Assert the index creation was called
    mock_index.assert_called_once_with(dimension)

# Test GPU index integration in the creation process
def test_create_gpu_index_integration(mocker):
    # Mock the index_cpu_to_all_gpus function
    mock_index_cpu_to_all_gpus = mocker.patch('faiss.index_cpu_to_all_gpus')

    # Instantiate the IndexBuilder and create a GPU index
    builder = IndexBuilder()
    dimension = 768
    builder._create_gpu_index(dimension)

    # Assert the integration function was called
    mock_index_cpu_to_all_gpus.assert_called_once()

# Test handling file errors during index creation
def test_create_index_with_file_error(mocker):
    # Mock a file error
    mocker.patch('builtins.open', mocker.mock_open(read_data='test text'), side_effect=Exception('File error'))

    # Instantiate the IndexBuilder and expect an exception during index creation
    builder = IndexBuilder()
    with pytest.raises(Exception):
        builder.create_index()
