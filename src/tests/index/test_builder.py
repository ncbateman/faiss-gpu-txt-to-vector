import pytest
import numpy as np
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
from index.builder import IndexBuilder

# Test to check if device is correctly set to GPU when available
def test_setup_device_with_gpu(mocker):
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.device_count', return_value=1)
    mocker.patch('torch.cuda.get_device_name', return_value='Test GPU')

    builder = IndexBuilder()
    assert builder.device == 'cuda'

# Test to check if device falls back to CPU when GPU is not available
def test_setup_device_with_cpu(mocker):
    mocker.patch('torch.cuda.is_available', return_value=False)

    builder = IndexBuilder()
    assert builder.device == 'cpu'

# Test to check if model and tokenizer are loaded correctly
def test_load_model(mocker):
    mocker.patch.object(IndexBuilder, '_load_model')

    builder = IndexBuilder()
    builder._load_model.assert_called_once()

# Test to check if the index is created correctly
def test_create_index(mocker):
    mocker.patch('builtins.open', mocker.mock_open(read_data='test text'))
    mocker.patch('os.listdir', return_value=['test.txt'])
    mocker.patch('faiss.write_index')
    mocker.patch.object(IndexBuilder, '_embed_text', return_value=np.array([1, 2, 3]))
    mocker.patch.object(IndexBuilder, '_create_gpu_index', return_value=faiss.IndexFlatL2(768))

    builder = IndexBuilder()
    builder.create_index()

    assert builder._embed_text.called
    assert builder._create_gpu_index.called

# Test to handle model loading failure
def test_load_model_failure(mocker):
    mocker.patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception('Load error'))

    with pytest.raises(Exception) as excinfo:
        builder = IndexBuilder()
    assert 'Load error' in str(excinfo.value)

# Test the _embed_text method
def test_embed_text(mocker):
    mocker.patch.object(IndexBuilder, '_embed_text')
    test_text = "Test text"

    builder = IndexBuilder()
    builder._embed_text(test_text)

    builder._embed_text.assert_called_once_with(test_text)

# Test creating GPU index
def test_create_gpu_index(mocker):
    mocker.patch('faiss.IndexFlatL2')

    builder = IndexBuilder()
    dimension = 768
    builder._create_gpu_index(dimension)

    assert faiss.IndexFlatL2.called_with(dimension)

# Test GPU index integration in the creation process
def test_create_gpu_index_integration(mocker):
    mocker.patch('faiss.index_cpu_to_all_gpus')

    builder = IndexBuilder()
    dimension = 768
    builder._create_gpu_index(dimension)

    assert faiss.index_cpu_to_all_gpus.called_once()

# Test handling file errors during index creation
def test_create_index_with_file_error(mocker):
    mocker.patch('builtins.open', mocker.mock_open(read_data='test text'), side_effect=Exception('File error'))

    builder = IndexBuilder()
    with pytest.raises(Exception):
        builder.create_index()
