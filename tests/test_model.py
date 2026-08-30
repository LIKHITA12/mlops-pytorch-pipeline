import pytest
import torch

from src.model import SimpleCNN, SimpleClassifier, get_model


def test_simple_classifier_output_shape():
    model = SimpleClassifier(input_dim=10, hidden_dim=16, num_classes=2)
    x = torch.randn(4, 10)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_simple_cnn_output_shape():
    model = SimpleCNN(in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_get_model_resnet18():
    model = get_model(architecture="resnet18", num_classes=10, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_get_model_cnn():
    model = get_model(architecture="cnn", num_classes=10, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_get_model_mlp():
    model = get_model(architecture="mlp", input_dim=8, hidden_dim=16, num_classes=3)
    x = torch.randn(5, 8)
    logits = model(x)
    assert logits.shape == (5, 3)


def test_unsupported_architecture():
    with pytest.raises(ValueError):
        get_model(architecture="unsupported_arch")
