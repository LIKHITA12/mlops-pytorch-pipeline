import torch

from src.model import SimpleClassifier


def test_model_output_shape():
    model = SimpleClassifier(input_dim=10, hidden_dim=16, num_classes=2)
    x = torch.randn(4, 10)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_model_forward_runs_without_error():
    model = SimpleClassifier()
    x = torch.randn(1, 10)
    out = model(x)
    assert out.dim() == 2
    assert out.shape[0] == 1
