import torch
from torch import nn


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def get_model(architecture: str = "mlp", input_dim: int = 10, hidden_dim: int = 32, num_classes: int = 2) -> nn.Module:
    if architecture.lower() not in {"mlp", "simpleclassifier"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return SimpleClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)


if __name__ == "__main__":
    sample = torch.randn(2, 10)
    model = get_model()
    output = model(sample)
    print(output.shape)
