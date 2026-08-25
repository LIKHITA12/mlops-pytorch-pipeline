import torch
from torch import nn

try:
    # torchvision may not be available in minimal test envs
    from torchvision import models
except Exception:
    models = None


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
    arch = architecture.lower() if architecture else "mlp"
    if arch in {"mlp", "simpleclassifier"}:
        return SimpleClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    if arch in {"resnet18", "resnet-18", "resnet"}:
        if models is None:
            raise RuntimeError("torchvision.models is required for resnet architectures")
        # Use an un-pretrained ResNet-18 adapted to `num_classes`.
        # Use the modern `weights` API to avoid deprecation warnings.
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            # Older torchvision versions may not support `weights`; fall back
            model = models.resnet18(pretrained=False)
        # Adapt final fully-connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


if __name__ == "__main__":
    sample = torch.randn(2, 10)
    model = get_model()
    output = model(sample)
    print(output.shape)
