import torch
from torch import nn

try:
    from torchvision import models
except ImportError:
    models = None


class SimpleCNN(nn.Module):
    """Convolutional Neural Network for image classification (e.g., CIFAR-10, Fashion-MNIST)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleClassifier(nn.Module):
    """Basic MLP classifier for tabular/vector data."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    in_channels: int = 3,
    input_dim: int = 10,
    hidden_dim: int = 32,
    pretrained: bool = False,
) -> nn.Module:
    """Build and return a model instance based on architecture name."""
    arch = (architecture or "resnet18").lower()

    if arch in {"cnn", "simplecnn"}:
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if arch in {"resnet18", "resnet-18", "resnet"}:
        if models is None:
            raise RuntimeError("torchvision.models is required for resnet architectures")

        try:
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
            else:
                model = models.resnet18(weights=None)
        except (AttributeError, TypeError):
            # Fallback for older torchvision versions
            model = models.resnet18(pretrained=pretrained)

        # Adapt first conv layer if input channels != 3 (e.g. 1 channel for grayscale)
        if in_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Adapt final fully connected layer for num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if arch in {"mlp", "simpleclassifier"}:
        return SimpleClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    raise ValueError(f"Unsupported architecture: {architecture}")


if __name__ == "__main__":
    sample = torch.randn(2, 3, 32, 32)
    model = get_model("resnet18", num_classes=10)
    output = model(sample)
    print("ResNet-18 output shape:", output.shape)