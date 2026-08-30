import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import SimpleCNN
from src.train import evaluate, load_config, train_one_epoch


def test_load_config():
    config_content = """
    model:
      architecture: resnet18
      num_classes: 10
    training:
      epochs: 2
      batch_size: 16
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config["model"]["architecture"] == "resnet18"
        assert config["model"]["num_classes"] == 10
        assert config["training"]["epochs"] == 2
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_train_one_epoch_and_evaluate():
    model = SimpleCNN(in_channels=3, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    dummy_x = torch.randn(8, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4)

    train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, device)
    assert train_loss >= 0.0
    assert 0.0 <= train_acc <= 1.0

    val_loss, val_acc = evaluate(model, loader, criterion, device)
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0


def test_checkpoint_save_and_load():
    model = SimpleCNN(in_channels=3, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_model.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": 0.5,
                "val_accuracy": 0.85,
                "architecture": "cnn",
                "num_classes": 10,
            },
            save_path,
        )

        assert save_path.exists()
        loaded = torch.load(save_path, map_location="cpu")
        assert "model_state_dict" in loaded
        assert loaded["architecture"] == "cnn"
        assert loaded["num_classes"] == 10