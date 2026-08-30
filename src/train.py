import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work seamlessly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml

try:
    from dataset import get_dataloaders
    from model import get_model
except ImportError:
    from src.dataset import get_dataloaders
    from src.model import get_model


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress every 50 batches or on first/last batch
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches or batch_idx == 0:
            current_acc = correct / total if total > 0 else 0.0
            print(
                f"\rEpoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{total_batches}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {current_acc * 100:.2f}%",
                end="",
                flush=True,
            )

    print()  # newline after epoch batch progress
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy



@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def resolve_config_path(cli_path: str | None = None) -> Path:
    if cli_path and Path(cli_path).exists():
        return Path(cli_path)
    env_cfg = os.environ.get("TRAINING_CONFIG")
    if env_cfg and Path(env_cfg).exists():
        return Path(env_cfg)
    if Path("/app/configs/training_config.yaml").exists():
        return Path("/app/configs/training_config.yaml")
    if Path("configs/training_config.yaml").exists():
        return Path("configs/training_config.yaml")
    return Path(cli_path or "configs/training_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch image classification model")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--download", action="store_true", default=True, help="Download dataset if not present")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    # Determine device (supports CUDA, Apple Silicon MPS, or CPU)
    preferred_device = training_cfg.get("device")
    if preferred_device and preferred_device != "auto":
        device = torch.device(preferred_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Hyperparameters
    architecture = model_cfg.get("architecture", "resnet18")
    num_classes = int(model_cfg.get("num_classes", 10))
    dataset_name = data_cfg.get("dataset", "cifar10")
    data_dir = data_cfg.get("data_dir", "./data")

    epochs = args.epochs or int(training_cfg.get("epochs", 10))
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 64))
    lr = args.lr or float(training_cfg.get("learning_rate", 0.001))
    patience = int(training_cfg.get("early_stopping_patience", 3))
    # Default to 0 on macOS to avoid fork/multiprocessing lock issues, 2 on Linux/containers
    default_workers = 0 if sys.platform == "darwin" else 2
    num_workers = int(training_cfg.get("num_workers", default_workers))

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "checkpoints"))
    model_name = output_cfg.get("model_name", "classifier_v1.pt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting training: model={architecture}, dataset={dataset_name}, device={device}, "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}",
        flush=True,
    )

    # Input channels based on dataset
    d_name = dataset_name.lower().replace("-", "").replace("_", "")
    in_channels = 1 if d_name in {"fashionmnist", "mnist"} else 3

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
        download=args.download,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch=epoch + 1, total_epochs=epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)


        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / model_name
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "architecture": architecture,
                    "num_classes": num_classes,
                },
                save_path,
            )
            print(json.dumps({"event": "checkpoint_saved", "path": str(save_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                break

    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()