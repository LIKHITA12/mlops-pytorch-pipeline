import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running
# this file as a script (e.g. `python src/train.py`). When executing a file
# directly, Python sets sys.path[0] to the script's directory (src/), which
# prevents importing the `src` package by name. Insert the repository root
# (parent of `src`) at the front of sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml
import argparse

try:
    from dataset import get_dataloaders
    from model import get_model
except ImportError:
    from src.dataset import get_dataloaders
    from src.model import get_model


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
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

    avg_loss = total_loss / total
    accuracy = correct / total
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train the model using config file")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to training config YAML")
    parser.add_argument("--download", action="store_true", help="Allow dataset download if missing (default: False)")
    args = parser.parse_args()

    config_path = Path(args.config)

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    model = get_model(
        architecture=model_cfg.get("architecture", "mlp"),
        input_dim=model_cfg.get("input_dim", 10),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        num_classes=model_cfg.get("num_classes", 2),
    ).to(device)
    # choose image size heuristically for common architectures
    arch = model_cfg.get("architecture", "mlp")
    if isinstance(arch, str) and "resnet" in arch.lower():
        image_size = 224
    else:
        image_size = 32

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg.get("data_dir", "."),
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 2),
        dataset_name=data_cfg.get("dataset", "CIFAR10"),
        image_size=image_size,
        download=args.download,
        num_samples=data_cfg.get("num_samples", 1000),
        input_dim=model_cfg.get("input_dim", 10),
        train_split=data_cfg.get("train_split", 0.8),
        seed=data_cfg.get("seed", 42),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("learning_rate", 1e-3))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_cfg.get("early_stopping_patience", 3)
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_cfg.get("epochs", 5)):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
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
            save_path = checkpoint_dir / output_cfg.get("model_name", "model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
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
