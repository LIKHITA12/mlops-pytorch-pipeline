import os
import ssl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

# If certifi is installed, prefer its CA bundle to avoid macOS/system SSL issues
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    _certifi_ctx = ssl.create_default_context(cafile=certifi.where())
    # Ensure urllib/ssl uses this context by default
    ssl._create_default_https_context = lambda *a, **k: _certifi_ctx
except Exception:
    certifi = None


def get_transforms(train: bool = True, image_size: int = 32) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "CIFAR10",
    image_size: int = 32,
    download: bool = True,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders for supported datasets.

    Extra kwargs are ignored to preserve backward compatibility with callers.
    """
    if dataset_name.upper() == "CIFAR10":
        try:
            train_dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=download,
                transform=get_transforms(train=True, image_size=image_size),
            )
            val_dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=download,
                transform=get_transforms(train=False, image_size=image_size),
            )
        except Exception as e:
            # Common cause: SSL certificate verification failure when downloading
            msg = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate verify failed" in msg.lower():
                raise RuntimeError(
                    "Failed to download CIFAR10 due to SSL certificate verification error. "
                    "On macOS, run the 'Install Certificates.command' that comes with your Python installation, "
                    "or install certifi and set SSL_CERT_FILE. For example:\n"
                    "  pip install certifi\n"
                    "  python -c \"import certifi, ssl; print(certifi.where())\"\n"
                    "If you prefer offline use, set 'download=False' and place the dataset under the data directory."
                ) from e
            raise
    elif dataset_name.upper() in {"FASHIONMNIST", "FASHION-MNIST"}:
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
        val_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader