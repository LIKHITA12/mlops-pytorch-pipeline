import os
import ssl
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# If certifi is installed, prefer its CA bundle to avoid macOS/system SSL issues
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    _certifi_ctx = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda *a, **k: _certifi_ctx
except Exception:
    certifi = None
    _certifi_ctx = None

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Multiple mirror options for fast reliable CIFAR-10 downloads
CIFAR10_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://storage.googleapis.com/cvdf-datasets/cifar-10-python.tar.gz",
]


def ensure_cifar10_downloaded(data_dir: str):
    """Ensure CIFAR-10 archive is downloaded and unpacked using resilient fast download."""
    data_path = Path(data_dir)
    extracted_dir = data_path / "cifar-10-batches-py"
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        return

    data_path.mkdir(parents=True, exist_ok=True)
    tar_path = data_path / "cifar-10-python.tar.gz"

    if not tar_path.exists():
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        downloaded = False
        for url in CIFAR10_URLS:
            try:
                print(f"Downloading CIFAR-10 from {url} ...", flush=True)
                req = urllib.request.Request(url, headers=headers)
                ctx = _certifi_ctx or ssl.create_default_context()
                with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(tar_path, "wb") as out_file:
                    total_size = int(response.headers.get("content-length", 0))
                    chunk_size = 1024 * 1024  # 1MB chunks
                    downloaded_size = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}% ({downloaded_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)
                    print("\nDownload complete.", flush=True)
                downloaded = True
                break
            except Exception as e:
                print(f"\nMirror {url} failed: {e}. Trying next mirror...")
                if tar_path.exists():
                    tar_path.unlink()

        if not downloaded and not tar_path.exists():
            print("Direct mirror download failed; falling back to standard torchvision downloader.")
            return

    # Extract archive if not already extracted
    if tar_path.exists() and not extracted_dir.exists():
        print(f"Extracting {tar_path} to {data_dir} ...", flush=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.", flush=True)



def get_transforms(
    train: bool = True,
    dataset_name: str = "cifar10",
    image_size: int = 32,
) -> transforms.Compose:
    """Return appropriate image transforms for training or evaluation."""
    d_name = dataset_name.lower().replace("-", "").replace("_", "")

    if d_name == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
        if train:
            transform_list = []
            if image_size != 32:
                transform_list.append(transforms.Resize((image_size, image_size)))
            transform_list.extend([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            return transforms.Compose(transform_list)

        transform_list = []
        if image_size != 32:
            transform_list.append(transforms.Resize((image_size, image_size)))
        transform_list.extend([
            transforms.ToTensor(),
            normalize,
        ])
        return transforms.Compose(transform_list)

    if d_name in {"fashionmnist", "mnist"}:
        normalize = transforms.Normalize((0.5,), (0.5,))
        if train:
            transform_list = []
            if image_size != 28:
                transform_list.append(transforms.Resize((image_size, image_size)))
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            return transforms.Compose(transform_list)

        transform_list = []
        if image_size != 28:
            transform_list.append(transforms.Resize((image_size, image_size)))
        transform_list.extend([
            transforms.ToTensor(),
            normalize,
        ])
        return transforms.Compose(transform_list)

    # Generic fallback
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "cifar10",
    download: bool = True,
    image_size: int = 32,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders for supported datasets."""
    os.makedirs(data_dir, exist_ok=True)
    d_name = dataset_name.lower().replace("-", "").replace("_", "")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if d_name == "cifar10":
        if download:
            ensure_cifar10_downloaded(data_dir)

        train_transform = get_transforms(train=True, dataset_name="cifar10", image_size=image_size)
        val_transform = get_transforms(train=False, dataset_name="cifar10", image_size=image_size)
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform,
        )
        val_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=val_transform,
        )
    elif d_name in {"fashionmnist", "mnist"}:
        train_transform = get_transforms(train=True, dataset_name="fashionmnist", image_size=image_size)
        val_transform = get_transforms(train=False, dataset_name="fashionmnist", image_size=image_size)
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform,
        )
        val_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=val_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader