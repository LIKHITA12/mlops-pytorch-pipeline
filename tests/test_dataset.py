import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.dataset import CIFAR10_CLASSES, FASHION_MNIST_CLASSES, get_transforms


def test_cifar10_classes_count():
    assert len(CIFAR10_CLASSES) == 10
    assert "airplane" in CIFAR10_CLASSES
    assert "truck" in CIFAR10_CLASSES


def test_fashion_mnist_classes_count():
    assert len(FASHION_MNIST_CLASSES) == 10


def test_get_transforms_cifar10():
    train_tf = get_transforms(train=True, dataset_name="cifar10", image_size=32)
    val_tf = get_transforms(train=False, dataset_name="cifar10", image_size=32)

    assert isinstance(train_tf, transforms.Compose)
    assert isinstance(val_tf, transforms.Compose)


def test_get_transforms_fashionmnist():
    train_tf = get_transforms(train=True, dataset_name="fashionmnist", image_size=28)
    val_tf = get_transforms(train=False, dataset_name="fashionmnist", image_size=28)

    assert isinstance(train_tf, transforms.Compose)
    assert isinstance(val_tf, transforms.Compose)


def test_dataloader_batch_processing():
    # Test batch iteration with DataLoader
    dummy_x = torch.randn(16, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 3, 32, 32)
    assert batch_y.shape == (4,)
