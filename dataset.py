"""
MNIST Dataset loading and preprocessing utilities.

Key Concepts:
- Normalization: Raw pixel values are integers [0, 255].
  Transforming them to [0.0, 1.0] and standardizing with mean/std keeps
  gradients stable and speeds up training.
- DataLoader: Batches data (e.g. 64 images at a time) and shuffles
  training data so the network doesn't memorize the sequence of classes.
"""

import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Standard MNIST normalization constants
# mean: ~0.1307, std: ~0.3081 computed across the 60,000 training images
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and test transforms.
    1. ToTensor(): Converts PIL Image (H x W x C) [0, 255] to torch.FloatTensor (C x H x W) [0.0, 1.0]
    2. Normalize: Subtracts mean and divides by standard deviation: (x - mean) / std
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    return train_transform, test_transform


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads (if needed) and returns PyTorch DataLoader objects for Train and Test sets.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle every epoch so batches are random
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # No need to shuffle evaluation data
        num_workers=0
    )

    return train_loader, test_loader


def ascii_render(image_tensor: torch.Tensor, threshold: float = 0.2) -> str:
    """
    Visualizes a 28x28 grayscale image right inside the terminal using ASCII characters!
    Great for immediate visual feedback without needing an external window.
    """
    # Detach and un-normalize if needed
    if image_tensor.dim() == 3:  # (1, 28, 28)
        img = image_tensor.squeeze(0).detach().cpu().numpy()
    elif image_tensor.dim() == 2:  # (28, 28)
        img = image_tensor.detach().cpu().numpy()
    else:
        img = image_tensor.view(28, 28).detach().cpu().numpy()

    # Normalize values to 0.0 - 1.0 for rendering
    min_v, max_v = img.min(), img.max()
    if max_v > min_v:
        img = (img - min_v) / (max_v - min_v)

    chars = [" ", "░", "▒", "▓", "█"]
    lines = []
    lines.append("+" + "-" * 28 + "+")
    for row in img:
        row_str = "|"
        for val in row:
            idx = int(val * (len(chars) - 1))
            row_str += chars[idx]
        row_str += "|"
        lines.append(row_str)
    lines.append("+" + "-" * 28 + "+")
    return "\n".join(lines)

