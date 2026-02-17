"""
datasets.py — Download and export standard benchmark image datasets.

Uses torchvision to download datasets and exports them as individual image
files organized by class, ready for ingestion by the similarity engine.

Supported datasets:
    - cifar10:   60,000 images (32×32), 10 classes, ~170 MB
    - stl10:     113,000 images (96×96), 10 classes, ~2.6 GB
    - flowers102: 8,189 images, 102 classes, ~330 MB
    - caltech101: 9,146 images, 101 classes, ~130 MB
"""

import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


AVAILABLE_DATASETS = {
    "cifar10": {
        "description": "CIFAR-10 — 60,000 images (32×32), 10 classes",
        "size": "~170 MB",
        "classes": 10,
        "images": 60_000,
    },
    "stl10": {
        "description": "STL-10 — 113,000 images (96×96), 10+unlabeled classes",
        "size": "~2.6 GB",
        "classes": 10,
        "images": 113_000,
    },
    "flowers102": {
        "description": "Oxford Flowers 102 — 8,189 images, 102 flower species",
        "size": "~330 MB",
        "classes": 102,
        "images": 8_189,
    },
    "caltech101": {
        "description": "Caltech-101 — 9,146 images, 101 object categories",
        "size": "~130 MB",
        "classes": 101,
        "images": 9_146,
    },
}

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# STL-10 class names
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def list_datasets() -> str:
    """Return a formatted string listing all available datasets."""
    lines = ["\nAvailable benchmark datasets:\n"]
    for name, info in AVAILABLE_DATASETS.items():
        lines.append(f"  {name:14s}  {info['description']}")
        lines.append(f"  {'':14s}  Size: {info['size']}, "
                      f"Images: {info['images']:,}, Classes: {info['classes']}")
        lines.append("")
    return "\n".join(lines)


def download_dataset(
    name: str,
    dest_dir: str,
    split: str = "all",
) -> str:
    """Download a dataset and export images organized by class.

    Images are saved as: {dest_dir}/{dataset_name}/{class_name}/{index}.jpg

    Args:
        name: Dataset name (one of AVAILABLE_DATASETS keys).
        dest_dir: Root destination directory.
        split: Which split to download — 'train', 'test', or 'all'.

    Returns:
        Path to the exported dataset directory.
    """
    name = name.lower()
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(AVAILABLE_DATASETS.keys())}"
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    export_dir = dest / name
    cache_dir = dest / ".cache"

    if name == "cifar10":
        return _download_cifar10(cache_dir, export_dir, split)
    elif name == "stl10":
        return _download_stl10(cache_dir, export_dir, split)
    elif name == "flowers102":
        return _download_flowers102(cache_dir, export_dir, split)
    elif name == "caltech101":
        return _download_caltech101(cache_dir, export_dir)
    else:
        raise ValueError(f"Dataset '{name}' not yet implemented.")


def _download_cifar10(cache_dir: Path, export_dir: Path, split: str) -> str:
    """Download CIFAR-10 and export as individual images."""
    from torchvision.datasets import CIFAR10

    splits = []
    if split in ("train", "all"):
        splits.append(("train", True))
    if split in ("test", "all"):
        splits.append(("test", False))

    total = 0
    for split_name, is_train in splits:
        logger.info("Downloading CIFAR-10 %s split...", split_name)
        dataset = CIFAR10(root=str(cache_dir), train=is_train, download=True)

        for idx, (img, label) in enumerate(tqdm(
            dataset, desc=f"Exporting CIFAR-10 {split_name}", unit=" imgs",
        )):
            class_name = CIFAR10_CLASSES[label]
            class_dir = export_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{split_name}_{idx:05d}.jpg")
            total += 1

    logger.info("CIFAR-10: exported %d images to %s", total, export_dir)
    return str(export_dir)


def _download_stl10(cache_dir: Path, export_dir: Path, split: str) -> str:
    """Download STL-10 and export as individual images."""
    from torchvision.datasets import STL10

    splits = []
    if split in ("train", "all"):
        splits.append("train")
    if split in ("test", "all"):
        splits.append("test")
    if split == "all":
        splits.append("unlabeled")

    total = 0
    for split_name in splits:
        logger.info("Downloading STL-10 %s split...", split_name)
        dataset = STL10(root=str(cache_dir), split=split_name, download=True)

        for idx in tqdm(range(len(dataset)), desc=f"Exporting STL-10 {split_name}", unit=" imgs"):
            img, label = dataset[idx]
            if label >= 0 and label < len(STL10_CLASSES):
                class_name = STL10_CLASSES[label]
            else:
                class_name = "unlabeled"
            class_dir = export_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{split_name}_{idx:05d}.jpg")
            total += 1

    logger.info("STL-10: exported %d images to %s", total, export_dir)
    return str(export_dir)


def _download_flowers102(cache_dir: Path, export_dir: Path, split: str) -> str:
    """Download Oxford Flowers 102 and export as individual images."""
    from torchvision.datasets import Flowers102

    splits = []
    if split in ("train", "all"):
        splits.append("train")
    if split in ("test", "all"):
        splits.append("test")
    if split == "all":
        splits.append("val")

    total = 0
    for split_name in splits:
        logger.info("Downloading Flowers102 %s split...", split_name)
        dataset = Flowers102(root=str(cache_dir), split=split_name, download=True)

        for idx in tqdm(range(len(dataset)), desc=f"Exporting Flowers102 {split_name}", unit=" imgs"):
            img, label = dataset[idx]
            class_name = f"class_{label:03d}"
            class_dir = export_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img.convert("RGB").save(class_dir / f"{split_name}_{idx:05d}.jpg")
            total += 1

    logger.info("Flowers102: exported %d images to %s", total, export_dir)
    return str(export_dir)


def _download_caltech101(cache_dir: Path, export_dir: Path) -> str:
    """Download Caltech-101 and export as individual images."""
    from torchvision.datasets import Caltech101

    logger.info("Downloading Caltech-101...")
    dataset = Caltech101(root=str(cache_dir), download=True)

    total = 0
    for idx in tqdm(range(len(dataset)), desc="Exporting Caltech-101", unit=" imgs"):
        img, label = dataset[idx]
        class_name = dataset.categories[label]
        class_dir = export_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img.convert("RGB").save(class_dir / f"{idx:05d}.jpg")
        total += 1

    logger.info("Caltech-101: exported %d images to %s", total, export_dir)
    return str(export_dir)
