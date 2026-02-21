"""
ingestion.py — High-throughput, memory-efficient image ingestion pipeline.

Provides parallel disk I/O, image decoding, and batch embedding generation
optimized for multi-TB datasets with 1M+ images.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Supported image extensions (lowercase)
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp",
})


class ImageBatchIterator:
    """Lazily walks a directory tree and yields batches of image file paths.

    Never loads all paths into memory — ideal for multi-TB datasets.

    Args:
        root_dir: Root directory to scan.
        batch_size: Number of file paths per batch.
        extensions: Accepted file extensions (with leading dot).
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 256,
        extensions: Optional[frozenset] = None,
    ):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.extensions = extensions or IMAGE_EXTENSIONS

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

    def __iter__(self) -> Generator[List[Path], None, None]:
        batch: List[Path] = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if Path(fname).suffix.lower() in self.extensions:
                    batch.append(Path(dirpath) / fname)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch


def load_and_preprocess(
    paths: List[Path],
    transform,
    num_threads: int = 8,
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Thread-pooled image loading, decoding, and preprocessing.

    Args:
        paths: List of image file paths.
        transform: torchvision-compatible transform to apply.
        num_threads: Number of I/O threads.

    Returns:
        Tuple of (preprocessed_tensors, valid_ids, error_paths).
        - preprocessed_tensors: Successfully loaded and transformed images.
        - valid_ids: Corresponding string IDs (absolute path).
        - error_paths: Paths that failed to load.
    """
    results: List[Tuple[Optional[torch.Tensor], str]] = [None] * len(paths)
    errors: List[str] = []

    def _load_single(idx: int, path: Path):
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            return idx, tensor, str(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return idx, None, str(path)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {
            pool.submit(_load_single, i, p): i for i, p in enumerate(paths)
        }
        for future in as_completed(futures):
            idx, tensor, path_str = future.result()
            if tensor is not None:
                results[idx] = (tensor, path_str)
            else:
                errors.append(path_str)

    # Filter out Nones (failed loads) while preserving order
    tensors = []
    ids = []
    for item in results:
        if item is not None:
            tensors.append(item[0])
            ids.append(item[1])

    return tensors, ids, errors


@torch.no_grad()
def embed_batch(
    images: List[torch.Tensor],
    model,
    device: torch.device,
) -> np.ndarray:
    """Run batch CLIP inference and return L2-normalized float32 embeddings.

    Args:
        images: List of preprocessed image tensors.
        model: CLIP model (from open_clip).
        device: torch device (cpu / cuda / mps).

    Returns:
        Numpy array of shape (N, embedding_dim), dtype float32.
    """
    if not images:
        return np.empty((0, 0), dtype=np.float32)

    # pin_memory + non_blocking overlaps the CPU→GPU transfer with other work.
    batch = torch.stack(images)
    if device.type == "cuda":
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)

    # autocast enables FP16 math on CUDA (~1.5-2x faster) with minimal accuracy loss.
    if device.type == "cuda":
        with torch.autocast("cuda"):
            features = model.encode_image(batch)
    else:
        features = model.encode_image(batch)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().float().numpy()
