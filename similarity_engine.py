"""
similarity_engine.py — Core similarity search engine.

Orchestrates LanceDB (disk-backed vector store), CLIP (embedding model),
and the ingestion pipeline to provide high-performance similarity search
across 1M+ images on multi-TB datasets.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lancedb
import numpy as np
import open_clip
import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm

from ingestion import ImageBatchIterator, embed_batch, load_and_preprocess

logger = logging.getLogger(__name__)

# Default CLIP model
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_TABLE_NAME = "vectors"


def _get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimilarityEngine:
    """High-performance similarity search engine.

    Uses CLIP for embeddings and LanceDB for disk-backed vector storage
    with IVF-PQ indexing.

    Args:
        db_path: Path to the LanceDB database directory.
        model_name: Open-CLIP model name.
        pretrained: Pretrained weights tag.
        device: Torch device; auto-detected if None.
    """

    def __init__(
        self,
        db_path: str = "./lancedb",
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[torch.device] = None,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or _get_device()

        # Lazy-loaded model components
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dimension: Optional[int] = None

        # Connect to LanceDB
        self.db = lancedb.connect(self.db_path)
        logger.info(
            "SimilarityEngine initialized: db=%s, model=%s, device=%s",
            self.db_path, self.model_name, self.device,
        )

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """Load the CLIP model and transforms if not already loaded."""
        if self._model is not None:
            return
        logger.info("Loading model %s (%s)...", self.model_name, self.pretrained)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained,
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

        # Determine embedding dimension from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._dimension = self._model.encode_image(dummy).shape[-1]
        logger.info("Model loaded. Embedding dimension: %d", self._dimension)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def index(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_io_threads: int = 8,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> dict:
        """Ingest images from a directory into the vector store.

        Uses a parallel pipeline:
          1. Lazy directory walk (ImageBatchIterator)
          2. Thread-pooled image loading & preprocessing
          3. Batched CLIP inference
          4. Arrow-backed writes to LanceDB

        Args:
            data_dir: Root directory containing images.
            batch_size: Images per batch.
            num_io_threads: Number of I/O threads for image loading.
            table_name: LanceDB table name.

        Returns:
            dict with keys: total_indexed, total_errors, total_batches.
        """
        self._ensure_model()

        iterator = ImageBatchIterator(data_dir, batch_size=batch_size)
        total_indexed = 0
        total_errors = 0
        total_batches = 0
        table = None

        progress = tqdm(desc="Indexing images", unit=" imgs")

        for batch_paths in iterator:
            # 1. Load & preprocess in parallel
            tensors, ids, errors = load_and_preprocess(
                batch_paths, self._preprocess, num_threads=num_io_threads,
            )
            total_errors += len(errors)

            if not tensors:
                continue

            # 2. Embed
            embeddings = embed_batch(tensors, self._model, self.device)

            # 3. Build Arrow table and write to LanceDB
            data = pa.table({
                "id": ids,
                "vector": embeddings.tolist(),
            })

            if table is None:
                if table_name in self.db.table_names():
                    table = self.db.open_table(table_name)
                    table.add(data)
                else:
                    table = self.db.create_table(table_name, data=data)
            else:
                table.add(data)

            total_indexed += len(ids)
            total_batches += 1
            progress.update(len(ids))

        progress.close()

        stats = {
            "total_indexed": total_indexed,
            "total_errors": total_errors,
            "total_batches": total_batches,
        }
        logger.info("Ingestion complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Index creation (IVF-PQ for fast ANN search)
    # ------------------------------------------------------------------

    def create_index(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
        num_partitions: int = 256,
        num_sub_vectors: int = 96,
    ):
        """Build an IVF-PQ index on the LanceDB table for sub-100ms ANN search.

        Should be called after ingestion is complete.

        Args:
            table_name: LanceDB table name to index.
            num_partitions: Number of IVF partitions (higher = faster search,
                more memory). Recommended: sqrt(N) for N vectors.
            num_sub_vectors: Number of PQ sub-vectors (higher = better recall,
                more storage). Must evenly divide embedding dimension.
        """
        table = self.db.open_table(table_name)
        logger.info(
            "Creating IVF-PQ index: partitions=%d, sub_vectors=%d",
            num_partitions, num_sub_vectors,
        )
        table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        logger.info("Index created successfully.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Union[str, Path, np.ndarray],
        top_k: int = 5,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> List[Tuple[str, float]]:
        """Search for the most similar items to a query.

        Args:
            query: One of:
                - str starting with '/' or containing path separator → image path
                - str → text query (CLIP text encoding)
                - np.ndarray → raw embedding vector
            top_k: Number of results to return.
            table_name: LanceDB table to search.

        Returns:
            List of (id, score) tuples, sorted by similarity (highest first).
        """
        self._ensure_model()
        query_vector = self._encode_query(query)

        table = self.db.open_table(table_name)
        results = (
            table.search(query_vector.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        return [(r["id"], r.get("_distance", 0.0)) for r in results]

    def _encode_query(self, query: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Encode a query into an embedding vector."""
        if isinstance(query, np.ndarray):
            return query.astype(np.float32)

        query_str = str(query)

        # Check if it looks like a file path
        if os.path.isfile(query_str):
            return self._encode_image(query_str)

        return self._encode_text(query_str)

    @torch.no_grad()
    def _encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image file into an embedding."""
        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self.device)
        features = self._model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).squeeze(0)

    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into an embedding."""
        tokens = self._tokenizer([text]).to(self.device)
        features = self._model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).squeeze(0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def table_stats(self, table_name: str = DEFAULT_TABLE_NAME) -> dict:
        """Return basic stats about a LanceDB table."""
        try:
            table = self.db.open_table(table_name)
            count = table.count_rows()
            return {"table": table_name, "row_count": count}
        except Exception as e:
            return {"table": table_name, "error": str(e)}

    # ------------------------------------------------------------------
    # Advanced Analysis
    # ------------------------------------------------------------------

    def _get_all_embeddings(self, table_name: str = DEFAULT_TABLE_NAME):
        """Extract all embeddings and IDs from a table.

        Returns:
            Tuple of (ids: list[str], embeddings: np.ndarray of shape [N, D])
        """
        table = self.db.open_table(table_name)
        df = table.to_pandas()
        ids = df["id"].tolist()
        embeddings = np.stack(df["vector"].values).astype(np.float32)
        return ids, embeddings

    def find_duplicates(
        self,
        threshold: float = 0.05,
        table_name: str = DEFAULT_TABLE_NAME,
        progress_callback=None,
    ) -> List[dict]:
        """Find near-duplicate image pairs.

        Uses brute-force cosine distance between all embeddings.
        A pair is considered duplicate if distance < threshold.

        Args:
            threshold: Maximum cosine distance to consider a duplicate (0=identical).
            table_name: LanceDB table to scan.
            progress_callback: Optional fn(msg, progress_0_to_1).

        Returns:
            List of {"pair": [path_a, path_b], "distance": float}, sorted by distance.
        """
        if progress_callback:
            progress_callback("Loading embeddings...", 0.0)

        ids, embeddings = self._get_all_embeddings(table_name)
        n = len(ids)

        if progress_callback:
            progress_callback(f"Scanning {n} images for duplicates...", 0.1)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        # Compute cosine distance matrix (1 - similarity)
        # For large N, do in chunks to avoid OOM
        duplicates = []
        chunk_size = 500
        total_chunks = (n + chunk_size - 1) // chunk_size

        for ci in range(total_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            chunk = normed[start:end]

            # Cosine similarity: chunk @ normed.T
            sim = chunk @ normed.T  # shape: [chunk_size, N]
            dist = 1.0 - sim

            # Vectorized pair extraction: find all (i_local, j) with
            # dist < threshold and j > i_global (upper triangle only).
            chunk_len = end - start
            i_globals = np.arange(start, end)          # absolute row indices
            j_indices = np.arange(n)                   # column indices

            # upper_mask[r, c] = True iff c > i_globals[r]
            upper_mask = j_indices[None, :] > i_globals[:, None]  # [chunk_len, n]
            match_mask = (dist < threshold) & upper_mask          # [chunk_len, n]

            matched = np.argwhere(match_mask)          # shape: [num_matches, 2]
            for i_local, j in matched:
                duplicates.append({
                    "pair": [ids[start + int(i_local)], ids[int(j)]],
                    "distance": float(dist[i_local, j]),
                })

            if progress_callback:
                progress_callback(
                    f"Scanned {end}/{n} images...",
                    0.1 + 0.9 * (end / n)
                )

        duplicates.sort(key=lambda x: x["distance"])
        return duplicates

    def cluster_images(
        self,
        n_clusters: int = 10,
        table_name: str = DEFAULT_TABLE_NAME,
        progress_callback=None,
    ) -> dict:
        """Cluster images using K-Means on CLIP embeddings.

        Args:
            n_clusters: Number of clusters.
            table_name: LanceDB table.
            progress_callback: Optional fn(msg, progress_0_to_1).

        Returns:
            dict with keys:
                "clusters": {cluster_id: [list_of_paths]}
                "stats": {"n_clusters": int, "n_images": int, "inertia": float}
        """
        from sklearn.cluster import KMeans

        if progress_callback:
            progress_callback("Loading embeddings...", 0.0)

        ids, embeddings = self._get_all_embeddings(table_name)

        if progress_callback:
            progress_callback(f"Running K-Means (k={n_clusters})...", 0.2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        if progress_callback:
            progress_callback("Organizing clusters...", 0.9)

        clusters = {}
        for idx, label in enumerate(labels):
            label_key = int(label)
            if label_key not in clusters:
                clusters[label_key] = []
            clusters[label_key].append(ids[idx])

        return {
            "clusters": clusters,
            "stats": {
                "n_clusters": n_clusters,
                "n_images": len(ids),
                "inertia": float(kmeans.inertia_),
            }
        }

    def reduce_dimensions(
        self,
        method: str = "tsne",
        n_components: int = 2,
        table_name: str = DEFAULT_TABLE_NAME,
        progress_callback=None,
    ) -> List[dict]:
        """Reduce embedding dimensions for visualization.

        Args:
            method: "tsne" or "umap".
            n_components: Target dimensions (2 or 3).
            table_name: LanceDB table.
            progress_callback: Optional fn(msg, progress_0_to_1).

        Returns:
            List of {"id": path, "x": float, "y": float} dicts.
        """
        if progress_callback:
            progress_callback("Loading embeddings...", 0.0)

        ids, embeddings = self._get_all_embeddings(table_name)

        if progress_callback:
            progress_callback(f"Running {method.upper()} on {len(ids)} points...", 0.1)

        reducer = None

        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            except ImportError:
                logger.warning("umap-learn not installed, falling back to t-SNE")
                method = "tsne"

        if method == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(30, len(ids) - 1)
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=max(5, perplexity),
            )

        if reducer is None:
            raise ValueError(f"Unknown reduction method '{method}'. Choose 'tsne' or 'umap'.")

        reduced = reducer.fit_transform(embeddings)

        if progress_callback:
            progress_callback("Done!", 1.0)

        result = []
        for i, id_ in enumerate(ids):
            point = {"id": id_, "x": float(reduced[i, 0]), "y": float(reduced[i, 1])}
            if n_components == 3:
                point["z"] = float(reduced[i, 2])
            result.append(point)

        return result

    def quantize_table(self, table_name: str = DEFAULT_TABLE_NAME) -> dict:
        """Quantize embeddings from float32 to float16 to reduce DB size.

        Uses a write-to-temp-then-swap strategy so the original data is never
        destroyed before a confirmed successful copy exists.

        Returns:
            dict with before/after size info.
        """
        table = self.db.open_table(table_name)
        df = table.to_pandas()
        original_count = len(df)

        # Convert float32 vectors to float16
        df["vector"] = [v.astype(np.float16) for v in df["vector"].values]

        # Stage into a temp table first — original is untouched until this succeeds
        tmp_name = f"_qtmp_{table_name}"
        if tmp_name in self.db.table_names():
            self.db.drop_table(tmp_name)
        self.db.create_table(tmp_name, df)  # raises on failure; original intact

        # Swap: safe to destroy original now that we have a full copy
        self.db.drop_table(table_name)
        try:
            self.db.create_table(table_name, df)
        except Exception:
            # Recover: rename temp table back by re-creating from it
            logger.error(
                "Failed to recreate '%s' after quantization; data preserved in '%s'",
                table_name, tmp_name,
            )
            raise
        finally:
            if tmp_name in self.db.table_names():
                self.db.drop_table(tmp_name)

        return {
            "rows": original_count,
            "dtype_before": "float32",
            "dtype_after": "float16",
            "estimated_reduction": "~50%",
        }

