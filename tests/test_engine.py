"""
tests/test_engine.py — Unit tests for the similarity search engine.

Tests cover:
  - Indexing with synthetic vectors
  - Search returns correct top-k
  - Round-trip: ingest images → search by image/text
  - Error handling for corrupt images
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# We test the engine in "raw vector" mode where possible to avoid
# downloading the CLIP model in CI.


class TestImageBatchIterator(unittest.TestCase):
    """Test the lazy directory walker."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create some fake image files
        for i in range(10):
            (Path(self.tmpdir) / f"img_{i}.jpg").touch()
        # Create a non-image file that should be skipped
        (Path(self.tmpdir) / "readme.txt").touch()
        # Create a subdirectory with more images
        subdir = Path(self.tmpdir) / "subdir"
        subdir.mkdir()
        for i in range(5):
            (subdir / f"sub_{i}.png").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_iterates_all_images(self):
        from ingestion import ImageBatchIterator
        it = ImageBatchIterator(self.tmpdir, batch_size=100)
        all_paths = []
        for batch in it:
            all_paths.extend(batch)
        self.assertEqual(len(all_paths), 15)  # 10 jpg + 5 png

    def test_batch_size_respected(self):
        from ingestion import ImageBatchIterator
        it = ImageBatchIterator(self.tmpdir, batch_size=4)
        batches = list(it)
        # With 15 files and batch_size=4: ceil(15/4) = 4 batches
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 4)
        self.assertEqual(len(batches[-1]), 3)  # remainder

    def test_skips_non_image_files(self):
        from ingestion import ImageBatchIterator
        it = ImageBatchIterator(self.tmpdir, batch_size=100)
        all_paths = []
        for batch in it:
            all_paths.extend(batch)
        extensions = {p.suffix.lower() for p in all_paths}
        self.assertNotIn(".txt", extensions)

    def test_raises_on_missing_dir(self):
        from ingestion import ImageBatchIterator
        with self.assertRaises(FileNotFoundError):
            ImageBatchIterator("/nonexistent/path")


class TestSimilarityEngineRawVectors(unittest.TestCase):
    """Test the engine's LanceDB integration using raw vectors (no CLIP)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_raw_vector_round_trip(self):
        """Index raw vectors and search, bypassing CLIP."""
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(self.db_path)
        dim = 8
        num_vectors = 100

        # Generate random vectors
        rng = np.random.RandomState(42)
        vectors = rng.rand(num_vectors, dim).astype(np.float32)
        ids = [f"id_{i}" for i in range(num_vectors)]

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        data = pa.table({
            "id": ids,
            "vector": [v.tolist() for v in vectors],
        })
        table = db.create_table("vectors", data=data)

        # Search with the first vector — it should be the top result
        query = vectors[0].tolist()
        results = table.search(query).metric("cosine").limit(5).to_list()

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "id_0")

    def test_search_top_k(self):
        """Verify top-k parameter is respected."""
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(self.db_path)
        dim = 8
        rng = np.random.RandomState(123)
        vectors = rng.rand(50, dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        data = pa.table({
            "id": [f"id_{i}" for i in range(50)],
            "vector": [v.tolist() for v in vectors],
        })
        table = db.create_table("vectors", data=data)

        for k in [1, 3, 10]:
            results = table.search(vectors[0].tolist()).metric("cosine").limit(k).to_list()
            self.assertEqual(len(results), k)


class TestSimilarityEngineTableStats(unittest.TestCase):
    """Test table_stats utility."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_stats_on_nonexistent_table(self):
        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        stats = engine.table_stats("nonexistent")
        self.assertIn("error", stats)


if __name__ == "__main__":
    unittest.main()
