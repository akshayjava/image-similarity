"""
tests/test_advanced.py — Comprehensive tests for advanced features.

Tests cover:
  - Duplicate Detection (correctness + edge cases)
  - Image Clustering (K-Means correctness + cluster assignments)
  - Dimensionality Reduction (t-SNE output shape)
  - Vector Quantization (dtype conversion)
  - DB Config persistence
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa


def _create_test_db(db_path, dim=32, n=100, seed=42):
    """Create a LanceDB test database with random normalized vectors."""
    import lancedb

    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [f"/fake/img_{i}.jpg" for i in range(n)]

    db = lancedb.connect(db_path)
    data = pa.table({"id": ids, "vector": [v.tolist() for v in vectors]})
    db.create_table("vectors", data=data)
    return db, vectors, ids


class TestDuplicateDetection(unittest.TestCase):
    """Test find_duplicates logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_finds_exact_duplicates(self):
        """Two identical vectors should be found as duplicates."""
        import lancedb

        db = lancedb.connect(self.db_path)
        vec = np.ones(32, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        data = pa.table({
            "id": ["/a.jpg", "/b.jpg", "/c.jpg"],
            "vector": [vec.tolist(), vec.tolist(), np.random.rand(32).astype(np.float32).tolist()],
        })
        db.create_table("vectors", data=data)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        dups = engine.find_duplicates(threshold=0.01)

        # a.jpg and b.jpg should be duplicates (distance ~0)
        self.assertGreaterEqual(len(dups), 1)
        pair_ids = set(dups[0]["pair"])
        self.assertIn("/a.jpg", pair_ids)
        self.assertIn("/b.jpg", pair_ids)
        self.assertAlmostEqual(dups[0]["distance"], 0.0, places=4)

    def test_no_duplicates_with_random_vectors(self):
        """Diverse random vectors should have no near-duplicates with tight threshold."""
        _create_test_db(self.db_path, dim=32, n=50, seed=99)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        dups = engine.find_duplicates(threshold=0.001)

        # Random 32-D vectors are very unlikely to be near-duplicates
        self.assertEqual(len(dups), 0)

    def test_high_threshold_finds_many(self):
        """A high threshold should find many pairs."""
        _create_test_db(self.db_path, dim=32, n=30, seed=7)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        dups = engine.find_duplicates(threshold=0.99)

        # With threshold ~1.0, almost every pair should be a "duplicate"
        max_pairs = 30 * 29 // 2
        self.assertGreater(len(dups), max_pairs // 2)

    def test_duplicates_sorted_by_distance(self):
        """Results should be sorted by ascending distance."""
        _create_test_db(self.db_path, dim=32, n=30, seed=5)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        dups = engine.find_duplicates(threshold=0.5)

        for i in range(len(dups) - 1):
            self.assertLessEqual(dups[i]["distance"], dups[i + 1]["distance"])

    def test_progress_callback_invoked(self):
        """Progress callback should be called during scan."""
        _create_test_db(self.db_path, dim=32, n=20, seed=3)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)

        calls = []
        def cb(msg, p):
            calls.append((msg, p))

        engine.find_duplicates(threshold=0.1, progress_callback=cb)
        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(calls[0][1], 0.0, places=1)


class TestClustering(unittest.TestCase):
    """Test cluster_images logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_correct_number_of_clusters(self):
        """Should produce exactly n_clusters clusters."""
        _create_test_db(self.db_path, dim=32, n=100, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.cluster_images(n_clusters=5)

        self.assertEqual(result["stats"]["n_clusters"], 5)
        self.assertEqual(len(result["clusters"]), 5)

    def test_all_images_assigned(self):
        """Every image should be assigned to exactly one cluster."""
        _create_test_db(self.db_path, dim=32, n=60, seed=11)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.cluster_images(n_clusters=4)

        all_assigned = []
        for paths in result["clusters"].values():
            all_assigned.extend(paths)

        self.assertEqual(len(all_assigned), 60)
        self.assertEqual(len(set(all_assigned)), 60)  # no duplicates

    def test_stats_contain_inertia(self):
        """Stats should include inertia metric."""
        _create_test_db(self.db_path, dim=32, n=50, seed=8)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.cluster_images(n_clusters=3)

        self.assertIn("inertia", result["stats"])
        self.assertGreater(result["stats"]["inertia"], 0)

    def test_single_cluster(self):
        """n_clusters=1 should put everything in one cluster."""
        _create_test_db(self.db_path, dim=32, n=20, seed=1)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.cluster_images(n_clusters=1)

        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(len(list(result["clusters"].values())[0]), 20)


class TestDimensionalityReduction(unittest.TestCase):
    """Test reduce_dimensions logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_tsne_produces_2d_output(self):
        """t-SNE should reduce to 2 components."""
        _create_test_db(self.db_path, dim=32, n=30, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.reduce_dimensions(method="tsne", n_components=2)

        self.assertEqual(len(result), 30)
        self.assertIn("x", result[0])
        self.assertIn("y", result[0])
        self.assertIn("id", result[0])

    def test_tsne_3d(self):
        """t-SNE with 3 components should include z coordinate."""
        _create_test_db(self.db_path, dim=32, n=20, seed=10)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.reduce_dimensions(method="tsne", n_components=3)

        self.assertIn("z", result[0])

    def test_output_ids_match_input(self):
        """All IDs from the DB should appear in the output."""
        _, _, ids = _create_test_db(self.db_path, dim=32, n=15, seed=7)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.reduce_dimensions(method="tsne")

        result_ids = {p["id"] for p in result}
        self.assertEqual(result_ids, set(ids))


class TestQuantization(unittest.TestCase):
    """Test quantize_table logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_quantize_reduces_dtype(self):
        """After quantization, vectors should be float16."""
        _create_test_db(self.db_path, dim=32, n=50, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        result = engine.quantize_table()

        self.assertEqual(result["rows"], 50)
        self.assertEqual(result["dtype_after"], "float16")

    def test_quantize_preserves_row_count(self):
        """Row count should be identical after quantization."""
        _create_test_db(self.db_path, dim=32, n=30, seed=9)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=self.db_path)
        engine.quantize_table()
        stats = engine.table_stats()

        self.assertEqual(stats["row_count"], 30)


class TestDBConfig(unittest.TestCase):
    """Test db_config module."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "db_config.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_add_and_load(self):
        """Adding a database should persist to config."""
        import db_config
        # Patch CONFIG_FILE path
        original = db_config.CONFIG_FILE
        db_config.CONFIG_FILE = self.config_path

        try:
            db_config.add_database("test_db", "/tmp/test_lancedb")
            config = db_config.load_config()
            self.assertIn("test_db", config["databases"])
            self.assertEqual(config["databases"]["test_db"], "/tmp/test_lancedb")
        finally:
            db_config.CONFIG_FILE = original

    def test_remove_database(self):
        """Removing a database should delete it from config."""
        import db_config
        original = db_config.CONFIG_FILE
        db_config.CONFIG_FILE = self.config_path

        try:
            db_config.add_database("to_remove", "/tmp/remove_me")
            db_config.remove_database("to_remove")
            config = db_config.load_config()
            self.assertNotIn("to_remove", config["databases"])
        finally:
            db_config.CONFIG_FILE = original


if __name__ == "__main__":
    unittest.main()
