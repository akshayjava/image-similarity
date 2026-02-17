"""
tests/test_performance.py — Performance & speed benchmarks.

Measures throughput, latency, and scalability of core operations.
Uses pytest-benchmark style with manual timing for portability.

Run with:
    python -m pytest tests/test_performance.py -v -s
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa


def _create_test_db(db_path, dim=512, n=1000, seed=42):
    """Create a LanceDB test database with realistic-dimension vectors."""
    import lancedb

    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [f"/data/img_{i:06d}.jpg" for i in range(n)]

    db = lancedb.connect(db_path)
    data = pa.table({"id": ids, "vector": [v.tolist() for v in vectors]})
    db.create_table("vectors", data=data)
    return db, vectors, ids


class _TimingMixin:
    """Mixin providing timing utilities for performance tests."""

    def time_fn(self, fn, label, iterations=1):
        """Run fn `iterations` times and print timing stats."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        best = min(times)
        worst = max(times)
        print(f"\n  ⏱ {label}:")
        print(f"    avg={avg*1000:.1f}ms  best={best*1000:.1f}ms  worst={worst*1000:.1f}ms  (n={iterations})")
        return result, avg


class TestSearchPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark vector search latency at different scales."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_search_latency_1k_vectors(self):
        """Search latency with 1,000 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_1k")
        db, vectors, ids = _create_test_db(db_path, dim=512, n=1000)

        table = db.open_table("vectors")
        query = vectors[0].tolist()

        def do_search():
            return table.search(query).metric("cosine").limit(10).to_list()

        results, avg_ms = self.time_fn(do_search, "Search 1K vectors", iterations=10)
        self.assertGreater(len(results), 0)
        self.assertLess(avg_ms, 1.0, "Search over 1K vectors should take <1s")

    def test_search_latency_5k_vectors(self):
        """Search latency with 5,000 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_5k")
        db, vectors, ids = _create_test_db(db_path, dim=512, n=5000, seed=99)

        table = db.open_table("vectors")
        query = vectors[0].tolist()

        def do_search():
            return table.search(query).metric("cosine").limit(10).to_list()

        results, avg_ms = self.time_fn(do_search, "Search 5K vectors", iterations=5)
        self.assertGreater(len(results), 0)
        self.assertLess(avg_ms, 2.0, "Search over 5K vectors should take <2s")

    def test_search_latency_10k_vectors(self):
        """Search latency with 10,000 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_10k")
        db, vectors, ids = _create_test_db(db_path, dim=512, n=10000, seed=77)

        table = db.open_table("vectors")
        query = vectors[0].tolist()

        def do_search():
            return table.search(query).metric("cosine").limit(10).to_list()

        results, avg_ms = self.time_fn(do_search, "Search 10K vectors", iterations=5)
        self.assertGreater(len(results), 0)
        self.assertLess(avg_ms, 5.0, "Search over 10K vectors should take <5s")

    def test_search_top1_vs_top50(self):
        """Top-1 should not be significantly slower than top-50."""
        db_path = os.path.join(self.tmpdir, "db_topk")
        db, vectors, _ = _create_test_db(db_path, dim=512, n=5000, seed=33)

        table = db.open_table("vectors")
        query = vectors[0].tolist()

        _, avg_top1 = self.time_fn(
            lambda: table.search(query).metric("cosine").limit(1).to_list(),
            "Search top-1 (5K)", iterations=5,
        )
        _, avg_top50 = self.time_fn(
            lambda: table.search(query).metric("cosine").limit(50).to_list(),
            "Search top-50 (5K)", iterations=5,
        )

        # Top-50 shouldn't be more than 5x slower than top-1
        self.assertLess(avg_top50 / max(avg_top1, 1e-9), 5.0,
                        "Top-50 should not be >5x slower than top-1")


class TestDuplicateDetectionPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark duplicate detection at different scales."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_duplicate_scan_500(self):
        """Duplicate scan speed on 500 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_dup500")
        _create_test_db(db_path, dim=512, n=500, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        _, avg = self.time_fn(
            lambda: engine.find_duplicates(threshold=0.05),
            "Duplicate scan 500 vectors", iterations=3,
        )
        self.assertLess(avg, 5.0, "Duplicate scan of 500 vectors should take <5s")

    def test_duplicate_scan_1k(self):
        """Duplicate scan speed on 1,000 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_dup1k")
        _create_test_db(db_path, dim=512, n=1000, seed=88)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        _, avg = self.time_fn(
            lambda: engine.find_duplicates(threshold=0.05),
            "Duplicate scan 1K vectors", iterations=2,
        )
        self.assertLess(avg, 15.0, "Duplicate scan of 1K vectors should take <15s")


class TestClusteringPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark clustering speed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_kmeans_1k_10clusters(self):
        """K-Means on 1K vectors, k=10."""
        db_path = os.path.join(self.tmpdir, "db_clust1k")
        _create_test_db(db_path, dim=512, n=1000, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        result, avg = self.time_fn(
            lambda: engine.cluster_images(n_clusters=10),
            "K-Means 1K/k=10", iterations=3,
        )
        self.assertLess(avg, 10.0, "K-Means on 1K vectors should take <10s")
        self.assertEqual(len(result["clusters"]), 10)

    def test_kmeans_5k_20clusters(self):
        """K-Means on 5K vectors, k=20."""
        db_path = os.path.join(self.tmpdir, "db_clust5k")
        _create_test_db(db_path, dim=512, n=5000, seed=55)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        result, avg = self.time_fn(
            lambda: engine.cluster_images(n_clusters=20),
            "K-Means 5K/k=20", iterations=2,
        )
        self.assertLess(avg, 30.0, "K-Means on 5K vectors should take <30s")


class TestDimensionalityReductionPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark t-SNE and UMAP speed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_tsne_500(self):
        """t-SNE on 500 vectors (512-D)."""
        db_path = os.path.join(self.tmpdir, "db_tsne500")
        _create_test_db(db_path, dim=512, n=500, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        result, avg = self.time_fn(
            lambda: engine.reduce_dimensions(method="tsne"),
            "t-SNE 500 vectors", iterations=1,
        )
        self.assertEqual(len(result), 500)
        self.assertLess(avg, 60.0, "t-SNE on 500 vectors should take <60s")


class TestIngestionPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark the ingestion pipeline (vector insertion, not CLIP inference)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_bulk_insert_1k_vectors(self):
        """Insert 1,000 vectors into LanceDB."""
        import lancedb

        db_path = os.path.join(self.tmpdir, "db_insert")

        def do_insert():
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            db = lancedb.connect(db_path)
            rng = np.random.RandomState(42)
            vectors = rng.rand(1000, 512).astype(np.float32)
            ids = [f"/data/img_{i}.jpg" for i in range(1000)]
            data = pa.table({"id": ids, "vector": [v.tolist() for v in vectors]})
            db.create_table("vectors", data=data)
            return db

        _, avg = self.time_fn(do_insert, "Bulk insert 1K vectors (512-D)", iterations=3)
        self.assertLess(avg, 5.0, "Inserting 1K vectors should take <5s")

    def test_bulk_insert_10k_vectors(self):
        """Insert 10,000 vectors into LanceDB."""
        import lancedb

        db_path = os.path.join(self.tmpdir, "db_insert10k")

        def do_insert():
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            db = lancedb.connect(db_path)
            rng = np.random.RandomState(42)
            vectors = rng.rand(10000, 512).astype(np.float32)
            ids = [f"/data/img_{i}.jpg" for i in range(10000)]
            data = pa.table({"id": ids, "vector": [v.tolist() for v in vectors]})
            db.create_table("vectors", data=data)
            return db

        _, avg = self.time_fn(do_insert, "Bulk insert 10K vectors (512-D)", iterations=2)
        self.assertLess(avg, 30.0, "Inserting 10K vectors should take <30s")


class TestQuantizationPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark vector quantization speed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_quantize_1k_vectors(self):
        """Quantize 1K vectors float32→float16."""
        db_path = os.path.join(self.tmpdir, "db_quant")
        _create_test_db(db_path, dim=512, n=1000, seed=42)

        from similarity_engine import SimilarityEngine
        engine = SimilarityEngine(db_path=db_path)

        result, avg = self.time_fn(
            lambda: engine.quantize_table(),
            "Quantize 1K vectors", iterations=2,
        )
        self.assertLess(avg, 5.0, "Quantizing 1K vectors should take <5s")
        self.assertEqual(result["rows"], 1000)


class TestImageBatchIteratorPerformance(unittest.TestCase, _TimingMixin):
    """Benchmark the directory walker."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a large fake directory structure
        for i in range(1000):
            subdir = Path(self.tmpdir) / f"dir_{i // 100}"
            subdir.mkdir(exist_ok=True)
            (subdir / f"img_{i}.jpg").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_iterate_1k_files(self):
        """Walk 1,000 files in 10 subdirectories."""
        from ingestion import ImageBatchIterator

        def do_iterate():
            it = ImageBatchIterator(self.tmpdir, batch_size=256)
            count = 0
            for batch in it:
                count += len(batch)
            return count

        result, avg = self.time_fn(do_iterate, "Walk 1K files (10 dirs)", iterations=5)
        self.assertEqual(result, 1000)
        self.assertLess(avg, 1.0, "Walking 1K files should take <1s")


class TestEndToEndSearchScaling(unittest.TestCase, _TimingMixin):
    """Test that search time scales sub-linearly with data size."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_search_scaling(self):
        """Search time should scale sub-linearly from 1K → 5K → 10K."""
        import lancedb

        sizes = [1000, 5000, 10000]
        times = {}

        for n in sizes:
            db_path = os.path.join(self.tmpdir, f"db_{n}")
            db, vectors, _ = _create_test_db(db_path, dim=512, n=n, seed=42)
            table = db.open_table("vectors")
            query = vectors[0].tolist()

            def do_search(t=table, q=query):
                return t.search(q).metric("cosine").limit(10).to_list()

            _, avg = self.time_fn(do_search, f"Search {n} vectors", iterations=5)
            times[n] = avg

        print(f"\n  📈 Scaling: 1K={times[1000]*1000:.1f}ms  "
              f"5K={times[5000]*1000:.1f}ms  10K={times[10000]*1000:.1f}ms")

        # 10K should NOT be 10x slower than 1K (allow 8x margin)
        if times[1000] > 0:
            ratio = times[10000] / times[1000]
            print(f"  Scaling ratio (10K/1K): {ratio:.1f}x")
            self.assertLess(ratio, 8.0,
                            f"10K search should be <8x slower than 1K (got {ratio:.1f}x)")


if __name__ == "__main__":
    unittest.main()
