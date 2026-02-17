#!/usr/bin/env python3
"""
benchmarks/bench_search.py — Benchmark search latency and ingestion throughput.

Usage:
    python benchmarks/bench_search.py --scale 1000
    python benchmarks/bench_search.py --scale 100000

Measures:
    - Ingestion throughput (vectors/sec)
    - Query latency (ms) — mean, p50, p95, p99
    - Memory and disk usage
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate n random L2-normalized vectors."""
    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def benchmark_ingestion(db_path: str, vectors: np.ndarray, batch_size: int = 10000):
    """Benchmark ingestion throughput."""
    db = lancedb.connect(db_path)
    n = len(vectors)
    table = None

    start = time.perf_counter()
    for i in range(0, n, batch_size):
        batch = vectors[i : i + batch_size]
        ids = [f"id_{j}" for j in range(i, i + len(batch))]
        data = pa.table({
            "id": ids,
            "vector": [v.tolist() for v in batch],
        })
        if table is None:
            table = db.create_table("vectors", data=data)
        else:
            table.add(data)
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\n--- Ingestion Benchmark ---")
    print(f"  Vectors:    {n:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} vectors/sec")

    return table, elapsed


def benchmark_search(table, vectors: np.ndarray, num_queries: int = 100, top_k: int = 10):
    """Benchmark query latency."""
    rng = np.random.RandomState(99)
    query_indices = rng.randint(0, len(vectors), size=num_queries)

    latencies = []
    for idx in query_indices:
        q = vectors[idx].tolist()
        start = time.perf_counter()
        _ = table.search(q).metric("cosine").limit(top_k).to_list()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    print(f"\n--- Search Benchmark ---")
    print(f"  Queries:   {num_queries}")
    print(f"  Top-K:     {top_k}")
    print(f"  Mean:      {latencies.mean():.2f} ms")
    print(f"  Median:    {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:       {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:       {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min:       {latencies.min():.2f} ms")
    print(f"  Max:       {latencies.max():.2f} ms")


def measure_disk_usage(db_path: str):
    """Measure total disk usage of the LanceDB database."""
    total = 0
    for dirpath, _, filenames in os.walk(db_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    mb = total / (1024 * 1024)
    print(f"\n--- Disk Usage ---")
    print(f"  Total: {mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark similarity search")
    parser.add_argument("--scale", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=512, help="Vector dimension")
    parser.add_argument("--batch-size", type=int, default=10000, help="Ingestion batch size")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="bench_lancedb_")
    db_path = os.path.join(tmpdir, "bench_db")

    try:
        print(f"Benchmark: scale={args.scale:,}, dim={args.dim}, batch={args.batch_size:,}")
        vectors = generate_vectors(args.scale, args.dim)

        table, _ = benchmark_ingestion(db_path, vectors, batch_size=args.batch_size)
        benchmark_search(table, vectors, num_queries=args.queries, top_k=args.top_k)
        measure_disk_usage(db_path)
    finally:
        shutil.rmtree(tmpdir)
        print(f"\nCleaned up temp directory: {tmpdir}")


if __name__ == "__main__":
    main()
