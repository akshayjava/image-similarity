#!/usr/bin/env python3
"""
benchmarks/bench_datasets.py — End-to-end benchmark on real image datasets.

Downloads standard benchmark datasets, ingests them with the CLIP pipeline,
runs similarity searches, and reports performance metrics in a comparison table.

Usage:
    # Quick benchmark on smallest dataset only
    python benchmarks/bench_datasets.py --quick

    # Benchmark specific datasets
    python benchmarks/bench_datasets.py --datasets caltech101 flowers102

    # Benchmark all datasets
    python benchmarks/bench_datasets.py --all

    # Custom parameters
    python benchmarks/bench_datasets.py --datasets cifar10 --batch-size 512 --queries 200
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import download_dataset, AVAILABLE_DATASETS
from similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)

# Ordered by dataset size (smallest first)
# Note: caltech101 excluded from defaults — Caltech server is often offline (404)
BENCHMARK_ORDER = ["flowers102", "cifar10", "stl10"]
QUICK_DATASETS = ["cifar10"]

SAMPLE_QUERIES = {
    "caltech101": ["airplane", "car", "face", "butterfly", "dolphin"],
    "flowers102": ["sunflower", "rose", "daisy", "tulip", "orchid"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "ship"],
    "stl10": ["airplane", "bird", "car", "cat", "ship"],
}


def disk_usage_mb(path: str) -> float:
    """Measure total disk usage in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def count_files(path: str, extensions=None) -> int:
    """Count files in a directory tree."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
    count = 0
    for _, _, filenames in os.walk(path):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                count += 1
    return count


def benchmark_dataset(
    dataset_name: str,
    data_dir: str,
    db_dir: str,
    batch_size: int = 256,
    num_io_threads: int = 8,
    num_queries: int = 50,
    top_k: int = 10,
    progress_callback=None,
) -> dict:
    """Run a full benchmark on a single dataset.

    Returns a dict with all metrics.
    """
    export_dir = os.path.join(data_dir, dataset_name)
    db_path = os.path.join(db_dir, f"bench_{dataset_name}")

    result = {
        "dataset": dataset_name,
        "description": AVAILABLE_DATASETS[dataset_name]["description"],
    }

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(f"Downloading {dataset_name}...", 0.1)
    print(f"\n  [1/4] Downloading {dataset_name}...")
    t0 = time.perf_counter()
    export_path = download_dataset(dataset_name, dest_dir=data_dir)
    download_time = time.perf_counter() - t0

    num_images = count_files(export_path)
    dataset_size_mb = disk_usage_mb(export_path)
    result["num_images"] = num_images
    result["dataset_size_mb"] = round(dataset_size_mb, 1)
    result["download_time_s"] = round(download_time, 1)
    print(f"        {num_images:,} images, {dataset_size_mb:.1f} MB, "
          f"{download_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Ingest
    # ------------------------------------------------------------------
    # Step 2: Ingest
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(f"Ingesting {dataset_name}...", 0.3)
    print(f"  [2/4] Ingesting with CLIP...")
    engine = SimilarityEngine(db_path=db_path)

    t0 = time.perf_counter()
    stats = engine.index(
        data_dir=export_path,
        batch_size=batch_size,
        num_io_threads=num_io_threads,
    )
    ingest_time = time.perf_counter() - t0

    throughput = stats["total_indexed"] / ingest_time if ingest_time > 0 else 0
    result["indexed"] = stats["total_indexed"]
    result["errors"] = stats["total_errors"]
    result["ingest_time_s"] = round(ingest_time, 1)
    result["ingest_throughput_ips"] = round(throughput, 1)
    db_size_mb = disk_usage_mb(db_path)
    result["db_size_mb"] = round(db_size_mb, 1)
    print(f"        {stats['total_indexed']:,} indexed in {ingest_time:.1f}s "
          f"({throughput:.1f} imgs/sec)")
    print(f"        Errors: {stats['total_errors']}, DB size: {db_size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Step 3: Search latency
    # ------------------------------------------------------------------
    # Step 3: Search latency
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(f"Measuring search latency...", 0.8)
    print(f"  [3/4] Measuring search latency ({num_queries} queries)...")
    queries = SAMPLE_QUERIES.get(dataset_name, ["object"])

    latencies = []
    for _ in range(num_queries):
        q = queries[_ % len(queries)]
        t0 = time.perf_counter()
        _ = engine.search(query=q, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    lat = np.array(latencies)
    result["query_mean_ms"] = round(lat.mean(), 2)
    result["query_p50_ms"] = round(np.percentile(lat, 50), 2)
    result["query_p95_ms"] = round(np.percentile(lat, 95), 2)
    result["query_p99_ms"] = round(np.percentile(lat, 99), 2)
    print(f"        Mean: {lat.mean():.2f}ms, P50: {np.percentile(lat, 50):.2f}ms, "
          f"P95: {np.percentile(lat, 95):.2f}ms")

    # ------------------------------------------------------------------
    # Step 4: Sample results (qualitative)
    # ------------------------------------------------------------------
    # Step 4: Sample results (qualitative)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback(f"Running sample queries...", 0.95)
    print(f"  [4/4] Sample search results:")
    sample_query = queries[0]
    results = engine.search(query=sample_query, top_k=5)
    result["sample_query"] = sample_query
    result["sample_results"] = [
        {"id": r[0], "distance": round(r[1], 4)} for r in results
    ]
    print(f"        Query: \"{sample_query}\"")
    for rank, (rid, score) in enumerate(results, 1):
        # Show just the filename for readability
        short = "/".join(Path(rid).parts[-2:]) if "/" in rid else rid
        print(f"          {rank}. {short}  (dist: {score:.4f})")

    return result


def print_comparison_table(results: list):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK RESULTS COMPARISON")
    print(f"{'='*90}\n")

    # Header
    header = f"{'Dataset':<14} {'Images':>8} {'Ingest':>8} {'Throughput':>12} {'DB Size':>8} {'Query P50':>10} {'Query P95':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['dataset']:<14} "
            f"{r['num_images']:>7,} "
            f"{r['ingest_time_s']:>7.1f}s "
            f"{r['ingest_throughput_ips']:>9.1f} i/s "
            f"{r['db_size_mb']:>6.1f} MB "
            f"{r['query_p50_ms']:>8.2f}ms "
            f"{r['query_p95_ms']:>8.2f}ms"
        )

    print()

    # Detailed JSON
    print(f"\n{'='*90}")
    print(f"  DETAILED RESULTS (JSON)")
    print(f"{'='*90}\n")
    for r in results:
        summary = {k: v for k, v in r.items() if k != "sample_results"}
        print(json.dumps(summary, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark on real image datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Datasets to benchmark (e.g. caltech101 cifar10)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: benchmark only Caltech-101 (~9K images)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Benchmark all available datasets",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-dir", default=None, help="Data directory (default: temp)")
    parser.add_argument("--keep-data", action="store_true", help="Don't delete data after")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine datasets to benchmark
    if args.quick:
        datasets = QUICK_DATASETS
    elif args.all:
        datasets = BENCHMARK_ORDER
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = QUICK_DATASETS
        print("No datasets specified. Using --quick mode (Caltech-101).")
        print("Use --all for all datasets, or --datasets name1 name2\n")

    # Validate
    for d in datasets:
        if d not in AVAILABLE_DATASETS:
            print(f"Error: Unknown dataset '{d}'. "
                  f"Available: {list(AVAILABLE_DATASETS.keys())}")
            sys.exit(1)

    # Set up directories
    use_temp = args.data_dir is None
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="bench_data_")
    db_dir = tempfile.mkdtemp(prefix="bench_db_")

    print(f"╔{'═'*58}╗")
    print(f"║  Image Similarity — Dataset Benchmark                    ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Datasets:    {', '.join(datasets):<42} ║")
    print(f"║  Batch size:  {args.batch_size:<42} ║")
    print(f"║  I/O threads: {args.workers:<42} ║")
    print(f"║  Queries:     {args.queries:<42} ║")
    print(f"╚{'═'*58}╝")

    results = []
    try:
        for dataset_name in datasets:
            print(f"\n{'─'*60}")
            info = AVAILABLE_DATASETS[dataset_name]
            print(f"  Benchmarking: {info['description']}")
            print(f"{'─'*60}")

            result = benchmark_dataset(
                dataset_name,
                data_dir=data_dir,
                db_dir=db_dir,
                batch_size=args.batch_size,
                num_io_threads=args.workers,
                num_queries=args.queries,
                top_k=args.top_k,
            )
            results.append(result)

        print_comparison_table(results)

        # Save results to file
        results_file = os.path.join(
            Path(__file__).resolve().parent, "benchmark_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    finally:
        # Cleanup
        shutil.rmtree(db_dir, ignore_errors=True)
        if use_temp and not args.keep_data:
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"Cleaned up temp directories.")
        else:
            print(f"Data kept at: {data_dir}")


if __name__ == "__main__":
    main()
