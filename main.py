#!/usr/bin/env python3
"""
main.py — CLI interface for the Local Similarity Search Engine.

Commands:
    ingest        Ingest images from a directory into the vector store.
    search        Search by image path or text query.
    create-index  Build an IVF-PQ index for fast ANN search.
    stats         Show table statistics.
    download      Download a standard benchmark dataset.
    demo          Download, ingest, and search a benchmark dataset (one-shot).
"""

import argparse
import json
import logging
import sys

from similarity_engine import SimilarityEngine


def cmd_ingest(args):
    engine = SimilarityEngine(db_path=args.db_path)
    stats = engine.index(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_io_threads=args.workers,
    )
    print(json.dumps(stats, indent=2))


def cmd_search(args):
    engine = SimilarityEngine(db_path=args.db_path)
    results = engine.search(query=args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(
            [{"id": r[0], "distance": r[1]} for r in results], indent=2,
        ))
    else:
        print(f"\nTop {len(results)} results:\n")
        for rank, (item_id, score) in enumerate(results, 1):
            print(f"  {rank}. {item_id}  (distance: {score:.6f})")
        print()


def cmd_create_index(args):
    engine = SimilarityEngine(db_path=args.db_path)
    engine.create_index(
        num_partitions=args.partitions,
        num_sub_vectors=args.sub_vectors,
    )
    print("Index created successfully.")


def cmd_stats(args):
    engine = SimilarityEngine(db_path=args.db_path)
    stats = engine.table_stats()
    print(json.dumps(stats, indent=2))


def cmd_download(args):
    from datasets import download_dataset, list_datasets, AVAILABLE_DATASETS

    if args.list:
        print(list_datasets())
        return

    if not args.dataset:
        print("Error: --dataset is required. Use --list to see available datasets.")
        sys.exit(1)

    print(f"Downloading '{args.dataset}' to {args.dest}...")
    export_path = download_dataset(args.dataset, dest_dir=args.dest)
    print(f"\nDataset exported to: {export_path}")
    print(f"\nTo ingest, run:")
    print(f"  python main.py ingest --data-dir {export_path}")


def cmd_demo(args):
    from datasets import download_dataset, AVAILABLE_DATASETS

    dataset_name = args.dataset
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'.")
        print(f"Available: {list(AVAILABLE_DATASETS.keys())}")
        sys.exit(1)

    info = AVAILABLE_DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"  Demo: {info['description']}")
    print(f"{'='*60}\n")

    # Step 1: Download
    print("[1/3] Downloading dataset...")
    export_path = download_dataset(dataset_name, dest_dir=args.dest)
    print(f"  → Exported to: {export_path}\n")

    # Step 2: Ingest
    print("[2/3] Ingesting into vector store...")
    engine = SimilarityEngine(db_path=args.db_path)
    stats = engine.index(
        data_dir=export_path,
        batch_size=args.batch_size,
        num_io_threads=args.workers,
    )
    print(f"  → Indexed: {stats['total_indexed']:,} images, "
          f"Errors: {stats['total_errors']}\n")

    # Step 3: Search
    print(f"[3/3] Searching for: '{args.query}'\n")
    results = engine.search(query=args.query, top_k=args.top_k)

    print(f"Top {len(results)} results:\n")
    for rank, (item_id, score) in enumerate(results, 1):
        print(f"  {rank}. {item_id}")
        print(f"     distance: {score:.6f}")
    print()


def cmd_duplicates(args):
    engine = SimilarityEngine(db_path=args.db_path)
    print(f"Scanning for duplicates (threshold={args.threshold})...")
    dups = engine.find_duplicates(threshold=args.threshold)
    print(f"\nFound {len(dups)} duplicate pair(s):\n")
    if args.json:
        print(json.dumps(dups, indent=2))
    else:
        for i, d in enumerate(dups[:args.limit], 1):
            print(f"  {i}. [{d['distance']:.6f}]")
            print(f"     A: {d['pair'][0]}")
            print(f"     B: {d['pair'][1]}")
        if len(dups) > args.limit:
            print(f"  ... and {len(dups) - args.limit} more pairs")
    print()


def cmd_cluster(args):
    engine = SimilarityEngine(db_path=args.db_path)
    print(f"Clustering into {args.n_clusters} groups...")
    result = engine.cluster_images(n_clusters=args.n_clusters)
    stats = result["stats"]
    print(f"\nClustered {stats['n_images']} images into {stats['n_clusters']} groups")
    print(f"Inertia: {stats['inertia']:.2f}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for cid, paths in sorted(result["clusters"].items()):
            print(f"  Cluster {cid} ({len(paths)} images):")
            for p in paths[:3]:
                print(f"    - {p}")
            if len(paths) > 3:
                print(f"    ... and {len(paths) - 3} more")
    print()


def cmd_quantize(args):
    engine = SimilarityEngine(db_path=args.db_path)
    print("Quantizing embeddings (float32 → float16)...")
    result = engine.quantize_table()
    print(f"Done! {result['rows']} vectors quantized.")
    print(json.dumps(result, indent=2))


def cmd_export_onnx(args):
    from onnx_export import export_clip_to_onnx
    export_clip_to_onnx(output_dir=args.output_dir)
    print(f"ONNX models exported to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="image-similarity",
        description="Local Similarity Search Engine — privacy-focused, high-performance",
    )
    parser.add_argument(
        "--db-path", default="./lancedb",
        help="Path to the LanceDB database directory (default: ./lancedb)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest images from a directory")
    p_ingest.add_argument("--data-dir", required=True, help="Root image directory")
    p_ingest.add_argument("--batch-size", type=int, default=256, help="Images per batch")
    p_ingest.add_argument("--workers", type=int, default=8, help="I/O threads")
    p_ingest.set_defaults(func=cmd_ingest)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search by image or text")
    p_search.add_argument("--query", required=True, help="Image path or text query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--json", action="store_true", help="Output as JSON")
    p_search.set_defaults(func=cmd_search)

    # --- create-index ---
    p_index = subparsers.add_parser("create-index", help="Build IVF-PQ index")
    p_index.add_argument("--partitions", type=int, default=256, help="IVF partitions")
    p_index.add_argument("--sub-vectors", type=int, default=96, help="PQ sub-vectors")
    p_index.set_defaults(func=cmd_create_index)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show table statistics")
    p_stats.set_defaults(func=cmd_stats)

    # --- download ---
    p_dl = subparsers.add_parser("download", help="Download a benchmark dataset")
    p_dl.add_argument("--dataset", help="Dataset name (cifar10, stl10, flowers102, caltech101)")
    p_dl.add_argument("--dest", default="./data", help="Destination directory (default: ./data)")
    p_dl.add_argument("--list", action="store_true", help="List available datasets")
    p_dl.set_defaults(func=cmd_download)

    # --- demo ---
    p_demo = subparsers.add_parser("demo", help="Download, ingest, and search (one-shot)")
    p_demo.add_argument("--dataset", required=True, help="Dataset name")
    p_demo.add_argument("--query", default="airplane", help="Text query (default: airplane)")
    p_demo.add_argument("--dest", default="./data", help="Data directory (default: ./data)")
    p_demo.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_demo.add_argument("--batch-size", type=int, default=256, help="Ingestion batch size")
    p_demo.add_argument("--workers", type=int, default=8, help="I/O threads")
    p_demo.set_defaults(func=cmd_demo)

    # --- duplicates ---
    p_dup = subparsers.add_parser("duplicates", help="Find near-duplicate images")
    p_dup.add_argument("--threshold", type=float, default=0.05, help="Max cosine distance (default: 0.05)")
    p_dup.add_argument("--limit", type=int, default=50, help="Max pairs to display (default: 50)")
    p_dup.add_argument("--json", action="store_true", help="Output as JSON")
    p_dup.set_defaults(func=cmd_duplicates)

    # --- cluster ---
    p_clust = subparsers.add_parser("cluster", help="Cluster images by similarity")
    p_clust.add_argument("--n-clusters", type=int, default=10, help="Number of clusters (default: 10)")
    p_clust.add_argument("--json", action="store_true", help="Output as JSON")
    p_clust.set_defaults(func=cmd_cluster)

    # --- quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize vectors float32→float16")
    p_quant.set_defaults(func=cmd_quantize)

    # --- export-onnx ---
    p_onnx = subparsers.add_parser("export-onnx", help="Export CLIP to ONNX format")
    p_onnx.add_argument("--output-dir", default="./models", help="Output directory (default: ./models)")
    p_onnx.set_defaults(func=cmd_export_onnx)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
