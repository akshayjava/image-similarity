#!/usr/bin/env python3
"""
main.py — CLI interface for the Local Similarity Search Engine.

Commands:
    ingest        Ingest images from a directory into the vector store.
    search        Search by image path or text query.
    create-index  Build an IVF-PQ index for fast ANN search.
    stats         Show table statistics.
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
