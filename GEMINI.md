# GEMINI.md

## Project: Local Similarity Search Engine

### Overview
A high-performance similarity search engine that runs entirely on the local machine. Finds the most similar items to a given query from a large dataset without cloud dependencies, prioritizing privacy and performance.

### Goals
1.  **Privacy-Focused**: All data and computations remain local.
2.  **High Performance & Scalability**:
    -   Support for **1M+ images/content items**.
    -   Efficient handling of **multi-TB datasets**.
    -   Optimized for high-throughput disk reads and low-latency search.
3.  **Simple API**: Easy-to-use Python API and CLI.

### Architecture

```
┌────────────────────────────────────────────────────┐
│  main.py (CLI: ingest / search / create-index)     │
├────────────────────────────────────────────────────┤
│  similarity_engine.py (SimilarityEngine)           │
│    ├── Lazy CLIP model loading (open_clip)         │
│    ├── Parallel ingestion pipeline                 │
│    ├── IVF-PQ index creation                       │
│    └── Cross-modal search (image + text)           │
├────────────────────────────────────────────────────┤
│  ingestion.py                                      │
│    ├── ImageBatchIterator (lazy dir walk)           │
│    ├── ThreadPool image loading & preprocessing    │
│    └── Batched CLIP inference (GPU/CPU)             │
├────────────────────────────────────────────────────┤
│  LanceDB (disk-backed, memory-mapped, Arrow)       │
│    └── IVF-PQ index for sub-100ms ANN search       │
└────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Why |
|---|---|---|
| Vector Store | LanceDB | Disk-native, memory-mapped, zero-copy, handles data >> RAM |
| Embeddings | CLIP ViT-B/32 (open_clip) | Strong image+text embeddings, runs locally |
| Indexing | IVF-PQ (LanceDB built-in) | Sub-100ms search at million scale |
| Parallel I/O | concurrent.futures | ThreadPool for I/O, batched GPU inference |

### Performance Requirements
-   **Query Latency**: < 100ms for ANN search at 1M+ scale.
-   **Indexing Throughput**: Saturate local I/O and compute during ingestion.
-   **Scalability**: Linear scaling up to multi-TB datasets.

### Project Structure
```
image-similarity/
├── main.py                  # CLI entry point
├── similarity_engine.py     # Core engine (LanceDB + CLIP)
├── ingestion.py             # Parallel ingestion pipeline
├── datasets.py              # Benchmark dataset downloader
├── requirements.txt         # Python dependencies
├── tests/
│   └── test_engine.py       # Unit tests
├── benchmarks/
│   └── bench_search.py      # Latency & throughput benchmarks
├── GEMINI.md                # This file
└── README.md                # User-facing docs
```

### CLI Usage
```sh
# Ingest images
python main.py ingest --data-dir /path/to/images --batch-size 256 --workers 8

# Search by text
python main.py search --query "a red car" --top-k 10

# Search by image
python main.py search --query /path/to/query.jpg --top-k 10

# Build ANN index (after ingestion)
python main.py create-index

# Show stats
python main.py stats

# Download benchmark datasets
python main.py download --list
python main.py download --dataset cifar10 --dest ./data

# One-shot demo: download → ingest → search
python main.py demo --dataset cifar10 --query "airplane" --top-k 5
```

### Current Status
-   **Core Implementation**: Complete — `similarity_engine.py`, `ingestion.py`, `main.py`.
-   **Tests**: `tests/test_engine.py` — unit tests for iterator, LanceDB integration.
-   **Benchmarks**: `benchmarks/bench_search.py` — latency & throughput measurement.
-   **Pending**: Install deps, run tests, run benchmarks, validate at scale.

### Development Notes
-   Model is lazy-loaded to keep import time fast.
-   `ImageBatchIterator` uses `os.walk` with lazy yielding to handle multi-TB directories without memory pressure.
-   Embeddings are L2-normalized before storage for cosine similarity.
-   IVF-PQ index should be created after ingestion for optimal search performance.
