# Local Similarity Search Engine

A high-performance, privacy-focused similarity search engine that runs entirely on your local machine. Find the most similar images to a query — by image or text — across millions of files, without sending your data to the cloud.

## Features

-   **Privacy-Focused**: All data and computations stay on your machine.
-   **High Performance**: LanceDB disk-backed vector store + CLIP embeddings, optimized for 1M+ images and multi-TB datasets.
-   **Cross-Modal Search**: Search by image path *or* text description (e.g., "a red car").
-   **Duplicate Detection**: Find near-duplicate images using cosine distance thresholds.
-   **Image Clustering**: Group similar images using K-Means on CLIP embeddings.
-   **Visual Explorer**: Interactive 2D scatter plot of your image collection (t-SNE / UMAP).
-   **Performance Optimizations**: ONNX export for 2-3x inference speedup, vector quantization for ~50% DB size reduction.
-   **Full GUI**: Streamlit-based graphical interface with search, benchmarks, database management, and analysis tools.
-   **Comprehensive CLI**: Everything the GUI can do, plus more, from the command line.

## Performance Estimates

### Search Latency (10K vectors benchmark)

| Metric | Value |
|---|---|
| Ingestion | **43,655 vectors/sec** |
| Query (median) | **3.30 ms** |
| Query (P95) | **3.66 ms** |
| Query (P99) | **4.82 ms** |

### Scaling Characteristics

| Dataset Size | Avg Search Latency | Scaling Factor |
|---|---|---|
| 1,000 vectors | 1.6 ms | 1.0x |
| 5,000 vectors | 2.9 ms | 1.8x |
| 10,000 vectors | 3.5 ms | 2.2x |

> Search scales **sub-linearly** — 10x more data only costs 2.2x in latency.

### Indexing Time for 1TB of Images

Assumes ~333K images at ~3MB average, or ~2M images at ~500KB average. CLIP embedding is the bottleneck, not disk I/O or vector storage.

#### ~333K images (3MB avg / 1TB total)

| Hardware | Throughput | Time |
|---|---|---|
| CPU only (M-series Mac) | ~40 imgs/sec | **~2.3 hours** |
| Apple MPS (M1/M2/M3) | ~300 imgs/sec | **~18 minutes** |
| NVIDIA GPU (RTX 3090+) | ~500 imgs/sec | **~11 minutes** |

#### ~2M images (500KB avg / 1TB total)

| Hardware | Throughput | Time |
|---|---|---|
| CPU only (M-series Mac) | ~40 imgs/sec | **~14 hours** |
| Apple MPS (M1/M2/M3) | ~300 imgs/sec | **~1.8 hours** |
| NVIDIA GPU (RTX 3090+) | ~500 imgs/sec | **~1.1 hours** |

> **Note**: The engine auto-detects the best available device (CUDA → MPS → CPU).

## Getting Started

### Prerequisites

Python 3.8 or higher.

### Installation

```sh
git clone <your-repository-url>
cd image-similarity
pip install -r requirements.txt
```

## Usage

### Core Commands

```sh
# Ingest images from a directory
python main.py ingest --data-dir /path/to/images --batch-size 256 --workers 8

# Search by text
python main.py search --query "a red car" --top-k 10

# Search by image
python main.py search --query /path/to/query.jpg --top-k 10

# Build ANN index (for faster search at scale)
python main.py create-index

# Show table stats
python main.py stats
```

### Advanced Analysis Commands

```sh
# Find near-duplicate images (threshold = cosine distance)
python main.py duplicates --threshold 0.05 --limit 100 --json

# Cluster images by similarity
python main.py cluster --n-clusters 10 --json

# Quantize vectors float32 → float16 (~50% DB size reduction)
python main.py quantize

# Export CLIP to ONNX format (2-3x faster inference)
python main.py export-onnx --output-dir ./models
```

### Benchmark Datasets

Download standard datasets and run similarity search demos.

**Available datasets:** CIFAR-10 (60K), STL-10 (113K), Oxford Flowers 102 (8K), Caltech-101 (9K)

```sh
# List available datasets
python main.py download --list

# Download a dataset
python main.py download --dataset cifar10 --dest ./data

# One-shot demo: download → ingest → search
python main.py demo --dataset cifar10 --query "airplane" --top-k 5
```

### Run search benchmarks

```sh
python benchmarks/bench_search.py --scale 100000 --dim 512
```

### Graphical User Interface (GUI)

Run the Streamlit app for an interactive search experience:

```sh
streamlit run app.py
```

The GUI includes:
- **🔍 Search** — Search by text or image with visual results
- **📊 Benchmarks** — Run dataset benchmarks with live progress, timing breakdown, and performance charts
- **📁 Manage Databases** — Create, switch, and ingest image collections
- **🛠️ Tools** — Duplicate detection, clustering, and visual explorer

## Test Suite

Run the comprehensive test suite:

```sh
# All tests (37 tests)
python -m pytest tests/ -v -s

# Feature tests only
python -m pytest tests/test_advanced.py -v

# Performance benchmarks only
python -m pytest tests/test_performance.py -v -s
```

| Test File | Tests | Coverage |
|---|---|---|
| `test_engine.py` | 8 | Iterator, raw vectors, top-k, table stats |
| `test_advanced.py` | 15 | Duplicate detection, clustering, dim-reduction, quantization, db config |
| `test_performance.py` | 14 | Search latency, dup scan speed, K-Means, t-SNE, ingestion, scaling |

## Python API

```python
from similarity_engine import SimilarityEngine

engine = SimilarityEngine(db_path="./mydb")

# Ingest
engine.index(data_dir="/path/to/images", batch_size=256, num_io_threads=8)

# Build index for fast ANN search
engine.create_index()

# Search by text
results = engine.search("a red car", top_k=5)

# Search by image
results = engine.search("/path/to/query.jpg", top_k=5)

for item_id, score in results:
    print(f"{item_id}: {score:.4f}")

# Find duplicates
dups = engine.find_duplicates(threshold=0.05)

# Cluster images
clusters = engine.cluster_images(n_clusters=10)

# Reduce dimensions for visualization
points = engine.reduce_dimensions(method="tsne")

# Quantize embeddings
engine.quantize_table()
```

## Project Structure

```
image-similarity/
├── similarity_engine.py  # Core engine (CLIP + LanceDB)
├── ingestion.py          # Batch image loader pipeline
├── datasets.py           # Dataset download & export
├── main.py               # CLI entry point
├── app.py                # Streamlit GUI
├── db_config.py          # Database configuration
├── onnx_export.py        # ONNX model export
├── requirements.txt      # Dependencies
├── benchmarks/
│   ├── bench_search.py   # Synthetic vector benchmarks
│   └── bench_datasets.py # Real dataset benchmarks
└── tests/
    ├── test_engine.py    # Core engine tests
    ├── test_advanced.py  # Advanced feature tests
    └── test_performance.py # Performance benchmarks
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.