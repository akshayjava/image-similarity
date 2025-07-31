# Local Similarity Search Engine

This project is a high-performance similarity search engine designed to run entirely on your local machine. It allows you to find the most similar items to a given query from a large dataset, without needing to send your data to the cloud.

## Features

-   **Privacy-Focused**: All data and computations stay on your machine, ensuring privacy and security.
-   **High Performance**: Optimized for fast similarity searches on large datasets.
-   **Simple API**: An easy-to-use interface for indexing data and performing searches.

## Getting Started

Follow these instructions to get the project set up on your local machine.

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

### Installation

1.  Clone the repository:
    ```sh
    git clone <your-repository-url>
    ```
2.  Navigate into the project directory:
    ```sh
    cd <repository-name>
    ```
3.  Install the necessary dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Here is a basic example of how to use the similarity search engine.

```python
# main.py
import numpy as np
from similarity_engine import SimilarityEngine

# Initialize the engine with the vector dimension
engine = SimilarityEngine(dimension=128)

# Create some random vectors to index
num_vectors = 1000
vector_dim = 128
vectors = np.random.rand(num_vectors, vector_dim).astype('float32')
ids = [f"id_{i}" for i in range(num_vectors)]

# Index the vectors
engine.index(vectors, ids)

# Create a query vector
query_vector = np.random.rand(1, vector_dim).astype('float32')

# Perform a search
results = engine.search(query_vector, top_k=5)

print("Top 5 most similar items:")
for item_id, score in results:
    print(f"ID: {item_id}, Score: {score:.4f}")
```

To run the example, execute the following command in your terminal:

```sh
python main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for