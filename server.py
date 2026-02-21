"""
server.py — FastAPI REST API for the Local Similarity Search Engine.

Exposes SimilarityEngine as an HTTP microservice so external applications,
scripts, or custom frontends can integrate with the search engine without
importing Python directly.

Usage:
    # Standalone
    python server.py --db-path ./lancedb --host 0.0.0.0 --port 8000

    # Via CLI
    python main.py serve --db-path ./lancedb --port 8000

Interactive API docs are available at http://<host>:<port>/docs once running.
"""

import argparse
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from similarity_engine import SimilarityEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Similarity Search API",
    description=(
        "Privacy-focused local image similarity search powered by CLIP + LanceDB. "
        "All data stays on the server — nothing is sent to external services."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level engine — injected at startup via _init_engine() or main()
_engine: Optional[SimilarityEngine] = None


def _init_engine(db_path: str) -> None:
    """Initialize the global engine. Called once at server startup."""
    global _engine
    _engine = SimilarityEngine(db_path=db_path)


def _get_engine() -> SimilarityEngine:
    if _engine is None:
        raise HTTPException(
            status_code=503, detail="Engine not initialized. Start the server with a valid --db-path."
        )
    return _engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    where: Optional[str] = None

    model_config = {"json_schema_extra": {
        "examples": [{"query": "a red sports car", "top_k": 10}]
    }}


class SearchResult(BaseModel):
    path: str
    score: float


class IngestRequest(BaseModel):
    data_dir: str
    batch_size: int = 256
    incremental: bool = True

    model_config = {"json_schema_extra": {
        "examples": [{"data_dir": "/data/photos", "incremental": True}]
    }}


class IngestResult(BaseModel):
    total_indexed: int
    total_errors: int
    total_batches: int
    total_skipped: int


class DeleteByPathsRequest(BaseModel):
    paths: List[str]

    model_config = {"json_schema_extra": {
        "examples": [{"paths": ["/data/photos/img001.jpg", "/data/photos/img002.jpg"]}]
    }}


class DeleteByPrefixRequest(BaseModel):
    prefix: str

    model_config = {"json_schema_extra": {
        "examples": [{"prefix": "/data/old_photos/"}]
    }}


class DeleteResult(BaseModel):
    deleted: int


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"], summary="Liveness probe")
def health():
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}


@app.get("/stats", tags=["system"], summary="Database statistics")
def stats():
    """Return row count and table name for the active LanceDB database."""
    return _get_engine().table_stats()


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/search/text",
    response_model=List[SearchResult],
    tags=["search"],
    summary="Search by text description",
)
def search_by_text(req: TextSearchRequest):
    """
    Encode a natural-language query with CLIP and return the top-K most
    similar images from the database.

    - **query**: any text description, e.g. "a sunset over the ocean"
    - **top_k**: number of results to return (default 5)
    - **where**: optional SQL WHERE clause to pre-filter candidates by
      metadata (e.g. `"width > 1920 AND file_size < 5000000"`).
      Only effective when the DB was ingested with metadata columns.
    """
    results = _get_engine().search(req.query, top_k=req.top_k, where=req.where)
    return [{"path": p, "score": s} for p, s in results]


@app.post(
    "/search/image",
    response_model=List[SearchResult],
    tags=["search"],
    summary="Search by uploading an image",
)
async def search_by_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, …)"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results"),
    where: Optional[str] = Query(
        None, description="SQL WHERE filter on metadata columns"
    ),
):
    """
    Upload an image and return the top-K most visually similar images from
    the database. The uploaded file is processed in memory and never stored
    on disk beyond the duration of the request.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = _get_engine().search(tmp_path, top_k=top_k, where=where)
        return [{"path": p, "score": s} for p, s in results]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Management endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResult,
    tags=["management"],
    summary="Index images from a server-side directory",
)
def ingest(req: IngestRequest):
    """
    Walk `data_dir` on the server, embed each image with CLIP, and store
    the vectors in LanceDB. Skips already-indexed images when
    `incremental=true` (default).
    """
    if not os.path.isdir(req.data_dir):
        raise HTTPException(
            status_code=400, detail=f"Directory not found: {req.data_dir}"
        )
    return _get_engine().index(
        data_dir=req.data_dir,
        batch_size=req.batch_size,
        incremental=req.incremental,
    )


@app.delete(
    "/images/by-paths",
    response_model=DeleteResult,
    tags=["management"],
    summary="Remove images by exact file paths",
)
def delete_by_paths(req: DeleteByPathsRequest):
    """Remove specific images from the vector store by their absolute file paths."""
    return _get_engine().delete(req.paths)


@app.delete(
    "/images/by-prefix",
    response_model=DeleteResult,
    tags=["management"],
    summary="Remove all images under a directory prefix",
)
def delete_by_prefix(req: DeleteByPrefixRequest):
    """
    Remove every indexed image whose path starts with `prefix`.
    Useful for bulk-removing an entire directory without rebuilding the DB.
    """
    return _get_engine().delete_by_prefix(req.prefix)


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/duplicates",
    tags=["analysis"],
    summary="Find near-duplicate image pairs",
)
def find_duplicates(
    threshold: float = Query(
        0.05, ge=0.0, le=1.0,
        description="Max cosine distance to consider a duplicate (0 = identical)",
    ),
):
    """
    Scan the entire database for near-duplicate pairs using brute-force
    cosine similarity. Returns pairs sorted by distance (closest first).

    **Warning:** O(N²) — may be slow for very large databases.
    """
    return _get_engine().find_duplicates(threshold=threshold)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install 'uvicorn[standard]'")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="Image Similarity Search — REST API server"
    )
    parser.add_argument(
        "--db-path", default="./lancedb",
        help="Path to the LanceDB database directory (default: ./lancedb)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose on the network)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-reload on code changes (development only)",
    )
    args = parser.parse_args()

    print(f"Loading engine from '{args.db_path}'...")
    _init_engine(args.db_path)
    print(f"Engine ready.")
    print(f"Starting API server  →  http://{args.host}:{args.port}")
    print(f"Interactive docs     →  http://{args.host}:{args.port}/docs")

    if args.reload:
        # reload mode requires passing the import string, not the app object
        uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
