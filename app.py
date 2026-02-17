import logging
import os
import time
import shutil
import tempfile
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Import project modules
from similarity_engine import SimilarityEngine, DEFAULT_TABLE_NAME
from datasets import download_dataset, AVAILABLE_DATASETS, list_datasets
from benchmarks.bench_datasets import benchmark_dataset

# Page configuration
st.set_page_config(
    page_title="Image Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for the engine
if "engine" not in st.session_state:
    st.session_state["engine"] = None

@st.cache_resource
def get_engine(db_path: str):
    """Cached engine initialization to avoid reloading model on every rerun."""
    return SimilarityEngine(db_path=db_path)

def render_search_page(engine, top_k):
    """Render the Search page."""
    # --- Search Mode ---
    search_mode = st.radio("Search Mode", ["Text Query", "Image Query"], horizontal=True)
    st.divider()

    query = None
    
    if search_mode == "Text Query":
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_input("Describe what you're looking for:", placeholder="e.g., 'a red sports car in the rain'")
        with col2:
            st.write("") # spacer
            st.write("")
            search_btn = st.button("Search", type="primary", use_container_width=True)
            
        if text_input and (search_btn or text_input):
            query = text_input
            
    elif search_mode == "Image Query":
        uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", width=250)
            
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file, format="JPEG")
                query = tmp_file.name
    
    # --- Perform Search ---
    if query:
        st.markdown(f"### Results")
        
        with st.spinner("Searching..."):
            try:
                start_time = time.perf_counter()
                results = engine.search(query, top_k=top_k)
                elapsed = (time.perf_counter() - start_time) * 1000
                
                st.caption(f"Found {len(results)} results in {elapsed:.2f} ms")
                
                if not results:
                    st.warning("No results found.")
                else:
                    # Display results in a grid
                    cols = st.columns(4)
                    for idx, (path, score) in enumerate(results):
                        col = cols[idx % 4]
                        
                        # Show image
                        with col:
                            # Verify path exists
                            if os.path.exists(path):
                                img = Image.open(path)
                                st.image(img, use_column_width=True)
                                st.caption(f"**{score:.4f}**\n`{Path(path).name}`")
                            else:
                                st.error(f"File not found: {path}")
                                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                
    # Cleanup temp file if it was an image query
    if search_mode == "Image Query" and query and isinstance(query, str) and os.path.exists(query) and "tmp" in query:
        try:
            os.remove(query)
        except:
            pass

def render_benchmarks_page():
    """Render the Benchmarks page."""
    st.header("📊 Dataset Benchmarks")
    st.markdown("""
    Run benchmarks on standard datasets to evaluate ingestion throughput and search latency.
    This will download the dataset, ingest it using CLIP, and run sample queries.
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Dataset Selection
        dataset_options = list(AVAILABLE_DATASETS.keys())
        selected_datasets = st.multiselect(
            "Select Datasets", 
            dataset_options, 
            default=["cifar10"]
        )
        
        # Benchmark Parameters
        batch_size = st.number_input("Batch Size", value=256, step=64)
        workers = st.number_input("I/O Threads", value=8, min_value=1, max_value=32)
        
        run_btn = st.button("Run Benchmark", type="primary", use_container_width=True)

    with col2:
        if run_btn and selected_datasets:
            results = []
            
            # Create progress indicators
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Create temp directories
            data_dir = tempfile.mkdtemp(prefix="bench_data_gui_")
            db_dir = tempfile.mkdtemp(prefix="bench_db_gui_")
            
            try:
                total_steps = len(selected_datasets)
                
                for i, dataset_name in enumerate(selected_datasets):
                    status_text.markdown(f"**Running benchmark: `{dataset_name}`...**")
                    progress_bar.progress((i) / total_steps)
                    
                    # Capture stdout to show logs? Streamlit doesn't easily capture stdout live.
                    # We'll rely on the final result for now, maybe show a spinner.
                    
                    with st.spinner(f"Processing {dataset_name} (Download → Ingest → Search)..."):
                        # Run the benchmark function from bench_datasets.py
                        # We need to import it. Added import at top.
                        
                        # Note: benchmark_dataset prints to stdout. 
                        # We could redirect stdout to a StringIO if we want to show logs.
                        
                        try:
                            res = benchmark_dataset(
                                dataset_name,
                                data_dir=data_dir,
                                db_dir=db_dir,
                                batch_size=batch_size,
                                num_io_threads=workers,
                                num_queries=50,
                                top_k=10,
                            )
                            results.append(res)
                            st.success(f"✅ {dataset_name} completed!")
                            
                            # Show immediate stats for this dataset
                            with st.expander(f"Details: {dataset_name}", expanded=True):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Images", f"{res['num_images']:,}")
                                c2.metric("Ingestion Rate", f"{res['ingest_throughput_ips']:.1f} img/s")
                                c3.metric("Search Latency (P50)", f"{res['query_p50_ms']:.2f} ms")
                                
                                st.markdown("**Sample Search:**")
                                st.write(f"Query: *{res['sample_query']}*")
                                st.json(res['sample_results'])

                        except Exception as e:
                            st.error(f"Benchmark failed for {dataset_name}: {e}")
                            logging.exception("Benchmark failed")

                progress_bar.progress(1.0)
                status_text.markdown("**All benchmarks completed!**")
                
                # --- Aggregate Results ---
                if results:
                    st.divider()
                    st.subheader("Results Comparison")
                    
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    
                    # Select columns for display
                    display_cols = [
                        "dataset", "num_images", "ingest_throughput_ips", 
                        "query_p50_ms", "query_p95_ms", "db_size_mb"
                    ]
                    
                    # Rename for nicer display
                    column_config = {
                        "dataset": "Dataset",
                        "num_images": "Images",
                        "ingest_throughput_ips": "Throughput (img/s)",
                        "query_p50_ms": "Latency P50 (ms)",
                        "query_p95_ms": "Latency P95 (ms)",
                        "db_size_mb": "DB Size (MB)"
                    }
                    
                    st.dataframe(
                        df[display_cols], 
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Charts
                    st.subheader("Performance Visualization")
                    tab1, tab2 = st.tabs(["Throughput", "Latency"])
                    
                    with tab1:
                        st.bar_chart(df.set_index("dataset")["ingest_throughput_ips"])
                        st.caption("Ingestion Throughput (higher is better)")
                        
                    with tab2:
                        st.bar_chart(df.set_index("dataset")[["query_p50_ms", "query_p95_ms"]])
                        st.caption("Search Latency (lower is better)")

            finally:
                # Cleanup
                shutil.rmtree(db_dir, ignore_errors=True)
                shutil.rmtree(data_dir, ignore_errors=True)
                
        elif run_btn and not selected_datasets:
            st.warning("Please select at least one dataset.")


def main():
    st.title("🔍 Local Image Semantic Search")

    # --- Sidebar Configuration ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "Benchmarks"])
    
    st.sidebar.divider()
    st.sidebar.header("Configuration")
    
    # DB Path Selection
    default_db = "./lancedb"
    db_path = st.sidebar.text_input("Database Path", value=default_db)
    
    # Load Engine
    try:
        engine = get_engine(db_path)
        if page == "Search":
            st.sidebar.success(f"Connected to {db_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to load DB: {e}")
        st.stop()

    # Table Selection (if multiple tables supported in future)
    table_name = DEFAULT_TABLE_NAME

    # Show Stats
    if page == "Search" and st.sidebar.checkbox("Show DB Stats"):
        try:
            stats = engine.table_stats(table_name)
            st.sidebar.json(stats)
        except Exception:
            st.sidebar.warning("Table not found or empty.")

    st.sidebar.divider()
    
    # --- Page Routing ---
    if page == "Search":
        top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=50, value=12)
        render_search_page(engine, top_k)
    elif page == "Benchmarks":
        render_benchmarks_page()

    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Privacy-focused, local image similarity search using CLIP + LanceDB.\n"
        "No data leaves your machine."
    )

if __name__ == "__main__":
    main()
