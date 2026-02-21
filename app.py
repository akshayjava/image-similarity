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
import db_config

# Page configuration
st.set_page_config(
    page_title="Image Semantic Search",
    page_icon="\U0001f50d",
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

def render_manage_page():
    """Render the Database Management page."""
    st.header("🗄️ Database Management")
    
    config = db_config.load_config()
    databases = config.get("databases", {})
    
    # --- Check DBs ---
    st.subheader("Registered Databases")
    if not databases:
        st.warning("No databases configured.")
    else:
        # Create a dataframe for display
        db_list = []
        for name, path in databases.items():
            db_list.append({"Name": name, "Path": path})
            
        st.dataframe(pd.DataFrame(db_list), use_container_width=True, hide_index=True)
        
        # Removal UI
        st.caption("Remove a database from configuration:")
        col1, col2 = st.columns([3, 1])
        with col1:
            to_remove = st.selectbox("Select to remove", list(databases.keys()), key="remove_select")
        with col2:
            if st.button("Remove", type="primary"):
                if to_remove == "Default":
                    st.error("Cannot remove Default database.")
                else:
                    db_config.remove_database(to_remove)
                    st.success(f"Removed '{to_remove}'")
                    st.rerun()

    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Add Existing", "Create New (Ingest)", "Create from Dataset"])
    
    with tab1:
        st.subheader("Add Existing Database")
        ae_name = st.text_input("Name", placeholder="e.g., MyVacationPhotos")
        ae_path = st.text_input("Path to LanceDB", placeholder="/path/to/lancedb")
        
        if st.button("Add Existing"):
            if ae_name and ae_path:
                if os.path.exists(ae_path):
                    db_config.add_database(ae_name, ae_path)
                    st.success(f"Added '{ae_name}'")
                    st.rerun()
                else:
                    st.error("Path does not exist.")
            else:
                st.error("Name and Path are required.")
                
    with tab2:
        st.subheader("Create New Database (Ingest)")
        cn_name = st.text_input("New DB Name", placeholder="e.g., MyPhotos_DB")
        cn_source = st.text_input("Source Image Directory", placeholder="/path/to/images")
        cn_dest = st.text_input("Destination DB Path (Optional)", placeholder="Leave empty to use ./dbs/<name>")
        
        if st.button("Ingest & Create"):
            if cn_name and cn_source:
                if not os.path.exists(cn_source):
                    st.error("Source directory not found.")
                else:
                    target_db_path = cn_dest if cn_dest else f"./dbs/{cn_name}"
                    os.makedirs(target_db_path, exist_ok=True)
                    
                    import sys, io
                    console_box = st.empty()
                    
                    class _IngestCapture(io.TextIOBase):
                        def __init__(self, el, max_lines=30):
                            self._lines, self._cur, self._el, self._max = [], "", el, max_lines
                        def write(self, t):
                            if not t: return 0
                            for j, part in enumerate(t.split("\r")):
                                if j > 0: self._cur = ""
                                for k, sub in enumerate(part.split("\n")):
                                    if k > 0: self._lines.append(self._cur); self._cur = ""
                                    self._cur += sub
                            d = self._lines[-self._max:] if len(self._lines) > self._max else self._lines
                            out = "\n".join(d) + ("\n" + self._cur if self._cur else "")
                            self._el.code(out, language=None)
                            return len(t)
                        def flush(self): pass
                    
                    cap = _IngestCapture(console_box)
                    old_out, old_err = sys.stdout, sys.stderr
                    try:
                        sys.stdout = cap
                        sys.stderr = cap
                        ingest_engine = SimilarityEngine(db_path=target_db_path)
                        stats = ingest_engine.index(cn_source)
                    finally:
                        sys.stdout = old_out
                        sys.stderr = old_err
                        
                    st.success(f"✅ Ingestion complete! {stats['total_indexed']} images indexed.")
                    db_config.add_database(cn_name, target_db_path)
                    st.success(f"Database '{cn_name}' registered. Switch to it in the sidebar.")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("Name and Source Directory are required.")
    
    with tab3:
        st.subheader("Create from Benchmark Dataset")
        st.caption("Download a standard dataset, ingest it, and create a searchable database.")
        
        ds_name = st.selectbox("Dataset", list(AVAILABLE_DATASETS.keys()), key="manage_ds_select")
        ds_info = AVAILABLE_DATASETS.get(ds_name, {})
        st.info(f"{ds_info.get('description', ds_name)} — Size: {ds_info.get('size', '?')}")
        
        ds_db_name = st.text_input("Database Name", value=f"{ds_name}_db", key="manage_ds_db_name")
        
        if st.button("📥 Download & Create DB", type="primary", key="manage_ds_create"):
            if not ds_db_name:
                st.error("Database name is required.")
            else:
                import sys, io
                target_db_path = f"./dbs/{ds_db_name}"
                data_dir = os.path.join("./data", ds_db_name)
                os.makedirs(data_dir, exist_ok=True)
                
                console_box = st.empty()
                progress_bar = st.progress(0)
                
                class _DSCapture(io.TextIOBase):
                    def __init__(self, el, max_lines=30):
                        self._lines, self._cur, self._el, self._max = [], "", el, max_lines
                    def write(self, t):
                        if not t: return 0
                        for j, part in enumerate(t.split("\r")):
                            if j > 0: self._cur = ""
                            for k, sub in enumerate(part.split("\n")):
                                if k > 0: self._lines.append(self._cur); self._cur = ""
                                self._cur += sub
                        d = self._lines[-self._max:] if len(self._lines) > self._max else self._lines
                        out = "\n".join(d) + ("\n" + self._cur if self._cur else "")
                        self._el.code(out, language=None)
                        return len(t)
                    def flush(self): pass
                
                cap = _DSCapture(console_box)
                old_out, old_err = sys.stdout, sys.stderr
                
                try:
                    sys.stdout = cap
                    sys.stderr = cap
                    
                    # Step 1: Download
                    print(f"[1/2] Downloading {ds_name}...")
                    def dl_cb(msg, p):
                        progress_bar.progress(min(p * 0.5, 0.49))
                    
                    export_path = download_dataset(ds_name, dest_dir=data_dir, progress_callback=dl_cb)
                    num_imgs = sum(1 for _ in Path(export_path).rglob("*.jpg"))
                    print(f"       Downloaded {num_imgs:,} images")
                    
                    # Step 2: Ingest
                    print(f"[2/2] Ingesting into '{ds_db_name}'...")
                    progress_bar.progress(0.5)
                    os.makedirs(target_db_path, exist_ok=True)
                    ingest_engine = SimilarityEngine(db_path=target_db_path)
                    stats = ingest_engine.index(export_path)
                    print(f"       Indexed {stats['total_indexed']:,} images")
                    progress_bar.progress(1.0)
                    
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                
                db_config.add_database(ds_db_name, target_db_path)
                st.success(f"✅ Database '{ds_db_name}' created with {stats['total_indexed']:,} images!")
                st.info("Switch to this database in the sidebar to search it.")
                time.sleep(1)
                st.rerun()


def _show_db_preview(engine, label="Database Preview", n=8):
    """Show a compact row of sample images from the current database."""
    try:
        table = engine.db.open_table(DEFAULT_TABLE_NAME)
        count = table.count_rows()
        if count == 0:
            return
        import random
        sample_n = min(n, count)
        # Read only a small head of the table instead of loading all rows —
        # avoids an expensive full scan on large databases.
        head_n = min(sample_n * 8, count)
        raw = table.head(head_n)
        df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        sample = df.sample(n=sample_n, random_state=random.randint(0, 9999))
        paths = sample["id"].tolist()
        valid = [p for p in paths if os.path.exists(p)]
        if not valid:
            return
        with st.expander(f"🖼️ {label} ({count:,} images)", expanded=False):
            cols = st.columns(min(len(valid), 8))
            for j, path in enumerate(valid[:8]):
                with cols[j]:
                    st.image(Image.open(path), use_container_width=True)
                    st.caption(f"`{Path(path).name}`")
    except Exception:
        pass


def render_tools_page(engine):
    """Render the Tools page with Duplicate Detection, Clustering, and Explorer."""
    st.header("🛠️ Analysis Tools")

    tab_dup, tab_clust, tab_explore = st.tabs(
        ["🔍 Duplicate Detection", "📊 Clustering", "🗺️ Visual Explorer"]
    )

    # ---- Duplicate Detection ----
    with tab_dup:
        st.subheader("Find Near-Duplicate Images")
        st.caption("Scan your database for image pairs with very similar CLIP embeddings.")
        _show_db_preview(engine, "Sample Images from Database")

        col1, col2 = st.columns([1, 3])
        with col1:
            threshold = st.slider(
                "Distance Threshold", min_value=0.01, max_value=0.30,
                value=0.05, step=0.01,
                help="Lower = stricter matching. 0.05 is a good default."
            )
            scan_btn = st.button("🔎 Scan for Duplicates", type="primary", use_container_width=True)

        with col2:
            if scan_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def dup_cb(msg, p):
                    progress_bar.progress(min(p, 0.99))
                    status_text.text(msg)

                try:
                    dups = engine.find_duplicates(
                        threshold=threshold, progress_callback=dup_cb
                    )
                    progress_bar.progress(1.0)
                    status_text.text(f"Done! Found {len(dups)} duplicate pair(s).")

                    if not dups:
                        st.success("No duplicates found! Your dataset is clean.")
                    else:
                        st.warning(f"Found **{len(dups)}** potential duplicate pairs.")
                        for i, d in enumerate(dups[:20]):
                            with st.expander(
                                f"Pair {i+1} — distance: {d['distance']:.6f}",
                                expanded=(i < 3)
                            ):
                                c1, c2 = st.columns(2)
                                for ci, path in enumerate([d["pair"][0], d["pair"][1]]):
                                    with [c1, c2][ci]:
                                        if os.path.exists(path):
                                            st.image(Image.open(path), use_container_width=True)
                                        st.caption(f"`{Path(path).name}`")
                        if len(dups) > 20:
                            st.info(f"Showing first 20 of {len(dups)} pairs.")
                except Exception as e:
                    st.error(f"Scan failed: {e}")

    # ---- Clustering ----
    with tab_clust:
        st.subheader("Image Clustering (K-Means)")
        st.caption("Group similar images into clusters based on their CLIP embeddings.")
        _show_db_preview(engine, "Sample Images to Cluster")

        col1, col2 = st.columns([1, 3])
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=10)
            cluster_btn = st.button("📊 Run Clustering", type="primary", use_container_width=True)

        with col2:
            if cluster_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def clust_cb(msg, p):
                    progress_bar.progress(min(p, 0.99))
                    status_text.text(msg)

                try:
                    result = engine.cluster_images(
                        n_clusters=n_clusters, progress_callback=clust_cb
                    )
                    progress_bar.progress(1.0)
                    stats = result["stats"]
                    status_text.text(
                        f"Done! {stats['n_images']} images → {stats['n_clusters']} clusters"
                    )

                    # Show cluster stats
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Images", f"{stats['n_images']:,}")
                    c2.metric("Clusters", stats['n_clusters'])
                    c3.metric("Inertia", f"{stats['inertia']:.0f}")

                    # Show each cluster
                    for cid in sorted(result["clusters"].keys()):
                        paths = result["clusters"][cid]
                        with st.expander(
                            f"Cluster {cid} ({len(paths)} images)",
                            expanded=(cid < 3)
                        ):
                            cols = st.columns(min(6, len(paths)))
                            for j, path in enumerate(paths[:6]):
                                with cols[j]:
                                    if os.path.exists(path):
                                        st.image(Image.open(path), use_container_width=True)
                                    st.caption(f"`{Path(path).name}`")
                            if len(paths) > 6:
                                st.caption(f"... and {len(paths) - 6} more")
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

    # ---- Visual Explorer ----
    with tab_explore:
        st.subheader("Embedding Space Explorer")
        st.caption("Visualize your image database as a 2D scatter plot using t-SNE or UMAP.")
        _show_db_preview(engine, "Sample Images to Visualize")

        col1, col2 = st.columns([1, 3])
        with col1:
            method = st.selectbox("Reduction Method", ["tsne", "umap"])
            explore_btn = st.button("🗺️ Generate Map", type="primary", use_container_width=True)

        with col2:
            if explore_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def explore_cb(msg, p):
                    progress_bar.progress(min(p, 0.99))
                    status_text.text(msg)

                try:
                    points = engine.reduce_dimensions(
                        method=method, progress_callback=explore_cb
                    )
                    progress_bar.progress(1.0)
                    status_text.text(f"Done! {len(points)} points plotted.")

                    import plotly.express as px

                    df = pd.DataFrame(points)
                    df["label"] = df["id"].apply(lambda x: Path(x).name)

                    fig = px.scatter(
                        df, x="x", y="y",
                        hover_name="label",
                        hover_data={"id": True, "x": ":.2f", "y": ":.2f"},
                        title=f"{method.upper()} Projection of Image Embeddings",
                        width=900, height=700,
                    )
                    fig.update_traces(marker=dict(size=5, opacity=0.7))
                    fig.update_layout(
                        xaxis_title="", yaxis_title="",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Explorer failed: {e}")


def main():
    st.title("🔍 Local Image Semantic Search")

    # --- Sidebar Configuration ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "Tools", "Benchmarks", "Manage Databases"])
    
    st.sidebar.divider()
    st.sidebar.header("Configuration")
    
    # Load config
    config = db_config.load_config()
    databases = config.get("databases", {"Default": "./lancedb"})
    
    # DB Selection
    db_names = list(databases.keys())
    # Try to preserve selection or default
    index = 0
    if "active_db" in st.session_state:
        if st.session_state["active_db"] in db_names:
            index = db_names.index(st.session_state["active_db"])
            
    selected_db_name = st.sidebar.selectbox("Active Database", db_names, index=index)
    st.session_state["active_db"] = selected_db_name
    
    db_path = databases[selected_db_name]
    
    # Load Engine
    try:
        engine = get_engine(db_path)
        if page == "Search":
            st.sidebar.success(f"Connected to {selected_db_name}")
            st.sidebar.caption(f"`{db_path}`")
    except Exception as e:
        st.sidebar.error(f"Failed to load DB: {e}")
        # Don't stop here, user might want to go to Manage page to fix it
        if page == "Search":
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
    elif page == "Tools":
        render_tools_page(engine)
    elif page == "Benchmarks":
        render_benchmarks_page()
    elif page == "Manage Databases":
        render_manage_page()

    if page not in ("Manage Databases", "Tools"):
        st.sidebar.divider()
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "Privacy-focused, local image similarity search using CLIP + LanceDB.\n"
            "No data leaves your machine."
        )



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
            
        if search_btn and text_input:
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
                                st.image(img, use_container_width=True)
                                st.caption(f"**{score:.4f}**\n`{Path(path).name}`")
                            else:
                                st.error(f"File not found: {path}")
                                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                
    # Cleanup temp file if it was an image query
    if search_mode == "Image Query" and query and isinstance(query, str) and os.path.exists(query) and "tmp" in query:
        try:
            os.remove(query)
        except OSError:
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
        
        st.divider()
        save_db = st.checkbox(
            "Save as searchable database",
            value=False,
            help="Keep the benchmark DB so you can search it from the Search page."
        )
        
        run_btn = st.button("Run Benchmark", type="primary", use_container_width=True)

    with col2:
        if run_btn and selected_datasets:
            results = []
            
            # ── Live progress dashboard ──
            status_text = st.empty()
            progress_bar = st.progress(0)
            console_box = st.empty()     # live CLI output
            
            # Create directories — persistent if save_db is checked
            if save_db:
                data_dir = os.path.join("./data", "benchmarks")
                db_dir = os.path.join("./dbs", "benchmarks")
                os.makedirs(data_dir, exist_ok=True)
                os.makedirs(db_dir, exist_ok=True)
            else:
                data_dir = tempfile.mkdtemp(prefix="bench_data_gui_")
                db_dir = tempfile.mkdtemp(prefix="bench_db_gui_")
            
            try:
                import time as _time
                import io
                import sys
                
                # --- Stream capture for live CLI output ---
                class StreamCapture(io.TextIOBase):
                    """Captures writes (print + tqdm) and updates a Streamlit code block."""
                    def __init__(self, st_element, max_lines=40):
                        self._lines = []          # finished lines
                        self._current = ""        # line being built (tqdm uses \r)
                        self._el = st_element
                        self._max = max_lines
                    
                    def write(self, text):
                        if not text:
                            return 0
                        # tqdm writes \r to overwrite the current line
                        parts = text.split("\r")
                        for j, part in enumerate(parts):
                            if j > 0:
                                # \r  → overwrite current line
                                self._current = ""
                            sub = part.split("\n")
                            self._current += sub[0]
                            for k in range(1, len(sub)):
                                self._lines.append(self._current)
                                self._current = sub[k]
                        self._refresh()
                        return len(text)
                    
                    def flush(self):
                        pass
                    
                    def _refresh(self):
                        display = self._lines[-self._max:] if len(self._lines) > self._max else self._lines
                        out = "\n".join(display)
                        if self._current:
                            out += "\n" + self._current
                        self._el.code(out, language=None)
                    
                    def get_text(self):
                        out = "\n".join(self._lines)
                        if self._current:
                            out += "\n" + self._current
                        return out
                
                total_steps = len(selected_datasets)
                
                for i, dataset_name in enumerate(selected_datasets):
                    ds_info = AVAILABLE_DATASETS.get(dataset_name, {})
                    expected_imgs = ds_info.get("images", "?")
                    ds_size = ds_info.get("size", "?")
                    
                    status_text.markdown(
                        f"### 🔄 `{dataset_name}` ({i+1}/{total_steps})\n"
                        f"**{ds_info.get('description', dataset_name)}** &nbsp;|&nbsp; "
                        f"Size: {ds_size} &nbsp;|&nbsp; ~{expected_imgs:,} images"
                    )
                    progress_bar.progress(i / total_steps)
                    
                    bench_start = _time.perf_counter()
                    
                    # Set up stream capture
                    capture = StreamCapture(console_box)
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    try:
                        # Redirect stdout + stderr so print() and tqdm output
                        # both appear in the live console
                        sys.stdout = capture
                        sys.stderr = capture
                        
                        def progress_cb(msg, p, _i=i):
                            global_p = min((_i + p) / total_steps, 0.99)
                            progress_bar.progress(global_p)

                        res = benchmark_dataset(
                            dataset_name,
                            data_dir=data_dir,
                            db_dir=db_dir,
                            batch_size=batch_size,
                            num_io_threads=workers,
                            num_queries=50,
                            top_k=10,
                            progress_callback=progress_cb,
                        )
                        results.append(res)
                        
                    except Exception as e:
                        st.error(f"❌ Benchmark failed for {dataset_name}: {e}")
                        logging.exception("Benchmark failed")
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    bench_elapsed = _time.perf_counter() - bench_start
                    
                    # Show the final captured output in an expander
                    final_output = capture.get_text()
                    
                    st.success(f"✅ **{dataset_name}** completed in {bench_elapsed:.1f}s")
                    
                    if res:
                        # Register as searchable DB if requested
                        if save_db:
                            bench_db_name = f"{dataset_name}_bench"
                            bench_db_path = os.path.join(db_dir, dataset_name)
                            db_config.add_database(bench_db_name, bench_db_path)
                            st.info(f"💾 Saved as database **{bench_db_name}** — switch to it in the sidebar to search.")
                        
                        with st.expander(f"📋 Details: {dataset_name}", expanded=True):
                            # Show the CLI output
                            st.code(final_output, language=None)
                            
                            st.divider()
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Images", f"{res['num_images']:,}")
                            c2.metric("Throughput", f"{res['ingest_throughput_ips']:.1f} img/s")
                            c3.metric("P50 Latency", f"{res['query_p50_ms']:.2f} ms")
                            c4.metric("DB Size", f"{res['db_size_mb']:.1f} MB")

                progress_bar.progress(1.0)
                status_text.markdown("### ✅ All benchmarks completed!")
                console_box.empty()
                
                # ── Aggregate Results ──
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
                # Cleanup — keep both dirs if saving for searchable DB
                if not save_db:
                    shutil.rmtree(db_dir, ignore_errors=True)
                    shutil.rmtree(data_dir, ignore_errors=True)
                
        elif run_btn and not selected_datasets:
            st.warning("Please select at least one dataset.")




if __name__ == "__main__":
    main()
