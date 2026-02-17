import logging
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from similarity_engine import SimilarityEngine, DEFAULT_TABLE_NAME

# Page configuration
st.set_page_config(
    page_title="Image Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for the engine
if "engine" not in st.session_state:
    # Lazy load the engine only when needed
    pass

@st.cache_resource
def get_engine(db_path: str):
    """Cached engine initialization to avoid reloading model on every rerun."""
    return SimilarityEngine(db_path=db_path)

def main():
    st.title("🔍 Local Image Semantic Search")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # DB Path Selection
    default_db = "./lancedb"
    db_path = st.sidebar.text_input("Database Path", value=default_db)
    
    # Load Engine
    try:
        engine = get_engine(db_path)
        st.sidebar.success(f"Connected to {db_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to load DB: {e}")
        st.stop()

    # Table Selection (if multiple tables supported in future)
    table_name = DEFAULT_TABLE_NAME

    # Show Stats
    if st.sidebar.checkbox("Show DB Stats"):
        try:
            stats = engine.table_stats(table_name)
            st.sidebar.json(stats)
        except Exception:
            st.sidebar.warning("Table not found or empty.")

    st.sidebar.divider()
    
    # --- Search Mode ---
    search_mode = st.sidebar.radio("Search Mode", ["Text Query", "Image Query"])
    
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=50, value=12)

    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Privacy-focused, local image similarity search using CLIP + LanceDB.\n"
        "No data leaves your machine."
    )

    # --- Main Content ---
    
    query = None
    
    if search_mode == "Text Query":
        text_input = st.text_input("Describe what you're looking for:", placeholder="e.g., 'a red sports car in the rain'")
        if text_input:
            query = text_input
            
    elif search_mode == "Image Query":
        uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", width=250)
            
            # Convert to RGB and numpy for the engine (if needed, but engine handles PIL/paths)
            # The engine expects a path or a PIL image (actually engine._encode_query handles raw image?)
            # Let's check engine.search signature. It takes `Union[str, Path, np.ndarray, Image.Image]`.
            # Wait, engine.search signature in previous turns showed `Union[str, Path, np.ndarray]`.
            # I should verify if I can pass a PIL image directly or if I need to save it / convert it.
            # Looking at previous logs, `_encode_query` handles `str` (text/path) or `np.ndarray`.
            # If I pass a PIL image, I might need to convert it to a temp path or update engine.
            # Let's stick to what we know works: save to temp or convert to ndarray?
            # Actually, `engine.search` calls `_encode_query`.
            # Let's perform a quick fix: save uploaded file to temp for robustness, or assume engine handles it.
            # To be safe and avoid modifying engine right now, I'll save to a temp file.
            
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file, format="JPEG")
                query = tmp_file.name
    
    # --- Perform Search ---
    if query:
        st.divider()
        st.markdown(f"### Results for: *{query if isinstance(query, str) and len(query) < 50 else 'Image Query'}*")
        
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

if __name__ == "__main__":
    main()
