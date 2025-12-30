import streamlit as st
import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Paths configuration
MODEL_DIR = REPO_ROOT / "models"
REID_DIR = MODEL_DIR / "reid"
REID_OPTS = REID_DIR / "opts.yaml"
REID_CKPT = REID_DIR / "net.pth"
DET_WEIGHTS = MODEL_DIR / "best.pt"
BATCH_SIZE = 1 # for streamlit, keep it simple

# Import ReID utilities
# We need to hack sys.path for internal imports in search.py if necessary, but it seems it uses relative imports from src...
# However, search.py imports `src.pipeline...` so we are good if REPO_ROOT is in path.
try:
    from src.reid.search import (
        _load_reid_model, 
        _load_gallery_embeddings, 
        _collect_candidates, 
        _score_candidates, 
        _filter_matches,
        GalleryEntry,
        Candidate
    )
    from src.reid.visualize import draw_reid_debug
except ImportError as e:
    st.error(f"Failed to import ReID modules: {e}")

def render():
    st.header("Vehicle Re-Identification")
    
    st.info("Identify a vehicle in a query image by matching it against a gallery of known vehicles (by Plate ID).")
    
    # User Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Settings")
        plate_id = st.text_input("Target Plate ID (e.g. ABC 123)", "ABC-1234")
        
        # Gallery Directory Selection
        # Allows user to type a path, default to 'data/gallery'
        default_gallery = str(REPO_ROOT / "data" / "gallery")
        gallery_path_str = st.text_input("Gallery Directory Path", default_gallery)
        gallery_dir = Path(gallery_path_str)

        # Thresholds
        min_score = st.slider("Min ReID Score", 0.0, 1.0, 0.4)
        
    with col2:
        st.subheader("2. Upload Query Image")
        uploaded_file = st.file_uploader("Choose a query image...", type=["jpg", "png", "jpeg"], key="reid_uploader")

    if uploaded_file is not None and st.button("Run ReID Search"):
        if not gallery_dir.exists():
            st.error(f"Gallery directory not found: {gallery_dir}")
            return
            
        # Verify models
        if not REID_OPTS.exists() or not REID_CKPT.exists():
            st.error(f"ReID model not found at {REID_DIR}")
            return
        if not DET_WEIGHTS.exists():
            st.error(f"Detection weights not found at {DET_WEIGHTS}")
            return

        with st.spinner("Initializing models and processing..."):
            # creating a temp query image
            temp_dir = Path("temp/reid_query")
            temp_dir.mkdir(parents=True, exist_ok=True)
            query_path = temp_dir / uploaded_file.name
            
            image = Image.open(uploaded_file)
            image.save(query_path)
            
            # 1. Load ReID Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # Input size 224 matches third_party/vehicle_reid/extract_features.py
                input_size = 224 
                model = _load_reid_model(REID_OPTS, REID_CKPT, device)
            except Exception as e:
                st.error(f"Error loading ReID model: {e}")
                return

            # 2. Load/Compute Gallery Embeddings for the Target Plate ID
            st.write(f"Loading gallery for Plate ID: `{plate_id}`...")
            gallery_entries = _load_gallery_embeddings(
                gallery_dir,
                plate_id,
                model,
                input_size,
                device,
                BATCH_SIZE
            )
            
            if not gallery_entries:
                st.warning(f"No gallery images found for Plate ID `{plate_id}` in {gallery_dir}")
                # We can stop here or proceed to just detect cars
                return
            
            st.success(f"Loaded {len(gallery_entries)} gallery images.")

            # 3. Detect & Extract Candidates from Query Image
            st.write("Detecting cars in query image...")
            try:
                candidates = _collect_candidates(
                    DET_WEIGHTS,
                    [query_path],
                    conf=0.25, # Default conf
                    iou=0.45,
                    device=None, # Auto
                    pad=0.0,
                    car_label="car" # Assume class name check
                )
            except Exception as e:
                st.error(f"Detection failed: {e}")
                return
                
            if not candidates:
                st.warning("No cars detected in query image.")
                return

            # 4. Score Candidates
            matches = _score_candidates(
                gallery_entries,
                candidates,
                model,
                input_size,
                device,
                BATCH_SIZE
            )
            
            # Filter matches
            matches = _filter_matches(matches, min_score=min_score, top_k=5)
            
            # 5. Visualisation
            img_bgr = cv2.imread(str(query_path))
            
            # Convert matches to dict format expected by verify_visualize
            # Actually reuse the visualize function
            match_dicts = []
            for m in matches:
                 match_dicts.append({
                     "bbox_xyxy": m.bbox_xyxy,
                     "score": m.score,
                     "kept": m.kept
                 })
            
            annotated = draw_reid_debug(img_bgr, match_dicts, plate_id)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, caption="ReID Results", use_column_width=True)
            
            st.divider()
            st.subheader("Detailed Matches")
            for m in matches:
                if m.kept:
                    with st.expander(f"Match: Score {m.score:.4f}"):
                        st.write(f"Matched with gallery image: `{os.path.basename(m.best_gallery_path)}`")
                        # Show crop
                        st.image(m.best_gallery_path, caption="Matched Gallery Image", width=200)

