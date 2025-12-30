import streamlit as st
import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import tempfile

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
OCR_WEIGHTS = MODEL_DIR / "yolo11m_car_plate_ocr.pt"
GALLERY_DIR = REPO_ROOT / "data" / "gallery"
BATCH_SIZE = 1

# Import ReID utilities
try:
    from src.reid.search import (
        _load_reid_model, 
        _load_gallery_embeddings, 
        _score_candidates, 
        _filter_matches,
        Candidate
    )
    from src.reid.visualize import draw_reid_debug
    from src.pipeline.yolo_ocr import load_model as load_yolo_ocr_model, read_plate_text
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Failed to import ReID modules: {e}")

@st.cache_resource
def load_models():
    """Load Detection and OCR models cached."""
    det_model = YOLO(DET_WEIGHTS)
    ocr_model = load_yolo_ocr_model(OCR_WEIGHTS)
    return det_model, ocr_model

def get_gallery_plates():
    """Get list of registered plate IDs from gallery."""
    if not GALLERY_DIR.exists():
        return []
    return [d.name for d in GALLERY_DIR.iterdir() if d.is_dir() and list(d.glob("*.jpg"))]

def get_plate_thumbnail(plate_id):
    """Get first image from plate folder as thumbnail."""
    plate_dir = GALLERY_DIR / plate_id
    images = list(plate_dir.glob("*.jpg"))
    if images:
        return str(images[0])
    return None

def save_to_gallery(car_crop, plate_text):
    """Save car crop to data/gallery/{plate_text}/{uuid}.jpg"""
    import uuid
    safe_plate = "".join([c for c in plate_text if c.isalnum() or c in "_-"])
    if not safe_plate:
        safe_plate = "unknown"
    save_dir = GALLERY_DIR / safe_plate
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(save_dir / filename), car_crop)
    return save_dir

def process_registration(file_path, is_video=False):
    """Run Detection -> OCR -> Save to Gallery."""
    det_model, ocr_model = load_models()
    cap = cv2.VideoCapture(str(file_path)) if is_video else None
    saved_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_idx = 0
    
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 != 0:
                frame_idx += 1
                continue
        else:
            frame = cv2.imread(str(file_path))
            if frame is None:
                break
        
        results = det_model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes
        names = det_model.names
        
        cars = []
        plates = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id].lower()
            coords = box.xyxy[0].cpu().numpy()
            if "car" in name or "truck" in name or "bus" in name:
                cars.append(coords)
            elif "plate" in name:
                plates.append(coords)
        
        for px1, py1, px2, py2 in plates:
            best_car = None
            p_center = ((px1+px2)/2, (py1+py2)/2)
            for cx1, cy1, cx2, cy2 in cars:
                if cx1 < p_center[0] < cx2 and cy1 < p_center[1] < cy2:
                    best_car = (cx1, cy1, cx2, cy2)
                    break
            
            if best_car:
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
                text, conf = read_plate_text(plate_crop, model=ocr_model)
                if text and conf > 0.4:
                    cx1, cy1, cx2, cy2 = map(int, best_car)
                    car_crop = frame[cy1:cy2, cx1:cx2]
                    save_to_gallery(car_crop, text)
                    saved_count += 1
                    status_text.text(f"✅ Registered: {text}")
        
        frame_idx += 1
        if not is_video:
            break
            
    if cap:
        cap.release()
    return saved_count

def render():
    st.markdown("# 🔍 Vehicle Re-Identification")
    st.markdown("*Find your car across cameras using AI*")
    
    # Architecture diagram
    ASSETS_DIR = Path(__file__).parent / "assets"
    reid_img = ASSETS_DIR / "reid_pipeline.png"
    
    with st.expander("📐 ReID Architecture", expanded=False):
        if reid_img.exists():
            st.image(str(reid_img), use_container_width=True)
        st.caption("ResNet50-IBN backbone with contrastive learning for vehicle matching")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📥 Register Vehicle", "🔎 Search", "🖼️ Gallery"])
    
    # --- TAB 1: REGISTER ---
    with tab1:
        st.markdown("### Add a New Vehicle")
        st.info("Upload an image or video. We'll detect cars, read plates, and save to gallery.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("📁 Upload Image/Video", type=['jpg', 'png', 'mp4', 'mov', 'avi'], key="reg_upload")
        with col2:
            st.markdown("**Supported formats:**")
            st.markdown("- Images: JPG, PNG")
            st.markdown("- Videos: MP4, MOV, AVI")
        
        if uploaded_file:
            if st.button("🚀 Start Registration", use_container_width=True):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                is_video = uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi'))
                
                with st.spinner("🔄 Processing..."):
                    count = process_registration(Path(tfile.name), is_video)
                
                if count > 0:
                    st.success(f"🎉 Registration Complete! Added {count} vehicle(s).")
                    st.balloons()
                else:
                    st.warning("No plates detected. Try a clearer image.")
    
    # --- TAB 2: SEARCH ---
    with tab2:
        st.markdown("### Search for a Vehicle")
        
        # Gallery selector - simple dropdown
        plates = get_gallery_plates()
        
        if not plates:
            st.warning("No vehicles registered yet. Go to 'Register Vehicle' tab first.")
        else:
            # Simple selectbox instead of cards
            plate_input = st.selectbox(
                "🚗 Select Vehicle from Gallery",
                plates,
                index=0
            )
            
            st.divider()
            
            uploaded_file = st.file_uploader("📁 Upload Search Image/Video", type=['mp4', 'mov', 'avi', 'jpg', 'png', 'jpeg'], key="search_file")
            min_score = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.6)
            
            if uploaded_file and st.button("🔍 Start Search", use_container_width=True):
                if not REID_OPTS.exists() or not REID_CKPT.exists():
                    st.error("ReID models missing.")
                    return
                
                tfile_input = tempfile.NamedTemporaryFile(delete=False) 
                tfile_input.write(uploaded_file.read())
                input_path = Path(tfile_input.name)
                is_video = uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi'))
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                with st.spinner("🔄 Loading ReID Model & Gallery..."):
                    input_size = 224
                    model = _load_reid_model(REID_OPTS, REID_CKPT, device)
                    gallery_entries = _load_gallery_embeddings(
                        GALLERY_DIR, plate_input, model, input_size, device, BATCH_SIZE
                    )
                
                if not gallery_entries:
                    st.error(f"No gallery images for: {plate_input}")
                else:
                    st.success(f"✅ Gallery loaded ({len(gallery_entries)} templates)")
                    det_model_search, _ = load_models()
                    
                    def process_frame(frame):
                        results = det_model_search.predict(frame, conf=0.25, verbose=False)
                        candidates = []
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            name = det_model_search.names[cls_id]
                            if "car" in name.lower():
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                h, w = frame.shape[:2]
                                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                                car_crop = frame[y1:y2, x1:x2]
                                if car_crop.size > 0:
                                    candidates.append(Candidate(
                                        image_path=Path("frame"),
                                        bbox_xyxy=(x1, y1, x2, y2),
                                        det_conf=float(box.conf[0]),
                                        crop=car_crop
                                    ))
                        return candidates
                    
                    if is_video:
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        cap = cv2.VideoCapture(str(input_path))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        st_frame = st.empty()
                        progress_bar = st.progress(0)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            candidates = process_frame(frame)
                            matches = _score_candidates(gallery_entries, candidates, model, input_size, device, BATCH_SIZE)
                            matches = _filter_matches(matches, min_score=min_score, top_k=5)
                            
                            match_dicts = [{"bbox_xyxy": m.bbox_xyxy, "score": m.score, "kept": m.kept} for m in matches]
                            annotated = draw_reid_debug(frame, match_dicts, plate_input)
                            out.write(annotated)
                            
                            if frame_count % 10 == 0:
                                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                            frame_count += 1
                        
                        cap.release()
                        out.release()
                        
                        st.success("✅ Search Complete!")
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button("📥 Download Video", f, "reid_result.mp4", "video/mp4")
                    else:
                        frame = cv2.imread(str(input_path))
                        if frame is not None:
                            candidates = process_frame(frame)
                            matches = _score_candidates(gallery_entries, candidates, model, input_size, device, BATCH_SIZE)
                            matches = _filter_matches(matches, min_score=min_score, top_k=5)
                            
                            match_dicts = [{"bbox_xyxy": m.bbox_xyxy, "score": m.score, "kept": m.kept} for m in matches]
                            annotated = draw_reid_debug(frame, match_dicts, plate_input)
                            
                            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="🎯 Search Result", use_container_width=True)
                            st.success("✅ Search Complete!")
                            
                            is_success, buffer = cv2.imencode(".jpg", annotated)
                            if is_success:
                                st.download_button("📥 Download Image", buffer.tobytes(), "reid_result.jpg", "image/jpeg")
    
    # --- TAB 3: GALLERY ---
    with tab3:
        st.markdown("### 🖼️ Registered Vehicles Gallery")
        
        plates = get_gallery_plates()
        
        if not plates:
            st.info("No vehicles registered yet. Upload images/videos in the 'Register Vehicle' tab.")
        else:
            st.success(f"Found **{len(plates)}** registered vehicles")
            
            for plate_id in plates:
                with st.expander(f"🚗 {plate_id}", expanded=False):
                    plate_dir = GALLERY_DIR / plate_id
                    images = list(plate_dir.glob("*.jpg"))[:6]  # Max 6 per plate
                    
                    cols = st.columns(min(len(images), 3))
                    for idx, img_path in enumerate(images):
                        with cols[idx % 3]:
                            st.image(str(img_path), use_container_width=True)
                    
                    st.caption(f"{len(list(plate_dir.glob('*.jpg')))} image(s) total")
