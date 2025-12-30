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
# Models paths
MODEL_DIR = REPO_ROOT / "models"
REID_DIR = MODEL_DIR / "reid"
REID_OPTS = REID_DIR / "opts.yaml"
REID_CKPT = REID_DIR / "net.pth"
DET_WEIGHTS = MODEL_DIR / "best.pt"
OCR_WEIGHTS = MODEL_DIR / "yolo11m_car_plate_ocr.pt"
BATCH_SIZE = 1

# Import ReID utilities
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
    # Import pipeline utils
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

def get_car_crop(img, car_box, plate_box):
    """
    Refines car crop. 
    Ideally, we trust the 'car' detection. 
    If plate is inside car, great.
    """
    x1, y1, x2, y2 = map(int, car_box)
    return img[y1:y2, x1:x2]

def save_to_gallery(car_crop, plate_text):
    """Save car crop to data/gallery/{plate_text}/{uuid}.jpg"""
    import uuid
    safe_plate = "".join([c for c in plate_text if c.isalnum() or c in "_-"])
    if not safe_plate:
        safe_plate = "unknown"
        
    gallery_dir = REPO_ROOT / "data" / "gallery"
    save_dir = gallery_dir / safe_plate
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(save_dir / filename), car_crop)
    return save_dir

def process_registration(file_path, is_video=False):
    """Run Detection -> OCR -> Save to Gallery."""
    det_model, ocr_model = load_models()
    
    cap = cv2.VideoCapture(str(file_path)) if is_video else None
    
    # Process
    saved_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            # Skip frames to speed up registration
            if frame_idx % 5 != 0:
                frame_idx += 1
                continue
        else:
            frame = cv2.imread(str(file_path))
            if frame is None:
                break
        
        # Detect
        results = det_model.predict(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes
        names = det_model.names
        
        # Separate cars and plates
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
                
        # Match plates to cars (simple containment or distance)
        for px1, py1, px2, py2 in plates:
            best_car = None
            min_dist = float('inf')
            
            p_center = ((px1+px2)/2, (py1+py2)/2)
            
            for cx1, cy1, cx2, cy2 in cars:
                # Check containment
                if cx1 < p_center[0] < cx2 and cy1 < p_center[1] < cy2:
                    best_car = (cx1, cy1, cx2, cy2)
                    break
            
            if best_car:
                # OCR the plate
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
                text, conf = read_plate_text(plate_crop, model=ocr_model)
                
                if text and conf > 0.4: # Filter weak OCR
                    # Save Car Crop
                    cx1, cy1, cx2, cy2 = map(int, best_car)
                    car_crop = frame[cy1:cy2, cx1:cx2]
                    save_to_gallery(car_crop, text)
                    saved_count += 1
                    status_text.text(f"Registered: {text}")
        
        frame_idx += 1
        if not is_video:
            break
            
    if cap:
        cap.release()
    
    return saved_count

def render():
    st.header("Vehicle Re-Identification Pipeline")
    
    tab1, tab2 = st.tabs(["1. Register to Gallery", "2. Search in Video"])
    
    # --- TAB 1: REGISTER ---
    with tab1:
        st.subheader("Add Vehicles to Gallery")
        st.info("Upload an image or video. The system will detect cars, read their plates, and save the car image to the gallery.")
        
        uploaded_file = st.file_uploader("Upload Image/Video", type=['jpg', 'png', 'mp4', 'mov', 'avi'])
        
        if uploaded_file and st.button("Start Registration"):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            is_video = uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi'))
            
            with st.spinner("Processing registration..."):
                count = process_registration(Path(tfile.name), is_video)
            
            st.success(f"Registration Complete. Added {count} entries.")
            st.balloons()

    # --- TAB 2: SEARCH ---
    with tab2:
        st.subheader("Search for a Vehicle")
        
        plate_input = st.text_input("Target Plate ID", "ABC-1234")
        uploaded_file = st.file_uploader("Upload Search Image/Video", type=['mp4', 'mov', 'avi', 'jpg', 'png', 'jpeg'], key="search_file")
        min_score = st.slider("ReID Confidence Threshold", 0.0, 1.0, 0.6)
        
        if uploaded_file and st.button("Start Search"):
            if not REID_OPTS.exists() or not REID_CKPT.exists():
                st.error("ReID models missing.")
                return

            tfile_input = tempfile.NamedTemporaryFile(delete=False) 
            tfile_input.write(uploaded_file.read())
            input_path = Path(tfile_input.name)
            
            is_video = uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi'))
            
            gallery_dir = REPO_ROOT / "data" / "gallery"
            
            # Load ReID
            device = torch.device("cpu") # Use CPU for safety or cuda if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                
            with st.spinner("Loading ReID Model & Gallery..."):
                 # Input size 224 matches third_party/vehicle_reid/extract_features.py
                input_size = 224
                model = _load_reid_model(REID_OPTS, REID_CKPT, device)
                
                gallery_entries = _load_gallery_embeddings(
                    gallery_dir,
                    plate_input,
                    model,
                    input_size,
                    device,
                    BATCH_SIZE
                )
            
            if not gallery_entries:
                st.error(f"No gallery images found for plate: {plate_input}")
            else:
                st.success(f"Gallery loaded with {len(gallery_entries)} templates.")
                
                det_model_search, _ = load_models()
                
                def process_frame(frame, det_model):
                    # Run inference
                    results = det_model.predict(frame, conf=0.25, verbose=False)
                    
                    # Convert to Candidates
                    candidates = []
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        name = det_model.names[cls_id]
                        if "car" in name.lower():
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Clamp
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
                    # Video Processing
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    
                    cap = cv2.VideoCapture(str(input_path))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Use 'avc1' for better browser compatibility (H.264)
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    st_frame = st.empty()
                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        candidates = process_frame(frame, det_model_search)
                        
                        # Score
                        matches = _score_candidates(
                            gallery_entries,
                            candidates,
                            model,
                            input_size,
                            device,
                            BATCH_SIZE
                        )
                        
                        # Filter
                        matches = _filter_matches(matches, min_score=min_score, top_k=5)
                        
                        # Draw
                        match_dicts = []
                        for m in matches:
                            match_dicts.append({
                                "bbox_xyxy": m.bbox_xyxy,
                                "score": m.score,
                                "kept": m.kept
                            })
                        
                        annotated = draw_reid_debug(frame, match_dicts, plate_input)
                        out.write(annotated)
                        
                        # Preview
                        if frame_count % 10 == 0:
                            st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                        
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    
                    st.success("Search Complete!")
                    st.video(output_path)
                    
                    # Add Download Button for Video
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Search Result Video",
                            data=f,
                            file_name="reid_search_result.mp4",
                            mime="video/mp4"
                        )
                else:
                    # Image Processing
                    frame = cv2.imread(str(input_path))
                    if frame is not None:
                        candidates = process_frame(frame, det_model_search)
                        
                        matches = _score_candidates(
                            gallery_entries,
                            candidates,
                            model,
                            input_size,
                            device,
                            BATCH_SIZE
                        )
                        
                        matches = _filter_matches(matches, min_score=min_score, top_k=5)
                        
                        match_dicts = []
                        for m in matches:
                            match_dicts.append({
                                "bbox_xyxy": m.bbox_xyxy,
                                "score": m.score,
                                "kept": m.kept
                            })
                        
                        annotated = draw_reid_debug(frame, match_dicts, plate_input)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        
                        st.image(annotated_rgb, caption="Search Result", width="stretch")
                        st.success("Search Complete!")

                        # Add Download Button for Image (Optional but good)
                         # Convert Result Image to Bytes
                        is_success, buffer = cv2.imencode(".jpg", annotated)
                        if is_success:
                            st.download_button(
                                label="Download Search Result Image",
                                data=buffer.tobytes(),
                                file_name="reid_search_result.jpg",
                                mime="image/jpeg"
                            )


