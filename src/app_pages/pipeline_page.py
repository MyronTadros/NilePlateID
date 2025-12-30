import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from src.pipeline.yolo_ocr import load_model as load_yolo_ocr_model, read_plate_text as read_yolo_plate_text
from src.pipeline.ocr import read_plate_text as read_easyocr_plate_text

# Model paths configuration
DET_MODEL_PATH = "models/best.pt"
OCR_YOLO_PATH = "models/yolo11m_car_plate_ocr.pt"

@st.cache_resource
def load_det_model():
    return YOLO(DET_MODEL_PATH)

@st.cache_resource
def load_ocr_model_cached():
    return load_yolo_ocr_model(Path(OCR_YOLO_PATH))

def render():
    st.markdown("# 🎯 YOLO Detection Pipeline")
    st.markdown("*State-of-the-art deep learning for car and plate detection*")
    
    # Architecture diagram
    ASSETS_DIR = Path(__file__).parent / "assets"
    pipeline_img = ASSETS_DIR / "yolo_pipeline.png"
    
    with st.expander("📐 Pipeline Architecture", expanded=False):
        if pipeline_img.exists():
            st.image(str(pipeline_img), use_container_width=True)
        st.caption("YOLO v11 for detection + Custom YOLO OCR for Arabic character recognition")
    
    st.divider()
    
    st.subheader("📤 1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    st.subheader("🔧 2. Select OCR Model")
    ocr_option = st.radio(
        "Choose OCR Engine:",
        ("YOLO OCR", "EasyOCR")
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        # Convert to CV2 format (BGR)
        img_np = np.array(image.convert('RGB')) 
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")
            
        with st.spinner("Detecting cars and plates..."):
            det_model = load_det_model()
            results = det_model.predict(img_bgr, conf=0.25)
            
            # Draw detections
            res_plotted = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(res_plotted_rgb, caption="Detections (YOLO)", width="stretch")

            # Extract Plates
            boxes = results[0].boxes
            names = det_model.names
            
            plates_found = []
            
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                
                if 'plate' in cls_name.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img_bgr[y1:y2, x1:x2]
                    plates_found.append(crop)
            
            st.divider()
            st.subheader(f"Found {len(plates_found)} Plate(s)")
            
            if plates_found:
                for idx, plate_crop in enumerate(plates_found):
                    st.write(f"### Plate #{idx+1}")
                    
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    
                    c1, c2, c3 = st.columns([1, 1, 2])
                    
                    with c1:
                        st.image(plate_rgb, caption="Cropped Plate", width="stretch")
                    
                    # Run OCR
                    text = ""
                    conf = 0.0
                    
                    if ocr_option == "YOLO OCR":
                        # Preprocessing for display purpose if any? YOLO usually takes raw crop
                        # But user asked to "show the preprocessing steps". 
                        # YOLO OCR pipeline in `yolo_ocr.py` takes raw crop.
                        try:
                            ocr_model = load_ocr_model_cached()
                            text, conf = read_yolo_plate_text(plate_crop, model=ocr_model)
                            with c2:
                                st.info("YOLO OCR uses raw crop.")
                        except Exception as e:
                            st.error(f"Error loading YOLO OCR: {e}")

                    elif ocr_option == "EasyOCR":
                        # EasyOCR pipeline in `ocr.py` has preprocessing
                        # We can call `_preprocess` from `ocr.py` to show it.
                        from src.pipeline.ocr import _preprocess
                        preprocessed = _preprocess(plate_crop, use_enhancement=True)
                        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                        
                        with c2:
                            st.image(preprocessed_rgb, caption="Preprocessed (Enhanced)", width="stretch")
                        
                        text, conf = read_easyocr_plate_text(plate_crop, use_enhancement=True)



                    with c3:
                        st.success(f"**Detected Text:** {text}")
                        if conf > 0:
                            st.write(f"Confidence: {conf:.1f}%")

            else:
                st.warning("No plates detected.")
