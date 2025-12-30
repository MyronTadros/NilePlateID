import streamlit as st
import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path
from PIL import Image

# Add Classical Method to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_DIR = REPO_ROOT / "Classical Detection Method"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

try:
    from plate_detection import detect, process
    from ocr_processing import recognise_easyocr, post_process
    from canny_plate_detection import process_and_crop_plate
except ImportError as e:
    st.error(f"Failed to import Classical modules: {e}")
    def detect(img): return img, []
    def process(crop, count): return crop
    def process_and_crop_plate(path): return None

def render():
    st.markdown("# 🔬 Classical Detection + EasyOCR")
    st.markdown("*Traditional computer vision techniques for license plate detection*")
    
    # Detection method selector
    detection_method = st.radio(
        "🔧 Select Detection Method:",
        ("Morphological (Color-based)", "Canny Edge Detection"),
        horizontal=True
    )
    
    if detection_method == "Morphological (Color-based)":
        st.info("🔍 Uses HSV color segmentation, morphological closing, and contour analysis to detect blue Egyptian plates.")
    else:
        st.info("🔍 Uses bilateral filtering, Canny edge detection, contour approximation, and Harris corner validation.")

    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"], key="classical_uploader")

    if uploaded_file is not None:
        # Prepare temp directories
        if not os.path.exists('temp'):
            os.makedirs('temp')
        if not os.path.exists('temp/steps'):
            os.makedirs('temp/steps')
            
        # Clean up old steps
        for f in glob.glob('temp/steps/*'):
            try:
                os.remove(f)
            except:
                pass

        image = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🚀 Run Detection", use_container_width=True):
            
            if detection_method == "Canny Edge Detection":
                # Save temp image for Canny function
                temp_path = "temp/canny_input.jpg"
                cv2.imwrite(temp_path, img_bgr)
                
                with st.spinner("Running Canny edge detection..."):
                    # The Canny function uses matplotlib, so we need to modify it
                    # Let's do the processing inline here
                    
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
                    edged = cv2.Canny(filtered, 30, 200)
                    
                    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
                    
                    plate_found = False
                    for c in contours:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                        
                        if len(approx) == 4:
                            x, y, w, h = cv2.boundingRect(approx)
                            aspect_ratio = w / float(h)
                            
                            if 2.0 <= aspect_ratio <= 4.0:
                                plate_roi = gray[y:y+h, x:x+w]
                                harris_dst = cv2.cornerHarris(plate_roi, blockSize=2, ksize=3, k=0.04)
                                harris_corners_count = np.sum(harris_dst > 0.01 * harris_dst.max())
                                
                                if harris_corners_count > 20:
                                    # Draw result
                                    result_img = img_bgr.copy()
                                    cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 3)
                                    cropped_plate = img_bgr[y:y+h, x:x+w]
                                    
                                    # Harris visualization
                                    harris_vis = cropped_plate.copy()
                                    harris_vis[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]
                                    
                                    with col2:
                                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
                                    
                                    st.success("✅ Plate detected!")
                                    
                                    # Show steps
                                    st.subheader("📊 Detection Steps")
                                    cols = st.columns(4)
                                    with cols[0]:
                                        st.image(gray, caption="Grayscale", use_container_width=True)
                                    with cols[1]:
                                        st.image(filtered, caption="Bilateral Filter", use_container_width=True)
                                    with cols[2]:
                                        st.image(edged, caption="Canny Edges", use_container_width=True)
                                    with cols[3]:
                                        st.image(cv2.cvtColor(harris_vis, cv2.COLOR_BGR2RGB), caption="Harris Corners", use_container_width=True)
                                    
                                    st.subheader("📋 Cropped Plate")
                                    st.image(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB), caption="Final Crop", use_container_width=True)
                                    
                                    # Run OCR
                                    cv2.imwrite("temp/canny_plate.jpg", cropped_plate)
                                    easyocr_out = "temp/canny_plate_ocr"
                                    result = recognise_easyocr("temp/canny_plate.jpg", easyocr_out)
                                    if result == 0 and os.path.exists(easyocr_out + ".txt"):
                                        text = post_process(easyocr_out + ".txt")
                                        st.info(f"**EasyOCR Result**: {text}")
                                    
                                    plate_found = True
                                    break
                    
                    if not plate_found:
                        with col2:
                            st.image(edged, caption="Canny Edges (No plate found)", use_container_width=True)
                        st.warning("No valid plate detected with Canny method.")
            
            else:
                # Morphological method
                with st.spinner("Running morphological detection..."):
                    result_img, crops = detect(img_bgr)
                    step_images = sorted(glob.glob("temp/steps/*.png"))
                    
                    with col2:
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)

                    if crops:
                        st.success(f"✅ Detected {len(crops)} potential plates.")
                        
                        st.subheader("📊 Detection Steps")
                        det_steps = [s for s in step_images if "plate" not in os.path.basename(s)]
                        if det_steps:
                            cols = st.columns(min(len(det_steps), 3))
                            for idx, step_path in enumerate(det_steps):
                                step_name = os.path.basename(step_path).replace(".png", "").replace("_", " ").title()
                                with cols[idx % 3]:
                                    st.image(step_path, caption=step_name, use_container_width=True)
                        
                        st.subheader("📋 Plate Processing & OCR")
                        for i, crop in enumerate(crops):
                            plate_num = i + 1
                            st.markdown(f"**Plate #{plate_num}**")
                            
                            processed_crop = process(crop, plate_num)
                            plate_steps = sorted(glob.glob(f"temp/steps/plate{plate_num}_*.png"))
                            
                            if plate_steps:
                                cols = st.columns(min(len(plate_steps), 3))
                                for idx, step_path in enumerate(plate_steps):
                                    step_name = os.path.basename(step_path).split('_', 2)[-1].replace(".png", "").replace("_", " ").title()
                                    with cols[idx % 3]:
                                        st.image(step_path, caption=step_name, use_container_width=True)
                            
                            threshold_path = f"temp/steps/plate{plate_num}_6_threshold.png"
                            if os.path.exists(threshold_path):
                                easyocr_out = f"temp/crop{plate_num}_easyocr"
                                result = recognise_easyocr(threshold_path, easyocr_out)
                                if result == 0 and os.path.exists(easyocr_out + ".txt"):
                                    text = post_process(easyocr_out + ".txt")
                                    st.info(f"**EasyOCR**: {text}")
                    else:
                        st.warning("No plates detected with morphological method.")
