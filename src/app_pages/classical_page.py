import streamlit as st
import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path
from PIL import Image

# Add Classical Method to sys.path
# We use absolute path to be safe
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_DIR = REPO_ROOT / "Classical Detection Method"
if str(CLASSICAL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSICAL_DIR))

try:
    from plate_detection import detect, process
    from ocr_processing import recognise_easyocr, recognise_tesseract, post_process
except ImportError as e:
    st.error(f"Failed to import Classical modules: {e}")
    # Define dummy functions so the app doesn't crash immediateley
    def detect(img): return img, []
    def process(crop, count): return crop

def render():
    st.header("Classical Approaches")
    
    st.info("Using methods from `Classical Detection Method` folder.")

    uploaded_file = st.file_uploader("Choose an image for Classical Detection...", type=["jpg", "png", "jpeg"], key="classical_uploader")

    if uploaded_file is not None:
        # Save temp file because the classical script assumes reading from file sometimes or writing to temp
        # But `detect` takes an image array (RGB/BGR?) 
        # Let's check plate_detection.py: `detect(img_rgb)` and it does `cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)`
        # Wait, if it converts BGR2HSV, it expects BGR. 
        # But the argument name is `img_rgb`. Variable says `img_rgb`. 
        # CV2 usually reads BGR. 
        # Let's verify `main.py` calls `img = cv2.imread(args.i)`. cv2.imread returns BGR.
        # So `detect` expects BGR.
        
        # Prepare temp directories
        if not os.path.exists('temp'):
            os.makedirs('temp')
        if not os.path.exists('temp/steps'):
            os.makedirs('temp/steps')
            
        # Clean up old steps
        files = glob.glob('temp/steps/*')
        for f in files:
            try:
                os.remove(f)
            except:
                pass

        image = Image.open(uploaded_file)
        # Convert PIL to BGR for OpenCV
        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        if st.button("Run Classical Pipeline"):
            with st.spinner("Running classical detection algorithms..."):
                # 1. Detection
                # Redirect stdout to capture prints if needed, but not critical
                result_img, crops = detect(img_bgr)
                
                # Use glob to find generated step images in temp/steps
                # The names are hardcoded in plate_detection.py:
                # "1_blue_color_detection.png", "2_closing_morphology.png"
                
                step_images = sorted(glob.glob("temp/steps/*.png"))
                
                with col2:
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

                if crops:
                    st.success(f"Detected {len(crops)} potential plates.")
                    
                    st.divider()
                    st.subheader("Intermediate Steps (Detection)")
                    
                    # Display detection steps
                    # We look for steps that don't start with "plate" (those are per-plate steps)
                    det_steps = [s for s in step_images if "plate" not in os.path.basename(s)]
                    
                    if det_steps:
                        cols = st.columns(min(len(det_steps), 3))
                        for idx, step_path in enumerate(det_steps):
                            step_name = os.path.basename(step_path).replace(".png", "").replace("_", " ").title()
                            with cols[idx % 3]:
                                st.image(step_path, caption=step_name, use_column_width=True)
                    
                    st.divider()
                    st.subheader("Plate Processing & OCR")
                    
                    for i, crop in enumerate(crops):
                        plate_num = i + 1
                        st.markdown(f"### Plate #{plate_num}")
                        
                        # Process (Enhancement, thresholding)
                        # The `process` function saves steps like `plate{num}_3_...`
                        processed_crop = process(crop, plate_num)
                        
                        # Find steps for this plate
                        plate_steps = sorted(glob.glob(f"temp/steps/plate{plate_num}_*.png"))
                        
                        if plate_steps:
                            cols = st.columns(len(plate_steps))
                            for idx, step_path in enumerate(plate_steps):
                                step_name = os.path.basename(step_path).split('_', 2)[-1].replace(".png", "").replace("_", " ").title()
                                with cols[idx]:
                                    st.image(step_path, caption=step_name, use_column_width=True)

                        # Final OCR
                        # `processed_crop` is the thresholded image
                        
                        col_ocr1, col_ocr2 = st.columns(2)
                        
                        # EasyOCR
                        # We can pass the path to the thresholded image saved by `process`
                        threshold_path = f"temp/steps/plate{plate_num}_6_threshold.png"
                        if os.path.exists(threshold_path):
                            e_text = recognise_easyocr(threshold_path)
                            e_text_clean = post_process(e_text)
                            
                            t_text = recognise_tesseract(threshold_path, f"temp/crop{plate_num}_tesseract")
                            t_text_clean = post_process(t_text)
                            
                            with col_ocr1:
                                st.info(f"**EasyOCR**: {e_text_clean}")
                            with col_ocr2:
                                st.warning(f"**Tesseract**: {t_text_clean}")
                        else:
                            st.error("Could not find processed image for OCR.")

                else:
                    st.warning("No plates detected with classical method.")
