import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"

def render():
    st.markdown("# 📊 Training Dashboard")
    st.markdown("*Model training metrics from Ultralytics HUB*")
    
    # Architecture and references
    with st.expander("📚 Training Architecture & References", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Detection Model
            - **Architecture**: YOLOv11
            - **Classes**: Car, License Plate
            - **Dataset**: Custom Egyptian plates
            - **Platform**: [Ultralytics HUB](https://hub.ultralytics.com/)
            """)
        
        with col2:
            st.markdown("""
            ### 🔤 OCR Model  
            - **Architecture**: YOLOv11 (Character Detection)
            - **Classes**: Arabic letters + Numerals
            - **Training**: Fine-tuned on Egyptian plates
            """)
        
        st.divider()
        
        st.markdown("""
        ### 🚗 ReID Model
        - **Base**: ResNet50-IBN
        - **Loss**: Contrastive + Circle Loss
        - **Paper**: [Zheng et al., CVPR 2019](https://arxiv.org/abs/1904.07223)
        - **Code**: [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
        - **Tutorial**: [Kaggle Vehicle ReID](https://www.kaggle.com/code/sosperec/vehicle-reid-tutorial/)
        """)
        
        training_arch_img = ASSETS_DIR / "training_architecture.png"
        if training_arch_img.exists():
            st.image(str(training_arch_img), use_container_width=True)
    
    # ReID Loss Functions Explanation
    with st.expander("📐 ReID Loss Functions (Math)", expanded=False):
        st.markdown("""
        ### Contrastive Loss
        
        Pulls similar samples together and pushes dissimilar samples apart:
        """)
        
        st.latex(r"L_{contrastive} = \frac{1}{2N} \sum_{n=1}^{N} (y_n \cdot d_n^2 + (1-y_n) \cdot \max(0, m - d_n)^2)")
        
        st.markdown("""
        Where:
        - $d_n$ = Euclidean distance between feature embeddings
        - $y_n$ = 1 if same identity, 0 otherwise
        - $m$ = margin (typically 0.5-1.0)
        
        **Effect**: Same vehicles → close embeddings; Different vehicles → far embeddings
        
        ---
        
        ### Circle Loss
        
        A unified loss that provides better convergence:
        """)
        
        st.latex(r"L_{circle} = \log \left[ 1 + \sum_{j=1}^{L} e^{\gamma \alpha_j^n (s_j^n - \Delta_n)} \cdot \sum_{k=1}^{K} e^{-\gamma \alpha_k^p (s_k^p - \Delta_p)} \right]")
        
        st.markdown("""
        Where:
        - $s_j^n$ = similarity score to negative samples
        - $s_k^p$ = similarity score to positive samples
        - $\\alpha$ = weighting factor
        - $\\gamma$ = scale factor
        - $\\Delta$ = margin for decision boundary
        
        **Effect**: More robust to noisy labels, better at handling class imbalance
        
        ---
        
        ### Combined Training
        
        The final loss combines multiple objectives:
        """)
        
        st.latex(r"L_{total} = L_{ID} + \lambda_1 L_{contrastive} + \lambda_2 L_{circle}")
        
        st.markdown("""
        Where $L_{ID}$ is the cross-entropy loss for identity classification, and $\\lambda$ are balancing weights.
        """)
    
    st.divider()
    
    # Training screenshots in expanders
    detection_img = ASSETS_DIR / "yolo model1.png"
    ocr_img = ASSETS_DIR / "yolo ocr.png"
    
    with st.expander("🚗 Car & License Plate Detection Training", expanded=False):
        if detection_img.exists():
            st.image(str(detection_img), use_container_width=True)
        else:
            st.warning(f"Image not found: {detection_img}")
    
    with st.expander("🔤 License Plate OCR Training", expanded=False):
        if ocr_img.exists():
            st.image(str(ocr_img), use_container_width=True)
        else:
            st.warning(f"Image not found: {ocr_img}")
    
    st.divider()
    st.caption("[Open Ultralytics HUB](https://hub.ultralytics.com/models/AC7YrPmQf90pOdPZDkzd?tab=train) for live training data.")
