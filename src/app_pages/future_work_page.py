import streamlit as st

def render():
    st.markdown("# 🔮 Future Work & Limitations")
    st.markdown("*Roadmap for NilePlateID development*")
    
    # Vision Section
    st.markdown("""
    ## 🎯 Project Vision
    
    > **This is a Proof of Concept (PoC)** demonstrating the feasibility of an AI-powered parking management system.
    
    Our goal is to develop this into a **full-scale parking service** for malls and commercial spaces:
    
    - 📱 **Mobile App Integration** - Users receive notifications when their car is detected
    - 🅿️ **Smart Parking Guidance** - Direct drivers to available spots
    - 💳 **Seamless Payment** - Automated billing based on license plate recognition
    - 🔍 **Car Finder** - Help users locate their parked vehicles via ReID
    """)
    
    st.divider()
    
    # Limitations Section
    st.markdown("""
    ## ⚠️ Current Limitations
    
    ### 🔤 Arabic OCR Challenges
    
    Arabic license plate recognition remains an **open research problem**:
    
    | Challenge | Description |
    |-----------|-------------|
    | **Right-to-Left** | Text direction differs from Western plates |
    | **Limited Datasets** | Few public Arabic plate datasets available |
    | **Regional Variations** | Egyptian plates differ from Saudi, UAE, etc. |
    | **Generalization** | Models trained on one region often fail on others |
    
    Many state-of-the-art methods that work well on Latin characters **fail to generalize** to Arabic plates due to:
    - Unique character morphology
    - Similar-looking characters (ب، ت، ث، ن)
    
    ### 🚗 ReID Limitations
    
    - **Lighting Variations** - Performance drops in low-light conditions
    - **Occlusion** - Partially visible vehicles are harder to match
    - **Similar Vehicles** - Same model/color cars can be confused
    - **Cross-Camera** - Angle differences affect matching accuracy
    """)
    
    st.divider()
    
    # Future Improvements
    st.markdown("""
    ## 🚀 Proposed Improvements
    
    ### Short-Term (PoC Enhancements)
    - [ ] Data augmentation for Arabic character robustness
    - [ ] Multi-scale detection for distant plates
    - [ ] Ensemble OCR (YOLO + EasyOCR voting)
    - [ ] Real-time video streaming support
    
    ### Medium-Term (Product Development)  
    - [ ] Mobile app with push notifications
    - [ ] Database for vehicle tracking history
    - [ ] Admin dashboard for parking operators
    - [ ] API for third-party integrations
    
    ### Long-Term (Research Directions)
    - [ ] Transformer-based Arabic OCR (attention mechanisms)
    - [ ] Self-supervised learning for plate recognition
    - [ ] Multi-modal fusion (plate + vehicle features)
    - [ ] Edge deployment on parking cameras
    """)
    
    st.divider()
    
    # Research References
    st.markdown("""
    ## 📚 Research References
    
    **Vehicle ReID:**
    - Zheng et al., "Joint Discriminative and Generative Learning for Person Re-identification", CVPR 2019
    - He et al., "Bag of Tricks for Vehicle Re-Identification", CVPR Workshop 2019
    
    **Arabic OCR:**
    - Current approaches struggle with Egyptian plate format
    - Active research area with limited standardized benchmarks
    
    **Object Detection:**
    - Ultralytics YOLOv11 - [Documentation](https://docs.ultralytics.com/)
    """)
    
    st.divider()
        # Contributors
    st.markdown("## 👥 Team Contributors")
    
    st.markdown("""
    | Name | Contribution |
    |------|-------------|
    | **Hamdy Awad** | Morphological Detection + EasyOCR |
    | **Mohamed Alaa** | YOLO Pipeline + ReID  + Streamlit|
    | **Myron Milad** | ReID + CLI Pipeline |
    | **Bola Warsy** | Canny Edge Detection  + Integration|
    """)
    st.info("💡 **Contribute:** This project is open for collaboration. Reach out if you're interested in advancing Arabic license plate recognition!")
