import streamlit as st
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app_pages import pipeline_page, classical_page, reid_page, training_page, future_work_page

st.set_page_config(
    page_title="NilePlateID",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Adaptive to light/dark with glowing effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient text for headers */
    h1 {
        background: linear-gradient(90deg, #3498DB, #2ECC71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Glowing buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3498DB 0%, #2ECC71 100%);
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 15px rgba(52, 152, 219, 0.4);
        }
        to {
            box-shadow: 0 0 25px rgba(46, 204, 113, 0.6);
        }
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 0 35px rgba(52, 152, 219, 0.7);
    }
    
    /* Sidebar navigation pills */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    
    /* Radio as pills */
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.2) 0%, rgba(46, 204, 113, 0.2) 100%);
        border-color: rgba(52, 152, 219, 0.6);
        box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(52, 152, 219, 0.5);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(52, 152, 219, 0.8);
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 10px;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3498DB, #2ECC71) !important;
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #3498DB;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498DB 0%, #2ECC71 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar branding
    st.sidebar.markdown("""
    # 🚗 NilePlateID
    ### Egyptian License Plate Recognition
    ---
    """)
    
    # Navigation with better styling
    page = st.sidebar.radio(
        "📍 Navigate",
        [
            "🔬 Classical + EasyOCR",
            "📊 Training Dashboard", 
            "🎯 YOLO Pipeline",
            "🔍 Vehicle ReID",
            "🔮 Future Work"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📚 Resources:**
    - [YOLO Docs](https://docs.ultralytics.com/)
    - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
    """)
    st.sidebar.caption("Made with ❤️ for CV Project")

    # Route to pages
    if page == "🔬 Classical + EasyOCR":
        classical_page.render()
    elif page == "📊 Training Dashboard":
        training_page.render()
    elif page == "🎯 YOLO Pipeline":
        pipeline_page.render()
    elif page == "🔍 Vehicle ReID":
        reid_page.render()
    elif page == "🔮 Future Work":
        future_work_page.render()

if __name__ == "__main__":
    main()
