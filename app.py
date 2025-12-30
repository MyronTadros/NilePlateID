import streamlit as st
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app_pages import pipeline_page, classical_page, reid_page

st.set_page_config(
    page_title="NilePlateID",
    page_icon="🚗",
    layout="wide"
)

def main():
    st.sidebar.title("NilePlateID Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Parking System Pipeline", "Classical Approaches", "Vehicle ReID"]
    )

    if page == "Parking System Pipeline":
        pipeline_page.render()
    elif page == "Classical Approaches":
        classical_page.render()
    elif page == "Vehicle ReID":
        reid_page.render()

if __name__ == "__main__":
    main()
