import streamlit as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "data" / "demo"

def render():
    st.markdown("# 🎬 Demo Videos")
    st.markdown("*Sample outputs from the NilePlateID pipeline*")
    
    st.info("These videos demonstrate the parking system in action with YOLO detection and OCR.")
    
    # Prefer h264 versions for browser compatibility
    all_videos = list(DEMO_DIR.glob("*.mp4"))
    videos = [v for v in all_videos if "_h264" in v.name]
    
    # Fall back to all videos if no h264 versions exist
    if not videos:
        videos = all_videos
    
    if not videos:
        st.warning("No demo videos found in data/demo/")
        return
    
    st.success(f"Found **{len(videos)}** demo video(s)")
    
    for idx, video_path in enumerate(videos):
        st.markdown(f"### Video {idx + 1}: {video_path.name}")
        
        # Display video
        st.video(str(video_path))
        
        # Download button
        with open(video_path, "rb") as f:
            st.download_button(
                f"📥 Download {video_path.name}",
                f,
                video_path.name,
                "video/mp4",
                key=f"download_{idx}"
            )
        
        st.divider()
