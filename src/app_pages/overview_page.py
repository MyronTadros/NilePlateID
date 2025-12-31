import streamlit as st

def render():
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🚗 NilePlateID</h1>
        <p style="font-size: 1.3rem; opacity: 0.8;">AI-powered Egyptian license plate recognition and vehicle re-identification for smart parking systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Motivation Story
    st.markdown("## 🎯 The Problem We Solve")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### The Mall Parking Challenge
        
        Imagine parking your car in a busy shopping mall garage. Hours later, 
        you're struggling to remember the exact floor and spot where you parked.
        
        **Traditional solutions fail because:**
        - 📍 Manual ticket systems are easily lost
        - 🎫 Numbered spots are hard to remember
        - 📱 Existing apps require manual check-in
        """)
    
    with col2:
        st.markdown("""
        ### Our AI Solution
        
        NilePlateID automatically **enrolls your vehicle** when you enter 
        (plate clearly visible), then helps you **find it later** from any 
        CCTV angle—even when the plate is hidden.
        
        **How it works:**
        - 🚘 Automatic plate detection & OCR on entry
        - 🧠 Vehicle appearance learning (ReID)
        - 🔍 Search by plate ID across all cameras
        """)
    
    st.divider()
    
    # Use Cases
    st.markdown("## 💼 Applications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏢 Shopping Malls**
        - Car finder kiosks
        - Automated parking guidance
        
        **🏠 Residential Compounds**
        - Visitor management
        - Resident vehicle tracking
        """)
    
    with col2:
        st.markdown("""
        **🅿️ Parking Garages**
        - Ticketless entry/exit
        - Payment automation
        
        **🎪 Events & Venues**
        - VIP vehicle recognition
        - Security monitoring
        """)
    
    with col3:
        st.markdown("""
        **🚛 Fleet Management**
        - Vehicle tracking
        - Route verification
        
        **🏭 Industrial Sites**
        - Access control
        - Delivery logging
        """)
    
    
    st.divider()
    
    # Pipeline Diagrams
    st.markdown("## 🔄 System Architecture")
    
    tab1, tab2 = st.tabs(["📥 Enrollment Pipeline", "🔍 Search Pipeline"])
    
    with tab1:
        st.markdown("### Crop + OCR (Enrollment)")
        st.caption("Registers vehicles by detecting plates, reading text, and saving car crops to the gallery.")
        
        enrollment_diagram = """
        digraph enrollment {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11];
            edge [fontname="Inter", fontsize=10];
            
            input [label="Input\\nImage/Video", fillcolor="#E3F2FD"];
            yolo [label="YOLO\\nDetect", fillcolor="#BBDEFB"];
            associate [label="Associate\\nPlate↔Car", fillcolor="#90CAF9"];
            crop [label="Crop\\nPlate + Car", fillcolor="#64B5F6"];
            ocr [label="OCR\\n(YOLO/EasyOCR)", fillcolor="#42A5F5"];
            normalize [label="Normalize\\nPlate ID", fillcolor="#2196F3"];
            save [label="Save to\\nGallery", fillcolor="#1E88E5", fontcolor="white"];
            
            input -> yolo -> associate -> crop -> ocr -> normalize -> save;
        }
        """
        st.graphviz_chart(enrollment_diagram)
        
        st.info("""
        **Outputs:**
        - `data/gallery/{plate_id}/` — Car crops indexed by plate
        - `data/plates/{plate_id}/` — Plate crops
        - `data/meta/index.csv` — Detection metadata
        """)
    
    with tab2:
        st.markdown("### ReID (Search)")
        st.caption("Finds vehicles across cameras by matching visual appearance against enrolled gallery.")
        
        search_diagram = """
        digraph search {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11];
            edge [fontname="Inter", fontsize=10];
            
            gallery [label="Gallery\\nCar Crops", fillcolor="#E8F5E9"];
            embed_g [label="Extract\\nEmbeddings", fillcolor="#C8E6C9"];
            index [label="Build Index\\n(per plate_id)", fillcolor="#A5D6A7"];
            
            query [label="Query\\nImage/Video", fillcolor="#FFF3E0"];
            detect [label="YOLO\\nCar Detect", fillcolor="#FFE0B2"];
            crop_q [label="Crop\\nCars", fillcolor="#FFCC80"];
            embed_q [label="Extract\\nEmbeddings", fillcolor="#FFB74D"];
            match [label="Similarity\\nMatch", fillcolor="#FFA726"];
            output [label="Ranked\\nResults", fillcolor="#FF9800", fontcolor="white"];
            
            gallery -> embed_g -> index;
            query -> detect -> crop_q -> embed_q -> match -> output;
            index -> match [style=dashed, label="compare"];
        }
        """
        st.graphviz_chart(search_diagram)
        
        st.info("""
        **Outputs:**
        - `data/meta/reid/index.npz` — Gallery embeddings
        - `data/meta/reid/results.csv` — Match results
        - `data/meta/reid/annotated/` — Visualized matches
        """)
    
    st.divider()
    
    # How to Use
    st.markdown("## 🚀 How to Use This App")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Step 1: Enroll
        
        Go to **🔬 Classical + EasyOCR** or **🎯 YOLO Pipeline**
        
        Upload images/videos to detect plates and save car crops to the gallery.
        """)
        
    with col2:
        st.markdown("""
        ### Step 2: Register
        
        Go to **🔍 Vehicle ReID** → **Register**
        
        Process your parking footage to build the searchable gallery index.
        """)
        
    with col3:
        st.markdown("""
        ### Step 3: Search
        
        Go to **🔍 Vehicle ReID** → **Search**
        
        Select a plate ID and search across query footage to find matches.
        """)
    
    st.divider()
    

    

    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; opacity: 0.7;">
            <p>Built with ❤️ using Streamlit, YOLO, and PyTorch</p>
            <p style="font-size: 0.8rem;">Egyptian University Informatics • Computer Vision Project</p>
        </div>
        """, unsafe_allow_html=True)
