# Components Overview

This section documents the core components of NilePlateID.

## Pipeline Components

| Component | Description | Page |
|-----------|-------------|------|
| **Classical CV** | Traditional image processing (morphological, Canny) + EasyOCR | [classical.md](classical.md) |
| **YOLO Detection** | YOLOv11 for car and plate detection | [yolo.md](yolo.md) |
| **YOLO OCR** | Custom YOLO model for Arabic character recognition | [yolo-ocr.md](yolo-ocr.md) |
| **Vehicle ReID** | ResNet50-IBN for vehicle re-identification | [reid.md](reid.md) |

## Architecture Diagram

```
Input Image/Video
    │
    ├─── Classical Pipeline ───► Morphological/Canny ───► EasyOCR
    │
    ├─── YOLO Pipeline ───► YOLOv11 Detection ───► YOLO OCR
    │
    └─── ReID Pipeline ───► Car Detection ───► Embeddings ───► Matching
```

## File Structure

```
src/
├── pipeline/               # Detection & OCR
│   ├── detection.py        # YOLO wrapper
│   ├── ocr.py              # EasyOCR backend
│   ├── yolo_ocr.py         # YOLO OCR
│   ├── enhancement.py      # Preprocessing
│   ├── association.py      # Plate-car matching
│   └── visualize.py        # Debug output
├── reid/                   # Re-identification
│   ├── search.py           # Gallery matching
│   └── visualize.py        # Result visualization
└── app_pages/              # Streamlit UI
```
