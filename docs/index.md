# NilePlateID

**Egyptian License Plate Recognition with AI**

NilePlateID is an end-to-end AI pipeline for detecting Egyptian license plates, recognizing Arabic text, and re-identifying vehicles across cameras.

![Pipeline Overview](assets/yolo_pipeline.png)

## Features

- 🔍 **YOLO Detection** - Car and license plate detection with YOLOv11
- 🔤 **Arabic OCR** - Custom YOLO OCR trained on Egyptian plates
- 🚗 **Vehicle ReID** - Re-identify cars across cameras using deep learning
- 🖥️ **Streamlit App** - Interactive web demo with premium UI

## Quick Links

| Documentation | Description |
|--------------|-------------|
| [Getting Started](getting-started.md) | Installation and setup |
| [Architecture](architecture.md) | System design overview |
| [Classical CV](classical.md) | Traditional CV approaches |
| [YOLO Detection](yolo.md) | Deep learning detection |
| [Vehicle ReID](reid.md) | Re-identification system |
| [API Reference](api.md) | CLI commands |

## Demo

Run the Streamlit app:

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.
