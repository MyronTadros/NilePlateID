# 🚗 NilePlateID - Egyptian License Plate Recognition

End-to-end AI pipeline for Egyptian license plate detection, OCR, and vehicle re-identification.

## ✨ Features

- **YOLO Detection** - Car and license plate detection with YOLOv11
- **Arabic OCR** - Custom YOLO OCR trained on Egyptian plates
- **Vehicle ReID** - Re-identify cars across cameras using deep learning
- **Streamlit App** - Interactive web demo with premium UI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install uv (package manager)
pip install uv

# Create environment and install
uv venv
uv sync
```

### 2. Download Models

```bash
uv run python -m src.download_weights
```

This downloads:
- `models/best.pt` - YOLO detection weights
- `models/yolo11m_car_plate_ocr.pt` - YOLO OCR weights
- `models/reid/net.pth` + `opts.yaml` - ReID model

### 3. Run Streamlit App

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```
NilePlateID/
├── app.py                      # Streamlit entry point
├── .streamlit/
│   └── config.toml             # Theme configuration
├── src/
│   ├── app_pages/              # Streamlit pages
│   │   ├── classical_page.py   # Classical CV + EasyOCR
│   │   ├── pipeline_page.py    # YOLO detection pipeline
│   │   ├── training_page.py    # Training dashboard
│   │   ├── reid_page.py        # Vehicle ReID
│   │   ├── future_work_page.py # Limitations & roadmap
│   │   └── assets/             # Architecture diagrams
│   ├── pipeline/               # Core detection & OCR logic
│   │   ├── detection.py        # YOLO detection wrapper
│   │   ├── ocr.py              # OCR backends (YOLO, EasyOCR)
│   │   ├── yolo_ocr.py         # YOLO OCR character mapping
│   │   ├── enhancement.py      # Image preprocessing
│   │   ├── association.py      # Plate-to-car matching
│   │   └── visualize.py        # Debug visualization
│   ├── reid/                   # ReID indexing & search
│   │   ├── search.py           # Gallery embeddings & matching
│   │   └── visualize.py        # ReID debug visualization
│   └── cli.py                  # CLI entrypoint
├── models/                     # Model weights (not committed)
│   ├── best.pt
│   ├── yolo11m_car_plate_ocr.pt
│   └── reid/
│       ├── net.pth
│       └── opts.yaml
├── data/
│   ├── gallery/                # Car crops by plate_id
│   ├── plates/                 # Plate crops by plate_id
│   └── meta/                   # Detection outputs
├── Classical Detection Method/ # Traditional CV approaches
└── third_party/
    └── vehicle_reid/           # ReID baseline code
```

## 🎯 Streamlit App Pages

| Page | Description |
|------|-------------|
| 🔬 Classical + EasyOCR | Traditional CV pipeline with morphological ops |
| 📊 Training Dashboard | Model training metrics and loss function math |
| 🎯 YOLO Pipeline | Detection + OCR with architecture diagram |
| 🔍 Vehicle ReID | Register cars to gallery and search by plate |
| 🔮 Future Work | Limitations and roadmap |

## 🛠️ CLI Commands (Optional)

For batch processing, the CLI is still available:

```bash
# Full pipeline (detect + OCR + save)
uv run python -m src.cli run \
    --weights models/best.pt \
    --input data/incoming \
    --gallery data/gallery \
    --plates data/plates \
    --index data/meta/index.csv

# Build ReID index
uv run python -m src.cli reid-index \
    --gallery_dir data/gallery \
    --reid_opts models/reid/opts.yaml \
    --reid_ckpt models/reid/net.pth

# Search by plate ID
uv run python -m src.cli reid-search \
    --plate_id ABC123 \
    --input_dir data/incoming

# Clean artifacts
uv run python -m src.cli clean --force
```

## 📦 Model Downloads

| Model | Description | Size |
|-------|-------------|------|
| `best.pt` | YOLO car + plate detection | ~50MB |
| `yolo11m_car_plate_ocr.pt` | YOLO Arabic OCR | ~40MB |
| `reid/net.pth` | ResNet50-IBN ReID | ~100MB |

Models are auto-downloaded with:
```bash
uv run python -m src.download_weights
```

## 📚 References

- **YOLO**: [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- **EasyOCR**: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **ReID Paper**: Zheng et al., "Joint Discriminative and Generative Learning", CVPR 2019
- **ReID Code**: [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)

## 📄 License

MIT License - see [LICENSE](LICENSE)
