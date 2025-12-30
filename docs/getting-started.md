# Getting Started

## Prerequisites

- Python 3.10-3.12
- pip or uv package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MyronTadros/NilePlateID.git
cd NilePlateID
```

### 2. Install Dependencies

**Using uv (Recommended):**

```bash
pip install uv
uv venv
uv sync
```

**Using pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Download Models

```bash
uv run python -m src.download_weights
```

This downloads:

| Model | Description | Size |
|-------|-------------|------|
| `best.pt` | YOLO car + plate detection | ~50MB |
| `yolo11m_car_plate_ocr.pt` | YOLO Arabic OCR | ~40MB |
| `reid/net.pth` | ResNet50-IBN ReID | ~100MB |

### 4. Run the App

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
NilePlateID/
├── app.py                      # Streamlit entry point
├── .streamlit/config.toml      # Theme configuration
├── src/
│   ├── app_pages/              # Streamlit pages
│   ├── pipeline/               # Detection & OCR logic
│   ├── reid/                   # ReID system
│   └── cli.py                  # CLI entrypoint
├── models/                     # Model weights
├── data/                       # Generated outputs
└── docs/                       # Documentation
```

## Next Steps

- [Architecture Overview](architecture.md)
- [Classical CV Methods](classical.md)
- [YOLO Detection](yolo.md)
- [Vehicle ReID](reid.md)
