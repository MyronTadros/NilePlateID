# API Reference

Command-line interface for batch processing.

## Installation

```bash
pip install uv
uv sync
```

## Commands

### Run Full Pipeline

Process images through detection + OCR:

```bash
uv run python -m src.cli run \
    --weights models/best.pt \
    --input data/incoming \
    --gallery data/gallery \
    --plates data/plates \
    --index data/meta/index.csv \
    --conf 0.25 \
    --iou 0.45 \
    --pad 0.05 \
    --ocr_min_conf 0.3
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `models/best.pt` | YOLO detection weights |
| `--input` | required | Input images directory |
| `--gallery` | `data/gallery` | Output car crops |
| `--plates` | `data/plates` | Output plate crops |
| `--index` | `data/meta/index.csv` | Output index file |
| `--conf` | 0.25 | Detection confidence |
| `--iou` | 0.45 | NMS IoU threshold |
| `--pad` | 0.05 | Crop padding fraction |
| `--ocr_min_conf` | 0.3 | Min OCR confidence |

---

### Detect Only

Run detection without OCR:

```bash
uv run python -m src.cli detect \
    --weights models/best.pt \
    --input data/incoming \
    --out data/meta/detections.json
```

---

### Video Frames

Extract frames from video:

```bash
uv run python -m src.cli video-frames \
    --video path/to/video.mp4 \
    --out_dir data/incoming_video/video \
    --fps 2
```

With OCR filtering (keep only readable plates):

```bash
uv run python -m src.cli video-frames \
    --video path/to/video.mp4 \
    --out_dir data/incoming_video/video \
    --fps 2 \
    --require_ocr \
    --weights models/best.pt \
    --ocr_min_conf 0.05
```

---

### Build ReID Index

Create gallery embeddings:

```bash
uv run python -m src.cli reid-index \
    --gallery_dir data/gallery \
    --reid_opts models/reid/opts.yaml \
    --reid_ckpt models/reid/net.pth \
    --index_dir data/meta/reid
```

---

### ReID Search

Search for a vehicle:

```bash
uv run python -m src.cli reid-search \
    --plate_id ABC123 \
    --input_dir data/incoming \
    --index_dir data/meta/reid \
    --min_score 0.6
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--plate_id` | required | Target plate ID |
| `--input_dir` | `data/incoming` | Query images |
| `--index_dir` | `data/meta/reid` | Gallery index |
| `--min_score` | 0.0 | Min cosine similarity |
| `--top_k` | 5 | Max matches per image |

---

### Clean Artifacts

Remove generated files:

```bash
# Dry run (preview)
uv run python -m src.cli clean --dry_run

# Force clean
uv run python -m src.cli clean --force
```

---

### Download Models

```bash
uv run python -m src.download_weights
```

Options:

| Flag | Description |
|------|-------------|
| `--force` | Overwrite existing |
| `--strict` | Fail on any error |

## Output Structure

```
data/
├── gallery/{plate_id}/     # Car crops indexed by plate
├── plates/{plate_id}/      # Plate crops
├── meta/
│   ├── index.csv           # Detection index
│   ├── detections.json     # Raw detections
│   ├── debug/              # Annotated images
│   └── reid/
│       ├── index.npz       # Gallery embeddings
│       ├── results.csv     # Search results
│       └── annotated/      # Match visualizations
```
