# YOLO License Plate Detection + EasyOCR Reading

End-to-end pipeline for car + license plate detection with Ultralytics YOLO, OCR via EasyOCR, and persistent storage of cropped cars/plates keyed by `plate_id`.

Key outputs:
- Per-plate galleries in `data/gallery/<plate_id>/`
- Plate crops in `data/plates/<plate_id>/`
- Structured metadata and debug visuals in `data/meta/`

## Pipeline (text diagram)

```
input images
  -> YOLO detect (cars + plates)
  -> associate plate to car (center-in-box, IoU tie-break)
  -> crop car + plate
  -> OCR preprocess (grayscale + 2x resize + denoise)
  -> OCR plate text
  -> normalize plate_id
  -> save crops + write index.csv + debug images
```

## Folder structure

```
.
??? data/
?   ??? incoming/            # input images (not committed)
?   ??? datasets/            # training datasets (not committed)
?   ??? gallery/             # car crops grouped by plate_id (generated)
?   ??? plates/              # plate crops grouped by plate_id (generated)
?   ??? meta/                # detections.json, index.csv, debug images (generated)
??? models/                  # YOLO weights (best.pt not committed)
?   ??? reid/                # ReID checkpoint + opts.yaml (not committed)
??? outputs/                 # optional generated outputs
??? runs/                    # Ultralytics training/inference outputs
??? scripts/                 # helper scripts (audit, check_*, clean, video)
??? src/
?   ??? pipeline/            # detection, OCR, association, visualization
?   ??? cli.py               # main CLI entrypoint
??? README.md
??? requirements.txt
```

## Setup

Recommended: Python 3.10-3.12 (3.13 may work but is less commonly tested).

### Quick Setup (Recommended)

1. **Install uv**:
   ```bash
   pip install uv
   ```

2. **Create virtual environment**:
   ```bash
   uv venv
   ```

3. **Sync environment**:
   ```bash
   uv sync
   ```

4. **Download models**:
   ```bash
   uv run python scripts/download_weights.py
   ```

### Manual Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Required files

- `models/best.pt`: YOLO weights file. This is not committed. Train a model or download from your experiment runs and place it here.
- `models/reid/opts.yaml` + `models/reid/net.pth`: ReID model files (not committed). Put the exported opts.yaml and checkpoint here.
- Input images: place JPG/PNG files in `data/incoming/`.

## Main commands

Check model:

```powershell
python scripts/check_model.py --weights models/best.pt --input_dir data/incoming
```

Check ReID model (loads model + embeds one image):

```powershell
python scripts/check_reid_model.py --reid_ckpt models/reid/net.pth --reid_opts models/reid/opts.yaml --image data/incoming/frame_000015.jpg
```


Detect only (writes JSON + optional debug images):

```powershell
python -m src.cli detect --weights models/best.pt --input data/incoming --out data/meta/detections.json --conf 0.25 --iou 0.45
```

Extract frames from a video:

```powershell
python -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2
```

Filter to frames with readable plates (OCR):

```powershell
python -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2 --require_ocr --weights models/best.pt --ocr_min_conf 0.05
```

Run full pipeline (crops + OCR + index):

ReID search can point to video frame folders (e.g., `--input_dir data/incoming_video/<video_name>`).

```powershell
python -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
```

Build ReID index (gallery embeddings):

```powershell
python -m src.cli reid-index --gallery_dir data/gallery --reid_opts models/reid/opts.yaml --reid_ckpt models/reid/net.pth --index_dir data/meta/reid
```

Search by plate_id using cached index:

```powershell
python -m src.cli reid-search --plate_id ABC123 --input_dir data/incoming --index_dir data/meta/reid
```

Convenience wrapper (builds index if missing):

```powershell
python -m src.cli reid --plate_id ABC123 --input data/incoming
```

Clean generated artifacts:

```powershell
python -m src.cli clean --dry_run
python -m src.cli clean --force
```

Clean and rerun in max debug mode:

```powershell
python scripts/clean_and_rerun.py
```

## Workflow flows (exact commands)

1) Detect only (detections JSON + optional debug):

```powershell
python -m src.cli detect --weights models/best.pt --input data/incoming --out data/meta/detections.json --conf 0.25 --iou 0.45
```

2) Build gallery + OCR from images in `data/incoming/`:

```powershell
python -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
```

3) Build ReID index from gallery crops:

```powershell
python -m src.cli reid-index --gallery_dir data/gallery --reid_opts models/reid/opts.yaml --reid_ckpt models/reid/net.pth --index_dir data/meta/reid
```

4) Search for a plate_id in a new image folder:

```powershell
python -m src.cli reid-search --plate_id ABC123 --input_dir data/incoming --index_dir data/meta/reid
```

5) One-shot convenience (builds index if missing, then searches):

```powershell
python -m src.cli reid --plate_id ABC123 --input data/incoming
```

6) Video -> frames -> (optional) search:

```powershell
python -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2
python -m src.cli reid-search --plate_id ABC123 --input_dir data/incoming_video/video --index_dir data/meta/reid
```

7) Video -> frames filtered by OCR readability:

```powershell
python -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2 --require_ocr --weights models/best.pt --ocr_min_conf 0.05
```

8) Reset outputs and rebuild:

```powershell
python -m src.cli clean --force
python -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
python -m src.cli reid-index --gallery_dir data/gallery --reid_opts models/reid/opts.yaml --reid_ckpt models/reid/net.pth --index_dir data/meta/reid
```

## CLI command reference (video + ReID)

`video-frames`: extract frames from a video (optionally keep only OCR-readable plates).

- `--video` (required): input video path
- `--out_dir` (required): output folder for extracted frames
- `--fps` (default: 2.0): target output FPS (mutually exclusive with `--every_n_frames`)
- `--every_n_frames` (default: unset): keep every Nth frame (mutually exclusive with `--fps`)
- `--require_ocr` (default: false): only save frames with readable plates
- `--weights` (default: `models/best.pt`): YOLO weights for plate detection
- `--conf` (default: 0.25): plate detection confidence threshold
- `--iou` (default: 0.45): plate detection IoU threshold
- `--pad` (default: 0.05): padding fraction applied to plate crops
- `--ocr_min_conf` (default: 0.05): minimum OCR confidence to keep a frame
- `--device` (default: unset): Ultralytics device string (e.g., `cpu`, `0`)
- `--force` (default: false): overwrite existing outputs

`reid-index`: build cached gallery embeddings + centroids.

- `--gallery_dir` (default: `data/gallery`): gallery car crops
- `--reid_opts` (default: `models/reid/opts.yaml`): ReID opts.yaml
- `--reid_ckpt` (default: `models/reid/net.pth`): ReID checkpoint
- `--index_dir` (default: `data/meta/reid`): output directory for index
- `--input_size` (default: 224): ReID input size
- `--batch_size` (default: 32): ReID embedding batch size
- `--reid_device` (default: unset): torch device for ReID (e.g., `cpu`, `cuda`, `0`)
- `--force` (default: false): overwrite existing index

`reid-search`: search query frames by `plate_id` using the cached index.

- `--plate_id` (required): target plate id
- `--weights` (default: `models/best.pt`): YOLO weights for car detection
- `--input_dir` (default: `data/incoming`): query images folder
- `--index_dir` (default: `data/meta/reid`): cached index folder
- `--debug_dir` (default: `data/meta/reid`): output folder for results/annotated images
- `--conf` (default: 0.25): car detection confidence threshold
- `--iou` (default: 0.45): car detection IoU threshold
- `--pad` (default: 0.05): padding fraction applied to car crops
- `--input_size` (default: 224): ReID input size
- `--batch_size` (default: 32): ReID embedding batch size
- `--min_score` (default: 0.0): minimum cosine similarity to keep a match
- `--top_k` (default: 5): keep top-k matches per image (0 keeps all)
- `--device` (default: unset): Ultralytics device string (e.g., `cpu`, `0`)
- `--reid_device` (default: unset): torch device for ReID (e.g., `cpu`, `cuda`, `0`)
- `--car_label` (default: `car`): substring used to identify car class names
- `--force` (default: false): overwrite existing outputs

Mac/Linux versions of the same commands:

```bash
python3 scripts/check_model.py --weights models/best.pt --input_dir data/incoming
python3 scripts/check_reid_model.py --reid_ckpt models/reid/net.pth --reid_opts models/reid/opts.yaml --image data/incoming/sample.jpg
python3 -m src.cli detect --weights models/best.pt --input data/incoming --out data/meta/detections.json --conf 0.25 --iou 0.45
python3 -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2
python3 -m src.cli video-frames --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2 --require_ocr --weights models/best.pt --ocr_min_conf 0.05
python3 -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
python3 -m src.cli reid-index --gallery_dir data/gallery --reid_opts models/reid/opts.yaml --reid_ckpt models/reid/net.pth --index_dir data/meta/reid
python3 -m src.cli reid-search --plate_id ABC123 --input_dir data/incoming --index_dir data/meta/reid
python3 -m src.cli reid --plate_id ABC123 --input data/incoming
python3 -m src.cli clean --dry_run
python3 -m src.cli clean --force
python3 scripts/clean_and_rerun.py
```

## Helper scripts

- `scripts/check_model.py`: YOLO smoke test with class names and optional preview (`python scripts/check_model.py --weights models/best.pt --input_dir data/incoming`).
- `scripts/check_ocr.py`: detect + crop one plate and run OCR (`python scripts/check_ocr.py --weights models/best.pt --input_dir data/incoming`).
- `scripts/check_reid_model.py`: load ReID model and embed one image (`python scripts/check_reid_model.py --reid_ckpt models/reid/net.pth --reid_opts models/reid/opts.yaml --image data/incoming/sample.jpg`).
- `scripts/audit_vehicle_reid_vendor.py`: verify vendor imports (`python scripts/audit_vehicle_reid_vendor.py`).
- `scripts/video_to_frames.py`: extract frames from a video (`python scripts/video_to_frames.py --video path/to/video.mp4 --out_dir data/incoming_video/video --fps 2`), add `--require_ocr` to keep readable plates only.
- `scripts/clean_artifacts.py`: safe cleanup dry-run/force (`python scripts/clean_artifacts.py --dry_run`).
- `scripts/clean_and_rerun.py`: clean and rerun with max debug (`python scripts/clean_and_rerun.py`).

## Outputs explained

- `data/gallery/<plate_id>/`: cropped car images associated with a plate_id.
- `data/plates/<plate_id>/`: cropped plate images associated with a plate_id.
- `data/meta/`:
  - `index.csv`: row per matched car+plate with paths, confidences, and bboxes.
  - `detections.json`: per-image detections (written in max debug mode).
  - `debug/`: annotated images with boxes and plate_id labels.
  - `crops_preview/`: sample crops (max debug mode).
  - `run_config.json`: CLI args and git commit (max debug mode).
- `data/meta/reid/`:
  - `index.npz`, `index.csv`: cached gallery embeddings + centroids.
  - `results.csv`, `results.json`: search outputs from `reid-search`.
  - `annotated/`: debug images with match scores.

## Troubleshooting

- No detections:
  - Check that `models/best.pt` matches your dataset classes and the images are readable.
  - Run `scripts/check_model.py` to confirm the model loads and has expected class names.
- OCR empty or low confidence:
  - Inspect plate crops in `data/plates/` and debug images in `data/meta/debug/`.
  - Adjust `--ocr_min_conf`, or improve crop padding with `--pad`.
  - OCR preprocessing uses grayscale + 2x resize + Gaussian blur before reading.
- GPU vs CPU:
  - EasyOCR is much faster with a GPU. CPU is supported but slower.
- Class name mismatch:
  - The pipeline expects class names containing "car" and "plate". If your model uses different labels, update the mapping in `src/cli.py` (`_split_by_role`).

## What's next

- Integrate vehicle ReID to connect plates across cameras or time. This can be added as a downstream step after cropping.

## Notes

- Ultralytics docs: https://docs.ultralytics.com/
- Ultralytics dataset format: https://docs.ultralytics.com/datasets/
- EasyOCR docs: https://www.jaided.ai/easyocr/documentation/
- EasyOCR GitHub: https://github.com/JaidedAI/EasyOCR

## License

MIT. See `LICENSE`.
