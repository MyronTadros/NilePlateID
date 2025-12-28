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
??? outputs/                 # optional generated outputs
??? runs/                    # Ultralytics training/inference outputs
??? scripts/                 # helper scripts (check_model, check_ocr, clean)
??? src/
?   ??? pipeline/            # detection, OCR, association, visualization
?   ??? cli.py               # main CLI entrypoint
??? README.md
??? requirements.txt
```

## Setup

Recommended: Python 3.10-3.12 (3.13 may work but is less commonly tested).

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
- Input images: place JPG/PNG files in `data/incoming/`.

## Main commands

Check model:

```powershell
python scripts/check_model.py --weights models/best.pt --input_dir data/incoming
```

Detect only (writes JSON + optional debug images):

```powershell
python -m src.cli detect --weights models/best.pt --input data/incoming --out data/meta/detections.json --conf 0.25 --iou 0.45
```

Run full pipeline (crops + OCR + index):

```powershell
python -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
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

Mac/Linux versions of the same commands:

```bash
python3 scripts/check_model.py --weights models/best.pt --input_dir data/incoming
python3 -m src.cli detect --weights models/best.pt --input data/incoming --out data/meta/detections.json --conf 0.25 --iou 0.45
python3 -m src.cli run --weights models/best.pt --input data/incoming --gallery data/gallery --plates data/plates --index data/meta/index.csv --conf 0.25 --iou 0.45 --pad 0.05 --ocr_min_conf 0.3
python3 -m src.cli clean --dry_run
python3 -m src.cli clean --force
python3 scripts/clean_and_rerun.py
```

## Outputs explained

- `data/gallery/<plate_id>/`: cropped car images associated with a plate_id.
- `data/plates/<plate_id>/`: cropped plate images associated with a plate_id.
- `data/meta/`:
  - `index.csv`: row per matched car+plate with paths, confidences, and bboxes.
  - `detections.json`: per-image detections (written in max debug mode).
  - `debug/`: annotated images with boxes and plate_id labels.
  - `crops_preview/`: sample crops (max debug mode).
  - `run_config.json`: CLI args and git commit (max debug mode).

## Troubleshooting

- No detections:
  - Check that `models/best.pt` matches your dataset classes and the images are readable.
  - Run `scripts/check_model.py` to confirm the model loads and has expected class names.
- OCR empty or low confidence:
  - Inspect plate crops in `data/plates/` and debug images in `data/meta/debug/`.
  - Adjust `--ocr_min_conf`, or improve crop padding with `--pad`.
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
