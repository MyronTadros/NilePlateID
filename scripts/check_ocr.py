"""Smoke-test OCR by detecting a plate and reading its text."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.crop import crop_image
from src.pipeline.detect import run_detection
from src.pipeline.normalize import normalize_plate_id
from src.pipeline.ocr import read_plate_text

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a plate, crop it, and run EasyOCR."
    )
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to YOLO weights (repo-relative by default).",
    )
    parser.add_argument(
        "--input_dir",
        default="data/incoming",
        help="Directory to read images from (repo-relative by default).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string (e.g., 'cpu', '0').",
    )
    return parser.parse_args()


def _pick_first_image(input_dir: Path) -> Path | None:
    if not input_dir.is_dir():
        return None
    candidates = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[0]


def _select_plate(detections: list[dict[str, object]]) -> dict[str, object] | None:
    plates = [
        det
        for det in detections
        if "plate" in str(det.get("cls_name", "")).lower()
    ]
    if not plates:
        return None
    return max(plates, key=lambda det: float(det.get("conf", 0.0)))


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    input_dir = Path(args.input_dir)

    if not weights_path.is_file():
        LOGGER.error("Weights not found: %s", weights_path)
        return 2

    image_path = _pick_first_image(input_dir)
    if image_path is None:
        LOGGER.error("No images found in: %s", input_dir)
        return 2

    LOGGER.info("Reading image: %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.error("Failed to read image: %s", image_path)
        return 2

    LOGGER.info("Running detection")
    results = run_detection(
        weights_path=weights_path,
        image_paths=[image_path],
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    per_image = results[0].get("detections", []) if results else []
    plate = _select_plate(list(per_image))
    if plate is None:
        LOGGER.error("No plate detections found.")
        return 2

    crop = crop_image(image, plate["bbox_xyxy"])
    if crop.size == 0:
        LOGGER.error("Empty crop produced.")
        return 2

    raw_text, mean_conf = read_plate_text(crop)
    normalized = normalize_plate_id(raw_text)
    print(f"raw_text: {raw_text}")
    print(f"normalized: {normalized}")
    print(f"mean_conf: {mean_conf:.4f}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
