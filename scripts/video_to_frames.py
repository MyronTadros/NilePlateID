"""Extract video frames for the pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.video_frames import extract_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for pipeline ingestion."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for extracted frames.",
    )
    parser.add_argument(
        "--require_ocr",
        action="store_true",
        help="Only keep frames where a plate is readable by OCR.",
    )
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="YOLO weights for plate detection (repo-relative by default).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for plate detection.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for plate detection.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Padding fraction applied to plate crops.",
    )
    
    parser.add_argument(
        "--ocr_backend",
        choices=["easyocr", "yolo"],
        default="yolo",
        help="OCR backend to use for filtering.",
    )
    parser.add_argument(
        "--ocr_weights",
        default="models/yolo11m_car_plate_ocr.pt",
        help="YOLO OCR weights (repo-relative by default).",
    )
    parser.add_argument(
        "--ocr_det_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO OCR.",
    )
    parser.add_argument(
        "--ocr_det_iou",
        type=float,
        default=0.45,
        help="IoU threshold for YOLO OCR.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string (e.g., 'cpu', '0').",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target output FPS.",
    )
    group.add_argument(
        "--every_n_frames",
        type=int,
        default=None,
        help="Extract every Nth frame.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = parser.parse_args()

    return extract_frames(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        fps=args.fps,
        every_n_frames=args.every_n_frames,
        force=args.force,
        require_ocr=args.require_ocr,
        weights_path=Path(args.weights),
        conf=args.conf,
        iou=args.iou,
        pad=args.pad,
        device=args.device,
        ocr_min_conf=args.ocr_min_conf,
        ocr_backend=args.ocr_backend,
        ocr_weights=Path(args.ocr_weights),
        ocr_det_conf=args.ocr_det_conf,
        ocr_det_iou=args.ocr_det_iou,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
