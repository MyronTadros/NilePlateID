"""Video-to-frames helper."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.pipeline.crop import clamp_bbox, validate_bbox

LOGGER = logging.getLogger(__name__)


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)):
        return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
    return str(class_id)


def _frame_has_readable_plate(
    frame,
    *,
    model,
    plate_label: str,
    conf: float,
    iou: float,
    pad: float,
    ocr_min_conf: float,
    device: str | None,
    read_plate_text,
) -> bool:
    results = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
        device=device,
    )
    if not results:
        return False

    names = getattr(model, "names", {})
    label = plate_label.lower()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False

    xyxy = boxes.xyxy.cpu().tolist()
    cls_ids = boxes.cls.cpu().tolist()

    for bbox, cls_id in zip(xyxy, cls_ids, strict=False):
        cls_name = _class_name(names, int(cls_id)).lower()
        if label not in cls_name:
            continue
        ok, _reason = validate_bbox(bbox, frame.shape)
        if not ok:
            continue
        x_min, y_min, x_max, y_max = clamp_bbox(bbox, frame.shape, pad=pad)
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        raw_text, mean_conf = read_plate_text(crop)
        if raw_text.strip() and mean_conf >= ocr_min_conf:
            return True

    return False


def extract_frames(
    *,
    video_path: Path,
    out_dir: Path,
    fps: float | None,
    every_n_frames: int | None,
    force: bool,
    require_ocr: bool = False,
    weights_path: Path | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    pad: float = 0.05,
    device: str | None = None,
    ocr_min_conf: float = 0.05,
    plate_label: str = "plate",
) -> int:
    if not video_path.is_file():
        LOGGER.error("Video not found: %s", video_path)
        return 2

    if every_n_frames is not None and every_n_frames <= 0:
        LOGGER.error("--every_n_frames must be >= 1")
        return 2

    if fps is not None and fps <= 0:
        LOGGER.error("--fps must be > 0")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = next(out_dir.glob("frame_*.jpg"), None)
        if existing is not None:
            LOGGER.error("Output exists (use --force): %s", out_dir)
            return 2

    ocr_model = None
    read_plate_text = None
    if require_ocr:
        if weights_path is None:
            LOGGER.error("--weights is required when --require_ocr is set")
            return 2
        if not weights_path.is_file():
            LOGGER.error("Weights not found: %s", weights_path)
            return 2
        from ultralytics import YOLO
        from src.pipeline.ocr import read_plate_text as _read_plate_text

        LOGGER.info("Loading YOLO model for OCR filtering: %s", weights_path)
        ocr_model = YOLO(str(weights_path))
        read_plate_text = _read_plate_text
        LOGGER.info("OCR filtering enabled (min_conf=%.2f)", ocr_min_conf)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error("Failed to open video: %s", video_path)
        return 2

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if every_n_frames is None:
        if source_fps <= 0:
            step = 1
            LOGGER.warning("Video FPS unknown; defaulting to every frame.")
        else:
            target_fps = fps if fps is not None else 2.0
            step = max(1, int(round(source_fps / target_fps)))
    else:
        step = every_n_frames

    LOGGER.info("Extracting frames every %d frame(s)", step)

    frame_index = 0
    saved_count = 0
    read_count = 0
    checked_count = 0
    skipped_ocr = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_count += 1
        if frame_index % step == 0:
            checked_count += 1
            if require_ocr:
                if not _frame_has_readable_plate(
                    frame,
                    model=ocr_model,
                    plate_label=plate_label,
                    conf=conf,
                    iou=iou,
                    pad=pad,
                    ocr_min_conf=ocr_min_conf,
                    device=device,
                    read_plate_text=read_plate_text,
                ):
                    skipped_ocr += 1
                    frame_index += 1
                    continue

            output_path = out_dir / f"frame_{saved_count + 1:06d}.jpg"
            if output_path.exists() and not force:
                LOGGER.error("Output exists (use --force): %s", output_path)
                cap.release()
                return 2
            if not cv2.imwrite(str(output_path), frame):
                LOGGER.error("Failed to write frame: %s", output_path)
                cap.release()
                return 2
            saved_count += 1
        frame_index += 1

    cap.release()
    LOGGER.info("Frames saved: %d", saved_count)
    LOGGER.info("Frames read: %d", read_count)
    if require_ocr:
        LOGGER.info("Frames checked for OCR: %d", checked_count)
        LOGGER.info("Frames skipped by OCR filter: %d", skipped_ocr)
    return 0
