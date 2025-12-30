"""YOLO-based OCR for license plate crops."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.pipeline.ocr import post_process

LOGGER = logging.getLogger(__name__)

_MODEL: Optional["YOLO"] = None
_MODEL_PATH: Optional[Path] = None


def load_model(weights_path: Path) -> "YOLO":
    """Load the YOLO OCR model (cached)."""
    global _MODEL, _MODEL_PATH

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install with: pip install ultralytics")

    weights_path = Path(weights_path)
    if _MODEL is None or _MODEL_PATH != weights_path:
        if not weights_path.is_file():
            raise FileNotFoundError(f"OCR weights not found: {weights_path}")
        LOGGER.info("Loading YOLO OCR model: %s", weights_path)
        _MODEL = YOLO(str(weights_path))
        _MODEL_PATH = weights_path

    return _MODEL


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)):
        return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
    return str(class_id)


def read_plate_text(
    crop: np.ndarray,
    *,
    model: Optional["YOLO"] = None,
    weights_path: Optional[Path] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> tuple[str, float]:
    """Read plate text using a YOLO OCR model.

    Returns (text, mean_confidence_percent).
    """
    if crop.size == 0:
        return "", 0.0

    if model is None:
        if weights_path is None:
            raise ValueError("weights_path is required when model is not provided")
        model = load_model(weights_path)

    results = model.predict(
        source=crop,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
        device=device,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return "", 0.0

    boxes = results[0].boxes
    names = getattr(model, "names", {})
    xyxy = boxes.xyxy.cpu().tolist()
    cls_ids = boxes.cls.cpu().tolist()
    confs = boxes.conf.cpu().tolist()

    items = []
    for bbox, cls_id, conf_value in zip(xyxy, cls_ids, confs, strict=False):
        x_min, _y_min, x_max, _y_max = bbox
        center_x = (x_min + x_max) / 2.0
        char = _class_name(names, int(cls_id))
        items.append((center_x, char, float(conf_value)))

    items.sort(key=lambda item: item[0])
    text = "".join(char for _center, char, _conf in items)
    text = post_process(text)

    mean_conf = sum(conf for _center, _char, conf in items) / len(items)
    return text, mean_conf * 100.0
