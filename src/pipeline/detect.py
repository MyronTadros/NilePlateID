"""Detection utilities using Ultralytics YOLO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)):
        return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)
    return str(class_id)


def run_detection(
    weights_path: str | Path,
    image_paths: Iterable[str | Path],
    conf: float,
    iou: float,
    device: str | None,
) -> list[dict[str, object]]:
    """Run detection and return per-image detections."""
    ordered_paths = sorted((Path(path) for path in image_paths), key=lambda p: p.name)
    LOGGER.info("Loading weights: %s", weights_path)
    model = YOLO(str(weights_path))
    names = getattr(model, "names", {})

    LOGGER.info("Running prediction on %d image(s)", len(ordered_paths))
    predict_kwargs = {
        "source": [str(path) for path in ordered_paths],
        "conf": conf,
        "iou": iou,
        "save": False,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    detections: list[dict[str, object]] = []

    for path, result in zip(ordered_paths, results, strict=False):
        per_image: list[dict[str, object]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            cls_ids = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            for bbox, cls_id, score in zip(xyxy, cls_ids, confs, strict=False):
                class_id = int(cls_id)
                per_image.append(
                    {
                        "bbox_xyxy": [float(coord) for coord in bbox],
                        "cls_id": class_id,
                        "cls_name": _class_name(names, class_id),
                        "conf": float(score),
                    }
                )

        detections.append(
            {
                "image_path": str(path),
                "detections": per_image,
            }
        )

    return detections
