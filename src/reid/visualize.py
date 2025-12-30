"""ReID visualization helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

MATCH_COLOR = (0, 200, 0)
SKIP_COLOR = (0, 0, 255)
BOX_THICKNESS = 2
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1
SCORE_THRESHOLD = 0.75
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_THICKNESS = 2


def _to_int_bbox(bbox_xyxy: Iterable[float]) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return int(x_min), int(y_min), int(x_max), int(y_max)


def draw_reid_debug(
    image: np.ndarray,
    matches: list[dict[str, object]],
    plate_id: str,
) -> np.ndarray:
    """Draw ReID match scores on a copy of the image."""
    annotated = image.copy()

    for match in matches:
        bbox = match.get("bbox_xyxy")
        if not bbox:
            continue
        try:
            score = float(match.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        kept = bool(match.get("kept", False))
        color = MATCH_COLOR if score >= SCORE_THRESHOLD else SKIP_COLOR
        label = f"{plate_id} {score:.3f}"
        x_min, y_min, x_max, y_max = _to_int_bbox(bbox)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, BOX_THICKNESS)
        origin = (x_min, max(0, y_min - 6))
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            OUTLINE_COLOR,
            OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return annotated





