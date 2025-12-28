"""Visual debugging helpers for detections."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

CAR_COLOR = (255, 128, 0)
PLATE_COLOR = (0, 200, 0)
MATCH_COLOR = (0, 0, 255)
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
BOX_THICKNESS = 2


def _to_int_bbox(bbox_xyxy: Iterable[float]) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _draw_box(
    image: np.ndarray,
    bbox_xyxy: Iterable[float],
    color: tuple[int, int, int],
) -> None:
    x_min, y_min, x_max, y_max = _to_int_bbox(bbox_xyxy)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, BOX_THICKNESS)


def _draw_label(
    image: np.ndarray,
    bbox_xyxy: Iterable[float],
    text: str,
    color: tuple[int, int, int],
) -> None:
    x_min, y_min, _, _ = _to_int_bbox(bbox_xyxy)
    origin = (x_min, max(0, y_min - 6))
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        color,
        TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def draw_debug(
    image: np.ndarray,
    cars: list[dict[str, object]],
    plates: list[dict[str, object]],
    matches: list[dict[str, object]],
) -> np.ndarray:
    """Draw cars, plates, and matched pair labels on a copy of the image."""
    annotated = image.copy()

    for car in cars:
        bbox = car.get("bbox_xyxy")
        if bbox:
            _draw_box(annotated, bbox, CAR_COLOR)

    for plate in plates:
        bbox = plate.get("bbox_xyxy")
        if bbox:
            _draw_box(annotated, bbox, PLATE_COLOR)

    for index, match in enumerate(matches):
        plate_bbox = match.get("plate", {}).get("bbox_xyxy")
        car_bbox = match.get("car", {}).get("bbox_xyxy")
        label = f"pair {index}"
        if plate_bbox:
            _draw_label(annotated, plate_bbox, label, MATCH_COLOR)
        if car_bbox:
            _draw_label(annotated, car_bbox, label, MATCH_COLOR)

    return annotated


def draw_plate_id_debug(
    image: np.ndarray,
    cars: list[dict[str, object]],
    plates: list[dict[str, object]],
    matches: list[dict[str, object]],
) -> np.ndarray:
    """Draw cars, plates, and matched plate IDs on a copy of the image."""
    annotated = image.copy()

    for car in cars:
        bbox = car.get("bbox_xyxy")
        if bbox:
            _draw_box(annotated, bbox, CAR_COLOR)

    for plate in plates:
        bbox = plate.get("bbox_xyxy")
        if bbox:
            _draw_box(annotated, bbox, PLATE_COLOR)

    for match in matches:
        plate_bbox = match.get("plate", {}).get("bbox_xyxy")
        plate_id = str(match.get("plate_id", "unknown"))
        if plate_bbox:
            _draw_label(annotated, plate_bbox, plate_id, MATCH_COLOR)

    return annotated
