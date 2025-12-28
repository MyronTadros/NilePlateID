"""Association utilities for matching plates to cars."""

from __future__ import annotations

from typing import Iterable


def _center_xy(bbox_xyxy: Iterable[float]) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _contains_point(bbox_xyxy: Iterable[float], point: tuple[float, float]) -> bool:
    x_min, y_min, x_max, y_max = bbox_xyxy
    x, y = point
    return x_min <= x <= x_max and y_min <= y <= y_max


def _iou(bbox_a: Iterable[float], bbox_b: Iterable[float]) -> float:
    ax_min, ay_min, ax_max, ay_max = bbox_a
    bx_min, by_min, bx_max, by_max = bbox_b

    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def match_plates_to_cars(
    cars: list[dict[str, object]],
    plates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Match plates to cars by center-in-box rule with IoU tie-break."""
    matches: list[dict[str, object]] = []

    for plate in plates:
        plate_bbox = plate.get("bbox_xyxy")
        if not plate_bbox:
            continue
        center = _center_xy(plate_bbox)
        best_car: dict[str, object] | None = None
        best_iou = 0.0

        for car in cars:
            car_bbox = car.get("bbox_xyxy")
            if not car_bbox:
                continue
            if not _contains_point(car_bbox, center):
                continue
            score = _iou(plate_bbox, car_bbox)
            if score > best_iou:
                best_iou = score
                best_car = car

        if best_car is not None:
            matches.append(
                {
                    "plate": plate,
                    "car": best_car,
                    "iou": best_iou,
                }
            )

    return matches
