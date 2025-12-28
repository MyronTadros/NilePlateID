"""Cropping helpers for detection outputs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def validate_bbox(
    bbox_xyxy: Iterable[float],
    image_shape: Iterable[int],
    min_size: float = 2.0,
) -> tuple[bool, str]:
    """Validate a bbox against image bounds and a minimum size."""
    height, width = int(image_shape[0]), int(image_shape[1])
    try:
        x_min, y_min, x_max, y_max = (float(value) for value in bbox_xyxy)
    except (TypeError, ValueError):
        return False, "non_finite"

    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return False, "non_finite"

    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))

    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
        return False, "too_small"

    if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
        return False, "out_of_bounds"

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        return True, "clamped"

    return True, "ok"


def clamp_bbox(
    bbox_xyxy: Iterable[float],
    image_shape: Iterable[int],
    pad: float = 0.05,
) -> tuple[int, int, int, int]:
    """Clamp a bbox to image bounds and apply padding."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x_min, y_min, x_max, y_max = (float(value) for value in bbox_xyxy)

    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad_x = box_w * pad
    pad_y = box_h * pad

    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(width), x_max + pad_x)
    y_max = min(float(height), y_max + pad_y)

    x_min_i = max(0, int(math.floor(x_min)))
    y_min_i = max(0, int(math.floor(y_min)))
    x_max_i = min(width, int(math.ceil(x_max)))
    y_max_i = min(height, int(math.ceil(y_max)))

    if width > 0 and x_max_i <= x_min_i:
        x_min_i = max(0, min(x_min_i, width - 1))
        x_max_i = min(width, x_min_i + 1)
    if height > 0 and y_max_i <= y_min_i:
        y_min_i = max(0, min(y_min_i, height - 1))
        y_max_i = min(height, y_min_i + 1)

    return x_min_i, y_min_i, x_max_i, y_max_i


def crop_image(
    image: np.ndarray,
    bbox_xyxy: Iterable[float],
    pad: float = 0.05,
) -> np.ndarray:
    """Return a cropped image region using a clamped bbox."""
    x_min, y_min, x_max, y_max = clamp_bbox(bbox_xyxy, image.shape, pad=pad)
    return image[y_min:y_max, x_min:x_max]
