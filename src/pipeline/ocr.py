"""OCR helpers using EasyOCR."""

from __future__ import annotations

import logging

import cv2
import easyocr
import numpy as np

LOGGER = logging.getLogger(__name__)
_READER: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        LOGGER.info("Initializing EasyOCR reader")
        _READER = easyocr.Reader(["ar", "en"])
    return _READER


def _preprocess(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    resized = cv2.resize(crop, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return denoised


def read_plate_text(crop: np.ndarray) -> tuple[str, float]:
    """Return raw text and mean confidence from an image crop."""
    if crop.size == 0:
        return "", 0.0

    processed = _preprocess(crop)
    reader = _get_reader()
    results = reader.readtext(processed)
    if not results:
        return "", 0.0

    texts: list[str] = []
    confs: list[float] = []
    for _, text, conf in results:
        texts.append(str(text))
        confs.append(float(conf))

    raw_text = " ".join(texts).strip()
    mean_conf = sum(confs) / len(confs)
    return raw_text, mean_conf
