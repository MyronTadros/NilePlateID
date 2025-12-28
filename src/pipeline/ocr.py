"""OCR helpers using EasyOCR."""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import easyocr
import numpy as np

LOGGER = logging.getLogger(__name__)
_READER: easyocr.Reader | None = None

ARABIC_LETTER_RANGES = [
    (0x0621, 0x064A),
    (0x066E, 0x06D3),
    (0x06FA, 0x06FC),
]
ARABIC_DIGIT_RANGES = [
    (0x0660, 0x0669),
    (0x06F0, 0x06F9),
]


def _build_allowlist(ranges: list[tuple[int, int]], extra: str = "") -> str:
    return extra + "".join(
        chr(code) for start, end in ranges for code in range(start, end + 1)
    )


ALLOWLIST_LETTERS = _build_allowlist(ARABIC_LETTER_RANGES, extra="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALLOWLIST_DIGITS = _build_allowlist(ARABIC_DIGIT_RANGES, extra="0123456789")


def _get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        LOGGER.info("Initializing EasyOCR reader")
        _READER = easyocr.Reader(["ar", "en"])
    return _READER


def _preprocess(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    height, width = gray.shape[:2]
    resized = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    return denoised


def _bbox_center_x(bbox: Iterable[Iterable[float]]) -> float:
    xs = [point[0] for point in bbox]
    return sum(xs) / len(xs)


def _read_side(
    reader: easyocr.Reader,
    image: np.ndarray,
    allowlist: str,
    rtl: bool,
) -> tuple[str, list[float]]:
    if image.size == 0:
        return "", []
    results = reader.readtext(image, allowlist=allowlist, detail=1, paragraph=False)
    if not results:
        return "", []

    ordered = sorted(results, key=lambda item: _bbox_center_x(item[0]), reverse=rtl)
    texts: list[str] = []
    confs: list[float] = []
    for _, text, conf in ordered:
        text = str(text).strip()
        if not text:
            continue
        texts.append(text)
        confs.append(float(conf))

    return "".join(texts).strip(), confs


def read_plate_text(crop: np.ndarray) -> tuple[str, float]:
    """Return raw text and mean confidence from an image crop."""
    if crop.size == 0:
        return "", 0.0

    processed = _preprocess(crop)
    height, width = processed.shape[:2]
    if height >= 4:
        processed = processed[height // 4 :, :]
    mid = max(1, processed.shape[1] // 2)
    left = processed[:, :mid]
    right = processed[:, mid:]

    reader = _get_reader()
    letters_text, letters_confs = _read_side(
        reader, right, allowlist=ALLOWLIST_LETTERS, rtl=True
    )
    digits_text, digits_confs = _read_side(
        reader, left, allowlist=ALLOWLIST_DIGITS, rtl=False
    )

    parts: list[str] = []
    if letters_text:
        parts.append(letters_text)
    if digits_text:
        parts.append(digits_text)
    raw_text = " ".join(parts).strip()

    confs = letters_confs + digits_confs
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return raw_text, mean_conf

