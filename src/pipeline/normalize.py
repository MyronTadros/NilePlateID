"""Normalization utilities for license plate text."""

from __future__ import annotations

import re
import uuid

ARABIC_INDIC_DIGITS = {
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}

ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u06FA-\u06FC]")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _normalize_digits(text: str) -> str:
    return "".join(ARABIC_INDIC_DIGITS.get(ch, ch) for ch in text)


def _is_allowed(char: str) -> bool:
    if "A" <= char <= "Z" or "0" <= char <= "9":
        return True
    return bool(ARABIC_LETTER_RE.match(char))


def normalize_plate_id(text: str) -> str:
    """Normalize OCR text into a clean plate identifier."""
    if not text:
        return f"unknown_{_short_uuid()}"

    cleaned = _normalize_digits(text.strip())
    cleaned = cleaned.upper()
    filtered = "".join(char for char in cleaned if _is_allowed(char))
    if not filtered:
        return f"unknown_{_short_uuid()}"
    return filtered
