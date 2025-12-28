"""Normalization utilities for license plate text."""

from __future__ import annotations

import re
import uuid

ARABIC_INDIC_DIGITS = {
    "\u0660": "0",
    "\u0661": "1",
    "\u0662": "2",
    "\u0663": "3",
    "\u0664": "4",
    "\u0665": "5",
    "\u0666": "6",
    "\u0667": "7",
    "\u0668": "8",
    "\u0669": "9",
    "\u06F0": "0",
    "\u06F1": "1",
    "\u06F2": "2",
    "\u06F3": "3",
    "\u06F4": "4",
    "\u06F5": "5",
    "\u06F6": "6",
    "\u06F7": "7",
    "\u06F8": "8",
    "\u06F9": "9",
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
