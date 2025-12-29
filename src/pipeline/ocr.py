"""OCR module for license plate text recognition using EasyOCR."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

from src.pipeline.enhancement import enhance_plate_crop

LOGGER = logging.getLogger(__name__)

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


def _preprocess(crop: np.ndarray, use_enhancement: bool = False, crop_number: int = 1) -> np.ndarray:
    """Preprocess cropped plate: resize, grayscale, denoise.
    
    Args:
        crop: Input BGR image crop
        use_enhancement: If True, apply enhanced preprocessing pipeline
        crop_number: Crop number for saving debug files
        
    Returns:
        Preprocessed grayscale image ready for OCR
    """
    if use_enhancement:
        enhanced = enhance_plate_crop(crop, crop_number=crop_number)
        if enhanced.size > 0:
            height, width = enhanced.shape[:2]
            scale = 4
            resized = cv2.resize(enhanced, (width * scale, height * scale), 
                               interpolation=cv2.INTER_CUBIC)
            return resized
    
    # Standard preprocessing: resize, grayscale, denoise
    height, width = crop.shape[:2]
    scale = 4
    resized = cv2.resize(crop, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return denoised


def _bbox_center_x(bbox: Iterable[Iterable[float]]) -> float:
    xs = [point[0] for point in bbox]
    return sum(xs) / len(xs)


_reader = None

def _get_reader():
    """Get or create EasyOCR reader instance (singleton pattern)."""
    global _reader
    if _reader is None:
        if easyocr is None:
            LOGGER.error("EasyOCR not installed. Install with: pip install easyocr")
            return None
        # Use Arabic language for OCR
        _reader = easyocr.Reader(['ar'], gpu=False)
    return _reader


def _read_with_easyocr(image: np.ndarray) -> tuple[str, float]:
    """Use EasyOCR for Arabic text recognition.
    
    Args:
        image: Grayscale or binary image
        
    Returns:
        Tuple of (text, confidence_percentage)
    """
    if image.size == 0:
        return "", 0.0
    
    reader = _get_reader()
    if reader is None:
        return "", 0.0
    
    try:
        results = reader.readtext(image, detail=1)
        
        texts = []
        confs = []
        
        for (bbox, text, confidence) in results:
            if text.strip():
                texts.append(text.strip())
                confs.append(confidence * 100)
        
        combined_text = ' '.join(texts)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        
        return combined_text, mean_conf
        
    except Exception as e:
        LOGGER.error(f"EasyOCR error: {e}")
        return "", 0.0


def post_process(text: str) -> str:
    """Remove OCR artifacts and special characters from text.
    
    Args:
        text: Raw OCR output
        
    Returns:
        Cleaned text with artifacts removed
    """
    text = text.strip()
    
    # Remove common OCR artifacts and special characters
    bad_chars = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', 
                 '[', ']', '{', '}', '|', '\\', ';', ':', '"', '<', '>', ',', '.', '?', '/', 
                 '°', '·', '•', '●', '○', '◦', '‣', '⁃', '※', '❖', '✓', '✔', '✗', '✘', '☐', '☑', 
                 '☒', '★', '☆', '♠', '♣', '♥', '♦', '♤', '♧', '♡', '♢', '▪', '▫', '◊', '◘', '◙', 
                 '▀', '▄', '█', '▌', '▐', '░', '▒', '▓', '■', '□', '▢', '▣', '▤', '▥', '▦', '▧', 
                 '▨', '▩', '▬', '▭', '▮', '▯', '▰', '▱', '◆', '◇', '◈', '◉', '◍', '◎', '●', '◐', 
                 '◑', '◒', '◓', '◔', '◕', '◖', '◗', '❍', '￮', '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', 
                 '⊜', '⊝', '⊞', '⊟', '⊠', '⊡', '⋄', '⋅', '∙', '·', '・', '∘', '○', '◦', '●', '•', 
                 '‣', '⁃', '∗', '∵', '∴', '∷', '∸', '∹', '∺', '∻', '∼', '∽', '∾', '∿', '≀', '≁', 
                 '≂', '≃', '≄', '≅', '≆', '≇', '≈', '≉', '≊', '≋', '≌', '≍', '≎', '≏', '≐', '≑', 
                 '≒', '≓', '≔', '≕', '≖', '≗', '≘', '≙', '≚', '≛', '≜', '≝', '≞', '≟', '≠', '≡', 
                 '≢', '≣', '≤', '≥', '≦', '≧', '≨', '≩', '⊂', '⊃', '⊄', '⊅', '⊆', '⊇', '⊈', '⊉', 
                 '⊊', '⊋', 'Ã', '€', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', 'Ž', ''', 
                 ''', '"', '"', '•', '–', '—', '˜', '™', 'š', '›', 'œ', 'ž', 'Ÿ', '¡', '¢', '£', '¤', 
                 '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', 
                 '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', '÷', 'Ø', '×', 'ø', '√', '∞', '─', '│', 
                 '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', 
                 '╦', '╩', '╬', '▼', '►', '◄', '↑', '↓', '→', '←', '↔', '↕', '▲', '△', '▴', '▵', '▶', 
                 '▷', '▸', '▹', '►', '▻', '▼', '▽', '▾', '▿', '◀', '◁', '◂', '◃', '◄', '◅', '、', '。', 
                 '〈', '〉', '《', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', '〖', '〗', '！', '（', 
                 '）', '，', '：', '；', '？']

    for char in bad_chars:
        if char in text:
            text = text.replace(char, '')
    
    return text


def recognise_easyocr(src_path: str) -> str:
    """Use EasyOCR to read text from plate image."""
    try:
        image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ""
        text, _ = _read_with_easyocr(image)
        return text
    except Exception as e:
        LOGGER.error(f"EasyOCR error: {e}")
        return ""


def read_plate_text(
    crop: np.ndarray, 
    use_enhancement: bool = False,
    crop_number: int = 1
) -> tuple[str, float]:
    """Read text from plate crop with optional enhanced preprocessing.
    
    Args:
        crop: Input plate crop (BGR image)
        use_enhancement: If True, use enhanced preprocessing and save steps to temp/steps/
        crop_number: Number for this crop (e.g., 1 for crop1.txt)
        
    Returns:
        Tuple of (text, confidence) where text is the recognized plate text
        and confidence is the mean OCR confidence
    """
    if crop.size == 0:
        return "", 0.0

    # Preprocess: resize, grayscale/threshold (saves steps if use_enhancement=True)
    processed = _preprocess(crop, use_enhancement=use_enhancement, crop_number=crop_number)
    
    # Remove top 35% to eliminate decorative text
    height, width = processed.shape[:2]
    if height >= 3:
        crop_amount = int(height * 0.35)
        processed = processed[crop_amount:, :]
    
    # Run EasyOCR
    raw_text, mean_conf = _read_with_easyocr(processed)
    
    # Clean artifacts
    raw_text = post_process(raw_text)
    
    # Save OCR result to file
    if crop_number > 0:
        Path("temp").mkdir(parents=True, exist_ok=True)
        with open(f"temp/crop{crop_number}.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
    
    return raw_text, mean_conf
