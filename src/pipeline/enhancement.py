"""Image enhancement and preprocessing for license plates.

This module contains functions for:
- Perspective correction (four-point transform)
- Automatic brightness and contrast adjustment
- Color-based detection helpers
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points in order: top-left, top-right, bottom-right, bottom-left."""
    result = np.zeros((4, 2), dtype="float32")

    # top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]
    result[2] = pts[np.argmax(s)]

    # top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(diff)]
    result[3] = pts[np.argmax(diff)]

    return result


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the plate.
    
    Args:
        image: Input image
        pts: Four corner points of the license plate region
        
    Returns:
        Warped image with corrected perspective
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def automatic_brightness_and_contrast(
    image: np.ndarray, 
    clip_hist_percent: float = 10
) -> tuple[np.ndarray, float, float]:
    """Automatically adjust brightness and contrast using histogram clipping.
    
    Args:
        image: Input BGR image
        clip_hist_percent: Percentage of histogram to clip (default 10)
        
    Returns:
        Tuple of (adjusted_image, alpha, beta) where alpha is contrast 
        and beta is brightness adjustment
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    acc = []
    acc.append(float(hist[0]))
    for index in range(1, hist_size):
        acc.append(acc[index - 1] + float(hist[index]))

    maximum = acc[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while acc[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while acc[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result, alpha, beta


def enhance_plate_crop(crop: np.ndarray, crop_number: int = 1) -> np.ndarray:
    """Enhanced preprocessing for license plate crops with step saving.
    
    Applies brightness/contrast adjustment, upscaling, grayscaling, and thresholding.
    Saves intermediate steps to temp/steps/ for debugging.
    
    Args:
        crop: Input BGR image of license plate crop
        crop_number: Number for this crop (e.g., 1 for crop1)
        
    Returns:
        Binary thresholded image ready for OCR
    """
    if crop.size == 0:
        return crop
    
    steps_dir = Path("temp/steps")
    steps_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(f"temp/crop_original_{crop_number}.jpg", crop)
    cv2.imwrite(f"temp/steps/crop{crop_number}_3_detected_plate.png", crop)
    
    adjusted, alpha, beta = automatic_brightness_and_contrast(crop)
    LOGGER.debug(f"Brightness/contrast adjusted: alpha={alpha:.2f}, beta={beta:.2f}")
    cv2.imwrite(f"temp/steps/crop{crop_number}_4_brightness_contrast_adjustment.png", adjusted)
    
    upscaled = upscale_image(adjusted, scale_factor=2)
    cv2.imwrite(f"temp/steps/crop{crop_number}_4.5_upscaled.png", upscaled)
    
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"temp/steps/crop{crop_number}_5_gray.png", gray)
    
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"temp/steps/crop{crop_number}_6_threshold.png", th)
    
    cv2.imwrite(f"temp/crop{crop_number}.jpg", th)
    
    return th


def detect_blue_region(image: np.ndarray) -> tuple[np.ndarray, list]:
    """Detect blue regions in image (for license plates with blue stripes).
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (blue_mask, contours) where blue_mask is binary mask 
        and contours is list of detected contours
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    low_blue = np.array([100, 150, 50])
    high_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    return closing, contours


def detect_white_gray_region(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect white/gray regions in image (for license plate background).
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (white_mask, white_sum) where white_mask is binary mask
        and white_sum is the total white pixel intensity
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    low_white = np.array([0, 0, 180])
    high_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, low_white, high_white)

    low_gray = np.array([0, 0, 150])
    high_gray = np.array([180, 100, 255])
    gray_mask = cv2.inRange(hsv, low_gray, high_gray)
    
    white_mask_bright = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 255, 255]))
    
    combined_mask = cv2.bitwise_or(white_mask, gray_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_mask_bright)
    
    white_sum = int(combined_mask.sum())
    
    return combined_mask, white_sum


def upscale_image(image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Upscale image using high-quality Lanczos interpolation.
    
    Args:
        image: Input image (grayscale or BGR)
        scale_factor: Scale multiplier (default 2x)
        
    Returns:
        Upscaled image
    """
    if image.size == 0:
        return image
    
    height, width = image.shape[:2]
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    upscaled = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=cv2.INTER_LANCZOS4
    )
    
    LOGGER.debug(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")
    return upscaled
