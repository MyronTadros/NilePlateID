"""Example script demonstrating Dutch ANPR enhancement integration.

This script shows how to use the enhanced preprocessing features
from the Dutch ANPR system within NilePlateID.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.enhancement import (
    automatic_brightness_and_contrast,
    enhance_plate_crop,
    four_point_transform,
    detect_blue_region,
    detect_white_gray_region,
)
from src.pipeline.ocr import read_plate_text


def example_basic_enhancement():
    """Example 1: Basic brightness/contrast enhancement."""
    print("=" * 60)
    print("Example 1: Basic Brightness/Contrast Enhancement")
    print("=" * 60)
    
    # Load a sample plate crop
    image_path = "data/incoming/sample.jpg"  # Replace with your image
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Apply automatic brightness and contrast adjustment
    enhanced, alpha, beta = automatic_brightness_and_contrast(image)
    
    print(f"Original image shape: {image.shape}")
    print(f"Adjustment parameters: alpha={alpha:.2f}, beta={beta:.2f}")
    print(f"Enhanced image shape: {enhanced.shape}")
    
    # Save result
    output_path = "temp/enhanced_brightness.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    print(f"Saved enhanced image to: {output_path}")


def example_full_enhancement():
    """Example 2: Full enhancement pipeline (brightness + grayscale + threshold)."""
    print("\n" + "=" * 60)
    print("Example 2: Full Enhancement Pipeline")
    print("=" * 60)
    
    # Load a sample plate crop
    image_path = "data/incoming/sample.jpg"  # Replace with your image
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Apply full enhancement (brightness + contrast + grayscale + threshold)
    enhanced_binary = enhance_plate_crop(image)
    
    print(f"Original image shape: {image.shape}")
    print(f"Enhanced binary image shape: {enhanced_binary.shape}")
    
    # Save result
    output_path = "temp/enhanced_binary.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, enhanced_binary)
    print(f"Saved binary image to: {output_path}")


def example_ocr_comparison():
    """Example 3: Compare standard vs enhanced OCR."""
    print("\n" + "=" * 60)
    print("Example 3: OCR Comparison (Standard vs Enhanced)")
    print("=" * 60)
    
    # Load a sample plate crop
    image_path = "data/plates/sample/plate.jpg"  # Replace with your plate crop
    if not Path(image_path).exists():
        print(f"Plate crop not found: {image_path}")
        print("Please provide a valid plate crop path")
        return
    
    plate_crop = cv2.imread(image_path)
    if plate_crop is None:
        print(f"Failed to load plate crop: {image_path}")
        return
    
    print(f"Plate crop shape: {plate_crop.shape}")
    
    # Standard OCR
    print("\n--- Standard OCR ---")
    text_standard, conf_standard = read_plate_text(plate_crop, use_enhancement=False)
    print(f"Text: {text_standard}")
    print(f"Confidence: {conf_standard:.4f}")
    
    # Enhanced OCR
    print("\n--- Enhanced OCR ---")
    text_enhanced, conf_enhanced = read_plate_text(plate_crop, use_enhancement=True)
    print(f"Text: {text_enhanced}")
    print(f"Confidence: {conf_enhanced:.4f}")
    
    # Comparison
    print("\n--- Comparison ---")
    if conf_enhanced > conf_standard:
        improvement = ((conf_enhanced - conf_standard) / conf_standard) * 100
        print(f"Enhanced OCR improved confidence by {improvement:.1f}%")
    elif conf_standard > conf_enhanced:
        degradation = ((conf_standard - conf_enhanced) / conf_standard) * 100
        print(f"Standard OCR was better by {degradation:.1f}%")
    else:
        print("Both methods yielded same confidence")
    
    if text_standard != text_enhanced:
        print(f"Different text detected:")
        print(f"  Standard: '{text_standard}'")
        print(f"  Enhanced: '{text_enhanced}'")


def example_color_detection():
    """Example 4: Detect blue and white regions (Dutch plate detection)."""
    print("\n" + "=" * 60)
    print("Example 4: Color-Based Region Detection")
    print("=" * 60)
    
    # Load a sample image
    image_path = "data/incoming/sample.jpg"  # Replace with your image
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Detect blue regions (Dutch plate blue stripe)
    blue_mask, blue_contours = detect_blue_region(image)
    print(f"Found {len(blue_contours)} blue contours")
    
    # Detect white/gray regions (plate background)
    white_mask, white_sum = detect_white_gray_region(image)
    print(f"White/gray pixel sum: {white_sum}")
    
    # Save masks
    Path("temp").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("temp/blue_mask.jpg", blue_mask)
    cv2.imwrite("temp/white_mask.jpg", white_mask)
    print("Saved masks to temp/")


def example_perspective_correction():
    """Example 5: Perspective correction for skewed plates."""
    print("\n" + "=" * 60)
    print("Example 5: Perspective Correction")
    print("=" * 60)
    
    # Load a sample image
    image_path = "data/incoming/sample.jpg"  # Replace with your image
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Example corner points for a skewed plate
    # In practice, you would detect these from contours or bounding boxes
    height, width = image.shape[:2]
    
    # Simulate skewed corners (replace with actual detected corners)
    pts = np.array([
        [width * 0.2, height * 0.3],  # top-left
        [width * 0.8, height * 0.25],  # top-right
        [width * 0.85, height * 0.7],  # bottom-right
        [width * 0.15, height * 0.75],  # bottom-left
    ], dtype="float32")
    
    print(f"Original image shape: {image.shape}")
    print(f"Corner points:\n{pts}")
    
    # Apply perspective transform
    warped = four_point_transform(image, pts)
    
    print(f"Warped image shape: {warped.shape}")
    
    # Save result
    output_path = "temp/perspective_corrected.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, warped)
    print(f"Saved corrected image to: {output_path}")


def main():
    """Run all examples."""
    print("Dutch ANPR Enhancement Integration Examples")
    print("=" * 60)
    print()
    
    # Run examples (comment out those you don't want to run)
    
    # example_basic_enhancement()
    # example_full_enhancement()
    # example_ocr_comparison()
    # example_color_detection()
    # example_perspective_correction()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print()
    print("To run specific examples, edit this file and uncomment the")
    print("corresponding example function calls in main().")


if __name__ == "__main__":
    main()
