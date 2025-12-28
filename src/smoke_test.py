"""Minimal import check for core dependencies."""

import cv2
import easyocr
import ultralytics


def main() -> None:
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"easyocr: {easyocr.__version__}")
    print(f"opencv-python: {cv2.__version__}")


if __name__ == "__main__":
    main()
