# Egypt Plate Detection

An automatic license plate detection and recognition system for Egyptian license plates. This system uses computer vision techniques to detect plates based on their distinctive blue stripe and performs OCR (Optical Character Recognition) to read the plate text in both Arabic and English.

## Features

- **Automatic Plate Detection**: Detects Egyptian license plates using color-based detection (blue stripe on top)
- **Perspective Correction**: Automatically warps detected plates to a front-facing view
- **Dual OCR Engines**: Uses both EasyOCR and Tesseract for comparison
- **Arabic & English Support**: Recognizes both Arabic and English characters
- **Image & Video Processing**: Works with both static images and video streams
- **Preprocessing Pipeline**: Automatic brightness/contrast adjustment and image enhancement
- **Debug Output**: Saves intermediate processing steps for analysis

## Project Structure

```
├── main.py                    # Main entry point
├── plate_detection.py         # Plate detection logic
├── ocr_processing.py          # OCR and text post-processing
├── image_preprocessing.py     # Image enhancement functions
├── image_transforms.py        # Perspective transformation utilities
├── temp/                      # Output directory for results
│   ├── crop*.jpg             # Detected plate crops
│   ├── detection.jpg         # Annotated input image
└── └── steps/                # Intermediate processing steps
```

## Requirements

### Python Dependencies

```bash
opencv-python
numpy
easyocr
pytesseract
```

### External Dependencies

- **Tesseract OCR**: Required for the Tesseract OCR engine
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

- **Arabic Language Support**: Install Arabic language data for Tesseract
  - macOS: `brew install tesseract-lang`
  - Ubuntu: `sudo apt-get install tesseract-ocr-ara`

## Installation

1. Clone the repository:
```bash
cd "Egypt Plate Detection"
```

2. Install Python dependencies:
```bash
pip install opencv-python numpy easyocr pytesseract
```

3. Install Tesseract OCR (see External Dependencies above)

4. For GPU acceleration with EasyOCR (optional but recommended):
```bash
pip install torch torchvision
```

## Usage

### Process an Image

```bash
python main.py --i path/to/image.jpg
```

This will:
- Detect all license plates in the image
- Save the detection result to `temp/detection.jpg`
- Save each detected plate crop to `temp/crop*.jpg`
- Perform OCR with both EasyOCR and Tesseract
- Save OCR results to `temp/crop*_easyocr.txt` and `temp/crop*_tesseract.txt`
- Save all intermediate processing steps to `temp/steps/`

### Process a Video

```bash
python main.py --v path/to/video.mp4
```

This will:
- Open a video window showing real-time plate detection
- Draw bounding boxes around detected plates
- Press 'Q' to quit

## How It Works

### 1. Plate Detection (`plate_detection.py`)

The detection algorithm leverages the distinctive blue stripe at the top of Egyptian license plates:

1. **Color Detection**: Converts image to HSV color space and detects blue regions
2. **Morphological Operations**: Closes small gaps in detected regions
3. **Contour Analysis**: Finds contours and filters by aspect ratio (plates are roughly 2:1)
4. **Blue Verification**: Checks if the top portion contains enough blue pixels
5. **White/Gray Detection**: Verifies the plate body has white or gray background
6. **Character Validation**: Ensures the region contains character-like shapes
7. **Aspect Ratio Correction**: Adjusts crop to match real plate dimensions (35cm × 17cm)
8. **Perspective Transform**: Warps the plate to a rectangular front-facing view

### 2. Image Preprocessing (`image_preprocessing.py`)

- **Automatic Brightness/Contrast Adjustment**: Analyzes image histogram to optimize visibility
- **Histogram Clipping**: Removes outliers for better contrast

### 3. OCR Processing (`ocr_processing.py`)

#### EasyOCR
- Supports both Arabic and English characters
- GPU acceleration available
- Better for Arabic text recognition

#### Tesseract
- Used for comparison and validation
- Script-based Arabic model (`ara`)
- DPI and PSM optimization for license plates

#### Post-Processing
- Removes punctuation and special characters
- Cleans up noise from OCR output
- Retains only alphanumeric characters

### 4. Image Transforms (`image_transforms.py`)

- **Point Ordering**: Ensures corner points are in the correct order
- **Perspective Transform**: Converts skewed plate regions to rectangular crops

## Output Files

After processing an image, the following files are created:

```
temp/
├── detection.jpg                          # Input image with detected plates marked
├── crop_original_1.jpg                    # Original detected plate crop
├── crop1.jpg                              # Processed plate ready for OCR
├── crop1_easyocr.txt                      # EasyOCR result
├── crop1_tesseract.txt                    # Tesseract result
└── steps/
    ├── 1_blue_color_detection.png         # Blue color mask
    ├── 2_closing_morphology.png           # After morphological closing
    ├── plate1_3_detected_plate.png        # Extracted plate region
    ├── plate1_4_Brigthness_contrast_adjustment.png
    ├── plate1_5_gray.png                  # Grayscale conversion
    └── plate1_6_threshold.png             # Binary threshold (input to OCR)
```

## Detection Parameters

### Color Ranges (HSV)

- **Blue Stripe**: `[100, 150, 50]` to `[130, 255, 255]`
- **White/Gray Background**: `[0, 0, 150]` to `[180, 100, 255]`

### Geometric Filters

- **Aspect Ratio**: Width should be between 2× and 6× the height
- **Minimum Size**: Area > 0.01% of image area
- **Character Count**: Expected 5-20 character-like regions

### Thresholds

- **Blue Sum**: > 200 (amount of blue pixels in top portion)
- **White Sum**: > 5000 (amount of white/gray pixels in plate body)

## Performance

Processing time varies based on:
- Image resolution
- Number of plates in the image
- Hardware (GPU availability for EasyOCR)
- OCR engine selection

Typical processing time for a single image: 2-5 seconds (CPU) or <1 second (GPU)

## Dataset

The project includes the EALPR Vehicles dataset with:
- Vehicle images with Egyptian license plates
- Ground truth labels in text files (YOLO format)

## Troubleshooting

### "need to install easyocr first"
```bash
pip install easyocr
```

### "tesseract not found"
Install Tesseract OCR and ensure it's in your system PATH, or use:
```bash
brew install tesseract  # macOS
```

### Poor OCR Results
- Ensure the image quality is good (not blurry, good lighting)
- Check intermediate steps in `temp/steps/` to diagnose issues
- The `6_threshold.png` file should show clear black text on white background
- Try adjusting brightness/contrast parameters

### No Plates Detected
- Check if the plate has a visible blue stripe
- Verify the plate is large enough in the image
- Adjust HSV color ranges if needed for different lighting conditions

## Acknowledgments

- EasyOCR for Arabic text recognition
- Tesseract OCR engine
- OpenCV for computer vision operations
- EALPR dataset for Egyptian plates
