# Classical Computer Vision

Traditional image processing approaches for license plate detection.

## Overview

The classical pipeline uses color-based segmentation and morphological operations to detect Egyptian blue license plates.

## Detection Methods

### 1. Morphological Detection (Color-based)

```python
# Convert to HSV for blue color detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Blue range for Egyptian plates
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Morphological closing to fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Processing Steps:**

1. **HSV Conversion** - Better color separation than RGB
2. **Blue Masking** - Isolate Egyptian plate color
3. **Morphological Closing** - Connect nearby regions
4. **Contour Detection** - Find plate boundaries
5. **Aspect Ratio Filtering** - Remove non-plate shapes

### 2. Canny Edge Detection

```python
# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(filtered, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Look for rectangles
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    if len(approx) == 4:  # Rectangle found
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # Egyptian plates: aspect ratio 2-4
        if 2.0 <= aspect_ratio <= 4.0:
            plate_roi = gray[y:y+h, x:x+w]
```

**Key Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Bilateral d | 11 | Filter diameter |
| Canny low | 30 | Lower threshold |
| Canny high | 200 | Upper threshold |
| Approx epsilon | 0.018 | Polygon approximation |

### Harris Corner Validation

To validate detected plates, we use Harris corner detection:

```python
harris_dst = cv2.cornerHarris(plate_roi, blockSize=2, ksize=3, k=0.04)
harris_corners_count = np.sum(harris_dst > 0.01 * harris_dst.max())

# Plates should have many corners (text)
if harris_corners_count > 20:
    # Valid plate detected
```

## OCR with EasyOCR

```python
import easyocr

reader = easyocr.Reader(['ar', 'en'])
result = reader.readtext(plate_image)

# Process results
for (bbox, text, confidence) in result:
    if confidence > 0.3:
        print(f"Detected: {text}")
```

## Limitations

- **Lighting Sensitivity** - Color-based methods fail in low light
- **Angle Dependence** - Requires near-frontal view
- **Color Variations** - Only works for blue Egyptian plates
- **Noise** - Complex backgrounds cause false positives

## When to Use

| Method | Best For |
|--------|----------|
| Morphological | Clear, well-lit images with blue plates |
| Canny | High-contrast images with visible edges |
| YOLO | General use, robust to variations |
