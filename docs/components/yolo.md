# YOLO Detection & OCR

Deep learning-based detection and text recognition for license plates.

## Overview

The YOLO pipeline uses YOLOv11 for object detection and a custom YOLO OCR model for Arabic character recognition.

![YOLO Pipeline](assets/yolo_pipeline.png)

## Detection Model

### Architecture

YOLOv11 consists of three main components:

- **Backbone**: CSPDarknet with C3k2 blocks for feature extraction
- **Neck**: PANet (Path Aggregation Network) for multi-scale fusion
- **Head**: Detection heads at P3, P4, P5 scales

### Classes

| Class | Description |
|-------|-------------|
| `car` | Vehicle detection |
| `plate` | License plate detection |

### Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

# Detect
results = model.predict(image, conf=0.25, iou=0.45)

# Process results
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    if 'plate' in cls_name.lower():
        plate_crop = image[y1:y2, x1:x2]
```

## OCR Model

### Custom YOLO OCR

Instead of traditional OCR, we use a YOLO model trained to detect individual characters:

```python
from src.pipeline.yolo_ocr import load_model, read_plate_text

# Load OCR model
ocr_model = load_model("models/yolo11m_car_plate_ocr.pt")

# Read plate
text, confidence = read_plate_text(plate_crop, model=ocr_model)
print(f"Plate: {text} ({confidence:.1%})")
```

### Character Classes

The OCR model detects Arabic letters and numerals:

| Type | Characters |
|------|------------|
| Arabic Letters | ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي |
| Numerals | 0 1 2 3 4 5 6 7 8 9 |

### Character Assembly

Characters are sorted left-to-right and assembled into the plate text:

```python
def read_plate_text(plate_crop, model, conf=0.25):
    results = model.predict(plate_crop, conf=conf, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        char = model.names[cls_id]
        detections.append((x_center, char))
    
    # Sort by x position (left to right)
    detections.sort(key=lambda x: x[0])
    
    # Assemble text
    text = ''.join([d[1] for d in detections])
    return text
```

## Configuration

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | 0.25 | Confidence threshold |
| `iou` | 0.45 | NMS IoU threshold |
| `device` | auto | GPU/CPU selection |

### Model Files

```
models/
├── best.pt                    # Detection model
└── yolo11m_car_plate_ocr.pt   # OCR model
```

## Training

Models were trained on Ultralytics HUB:

- **Detection**: Custom Egyptian plate dataset
- **OCR**: Character-level annotations on plate crops

See [Training Dashboard](../training_page.py) for metrics.

## Comparison with EasyOCR

| Feature | YOLO OCR | EasyOCR |
|---------|----------|---------|
| Speed | Faster | Slower |
| Arabic Support | Trained | General |
| Accuracy | Higher for Egyptian | Lower for Arabic |
| Dependencies | Ultralytics | easyocr, torch |
