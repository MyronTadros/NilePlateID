# YOLO OCR

Custom YOLO model trained to detect and recognize Arabic characters on Egyptian license plates.

## Overview

Instead of traditional OCR engines, we use a YOLO object detection model where each Arabic character and numeral is a separate class. This approach is more robust for the specific format of Egyptian plates.

## Architecture

The YOLO OCR model detects individual characters as objects:

```
Plate Crop → YOLO Inference → Character Boxes → Sort Left-to-Right → Assemble Text
```

## Character Classes

### Arabic Letters (17 classes)

| Character | Name | Character | Name |
|-----------|------|-----------|------|
| ا | Alef | ر | Ra |
| ب | Ba | س | Seen |
| ج | Jeem | ص | Sad |
| د | Dal | ط | Ta |
| ع | Ain | ف | Fa |
| ق | Qaf | ل | Lam |
| م | Meem | ن | Noon |
| ه | Ha | و | Waw |
| ي | Ya | | |

### Numerals (10 classes)

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

## Usage

```python
from src.pipeline.yolo_ocr import load_model, read_plate_text

# Load model
model = load_model("models/yolo11m_car_plate_ocr.pt")

# Read plate text
plate_crop = cv2.imread("plate.jpg")
text, confidence = read_plate_text(plate_crop, model=model)

print(f"Plate: {text} ({confidence:.1%})")
```

## Character Assembly

Characters are sorted by their x-coordinate (left to right) and concatenated:

```python
def read_plate_text(plate_crop, model, conf=0.25):
    results = model.predict(plate_crop, conf=conf, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        char = CLASS_MAP.get(model.names[cls_id], model.names[cls_id])
        detections.append((x_center, char, float(box.conf[0])))
    
    # Sort by x position
    detections.sort(key=lambda x: x[0])
    
    # Build text
    text = ''.join([d[1] for d in detections])
    avg_conf = sum(d[2] for d in detections) / len(detections) if detections else 0
    
    return text, avg_conf * 100
```

## Class Mapping

The model outputs class names that need to be mapped to actual characters:

```python
CLASS_MAP = {
    "alef": "ا", "ba": "ب", "jeem": "ج", "dal": "د",
    "ra": "ر", "seen": "س", "sad": "ص", "ta": "ط",
    "ain": "ع", "fa": "ف", "qaf": "ق", "lam": "ل",
    "meem": "م", "noon": "ن", "ha": "ه", "waw": "و",
    "ya": "ي",
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9"
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | 0.25 | Detection confidence threshold |
| `iou` | 0.45 | NMS IoU threshold |
| `verbose` | False | Print detection output |

## Model File

```
models/
└── yolo11m_car_plate_ocr.pt   # YOLO OCR weights (~40MB)
```

## Comparison with EasyOCR

| Feature | YOLO OCR | EasyOCR |
|---------|----------|---------|
| Speed | ⚡ Faster | Slower |
| Arabic Accuracy | ✅ Higher (trained) | Lower (general) |
| Setup | Ultralytics | easyocr package |
| Customization | ✅ Trainable | Limited |

## Training

The model was trained on Ultralytics HUB with:

- **Dataset**: Custom Egyptian plate character annotations
- **Base Model**: YOLOv11m
- **Classes**: 27 (17 Arabic letters + 10 numerals)
- **Augmentation**: Rotation, blur, brightness variation
