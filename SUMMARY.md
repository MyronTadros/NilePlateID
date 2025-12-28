# Summary: License Plate Detection + OCR Pipeline

## Problem statement
We need a repeatable pipeline to detect cars and license plates from images, read plate text, and store results in a structured way that supports audits and downstream analytics.

## Approach
We use Ultralytics YOLO for fast object detection and EasyOCR for plate text recognition. YOLO provides car and plate bounding boxes. Plates are associated to cars by a center-in-box rule with IoU tie-break. Crops are then saved and OCR results are normalized into a consistent plate_id used to group galleries.

Why YOLO + OCR:
- YOLO is fast and accurate for object detection with flexible deployment.
- EasyOCR provides multilingual OCR (Arabic + English) without custom training to start.
- The combination allows immediate end-to-end extraction and later model improvements.

## Current capabilities
- Detect cars and plates in batches of images.
- Match plates to cars per image.
- Crop car and plate images and save them under plate_id folders.
- OCR plate crops, normalize text, and write an index CSV.
- Generate debug images and run configuration logs for traceability.

## Assumptions and limitations
- Model class names include "car" and "plate" (or the mapping is updated).
- Plate association assumes the plate center lies within the car box.
- OCR accuracy depends on crop quality, resolution, and lighting.
- Multi-car scenes can contain ambiguous plate associations.

## How we evaluate correctness
- Visual debug images with boxes and plate_id labels.
- Consistency checks in `index.csv` (paths exist, bboxes valid, non-empty crops).
- End-of-run summaries with counts of detections, matches, and OCR fallbacks.

## Risks and mitigations
- OCR errors: use confidence thresholds, allow unknown_* fallback IDs, improve preprocessing.
- Matching errors: consider better association logic or multi-object tracking.
- Multiple cars / occlusion: rely on per-image debug review; plan stronger association rules.

## Future work
- Mask-based cutouts instead of rectangular crops.
- Better association using temporal or spatial cues.
- Plate formatting rules for country-specific standards.
- Vehicle ReID integration to connect detections across frames or cameras.
