
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import argparse
from typing import List, Tuple
from pathlib import Path
import sys
import os

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.normalize import normalize_plate_id
from src.pipeline.yolo_ocr import load_model as load_yolo_ocr_model, read_plate_text as read_yolo_plate_text

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking Management System (Dual Model)")
    parser.add_argument(
        "--video", 
        default="data/Video/parking2-cropped.mp4", 
        help="Path to input video"
    )
    parser.add_argument(
        "--output", 
        default="data/gallery/parking_output2.mp4", 
        help="Path to output video"
    )
    parser.add_argument(
        "--det-weights", 
        default="models/best.pt", 
        help="Path to Detection Model weights (Cars/Plates)"
    )
    parser.add_argument(
        "--ocr-weights", 
        default="models/yolo11m_car_plate_ocr.pt", 
        help="Path to OCR Model weights (Characters)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    # Load Models
    print(f"Loading Detection Model: {args.det_weights}")
    det_model = YOLO(args.det_weights)
    
    print(f"Loading OCR Model: {args.ocr_weights}")
    ocr_model = load_yolo_ocr_model(Path(args.ocr_weights))
    
    video_info = sv.VideoInfo.from_video_path(args.video)
    width, height = video_info.width, video_info.height
    
    # Define Parking Zones (Trapezoids for Perspective)
    # Position zones where cars actually park (lower portion of frame)
    
    center_x = width // 2
    center_y = height // 2 + 200  # Shift further down to bottom area
    
    # Perspective parameters - adjusted for better floor match
    top_w = 300
    bottom_w = 360
    h = 120
    gap = 30
    
    # Left Spot Trapezoid (shifted left)
    t1_x = center_x - gap +100
    t1_y = center_y -200
    
    spot1 = np.array([
        [t1_x - top_w, t1_y],
        [t1_x, t1_y],
        [t1_x + 20, t1_y + h],  # Perspective expansion
        [t1_x - bottom_w, t1_y + h]
    ])
    
    # Right Spot Trapezoid (shifted right)
    t2_x = center_x + gap +100
    t2_y = center_y - 200
    
    spot2 = np.array([
        [t2_x, t2_y],
        [t2_x + top_w, t2_y],
        [t2_x + bottom_w, t2_y + h],  # Perspective expansion
        [t2_x - 20, t2_y + h]
    ])

    zones = [spot1, spot2]
    
    polygon_zones = [
        sv.PolygonZone(
            polygon=zone
        ) for zone in zones
    ]
    
    # Annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1
    )
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.GREEN, 
            thickness=2,
            text_thickness=2,
            text_scale=0.8
        ) for zone in polygon_zones
    ]

    cap = cv2.VideoCapture(args.video)
    fps = int(video_info.fps)
    
    out = cv2.VideoWriter(
        args.output, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        1, 
        video_info.resolution_wh
    )

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only one frame per second
        if frame_count % fps != 0:
            frame_count += 1
            continue
        
        # 1. Detection Pass (Cars & Plates)
        det_results = det_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(det_results)
        
        # 2. Check Parking Occupancy (Cars only ideally)
        # Assuming 'car' is in names.
        
        # Create display labels
        labels = []
        
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            class_name = det_model.names[class_id]
            # confidence = detections.confidence[i]
            bbox = detections.xyxy[i]
            
            label = ""
            
            # If Plate -> Run OCR Model
            if 'plate' in class_name.lower():
                x1, y1, x2, y2 = bbox.astype(int)
                # Ensure within bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width, x2); y2 = min(height, y2)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    text, _conf = read_yolo_plate_text(crop, model=ocr_model)
                    if text:
                        label = text
            
            labels.append(label)

        # Draw Zones
        for i, zone in enumerate(polygon_zones):
            # Check occupancy
            # Ideally filter detections by 'car' class properly
            # Check if any detection center is in zone
            is_occupied = zone.trigger(detections=detections)
            
            zone_annotator = zone_annotators[i]
            if is_occupied.any():
                zone_annotator.color = sv.Color.RED
            else:
                zone_annotator.color = sv.Color.GREEN
                
            frame = zone_annotator.annotate(scene=frame)

        # Draw Detection Boxes & Labels
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        out.write(frame)
        print(f"Processed frame {frame_count} (Time: {frame_count/fps:.1f}s)")
        
        frame_count += 1

    cap.release()
    out.release()
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
