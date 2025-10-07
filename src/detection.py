import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging

from .config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS,
    PRIMARY_LIVING_BEINGS, LIVING_BEING_CLASSES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LivingBeingDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODEL_PATH)
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_living_beings(self, frame: np.ndarray) -> List[Dict]:
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            results = self.model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if class_id in PRIMARY_LIVING_BEINGS:
                            detection = {
                                'bbox': box.astype(int),  # [x1, y1, x2, y2]
                                'confidence': float(conf),
                                'class_id': int(class_id),
                                'class_name': PRIMARY_LIVING_BEINGS[class_id]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def filter_living_beings(self, detections: List[Dict]) -> List[Dict]:
        living_beings = []
        for detection in detections:
            class_id = detection.get('class_id')
            if class_id in PRIMARY_LIVING_BEINGS:
                living_beings.append(detection)
        return living_beings
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, int]:
        summary = {}
        
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            summary[class_name] = summary.get(class_name, 0) + 1
        
        return summary


class DetectionTracker:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.detection_history = []
        self.previous_objects: List[Dict] = []  # [{'id': int, 'center': (x,y), 'class_id': int}]
        self.next_id = 1

    def _center_of(self, bbox: np.ndarray) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def _assign_ids_and_speeds(self, detections: List[Dict], fps: float) -> None:
        # Simple nearest-center matching per class; computes speed in px/s.
        current_objects = []
        used_prev = set()
        speed_scale = fps if fps and fps > 0 else 0.0

        for det in detections:
            center = self._center_of(det['bbox'])
            class_id = det.get('class_id')

            # Find nearest previous object with same class
            best_idx = -1
            best_dist = float('inf')
            for idx, prev in enumerate(self.previous_objects):
                if idx in used_prev:
                    continue
                if prev.get('class_id') != class_id:
                    continue
                px, py = prev['center']
                dx = center[0] - px
                dy = center[1] - py
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                prev = self.previous_objects[best_idx]
                used_prev.add(best_idx)
                det['track_id'] = prev['id']
                if speed_scale > 0:
                    dx = center[0] - prev['center'][0]
                    dy = center[1] - prev['center'][1]
                    # pixels per frame -> pixels per second
                    speed_px_s = (dx * dx + dy * dy) ** 0.5 * speed_scale
                    det['speed_px_s'] = float(speed_px_s)
                else:
                    det['speed_px_s'] = 0.0
                current_objects.append({'id': prev['id'], 'center': center, 'class_id': class_id})
            else:
                # New object
                det['track_id'] = self.next_id
                det['speed_px_s'] = 0.0
                current_objects.append({'id': self.next_id, 'center': center, 'class_id': class_id})
                self.next_id += 1

        # Update previous objects for next frame
        self.previous_objects = current_objects

    def update(self, detections: List[Dict], fps: float = 0.0) -> List[Dict]:
        self._assign_ids_and_speeds(detections, fps)
        self.detection_history.append(detections)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        return detections

    def clear_history(self):
        self.detection_history = []
        self.previous_objects = []


def create_detector(model_path: str = None) -> LivingBeingDetector:
    return LivingBeingDetector(model_path)


 
