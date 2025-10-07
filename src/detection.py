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
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        self.detection_history.append(detections)
        
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        return detections
    
    def clear_history(self):
        self.detection_history = []


def create_detector(model_path: str = None) -> LivingBeingDetector:
    return LivingBeingDetector(model_path)


 
