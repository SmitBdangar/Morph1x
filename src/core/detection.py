"""
Core detection module for Morph1x.
Handles YOLOv8 inference and object tracking.
"""

import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """YOLOv8-based object detector with ByteTrack integration."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize detector with YOLO model.
        
        Args:
            model_path: Path to YOLOv8 model file.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IOU threshold for NMS.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info("Model loaded successfully")
    
    def detect(
        self,
        frame: np.ndarray,
        allowed_classes: set,
        persist: bool = True
    ) -> List[Dict]:
        """
        Run detection and tracking on frame.
        
        Args:
            frame: Input frame as NumPy array.
            allowed_classes: Set of class names to detect.
            persist: Enable object persistence tracking.
        
        Returns:
            List of detection dictionaries with bbox, class, confidence, and tracking info.
        
        Raises:
            ValueError: If frame is invalid.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame received")
        
        results = self.model.track(
            frame,
            persist=persist,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        
        if not hasattr(results, "boxes") or results.boxes is None:
            return detections
        
        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = self.model.names[cls_id]
            
            if class_name not in allowed_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            track_id = int(box.id.item()) if box.id is not None else -1
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": conf,
                "track_id": track_id,
                "unique_id": f"{track_id}-{class_name[0].upper()}" if track_id >= 0 else class_name
            })
        
        return detections
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata.
        """
        return {
            "model_name": self.model.model_name,
            "task": self.model.task,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }