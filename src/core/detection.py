import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List
import logging
import torch

logger = logging.getLogger("src.core.object_detector")

# Global counter for sequential IDs (since we are not using a persistent tracker)
# This will ensure the ID is ALWAYS a clean number.
current_id = 0 


class ObjectDetector:
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info("YOLO model loaded successfully.")
    
    def detect(
        self,
        frame: np.ndarray,
        allowed_classes: set
    ) -> List[Dict]:
        global current_id
        
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")
        
        # --- CRITICAL CHANGE: Use model() for basic detection (NO TRACKING) ---
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            # Force ultralytics not to draw anything internally
            boxes=True, # Ensure boxes are in the output
            show=False # Crucial: ensures ultralytics doesn't render to image
        )[0]
        
        detections = []
        
        if not hasattr(results, "boxes") or results.boxes is None or len(results.boxes) == 0:
            return detections
        
        # Reset ID counter for a fresh frame (simulate no tracking)
        current_id = 0
        
        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = self.model.names[cls_id]
            
            if class_name not in allowed_classes:
                continue
            
            # Manually increment and assign a unique ID for this frame
            current_id += 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            
            # --- Aggressively create the CLEAN unique_id (Layer 1) ---
            class_initial = class_name[0].upper()
            unique_id = f"ID-{current_id}-{class_initial}" # Now impossible to be ???
            track_id = current_id 
            
            logger.debug(f"ID processed: Assigned Frame ID: {track_id}, Final unique_id: {unique_id}")
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": conf,
                "track_id": track_id,
                "unique_id": unique_id
            })
            
        return detections
    
    def get_model_info(self) -> Dict[str, any]:
        return {
            "model_name": self.model.model_name,
            "task": self.model.task,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }