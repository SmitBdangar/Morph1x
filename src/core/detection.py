import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"import model first or the path is worong: {model_path}")
        
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info("successfully")
    
    def detect(
        self,
        frame: np.ndarray,
        allowed_classes: set,
        persist: bool = True
    ) -> List[Dict]:
        if frame is None or frame.size == 0:
            raise ValueError("there is a problem with frame")
        
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
        return {
            "model_name": self.model.model_name,
            "task": self.model.task,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }