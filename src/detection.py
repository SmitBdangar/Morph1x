"""
YOLOv8-based object detection for living beings in video streams.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging

from .config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS,
    PRIMARY_LIVING_BEINGS, LIVING_BEING_CLASSES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LivingBeingDetector:
    """
    YOLOv8-based detector for identifying living beings in video frames.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model file. If None, uses default from config.
        """
        self.model_path = model_path or str(MODEL_PATH)
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the YOLOv8 model."""
        try:
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_living_beings(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect living beings in a video frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'class_id', 'class_name'
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run YOLOv8 inference
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
                        # Check if this is a living being
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
        """
        Filter detections to only include living beings.
        
        Args:
            detections: List of all detections
            
        Returns:
            Filtered list containing only living beings
        """
        living_beings = []
        
        for detection in detections:
            class_id = detection.get('class_id')
            if class_id in PRIMARY_LIVING_BEINGS:
                living_beings.append(detection)
        
        return living_beings
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get a summary count of detected living beings by type.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with class names as keys and counts as values
        """
        summary = {}
        
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            summary[class_name] = summary.get(class_name, 0) + 1
        
        return summary


class DetectionTracker:
    """
    Simple tracker to maintain detection history and reduce flickering.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize the tracker.
        
        Args:
            max_history: Maximum number of frames to keep in history
        """
        self.max_history = max_history
        self.detection_history = []
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update detection history and return smoothed detections.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Smoothed detections based on history
        """
        # Add current detections to history
        self.detection_history.append(detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # For now, return current detections
        # TODO: Implement temporal smoothing to reduce flickering
        return detections
    
    def clear_history(self):
        """Clear detection history."""
        self.detection_history = []


def create_detector(model_path: str = None) -> LivingBeingDetector:
    """
    Factory function to create a LivingBeingDetector instance.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        LivingBeingDetector instance
    """
    return LivingBeingDetector(model_path)


def test_detection():
    """Test function to verify detection is working."""
    try:
        detector = create_detector()
        logger.info("Detection system initialized successfully")
        
        # Test with a simple black frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_living_beings(test_frame)
        logger.info(f"Test detection completed. Found {len(detections)} objects")
        
        return True
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    test_detection()
