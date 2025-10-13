"""
Visualization and rendering module for detections.
"""
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class FPSMeter:
    """FPS calculation utility."""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
    
    def update(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return round(self.fps, 2)
    
    def reset(self) -> None:
        """Reset counter."""
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0


class HUDRenderer:
    """Renders detections on frames."""
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_color = (0, 255, 0)  # Green
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw all detections on frame."""
        for det in detections:
            self._draw_box(frame, det)
        return frame
    
    def _draw_box(self, frame: np.ndarray, detection: Dict) -> None:
        """Draw bounding box with confidence and tracking ID."""
        x1, y1, x2, y2 = detection["bbox"]
        color = self.default_color  # Green
        
        # Label format: "confidence ID: track_id"
        label = f"{detection['confidence']:.2f} ID: {detection['track_id']}"
        
        # Draw green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label above the box
        self._draw_label(frame, label, x1, y1, color)

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int,
                    color: Tuple[int, int, int]) -> None:
        """Draw label above bounding box."""
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        
        label_y = max(y - 5, text_h + 5)
        
        # Draw colored background
        cv2.rectangle(frame, (x, label_y - text_h - 5),
                     (x + text_w + 6, label_y), color, -1)
        
        # Draw black text on colored background
        cv2.putText(frame, text, (x + 3, label_y - 3),
                   self.FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS on frame."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   self.FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def resize_frame(self, frame: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to max size while maintaining aspect ratio."""
        max_width, max_height = max_size
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        
        return frame