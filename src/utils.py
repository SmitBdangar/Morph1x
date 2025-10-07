import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging

from .config import (
    BOX_COLOR, TEXT_COLOR, BOX_THICKNESS, TEXT_THICKNESS,
    FONT_SCALE, TEXT_PADDING, SHOW_FPS, SHOW_DETECTION_COUNT
)

logger = logging.getLogger(__name__)


class FPSMeter:
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
    
    def update(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
    
    def get_fps(self) -> float:
        return self.fps
    
    def reset(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0


def draw_detection_box(frame: np.ndarray, detection: Dict, 
                      color: Tuple[int, int, int] = None) -> np.ndarray:
    if color is None:
        color = BOX_COLOR
    
    bbox = detection['bbox']  # [x1, y1, x2, y2]
    class_name = detection.get('class_name', 'Unknown')
    confidence = detection.get('confidence', 0.0)
    
    x1, y1, x2, y2 = bbox
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    speed_px_s = detection.get('speed_px_s')
    # Show speed only for persons where it makes sense
    if speed_px_s is not None and detection.get('class_name') == 'person':
        label = f"{class_name}: {confidence:.2f} | {speed_px_s:.0f}px/s"
    else:
        label = f"{class_name}: {confidence:.2f}"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS
    )
    
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - TEXT_PADDING * 2),
        (x1 + text_width + TEXT_PADDING * 2, y1),
        color,
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (x1 + TEXT_PADDING, y1 - TEXT_PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS
    )
    
    return frame


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for detection in detections:
        frame = draw_detection_box(frame, detection)
    
    return frame


def draw_info_panel(frame: np.ndarray, fps: float = None, 
                   detection_count: int = None, 
                   detection_summary: Dict[str, int] = None) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    panel_width = 300
    panel_height = 150
    panel_x = width - panel_width - 10
    panel_y = 10
    
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    y_offset = panel_y + 25
    line_height = 20
    
    if SHOW_FPS and fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
    
    if SHOW_DETECTION_COUNT and detection_count is not None:
        cv2.putText(frame, f"Detections: {detection_count}", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
    
    if detection_summary:
        cv2.putText(frame, "Living Beings:", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        for class_name, count in detection_summary.items():
            if y_offset < panel_y + panel_height - 10:
                cv2.putText(frame, f"  {class_name}: {count}", 
                           (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += line_height - 5
    
    return frame


def resize_frame(frame: np.ndarray, max_size: Tuple[int, int] = None) -> np.ndarray:
    if max_size is None:
        return frame
    
    max_width, max_height = max_size
    height, width = frame.shape[:2]
    
    scale = min(max_width / width, max_height / height)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame

def validate_frame(frame: np.ndarray) -> bool:
    if frame is None:
        return False
    
    if len(frame.shape) != 3:
        return False
    
    if frame.shape[2] != 3:
        return False
    
    if frame.size == 0:
        return False
    
    return True

