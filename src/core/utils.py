import cv2
import numpy as np
from typing import List, Dict, Tuple
import time

from .config import (
    BOX_COLOR, TEXT_COLOR, BOX_THICKNESS, TEXT_THICKNESS,
    FONT_SCALE, TEXT_PADDING
)

 


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


def _class_color(class_name: str) -> Tuple[int, int, int]:
    if class_name == 'person':
        return (40, 200, 40)  # professional green
    return (60, 160, 240)     # blue for others


def _draw_corner_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    w = x2 - x1
    h = y2 - y1
    lw = max(12, w // 6)
    lh = max(12, h // 6)
    # top-left
    cv2.line(img, (x1, y1), (x1 + lw, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + lh), color, thickness)
    # top-right
    cv2.line(img, (x2, y1), (x2 - lw, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + lh), color, thickness)
    # bottom-left
    cv2.line(img, (x1, y2), (x1 + lw, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - lh), color, thickness)
    # bottom-right
    cv2.line(img, (x2, y2), (x2 - lw, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - lh), color, thickness)


def _draw_label_pill(img: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE + 0.1, max(1, TEXT_THICKNESS))
    pad = TEXT_PADDING + 2
    w = tw + pad * 2
    h = th + pad * 2
    y0 = max(0, y - h - 4)
    x0 = max(0, x)
    # Shadow
    shadow = img.copy()
    cv2.rectangle(shadow, (x0 + 2, y0 + 2), (x0 + w + 2, y0 + h + 2), (0, 0, 0), -1)
    cv2.addWeighted(shadow, 0.3, img, 0.7, 0, img)
    # Badge fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    # Border
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    # Text
    cv2.putText(img, text, (x0 + pad, y0 + h - pad), cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE + 0.1, TEXT_COLOR, max(1, TEXT_THICKNESS))


def draw_detection_box(frame: np.ndarray, detection: Dict, 
                      color: Tuple[int, int, int] = None) -> np.ndarray:
    bbox = detection['bbox']  # [x1, y1, x2, y2]
    class_name = detection.get('class_name', 'Unknown')
    confidence = detection.get('confidence', 0.0)
    track_id = detection.get('track_id')
    speed_px_s = detection.get('speed_px_s')
    speed_m_s = detection.get('speed_m_s')

    if color is None:
        color = _class_color(class_name)

    x1, y1, x2, y2 = bbox

    # Corner-style professional box
    _draw_corner_box(frame, x1, y1, x2, y2, color, thickness=max(2, BOX_THICKNESS))

    # Label: ID only, centered above the box
    label = f"ID {track_id}" if track_id is not None else class_name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE + 0.1, max(1, TEXT_THICKNESS))
    badge_x = int((x1 + x2 - (tw + (TEXT_PADDING + 2) * 2)) / 2)
    badge_x = max(0, badge_x)
    _draw_label_pill(frame, badge_x, y1, label, color)

    return frame


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for detection in detections:
        frame = draw_detection_box(frame, detection)
    
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

