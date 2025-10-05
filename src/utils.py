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


def create_video_writer(output_path: str, frame_size: Tuple[int, int], 
                       fps: int = 30) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def get_video_source_info(source) -> Dict:
    info = {
        'type': 'unknown',
        'width': 0,
        'height': 0,
        'fps': 0
    }
    
    try:
        if isinstance(source, int):
            # Camera source
            cap = cv2.VideoCapture(source)
            info['type'] = 'camera'
        else:
            # File source
            cap = cv2.VideoCapture(str(source))
            info['type'] = 'file'
        
        if cap.isOpened():
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        
    except Exception as e:
        logger.error(f"Error getting video source info: {e}")
    
    return info


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

def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), 2)
    cv2.circle(frame, (400, 100), 50, (255, 0, 0), 2)
    cv2.putText(frame, "Test Frame", (250, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


if __name__ == "__main__":
    test_frame = create_test_frame()
    print(f"Test frame created: {test_frame.shape}")
    
    fps_meter = FPSMeter()
    for _ in range(10):
        fps_meter.update()
    print(f"FPS: {fps_meter.get_fps():.2f}")
