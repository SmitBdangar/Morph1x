"""
Visualization and rendering module for detections and HUD.
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
    """Renders detections and HUD panel on frames."""
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.panel_bg = tuple(self.config.get("panel_bg_color", [0, 0, 0]))
        self.header_color = tuple(self.config.get("header_color", [0, 255, 180]))
        self.accent_color = tuple(self.config.get("accent_color", [0, 255, 140]))
        self.divider_color = tuple(self.config.get("divider_color", [60, 60, 70]))
        self.class_colors = self.config.get("colors", {})
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for class or default."""
        color = self.class_colors.get(class_name.lower(), [200, 200, 200])
        return tuple(color)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw all detections on frame."""
        for det in detections:
            self._draw_box(frame, det)
        return frame
    
    def _draw_box(self, frame: np.ndarray, detection: Dict) -> None:
        """Draw single detection box with label."""
        x1, y1, x2, y2 = detection["bbox"]
        color = self.get_class_color(detection["class_name"])
        label = detection["unique_id"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._draw_label(frame, label, x1, y1, color)
    
    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int,
                    color: Tuple[int, int, int]) -> None:
        """Draw label above box."""
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        
        label_y = max(y - 5, text_h + 5)
        cv2.rectangle(frame, (x, label_y - text_h - 5),
                     (x + text_w + 6, label_y), color, -1)
        cv2.putText(frame, text, (x + 3, label_y - 3),
                   self.FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def draw_panel(self, panel: np.ndarray, title: str, 
                   count: int, ids: List[str]) -> np.ndarray:
        """Draw HUD info panel."""
        height = panel.shape[0]
        panel[:] = self.panel_bg
        
        y = 40
        cv2.putText(panel, title, (20, y), self.FONT, 0.8,
                   self.header_color, 2, cv2.LINE_AA)
        
        self._draw_divider(panel, y + 15)
        y += 50
        
        cv2.putText(panel, f"Active: {count}", (15, y), self.FONT,
                   0.55, self.accent_color, 1, cv2.LINE_AA)
        self._draw_divider(panel, y + 20)
        y += 50
        
        cv2.putText(panel, "Tracked IDs:", (15, y), self.FONT,
                   0.6, (200, 255, 200), 1, cv2.LINE_AA)
        y += 25
        
        if ids:
            for i, uid in enumerate(ids):
                if y > height - 60:
                    cv2.putText(panel, f"...+{len(ids) - i}", (25, y),
                               self.FONT, 0.5, (100, 100, 100), 1)
                    break
                cv2.putText(panel, f"• {uid}", (25, y), self.FONT,
                           0.5, self.accent_color, 1, cv2.LINE_AA)
                y += 20
        else:
            cv2.putText(panel, "No objects tracked", (25, y), self.FONT,
                       0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        self._draw_divider(panel, height - 40)
        cv2.putText(panel, "MORPH1X", (20, height - 15), self.FONT,
                   0.5, (120, 200, 255), 1, cv2.LINE_AA)
        
        return panel
    
    @staticmethod
    def _draw_divider(panel: np.ndarray, y: int, 
                      color: Tuple[int, int, int] = (60, 60, 70)) -> None:
        """Draw horizontal divider line."""
        cv2.line(panel, (10, y), (panel.shape[1] - 10, y), color, 1)
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS on frame."""
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                   self.FONT, 0.7, (0, 255, 0), 2)
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