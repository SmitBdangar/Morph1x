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
    """Renders detections and a table-style HUD panel on frames."""
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Simple color scheme with high contrast
        self.panel_bg = (40, 40, 40)  # Dark gray instead of pure black
        self.header_color = (255, 255, 255)  # White
        self.text_color = (220, 220, 220)  # Light gray
        self.divider_color = (100, 100, 100)  # Medium gray
        self.default_color = (0, 255, 0)  # Green
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for class - always green."""
        return self.default_color
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw all detections on frame."""
        for det in detections:
            self._draw_box(frame, det)
        return frame
    
    def _draw_box(self, frame: np.ndarray, detection: Dict) -> None:
        """Draw simple green box with only ID label - NO class name inside."""
        x1, y1, x2, y2 = detection["bbox"]
        color = self.default_color  # Always green
        
        # Label format: "0.55 ID: 34"
        label = f"{detection['confidence']:.2f} ID: {detection['track_id']}"
        
        # Draw simple green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw only the ID label above the box - NO class name inside
        self._draw_label(frame, label, x1, y1, color)

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int,
                    color: Tuple[int, int, int]) -> None:
        """Draw simple label above box."""
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        
        label_y = max(y - 5, text_h + 5)
        
        # Draw green background
        cv2.rectangle(frame, (x, label_y - text_h - 5),
                     (x + text_w + 6, label_y), color, -1)
        
        # Draw black text on green background
        cv2.putText(frame, text, (x + 3, label_y - 3),
                   self.FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def draw_panel(self, panel: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draws a simple table-style HUD with ID and Type - HIGHLY VISIBLE."""
        
        # Fill panel with dark gray (not pure black for visibility)
        panel[:] = self.panel_bg
        
        # Get panel dimensions
        panel_height, panel_width = panel.shape[:2]
        
        # Draw a white border around the entire panel for debugging visibility
        cv2.rectangle(panel, (0, 0), (panel_width-1, panel_height-1), (255, 255, 255), 3)
        
        # Define column positions with more spacing
        left_margin = 40
        id_col_x = left_margin
        type_col_x = left_margin + 100
        header_y = 60
        
        # Draw TITLE at the top
        cv2.putText(panel, "DETECTIONS", (left_margin, 35), self.FONT,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw table headers in WHITE with larger font
        cv2.putText(panel, "ID", (id_col_x, header_y), self.FONT,
                    0.9, self.header_color, 2, cv2.LINE_AA)
        cv2.putText(panel, "Type", (type_col_x, header_y), self.FONT,
                    0.9, self.header_color, 2, cv2.LINE_AA)
        
        # Draw thick divider line below header
        divider_y = header_y + 15
        cv2.line(panel, (left_margin - 10, divider_y), 
                (panel_width - left_margin, divider_y), 
                (255, 255, 255), 3)
        
        # Draw table rows from detections
        row_start_y = divider_y + 50
        row_spacing = 45
        
        if len(detections) == 0:
            # Show "No detections" message if empty
            cv2.putText(panel, "No detections", (left_margin, row_start_y), 
                       self.FONT, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        else:
            y = row_start_y
            for idx, det in enumerate(sorted(detections, key=lambda d: d['track_id'])):
                # Stop if we run out of space
                if y > panel_height - 50:
                    break
                
                track_id = str(det['track_id'])
                class_name = det['class_name'].upper()  # Uppercase for visibility
                
                # Draw ID in BRIGHT WHITE with larger font
                cv2.putText(panel, track_id, (id_col_x, y), self.FONT,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Draw Type in BRIGHT GREEN with larger font
                cv2.putText(panel, class_name, (type_col_x, y), self.FONT,
                            0.8, self.default_color, 2, cv2.LINE_AA)
                
                y += row_spacing
        
        # Draw count at bottom
        count_y = panel_height - 30
        count_text = f"Total: {len(detections)}"
        cv2.putText(panel, count_text, (left_margin, count_y), 
                   self.FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        return panel

    @staticmethod
    def _draw_divider(panel: np.ndarray, y: int, 
                      color: Tuple[int, int, int] = (60, 60, 70)) -> None:
        """Draw horizontal divider line."""
        cv2.line(panel, (10, y), (panel.shape[1] - 10, y), color, 1)
    
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