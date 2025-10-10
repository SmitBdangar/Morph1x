"""
HUD renderer for drawing detections and system information.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class HUDRenderer:
    """Renders heads-up display and detection visualization."""
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self, config: Dict = None):
        """
        Initialize HUD renderer.
        
        Args:
            config: Configuration dictionary with colors and settings.
        """
        self.config = config or {}
        self.panel_bg_color = tuple(self.config.get("panel_bg_color", [0, 0, 0]))
        self.header_color = tuple(self.config.get("header_color", [0, 255, 180]))
        self.accent_color = tuple(self.config.get("accent_color", [0, 255, 140]))
        self.divider_color = tuple(self.config.get("divider_color", [60, 60, 70]))
        self.class_colors = self.config.get("colors", {})
    
    @staticmethod
    def draw_divider(panel: np.ndarray, y: int, color: Tuple[int, int, int]) -> None:
        """Draw horizontal separator line."""
        cv2.line(panel, (10, y), (panel.shape[1] - 10, y), color, 1)
    
    def draw_panel(
        self,
        panel: np.ndarray,
        title: str,
        active_detections: int,
        active_ids: List[str]
    ) -> np.ndarray:
        """
        Draw the HUD panel with system information.
        
        Args:
            panel: Panel image to draw on.
            title: Panel title.
            active_detections: Number of active detections.
            active_ids: List of active tracking IDs.
        
        Returns:
            Rendered panel image.
        """
        height = panel.shape[0]
        panel[:] = self.panel_bg_color
        
        # Title
        cv2.putText(panel, title, (20, 40), self.FONT, 0.8, self.header_color, 2, cv2.LINE_AA)
        self.draw_divider(panel, 55, self.divider_color)
        
        # Detection Count
        cv2.putText(
            panel,
            f"Active: {active_detections}",
            (15, 85),
            self.FONT,
            0.55,
            self.accent_color,
            1,
            cv2.LINE_AA
        )
        self.draw_divider(panel, 105, self.divider_color)
        
        # IDs List
        cv2.putText(panel, "Tracked IDs:", (15, 130), self.FONT, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
        
        y = 155
        if active_ids:
            for idx, uid in enumerate(active_ids):
                uid = uid.strip()
                if not uid:
                    continue
                
                cv2.putText(panel, f"• {uid}", (25, y), self.FONT, 0.5, self.accent_color, 1, cv2.LINE_AA)
                y += 20
                
                if y > height - 60:
                    remaining = len(active_ids) - idx
                    cv2.putText(panel, f"...+{remaining}", (25, y), self.FONT, 0.5, (100, 100, 100), 1)
                    break
        else:
            cv2.putText(panel, "No objects tracked", (25, y), self.FONT, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        # Footer
        self.draw_divider(panel, height - 40, self.divider_color)
        cv2.putText(panel, "MORPH1X SYSTEM", (20, height - 15), self.FONT, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
        
        return panel
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for class name."""
        color_list = self.class_colors.get(class_name.lower(), [200, 200, 200])
        return tuple(color_list)
    
    def draw_detection_box(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw a detection bounding box with label.
        
        Args:
            frame: Frame to draw on.
            detection: Detection dictionary.
        
        Returns:
            Frame with drawn detection.
        """
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        color = self.get_class_color(class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = detection["unique_id"]
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, self.FONT, font_scale, thickness)
        
        label_y = max(y1 - 5, text_height + 5)
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 6, label_y),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - 3),
            self.FONT,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw all detection boxes on frame.
        
        Args:
            frame: Frame to draw on.
            detections: List of detection dictionaries.
        
        Returns:
            Frame with all detections drawn.
        """
        for detection in detections:
            frame = self.draw_detection_box(frame, detection)
        return frame