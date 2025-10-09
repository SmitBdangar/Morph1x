import cv2
import numpy as np
from typing import Dict, List, Tuple


def _draw_rounded_rect(img, rect, color, radius=8, thickness=-1):
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, thickness)
    cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)


 


 


 


def draw_stats_panel(frame: np.ndarray, stats: Dict[str, str]) -> np.ndarray:
    # Small side panel with key stats
    h, w = frame.shape[:2]
    panel_w = 220
    _draw_rounded_rect(frame, (10, 56, panel_w, 120), (25, 25, 25), radius=10, thickness=-1)
    y = 80
    for k, v in stats.items():
        cv2.putText(frame, f"{k}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(frame, f"{v}", (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        y += 22
    return frame


def draw_object_list(frame: np.ndarray, objects: List[Dict]) -> np.ndarray:
    # Right side panel listing tracked objects: ID, class, speed
    h, w = frame.shape[:2]
    panel_w = 280
    x0 = w - panel_w - 10
    y0 = 56
    panel_h = min(h - y0 - 46, 26 + 26 * max(1, len(objects)))
    _draw_rounded_rect(frame, (x0, y0, panel_w, panel_h), (25, 25, 25), radius=10, thickness=-1)
    cv2.putText(frame, "Objects", (x0 + 12, y0 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (220, 220, 220), 1)
    y = y0 + 44
    for obj in objects[: (panel_h - 44) // 26]:
        _id = obj.get('track_id', '-')
        cls = obj.get('class_name', '-')
        if obj.get('speed_m_s') is not None:
            spd_str = f"{obj['speed_m_s']:.2f} m/s"
        elif obj.get('speed_px_s') is not None:
            # Show px/s but label as m/s per request
            spd_str = f"{obj['speed_px_s']:.0f} m/s"
        else:
            spd_str = "-"
        line = f"ID {_id}  {cls}  {spd_str}"
        cv2.putText(frame, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 24
    return frame


def apply_hud(frame: np.ndarray, fps: float, det_count: int, summary: Dict[str, int], objects: List[Dict] = None) -> np.ndarray:
    stats = {
        "FPS": f"{fps:.1f}",
        "Detections": str(det_count),
    }
    if summary:
        top = sorted(summary.items(), key=lambda x: x[1], reverse=True)[:2]
        for i, (cls, cnt) in enumerate(top):
            stats[f"{cls}"] = str(cnt)
    draw_stats_panel(frame, stats)
    if objects:
        draw_object_list(frame, objects)
    return frame


