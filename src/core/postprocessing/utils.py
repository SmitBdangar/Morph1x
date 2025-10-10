"""
Postprocessing utilities for detection results.
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def filter_detections(
    detections: List[Dict],
    conf_threshold: float = 0.5,
    max_detections: int = 300
) -> List[Dict]:
    """
    Filter detections by confidence and max count.
    
    Args:
        detections: List of detection dictionaries.
        conf_threshold: Minimum confidence threshold.
        max_detections: Maximum number of detections to keep.
    
    Returns:
        Filtered list of detections.
    """
    # Filter by confidence
    filtered = [d for d in detections if d.get("confidence", 0) >= conf_threshold]
    
    # Sort by confidence descending
    filtered = sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Limit to max detections
    filtered = filtered[:max_detections]
    
    return filtered


def apply_nms(
    detections: List[Dict],
    iou_threshold: float = 0.45
) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        detections: List of detection dictionaries.
        iou_threshold: IOU threshold for NMS.
    
    Returns:
        Detections after NMS.
    """
    if not detections:
        return []
    
    # Group by class
    class_groups = {}
    for det in detections:
        cls = det.get("class_name", "unknown")
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
    
    # Apply NMS per class
    nms_detections = []
    for cls, dets in class_groups.items():
        nms_dets = _nms_per_class(dets, iou_threshold)
        nms_detections.extend(nms_dets)
    
    return nms_detections


def _nms_per_class(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """
    Apply NMS within a single class.
    
    Args:
        detections: Detections of same class.
        iou_threshold: IOU threshold.
    
    Returns:
        Detections after NMS.
    """
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
    
    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        # Remove detections with high IOU
        sorted_dets = [
            d for d in sorted_dets
            if _iou(current["bbox"], d["bbox"]) < iou_threshold
        ]
    
    return keep


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union.
    
    Args:
        box1: Bounding box (x1, y1, x2, y2).
        box2: Bounding box (x1, y1, x2, y2).
    
    Returns:
        IOU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def format_detections(detections: List[Dict]) -> Dict:
    """
    Format detections for API response.
    
    Args:
        detections: List of detection dictionaries.
    
    Returns:
        Formatted dictionary ready for JSON serialization.
    """
    return {
        "total_detections": len(detections),
        "detections": [
            {
                "id": d.get("unique_id"),
                "class": d.get("class_name"),
                "confidence": round(d.get("confidence", 0), 3),
                "bbox": d.get("bbox"),
                "track_id": d.get("track_id")
            }
            for d in detections
        ]
    }