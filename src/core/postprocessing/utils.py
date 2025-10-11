from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def filter_detections(
    detections: List[Dict],
    conf_threshold: float = 0.5,
    max_detections: int = 300
) -> List[Dict]:
    filtered = [d for d in detections if d.get("confidence", 0) >= conf_threshold]
    filtered = sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)
    filtered = filtered[:max_detections]
    return filtered

def apply_nms(
    detections: List[Dict],
    iou_threshold: float = 0.45
) -> List[Dict]:
    
    if not detections:
        return []
    
    class_groups = {}
    for det in detections:
        cls = det.get("class_name", "unknown")
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)

    nms_detections = []
    for cls, dets in class_groups.items():
        nms_dets = _nms_per_class(dets, iou_threshold)
        nms_detections.extend(nms_dets)
    
    return nms_detections


def _nms_per_class(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    
    if not detections:
        return []
    
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
    
    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        sorted_dets = [
            d for d in sorted_dets
            if _iou(current["bbox"], d["bbox"]) < iou_threshold
        ]
    
    return keep

def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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