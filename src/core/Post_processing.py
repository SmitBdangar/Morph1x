"""
Post-processing utilities for detections.
"""
from typing import Dict, List, Tuple, Set


class PostProcessor:
    """Post-processing utilities for detections."""
    
    @staticmethod
    def filter_by_confidence(detections: List[Dict], threshold: float = 0.5,
                           max_count: int = 300) -> List[Dict]:
        """Filter detections by confidence threshold and limit count."""
        filtered = [d for d in detections if d["confidence"] >= threshold]
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        return filtered[:max_count]
    
    @staticmethod
    def apply_nms(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
        """Apply non-maximum suppression per class."""
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            cls = det["class_name"]
            by_class.setdefault(cls, []).append(det)
        
        # Apply NMS per class
        nms_results = []
        for dets in by_class.values():
            nms_results.extend(PostProcessor._nms_single_class(dets, iou_threshold))
        
        return nms_results
    
    @staticmethod
    def _nms_single_class(detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """NMS for single class."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)
            sorted_dets = [
                d for d in sorted_dets
                if PostProcessor._iou(current["bbox"], d["bbox"]) < iou_threshold
            ]
        
        return keep
    
    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], 
             box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def format_output(detections: List[Dict]) -> Dict:
        """Format detections for output."""
        return {
            "total_detections": len(detections),
            "detections": [
                {
                    "id": d["unique_id"],
                    "class": d["class_name"],
                    "confidence": round(d["confidence"], 3),
                    "bbox": d["bbox"],
                    "track_id": d["track_id"]
                }
                for d in detections
            ]
        }
    
    @staticmethod
    def get_active_ids(detections: List[Dict]) -> List[str]:
        """Extract unique IDs from detections."""
        ids = list(set([d["unique_id"] for d in detections]))
        ids.sort()
        return ids
    
    @staticmethod
    def process_pipeline(detections: List[Dict], 
                        conf_threshold: float = 0.5,
                        iou_threshold: float = 0.45,
                        max_detections: int = 300) -> List[Dict]:
        """
        Full processing pipeline:
        1. Filter by confidence
        2. Apply NMS
        3. Limit detections
        """
        detections = PostProcessor.filter_by_confidence(
            detections, conf_threshold, max_detections
        )
        detections = PostProcessor.apply_nms(detections, iou_threshold)
        return detections