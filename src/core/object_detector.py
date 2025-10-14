import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.45):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, frame: np.ndarray, allowed_classes: Set[str]) -> List[Dict]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame")
        
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            tracker="bytetrack.yaml" 
        )[0]
        
        detections = []
        
        if not hasattr(results, "boxes") or results.boxes is None or results.boxes.id is None:
            return detections
        
        track_ids = results.boxes.id.cpu().numpy().astype(int) 
        
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            if class_name not in allowed_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            track_id = track_ids[i]
            
            class_initial = class_name[0].upper()
            unique_id = f"ID-{track_id}-{class_initial}" 
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": confidence,
                "track_id": track_id,
                "unique_id": unique_id
            })
        
        return detections
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model.model_name,
            "task": self.model.task,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }


class VideoCapture:    
    def __init__(self, source: str):
        self.source = source
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30
        self.total_frames = 0
    
    def open(self) -> bool:
        try:
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
                logger.info(f"Opened camera: {self.source}")
            else:
                video_path = Path(self.source)
                if not video_path.is_absolute():
                    video_path = Path.cwd() / video_path
                
                if not video_path.exists():
                    logger.error(f"Video file not found: {video_path}")
                    return False
                
                self.cap = cv2.VideoCapture(str(video_path))
                logger.info(f"Opened video: {video_path}")
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open source: {self.source}")
                return False
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
            return True
        
        except Exception as e:
            logger.error(f"Error opening source: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self) -> None:
        if self.cap:
            self.cap.release()
            logger.info("Video source released")
    
    def is_valid(self, frame: Optional[np.ndarray]) -> bool:
        return (
            frame is not None
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.size > 0
        )


class VideoWriter:    
    def __init__(self, output_path: str, frame_width: int, 
                 frame_height: int, fps: int):
        self.output_path = output_path
        self.writer = None
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            codecs = [
                ('mp4v', '.mp4'),
                ('H264', '.mp4'),
                ('MJPG', '.avi'),
                ('XVID', '.avi'),
            ]
            
            for codec_name, ext in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name) if codec_name != 'H264' else cv2.VideoWriter_fourcc(*'H264')
                    output_file = output_path.replace('.mp4', ext).replace('.avi', ext)
                    
                    self.writer = cv2.VideoWriter(
                        output_file, fourcc, fps, (frame_width, frame_height)
                    )
                    
                    if self.writer and self.writer.isOpened():
                        logger.info(f"Output writer initialized with {codec_name} codec: {output_file}")
                        break
                    else:
                        self.writer = None
                except Exception as e:
                    logger.warning(f"Codec {codec_name} failed: {e}")
                    self.writer = None
            
            if not self.writer:
                logger.warning("No suitable codec found, trying default...")
                self.writer = cv2.VideoWriter(
                    output_path, 0, fps, (frame_width, frame_height)
                )
    
    def write(self, frame: np.ndarray) -> None:
        if self.writer:
            self.writer.write(frame)
    
    def release(self) -> None:
        if self.writer:
            self.writer.release()
            logger.info("Video writer released")