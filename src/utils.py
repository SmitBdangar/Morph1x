import cv2
import numpy as np
import yaml
import time
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FPSMeter:
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
    
    def update(self) -> None:
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def get_fps(self) -> float:
        return round(self.fps, 2)
    
    def reset(self) -> None:
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0


def validate_frame(frame: Optional[np.ndarray]) -> bool:
    return (
        frame is not None
        and isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.size > 0
    )

def resize_frame(frame: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    if not validate_frame(frame):
        raise ValueError("Invalid frame")
    
    max_width, max_height = max_size
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    
    return frame


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict, config_path: str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {config_path}")


def create_directories(directory_structure: Dict) -> None:
    for name, path in directory_structure.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")