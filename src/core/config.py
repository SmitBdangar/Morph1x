import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
 

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

 

PRIMARY_LIVING_BEINGS = {
    0: "person",    
    15: "cat",     
    16: "dog",       
    17: "horse",     
    18: "sheep",    
    19: "cow",       
    20: "elephant",  
    21: "bear",     
    22: "zebra",      
    23: "giraffe",   
}

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
FONT_SCALE = 0.7
TEXT_PADDING = 5

DEFAULT_VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
OUTPUT_VIDEO_FPS = 30
SHOW_FPS = True
SHOW_DETECTION_COUNT = True

 

PROCESS_EVERY_N_FRAMES = 1
MAX_FRAME_SIZE = (1280, 720) 

LOG_LEVEL = "INFO"
LOG_DETECTIONS = True

# Real-world scale calibration
# Either set METERS_PER_PIXEL directly, or pass via CLI.
# Default 0.0 means unknown scale; speeds shown in px/s only.
METERS_PER_PIXEL = 0.0