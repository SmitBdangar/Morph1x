"""
Configuration settings for the video tracking system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
DATA_DIR = PROJECT_ROOT / "data"

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
IOU_THRESHOLD = 0.45  # Non-maximum suppression threshold
MAX_DETECTIONS = 100  # Maximum number of detections per frame

# Living being class IDs in COCO dataset
LIVING_BEING_CLASSES = {
    0: "person",           # person
    1: "bicycle",          # bicycle (optional, can be removed)
    2: "car",              # car (optional, can be removed)
    3: "motorcycle",       # motorcycle (optional, can be removed)
    5: "bus",              # bus (optional, can be removed)
    7: "truck",            # truck (optional, can be removed)
    15: "cat",             # cat
    16: "dog",             # dog
    17: "horse",           # horse
    18: "sheep",           # sheep
    19: "cow",             # cow
    20: "elephant",        # elephant
    21: "bear",            # bear
    22: "zebra",           # zebra
    23: "giraffe",         # giraffe
    24: "backpack",        # backpack (optional, can be removed)
    25: "umbrella",        # umbrella (optional, can be removed)
    26: "handbag",         # handbag (optional, can be removed)
    27: "tie",             # tie (optional, can be removed)
    28: "suitcase",        # suitcase (optional, can be removed)
    29: "frisbee",         # frisbee (optional, can be removed)
    30: "skis",            # skis (optional, can be removed)
    31: "snowboard",       # snowboard (optional, can be removed)
    32: "sports ball",     # sports ball (optional, can be removed)
    33: "kite",            # kite (optional, can be removed)
    34: "baseball bat",    # baseball bat (optional, can be removed)
    35: "baseball glove",  # baseball glove (optional, can be removed)
    36: "skateboard",      # skateboard (optional, can be removed)
    37: "surfboard",       # surfboard (optional, can be removed)
    38: "tennis racket",   # tennis racket (optional, can be removed)
    39: "bottle",          # bottle (optional, can be removed)
    40: "wine glass",      # wine glass (optional, can be removed)
    41: "cup",             # cup (optional, can be removed)
    42: "fork",            # fork (optional, can be removed)
    43: "knife",           # knife (optional, can be removed)
    44: "spoon",           # spoon (optional, can be removed)
    45: "bowl",            # bowl (optional, can be removed)
    46: "banana",          # banana (optional, can be removed)
    47: "apple",           # apple (optional, can be removed)
    48: "sandwich",        # sandwich (optional, can be removed)
    49: "orange",          # orange (optional, can be removed)
    50: "broccoli",        # broccoli (optional, can be removed)
    51: "carrot",          # carrot (optional, can be removed)
    52: "hot dog",         # hot dog (optional, can be removed)
    53: "pizza",           # pizza (optional, can be removed)
    54: "donut",           # donut (optional, can be removed)
    55: "cake",            # cake (optional, can be removed)
    56: "chair",           # chair (optional, can be removed)
    57: "couch",           # couch (optional, can be removed)
    58: "potted plant",    # potted plant (optional, can be removed)
    59: "bed",             # bed (optional, can be removed)
    60: "dining table",    # dining table (optional, can be removed)
    61: "toilet",          # toilet (optional, can be removed)
    62: "tv",              # tv (optional, can be removed)
    63: "laptop",          # laptop (optional, can be removed)
    64: "mouse",           # mouse (optional, can be removed)
    65: "remote",          # remote (optional, can be removed)
    66: "keyboard",        # keyboard (optional, can be removed)
    67: "cell phone",      # cell phone (optional, can be removed)
    68: "microwave",       # microwave (optional, can be removed)
    69: "oven",            # oven (optional, can be removed)
    70: "toaster",         # toaster (optional, can be removed)
    71: "sink",            # sink (optional, can be removed)
    72: "refrigerator",    # refrigerator (optional, can be removed)
    73: "book",            # book (optional, can be removed)
    74: "clock",           # clock (optional, can be removed)
    75: "vase",            # vase (optional, can be removed)
    76: "scissors",        # scissors (optional, can be removed)
    77: "teddy bear",      # teddy bear (optional, can be removed)
    78: "hair drier",      # hair drier (optional, can be removed)
    79: "toothbrush",      # toothbrush (optional, can be removed)
}

# Primary living beings (animals and humans)
PRIMARY_LIVING_BEINGS = {
    0: "person",      # person
    15: "cat",        # cat
    16: "dog",        # dog
    17: "horse",      # horse
    18: "sheep",      # sheep
    19: "cow",        # cow
    20: "elephant",   # elephant
    21: "bear",       # bear
    22: "zebra",      # zebra
    23: "giraffe",    # giraffe
}

# Display settings
BOX_COLOR = (0, 255, 0)  # Green color for bounding boxes (BGR format)
TEXT_COLOR = (255, 255, 255)  # White text
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
FONT_SCALE = 0.7
TEXT_PADDING = 5

# Video settings
DEFAULT_VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
OUTPUT_VIDEO_FPS = 30
SHOW_FPS = True
SHOW_DETECTION_COUNT = True

# Audio settings
ENABLE_AUDIO_FEEDBACK = True
AUDIO_ALERT_FILE = "detection_alert.wav"  # Will be created if not exists

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Process every frame (1) or skip frames for performance
MAX_FRAME_SIZE = (1280, 720)  # Resize frames for better performance

# Logging
LOG_LEVEL = "INFO"
LOG_DETECTIONS = True
