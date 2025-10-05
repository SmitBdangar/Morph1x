"""
Morph1x - Video Tracking for Living Beings

A real-time video tracking system that detects and tracks living beings
(humans, animals) using YOLOv8 object detection.
"""

__version__ = "1.0.0"
__author__ = "Morph1x Team"

from .main import VideoTracker, main
from .detection import LivingBeingDetector, create_detector
from .utils import FPSMeter, draw_detections, draw_info_panel
from .audio_feedback import AudioFeedback, create_audio_feedback
from .config import *

__all__ = [
    'VideoTracker',
    'main',
    'LivingBeingDetector',
    'create_detector',
    'FPSMeter',
    'draw_detections',
    'draw_info_panel',
    'AudioFeedback',
    'create_audio_feedback',
]
