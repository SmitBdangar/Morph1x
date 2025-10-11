import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import FPSMeter, validate_frame, resize_frame
import time


class TestFPSMeter:
    
    def test_fps_meter_initialization(self):
        fps_meter = FPSMeter()
        assert fps_meter.fps == 0.0
        assert fps_meter.frame_count == 0
    
    def test_fps_calculation(self):
        fps_meter = FPSMeter()
        
        for _ in range(30):
            fps_meter.update()
        
        time.sleep(1.1)
        
        fps = fps_meter.get_fps()
        assert fps >= 0
    
    def test_fps_reset(self):
        fps_meter = FPSMeter()
        fps_meter.update()
        fps_meter.reset()
        
        assert fps_meter.frame_count == 0
        assert fps_meter.fps == 0.0


class TestFrameValidation:
    
    def test_valid_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validate_frame(frame) is True
    
    def test_none_frame(self):
        assert validate_frame(None) is False
    
    def test_invalid_dimensions(self):
        frame = np.zeros((480, 640), dtype=np.uint8) 
        assert validate_frame(frame) is False
    
    def test_invalid_channels(self):
        frame = np.zeros((480, 640, 4), dtype=np.uint8) 
        assert validate_frame(frame) is False
    
    def test_empty_frame(self):
        frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        assert validate_frame(frame) is False


class TestFrameResize:
    
    def test_resize_upscaling_ignored(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        resized = resize_frame(frame, (1280, 720))
        
        assert resized.shape == frame.shape
    
    def test_resize_downscaling(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = resize_frame(frame, (640, 480))
        
        assert resized.shape[0] <= 480
        assert resized.shape[1] <= 640
    
    def test_resize_aspect_ratio(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        original_ratio = frame.shape[1] / frame.shape[0]
        
        resized = resize_frame(frame, (640, 480))
        resized_ratio = resized.shape[1] / resized.shape[0]
        assert abs(original_ratio - resized_ratio) < 0.01
    
    def test_resize_invalid_frame(self):
        with pytest.raises(ValueError):
            resize_frame(None, (640, 480))