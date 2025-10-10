"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import FPSMeter, validate_frame, resize_frame
import time


class TestFPSMeter:
    """Test FPSMeter class."""
    
    def test_fps_meter_initialization(self):
        """Test FPS meter initializes correctly."""
        fps_meter = FPSMeter()
        assert fps_meter.fps == 0.0
        assert fps_meter.frame_count == 0
    
    def test_fps_calculation(self):
        """Test FPS calculation."""
        fps_meter = FPSMeter()
        
        # Simulate 30 frames in 1 second
        for _ in range(30):
            fps_meter.update()
        
        time.sleep(1.1)  # Wait for FPS to update
        
        fps = fps_meter.get_fps()
        assert fps >= 0
    
    def test_fps_reset(self):
        """Test FPS reset."""
        fps_meter = FPSMeter()
        fps_meter.update()
        fps_meter.reset()
        
        assert fps_meter.frame_count == 0
        assert fps_meter.fps == 0.0


class TestFrameValidation:
    """Test frame validation function."""
    
    def test_valid_frame(self):
        """Test validation of valid frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validate_frame(frame) is True
    
    def test_none_frame(self):
        """Test validation of None."""
        assert validate_frame(None) is False
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        frame = np.zeros((480, 640), dtype=np.uint8)  # 2D instead of 3D
        assert validate_frame(frame) is False
    
    def test_invalid_channels(self):
        """Test validation of invalid channels."""
        frame = np.zeros((480, 640, 4), dtype=np.uint8)  # 4 channels instead of 3
        assert validate_frame(frame) is False
    
    def test_empty_frame(self):
        """Test validation of empty frame."""
        frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        assert validate_frame(frame) is False


class TestFrameResize:
    """Test frame resizing function."""
    
    def test_resize_upscaling_ignored(self):
        """Test that upscaling is ignored."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        resized = resize_frame(frame, (1280, 720))
        
        # Should not upscale
        assert resized.shape == frame.shape
    
    def test_resize_downscaling(self):
        """Test downscaling."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = resize_frame(frame, (640, 480))
        
        # Should downscale
        assert resized.shape[0] <= 480
        assert resized.shape[1] <= 640
    
    def test_resize_aspect_ratio(self):
        """Test that aspect ratio is maintained."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        original_ratio = frame.shape[1] / frame.shape[0]
        
        resized = resize_frame(frame, (640, 480))
        resized_ratio = resized.shape[1] / resized.shape[0]
        
        # Aspect ratios should be approximately equal
        assert abs(original_ratio - resized_ratio) < 0.01
    
    def test_resize_invalid_frame(self):
        """Test resize with invalid frame."""
        with pytest.raises(ValueError):
            resize_frame(None, (640, 480))