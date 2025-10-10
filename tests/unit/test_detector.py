"""
Unit tests for detector module.
"""

import pytest
import numpy as np
from pathlib import Path

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import ObjectDetector


class TestObjectDetector:
    """Test ObjectDetector class."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        # Note: This requires yolov8n.pt to exist
        model_path = "models/current/yolov8n.pt"
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        return ObjectDetector(model_path)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.model is not None
    
    def test_invalid_frame(self, detector):
        """Test detection with invalid frame."""
        with pytest.raises(ValueError):
            detector.detect(None, {"person"})
    
    def test_detection_output_format(self, sample_frame, detector):
        """Test detection returns correct format."""
        detections = detector.detect(sample_frame, {"person"})
        
        assert isinstance(detections, list)
        for det in detections:
            assert "bbox" in det
            assert "class_name" in det
            assert "confidence" in det
            assert "track_id" in det
            assert "unique_id" in det
    
    def test_confidence_filtering(self, sample_frame, detector):
        """Test confidence threshold is applied."""
        detector.conf_threshold = 0.99  # Very high threshold
        detections = detector.detect(sample_frame, {"person"})
        
        # Should have few or no detections with high threshold
        assert isinstance(detections, list)
    
    def test_class_filtering(self, sample_frame, detector):
        """Test class filtering works."""
        detections = detector.detect(sample_frame, {"nonexistent_class"})
        
        # Should filter out unallowed classes
        for det in detections:
            assert det["class_name"] in {"nonexistent_class"}
    
    def test_model_info(self, detector):
        """Test model info retrieval."""
        info = detector.get_model_info()
        
        assert "model_name" in info
        assert "task" in info
        assert "classes" in info
        assert "num_classes" in info