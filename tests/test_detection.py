import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.detection import create_detector, LivingBeingDetector, DetectionTracker
from src.utils import draw_detection_box, draw_detections, FPSMeter, validate_frame
from src.config import PRIMARY_LIVING_BEINGS, CONFIDENCE_THRESHOLD


class TestDetection:
    def setup_method(self):
        self.detector = create_detector()
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        assert self.detector is not None
        assert self.detector.model is not None
        assert self.detector.model_path is not None
    
    def test_detect_living_beings_empty_image(self):
        detections = self.detector.detect_living_beings(self.test_image)
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_detect_living_beings_with_content(self):
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (100, 100, 100), -1)
        
        detections = self.detector.detect_living_beings(self.test_image)
        assert isinstance(detections, list)
        assert len(detections) >= 0
    
    def test_detection_format(self):
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (100, 100, 100), -1)
        
        detections = self.detector.detect_living_beings(self.test_image)
        
        for detection in detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'class_id' in detection
            assert 'class_name' in detection
            
            bbox = detection['bbox']
            assert len(bbox) == 4
            assert all(isinstance(x, (int, np.integer)) for x in bbox)
            
            assert isinstance(detection['confidence'], (float, np.floating))
            assert 0.0 <= detection['confidence'] <= 1.0
            
            assert detection['class_id'] in PRIMARY_LIVING_BEINGS
            
            assert detection['class_name'] == PRIMARY_LIVING_BEINGS[detection['class_id']]
    
    def test_filter_living_beings(self):
        mock_detections = [
            {'class_id': 0, 'class_name': 'person', 'confidence': 0.8, 'bbox': [100, 100, 200, 200]},
            {'class_id': 1, 'class_name': 'bicycle', 'confidence': 0.7, 'bbox': [300, 300, 400, 400]},
            {'class_id': 15, 'class_name': 'cat', 'confidence': 0.9, 'bbox': [500, 500, 600, 600]}
        ]
        
        filtered = self.detector.filter_living_beings(mock_detections)
        assert len(filtered) == 2
        assert any(det['class_name'] == 'person' for det in filtered)
        assert any(det['class_name'] == 'cat' for det in filtered)
        assert not any(det['class_name'] == 'bicycle' for det in filtered)
    
    def test_detection_summary(self):
        mock_detections = [
            {'class_name': 'person', 'confidence': 0.8},
            {'class_name': 'person', 'confidence': 0.7},
            {'class_name': 'cat', 'confidence': 0.9}
        ]
        
        summary = self.detector.get_detection_summary(mock_detections)
        
        assert summary['person'] == 2
        assert summary['cat'] == 1
        assert len(summary) == 2


class TestDetectionTracker:
    def setup_method(self):
        self.tracker = DetectionTracker(max_history=3)
    
    def test_tracker_initialization(self):
        assert self.tracker.max_history == 3
        assert len(self.tracker.detection_history) == 0
    
    def test_tracker_update(self):
        detections = [{'class_name': 'person', 'confidence': 0.8}]
        
        result = self.tracker.update(detections)
        
        assert result == detections
        assert len(self.tracker.detection_history) == 1
    
    def test_tracker_history_limit(self):
        for i in range(5):
            detections = [{'class_name': 'person', 'confidence': 0.8}]
            self.tracker.update(detections)
        
        assert len(self.tracker.detection_history) == 3
    
    def test_tracker_clear_history(self):
        detections = [{'class_name': 'person', 'confidence': 0.8}]
        self.tracker.update(detections)
        
        assert len(self.tracker.detection_history) == 1
        
        self.tracker.clear_history()
        assert len(self.tracker.detection_history) == 0


class TestUtils:
    def test_draw_detection_box(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = {
            'bbox': [100, 100, 200, 200],
            'class_name': 'person',
            'confidence': 0.8
        }
        
        result = draw_detection_box(frame, detection)
        
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
    
    def test_draw_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                'bbox': [100, 100, 200, 200],
                'class_name': 'person',
                'confidence': 0.8
            },
            {
                'bbox': [300, 300, 400, 400],
                'class_name': 'cat',
                'confidence': 0.9
            }
        ]
        
        result = draw_detections(frame, detections)
        
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
    
    def test_fps_meter(self):
        fps_meter = FPSMeter()
        
        assert fps_meter.get_fps() == 0.0
        
        fps_meter.update()
        fps_meter.update()
        
        fps = fps_meter.get_fps()
        assert fps > 0.0
    
    def test_validate_frame(self):
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validate_frame(valid_frame) == True

        assert validate_frame(None) == False
        assert validate_frame(np.zeros((480, 640), dtype=np.uint8)) == False  # 2D
        assert validate_frame(np.zeros((480, 640, 1), dtype=np.uint8)) == False
        assert validate_frame(np.array([])) == False  # Empty


class TestConfig:
    
    def test_primary_living_beings(self):
        assert len(PRIMARY_LIVING_BEINGS) > 0
        assert 0 in PRIMARY_LIVING_BEINGS
        assert PRIMARY_LIVING_BEINGS[0] == 'person'
    
    def test_confidence_threshold(self):
        """Test confidence threshold is valid."""
        assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])