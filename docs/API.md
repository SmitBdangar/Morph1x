# Morph1x API Documentation

## Overview

Morph1x is a real-time video tracking system that detects and tracks living beings using YOLOv8 object detection. This document provides detailed API documentation for all components.

## Core Classes

### LivingBeingDetector

The main detection class that handles YOLOv8-based object detection.

```python
from src.detection import create_detector

detector = create_detector(model_path="path/to/model.pt")
```

#### Methods

##### `__init__(model_path: str = None)`
Initialize the detector with a YOLOv8 model.

**Parameters:**
- `model_path` (str, optional): Path to YOLOv8 model file. Defaults to config setting.

**Raises:**
- `Exception`: If model loading fails

##### `detect_living_beings(frame: np.ndarray) -> List[Dict]`
Detect living beings in a video frame.

**Parameters:**
- `frame` (np.ndarray): Input video frame in BGR format

**Returns:**
- `List[Dict]`: List of detection dictionaries with keys:
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `confidence`: Detection confidence score (0.0-1.0)
  - `class_id`: COCO class ID
  - `class_name`: Human-readable class name

**Example:**
```python
detections = detector.detect_living_beings(frame)
for det in detections:
    print(f"Found {det['class_name']} with confidence {det['confidence']:.2f}")
```

##### `filter_living_beings(detections: List[Dict]) -> List[Dict]`
Filter detections to only include living beings.

**Parameters:**
- `detections` (List[Dict]): List of all detections

**Returns:**
- `List[Dict]`: Filtered list containing only living beings

##### `get_detection_summary(detections: List[Dict]) -> Dict[str, int]`
Get a summary count of detected living beings by type.

**Parameters:**
- `detections` (List[Dict]): List of detection dictionaries

**Returns:**
- `Dict[str, int]`: Dictionary with class names as keys and counts as values

### DetectionTracker

Simple tracker to maintain detection history and reduce flickering.

```python
from src.detection import DetectionTracker

tracker = DetectionTracker(max_history=5)
```

#### Methods

##### `__init__(max_history: int = 5)`
Initialize the tracker.

**Parameters:**
- `max_history` (int): Maximum number of frames to keep in history

##### `update(detections: List[Dict]) -> List[Dict]`
Update detection history and return smoothed detections.

**Parameters:**
- `detections` (List[Dict]): Current frame detections

**Returns:**
- `List[Dict]`: Smoothed detections based on history

##### `clear_history()`
Clear detection history.

### VideoTracker

Main application class for video processing.

```python
from src.main import VideoTracker

tracker = VideoTracker(
    video_source=0,  # webcam
    enable_audio=True,
    output_path="output.mp4"
)
```

#### Methods

##### `__init__(video_source, model_path=None, enable_audio=True, enable_tts=False, output_path=None)`
Initialize the video tracker.

**Parameters:**
- `video_source` (Union[int, str]): Video source (camera index or file path)
- `model_path` (str, optional): Path to YOLOv8 model
- `enable_audio` (bool): Enable audio feedback
- `enable_tts` (bool): Enable text-to-speech announcements
- `output_path` (str, optional): Path to save output video

##### `run() -> bool`
Run the main video tracking loop.

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
success = tracker.run()
if success:
    print("Video tracking completed successfully")
```

## Utility Functions

### Drawing Functions

#### `draw_detection_box(frame: np.ndarray, detection: Dict, color: Tuple[int, int, int] = None) -> np.ndarray`
Draw a bounding box and label for a detection on the frame.

**Parameters:**
- `frame` (np.ndarray): Input video frame
- `detection` (Dict): Detection dictionary
- `color` (Tuple[int, int, int], optional): BGR color tuple for the box

**Returns:**
- `np.ndarray`: Frame with drawn detection box and label

#### `draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray`
Draw all detections on the frame.

**Parameters:**
- `frame` (np.ndarray): Input video frame
- `detections` (List[Dict]): List of detection dictionaries

**Returns:**
- `np.ndarray`: Frame with all detections drawn

#### `draw_info_panel(frame: np.ndarray, fps: float = None, detection_count: int = None, detection_summary: Dict[str, int] = None) -> np.ndarray`
Draw information panel on the frame.

**Parameters:**
- `frame` (np.ndarray): Input video frame
- `fps` (float, optional): Current FPS
- `detection_count` (int, optional): Number of detections
- `detection_summary` (Dict[str, int], optional): Summary of detections by type

**Returns:**
- `np.ndarray`: Frame with information panel

### Audio Functions

#### `create_audio_feedback(enabled: bool = None) -> AudioFeedback`
Factory function to create an AudioFeedback instance.

**Parameters:**
- `enabled` (bool, optional): Whether audio feedback is enabled

**Returns:**
- `AudioFeedback`: AudioFeedback instance

#### `create_detection_announcer(enabled: bool = False) -> DetectionAnnouncer`
Factory function to create a DetectionAnnouncer instance.

**Parameters:**
- `enabled` (bool, optional): Whether TTS is enabled

**Returns:**
- `DetectionAnnouncer`: DetectionAnnouncer instance

## Configuration

### Primary Living Beings

The system detects the following living beings:

```python
PRIMARY_LIVING_BEINGS = {
    0: "person",      # Human
    15: "cat",        # Cat
    16: "dog",        # Dog
    17: "horse",      # Horse
    18: "sheep",      # Sheep
    19: "cow",        # Cow
    20: "elephant",   # Elephant
    21: "bear",       # Bear
    22: "zebra",      # Zebra
    23: "giraffe",    # Giraffe
}
```

### Detection Settings

```python
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detections
IOU_THRESHOLD = 0.45          # Non-maximum suppression threshold
MAX_DETECTIONS = 100          # Maximum number of detections per frame
```

### Display Settings

```python
BOX_COLOR = (0, 255, 0)       # Green color for bounding boxes (BGR)
TEXT_COLOR = (255, 255, 255)  # White text
BOX_THICKNESS = 2             # Bounding box thickness
TEXT_THICKNESS = 2            # Text thickness
FONT_SCALE = 0.7              # Font scale
```

## Error Handling

All functions include proper error handling and logging. Common exceptions:

- `FileNotFoundError`: When model or video files are not found
- `cv2.error`: When OpenCV operations fail
- `ImportError`: When required dependencies are missing

## Performance Considerations

- Processing time depends on frame size and hardware
- GPU acceleration is recommended for real-time performance
- Adjust `PROCESS_EVERY_N_FRAMES` for lower-end systems
- Reduce `MAX_FRAME_SIZE` for faster processing

## Examples

### Basic Detection

```python
import cv2
from src.detection import create_detector

# Initialize detector
detector = create_detector()

# Load image
image = cv2.imread("test_image.jpg")

# Detect living beings
detections = detector.detect_living_beings(image)

# Print results
for det in detections:
    print(f"Found {det['class_name']} with confidence {det['confidence']:.2f}")
```

### Video Processing

```python
from src.main import VideoTracker

# Process video file
tracker = VideoTracker(
    video_source="input_video.mp4",
    output_path="output_video.mp4",
    enable_audio=True
)

# Run tracking
success = tracker.run()
```

### Custom Configuration

```python
from src.config import CONFIDENCE_THRESHOLD, BOX_COLOR

# Modify settings
CONFIDENCE_THRESHOLD = 0.7  # Higher confidence threshold
BOX_COLOR = (0, 0, 255)    # Red boxes instead of green
```
