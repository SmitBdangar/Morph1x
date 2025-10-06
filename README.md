# Morph1x - Living Being Tracker

A real-time object detection and tracking system focused on detecting living beings (humans and animals) using YOLOv8.

## Features

- **Real-time Detection**: Detect humans and animals in video streams or files
- **Multiple Input Sources**: Webcam, video files, or multiple cameras
- **Audio Feedback**: Sound alerts when detections are found
- **Text-to-Speech**: Optional voice announcements
- **Video Recording**: Save processed videos with detections
- **Configuration Presets**: Optimized settings for different use cases
- **Performance Monitoring**: FPS counter and detection statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 model (if not already present):
The model `yolov8n.pt` should be in the `models/` directory.

## Usage

### Basic Usage

**Webcam (default):**
```bash
python -m src.main
```

**Video file:**
```bash
python -m src.main --source "path/to/video.mp4"
```

**Different camera:**
```bash
python -m src.main --source 1
```

### Configuration Presets

**High Accuracy Mode:**
```bash
python -m src.main --preset high-accuracy
```

**Performance Mode (for low-end devices):**
```bash
python -m src.main --preset performance
```

**Development Mode (with debug info):**
```bash
python -m src.main --preset development
```

### Advanced Options

**Save processed video:**
```bash
python -m src.main --source "input.mp4" --output "output.mp4"
```

**Custom confidence threshold:**
```bash
python -m src.main --confidence 0.6
```

**Enable text-to-speech:**
```bash
python -m src.main --tts
```

**Disable audio feedback:**
```bash
python -m src.main --no-audio
```

**List detectable classes:**
```bash
python -m src.main --list-classes
```

**Verbose logging:**
```bash
python -m src.main --verbose
```

### Controls During Runtime

- `q` or `ESC` - Quit
- `p` - Pause/Resume
- `r` - Reset tracker

## Configuration Presets

| Preset | Confidence | IOU | Max Detections | Frame Size | Use Case |
|--------|------------|-----|----------------|------------|----------|
| balanced | 0.5 | 0.45 | 100 | 1280x720 | General use |
| high-accuracy | 0.3 | 0.3 | 200 | 1920x1080 | Maximum detection |
| performance | 0.7 | 0.6 | 50 | 640x480 | Fast processing |
| development | 0.5 | 0.45 | 100 | 1280x720 | Debug mode |

## Project Structure

```
Morph1x-1/
├── src/
│   ├── main.py          # Main entry point
│   ├── detection.py     # YOLOv8 detection logic
│   ├── audio_feedback.py # Audio and TTS functionality
│   ├── utils.py         # Utility functions
│   └── config.py        # Configuration settings
├── models/
│   └── yolov8n.pt      # YOLOv8 model file
├── tests/
│   └── test_detection.py # Unit tests
└── requirements.txt     # Dependencies
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pygame (for audio)
- pyttsx3 (for text-to-speech)

See `requirements.txt` for complete dependency list.
