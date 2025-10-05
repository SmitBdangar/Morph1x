# Morph1x

A real-time video tracking system for detecting and tracking living beings using YOLOv8 object detection.

## Overview

Morph1x provides real-time detection and tracking of humans and animals in video streams. The system uses YOLOv8 for accurate object detection and provides visual feedback with bounding boxes and labels.

## Features

- Real-time detection of humans and animals
- Visual feedback with bounding boxes and labels
- Support for webcam and video file input
- Audio alerts and text-to-speech notifications
- Video recording with detection overlays
- Configurable detection parameters
- Performance monitoring and statistics

## Installation

```bash
git clone <repository-url>
cd Morph1x
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run with webcam
python demo.py

# Process video file
python -m src.main --source video.mp4 --output result.mp4
```

### Advanced Options

```bash
# Use specific camera
python -m src.main --source 1

# Disable audio feedback
python -m src.main --no-audio

# Enable text-to-speech
python -m src.main --tts

# Verbose logging
python -m src.main --verbose
```

## Detected Objects

The system can detect the following living beings:
- Person
- Cat, Dog
- Horse, Sheep, Cow
- Elephant, Bear
- Zebra, Giraffe

## Configuration

Edit `src/config.py` to customize:
- Detection confidence threshold
- Visual styling and colors
- Audio settings
- Performance parameters

## Project Structure

```
Morph1x/
├── src/                    # Core application code
├── models/                 # YOLOv8 model files
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebook demos
├── docs/                   # Documentation
└── data/                   # Sample data and examples
```

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- Pygame (optional, for audio)

## Performance

- GPU acceleration recommended for real-time processing
- Adjustable frame processing rate for different hardware
- Configurable detection parameters for speed vs accuracy

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run examples
python data/examples/basic_usage.py

# Interactive demo
jupyter notebook notebooks/object_detection_demo.ipynb
```

## License

MIT License

## Contributing