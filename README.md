# Morph1x – YOLO Object Detection

A lightweight object detection project built on **YOLOv8** using OpenCV and Python.  
Supports real-time detection from webcam or video files with persistent object tracking.

---

## Features

- Real-time object detection using YOLOv8
- Persistent object tracking across frames
- Webcam and video file support
- Optional video output saving
- Configurable confidence and IoU thresholds
- Clean, minimal UI with bounding boxes and tracking IDs

---

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Webcam Detection

```bash
python run.py --source 0
```

### Video File Detection

```bash
python run.py --source "data/video.mp4"
```

### Save Output Video

```bash
python run.py --source 0 --output "output/result.mp4"
```

### Custom Configuration

```bash
python run.py --source 0 --conf 0.6 --iou 0.5 --classes person car
```

---

## Arguments

- `--source`: Video source (0 for webcam or path to video file) - Default: 0
- `--model`: Path to YOLO model file - Default: model/yolov8n.pt
- `--classes`: Allowed classes for detection - Default: person car dog
- `--output`: Optional output video path - Default: output/output.mp4
- `--conf`: Confidence threshold (0-1) - Default: 0.5
- `--iou`: IoU threshold for NMS (0-1) - Default: 0.45

---

## Exit

Press **q** to quit the application and close the detection window.

---

## Requirements

- Python 3.8+
- CUDA capable GPU (optional, CPU supported)
- YOLOv8 model weights