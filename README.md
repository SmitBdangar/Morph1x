# Morph1x – YOLO Object Detection

A lightweight object detection system built on **YOLOv8** with real-time tracking and video processing.

---

## Features

- Real-time object detection with YOLOv8
- Persistent object tracking using ByteTrack
- Webcam and video file support
- Video output with multiple codec support
- Configurable confidence and IoU thresholds
- Multi-class detection with filtering
- CUDA GPU acceleration
- YAML-based configuration
- FPS monitoring

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional)

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/morph1x-yolo-detection.git
cd morph1x-yolo-detection
pip install -r requirements.txt
```

---

## Usage

### Webcam Detection
```bash
python run.py --source 0
```

### Video File
```bash
python run.py --source "data/video.mp4"
```

### Save Output
```bash
python run.py --source 0 --output "output/result.mp4"
```

### Custom Settings
```bash
python run.py --source 0 --conf 0.6 --iou 0.5 --classes person car dog
```

### Batch Process Video
```bash
python -m src.lib.process_video --input "data/input.mp4" --output "output/processed.mp4"
```

Press **q** to quit.

---

## Arguments

### run.py
- `--source` - Video source (0 for webcam or file path)
- `--model` - YOLO model path (default: model/yolov8n.pt)
- `--classes` - Allowed classes (default: person car dog)
- `--output` - Output video path
- `--conf` - Confidence threshold (default: 0.5)
- `--iou` - IoU threshold (default: 0.45)

### process_video.py
- `--input` - Input video path (required)
- `--output` - Output video path (required)
- `--model` - YOLO model path
- `--classes` - Allowed classes
- `--conf` - Confidence threshold
- `--iou` - IoU threshold

---

## Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  path: "models/current/yolov8n.pt"

inference:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 300
  device: "cuda"  # or "cpu"

tracking:
  tracker: "bytetrack"
  max_age: 30
  min_hits: 3

visualization:
  colors:
    person: [0, 200, 0]
    car: [0, 180, 255]
    dog: [255, 120, 50]
```

---

## Project Structure

```
morph1x-yolo-detection/
├── config/
│   └── model_config.yaml
├── src/
│   ├── core/
│   │   ├── object_detector.py
│   │   ├── Post_processing.py
│   │   └── renderer.py
│   └── lib/
│       └── process_video.py
├── model/
│   └── yolov8n.pt
├── run.py
└── requirements.txt
```

---

## Supported Classes

Supports all 80 COCO classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

Filter with `--classes` argument.

---

## Key Features Explained

### Tracking
Objects maintain consistent IDs across frames using ByteTrack. Each detection includes `track_id` and `unique_id` (e.g., "ID-23-P").

### Post-Processing
- Confidence-based filtering
- IoU-based Non-Maximum Suppression per class
- Maximum detection count limiting

### Video Output
Automatic codec fallback: MP4V → H264 → MJPG → XVID → Default

---

## Performance

**Speed:**
```bash
python run.py --source 0 --model model/yolov8n.pt --conf 0.6
```

**Accuracy:**
```bash
python run.py --source video.mp4 --model model/yolov8x.pt --conf 0.3
```

Check GPU: `import torch; torch.cuda.is_available()`

---



## Troubleshooting

**Model not found:** Ensure model exists at `model/yolov8n.pt`

**Can't open video:** Check camera index (0, 1, 2) or file path

**Low FPS:** Use smaller model (yolov8n), lower confidence, enable GPU

**Codec error:** Try `.avi` extension or different output path
