# Morph1x – YOLO Object Detection

A lightweight object detection project built on **YOLOv8** using OpenCV and Python.  
Supports real-time detection from webcam or video files.

---

## Quick Start


▶ Webcam

python run.py --source 0

▶ Video File

python run.py --source "data/video.mp4

▶ Save Output

Copy code
python run.py --source 0 --output "output/result.mp4"


4️⃣ Quit
Press q to exit the OpenCV window.

Folder Structure

MORPH1X/
├── model/              # YOLO model weights
│   └── yolov8n.pt
├── src/core/           # Core detection logic
│   ├── object_detector.py
│   ├── Post_processing.py
│   ├── renderer.py
├── run.py              # Main entry point
├── requirements.txt
└── README.md