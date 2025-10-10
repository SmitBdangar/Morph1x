# Morph1x - Quick Start Guide

Get Morph1x running in 5 minutes!

## Prerequisites

- Python 3.8+
- Git
- ~3GB disk space (for models)

## 1️⃣ Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/morph1x.git
cd morph1x

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Configure (1 minute)

Edit `config/model_config.yaml`:

```yaml
model:
  path: "models/current/yolov8n.pt"  # Will auto-download

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
```

Edit `config/deployment.yaml`:

```yaml
video:
  source: "path/to/your/video.mp4"  # Or 0 for webcam
```

## 3️⃣ Run! (2 minutes)

### Option A: Real-time Detection with HUD

```bash
# Process video file
python src/scripts/run_detection.py "path/to/video.mp4"

# Or use webcam
python src/scripts/run_detection.py 0

# Or custom config
python src/scripts/run_detection.py "video.mp4" -c config/model_config.yaml
```

**Controls:**
- Press `Q` to quit
- Watch FPS counter and HUD panel in real-time

### Option B: Process Video & Save

```bash
# Process and save output
python src/scripts/process_video.py input.mp4 -o output.mp4

# Or without display (faster)
python src/scripts/process_video.py input.mp4 -o output.mp4 --no-display
```

### Option C: REST API Server

```bash
# Start API server
python src/api/main.py

# API running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## 🎯 What You'll See

### CLI Runner (Option A)

```
[INFO] 2024-01-15 10:23:45 - Initializing Morph1x Detection Runner...
[INFO] 2024-01-15 10:23:46 - Loading YOLOv8 model: models/current/yolov8n.pt
[INFO] 2024-01-15 10:23:50 - Model loaded successfully
[INFO] 2024-01-15 10:23:50 - Starting detection on: video.mp4
[INFO] 2024-01-15 10:23:50 - Video resolution: 1920x1080

[Video window opens with:]
- Detection bounding boxes
- Class labels and confidence scores
- Tracking IDs
- FPS counter
- HUD panel showing active objects
```

### API Server (Option C)

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete

# In browser:
http://localhost:8000/docs
```

## 📊 API Quick Test

```bash
# In another terminal:

# Health check
curl http://localhost:8000/health

# Upload image for detection
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"

# Get model info
curl http://localhost:8000/model/info

# View interactive docs
# Open in browser: http://localhost:8000/docs
```

## 🔧 Troubleshooting

### "Model not found" Error

```bash
# Don't worry! It will auto-download on first run
# Just run again or manually:
cd models/current
# YOLOv8 will cache automatically
```

### "No module named src" Error

```bash
# Option 1: Install package in dev mode
pip install -e .

# Option 2: Run from project root
cd /path/to/morph1x
python src/scripts/run_detection.py video.mp4
```

### Low FPS Performance

```yaml
# Edit config/model_config.yaml

visualization:
  frame_resize:
    max_width: 640      # Reduce from 1280
    max_height: 480     # Reduce from 720

inference:
  confidence_threshold: 0.6  # Increase from 0.5 for fewer detections
```

Or use faster model:
```yaml
model:
  name: "yolov8n"  # Already fastest
  # Or try: yolov8s (slightly slower, more accurate)
```

### CUDA/GPU Issues

```bash
# Check GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU support (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check CUDA installation
nvidia-smi
```

### OpenCV Display Issues (Linux)

```bash
# Install display backend
sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev

# Or use remote display
export DISPLAY=:0
```

## 📚 Common Tasks

### Change Detection Classes

Edit `config/model_config.yaml`:

```yaml
classes:
  allowed:
    - person
    - car
    # - dog    # Removed
    # - cow    # Removed
```

### Increase Detection Sensitivity

```yaml
inference:
  confidence_threshold: 0.3  # Lower = more sensitive (0.0-1.0)
```

### Use Different Model Size

```yaml
model:
  name: "yolov8m"  # Options: n, s, m, l, x (larger = slower but more accurate)
  path: "models/current/yolov8m.pt"
```

### Change Video Source

```bash
# Webcam (usually 0, 1, 2, etc.)
python src/scripts/run_detection.py 0

# Video file
python src/scripts/run_detection.py "/path/to/video.mp4"

# Or edit config/deployment.yaml
video:
  source: "rtsp://camera-ip/stream"  # IP camera
```

### Enable Output Video Saving

```yaml
# config/deployment.yaml
video:
  save_output: true
  output_dir: "data/processed"
```

### Change API Port

```bash
# Change in config/deployment.yaml
api:
  port: 8080  # Instead of 8000
```

Then start API:
```bash
python src/api/main.py
```

Access at: `http://localhost:8080`

## 🧪 Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_detector.py -v

# With coverage report
pytest tests/ --cov=src

# Only unit tests
pytest tests/unit/ -v
```

## 📖 Next Steps

### 1. Learn the Code Structure
- Read `STRUCTURE.md` for project layout
- Explore `src/core/` to understand detection pipeline

### 2. Customize
- Modify `config/model_config.yaml` for your use case
- Edit `src/core/visualization/renderer.py` to customize display

### 3. Build API
- Start `python src/api/main.py`
- Integrate with your application
- Deploy using Docker or cloud platform

### 4. Process Large Datasets

```bash
# Batch process multiple videos
for video in data/raw/*.mp4; do
    python src/scripts/process_video.py "$video" \
      -o "data/processed/$(basename $video)" \
      --no-display
done
```

### 5. Extend Functionality

Add new features to:
- `src/core/detection.py` - Detection logic
- `src/core/visualization/renderer.py` - Visualization
- `src/api/main.py` - API endpoints

## 🎓 Example Workflows

### Workflow 1: Quick Video Analysis

```bash
# 1. Place video in data/raw/
# 2. Process and save
python src/scripts/process_video.py data/raw/video.mp4 \
  -o data/processed/video_analyzed.mp4

# 3. View output video
# Output saved with detections!
```

### Workflow 2: Real-time Stream Processing

```bash
# 1. Start API server
python src/api/main.py

# 2. In another terminal, stream video via API
curl "http://localhost:8000/stream" > stream.mp4

# 3. View in media player
```

### Workflow 3: Web Application Integration

```python
# Your web app
import requests

# Send image to API
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/detect",
        files=files
    )

# Get detections
detections = response.json()
print(f"Found {detections['total_detections']} objects")
```

## 💡 Performance Metrics

### FPS by Model & Device

| Model | CPU | GPU (NVIDIA) |
|-------|-----|-------------|
| YOLOv8n | 5-10 FPS | 60-100 FPS |
| YOLOv8s | 2-5 FPS | 30-60 FPS |
| YOLOv8m | 1-3 FPS | 15-30 FPS |

### Inference Time

- YOLOv8n: ~20-40ms per frame
- GPU ~2-3x faster than CPU
- Larger frames = slower inference

## 🚀 Deployment

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/api/main.py"]
```

Build and run:
```bash
docker build -t morph1x .
docker run -p 8000:8000 morph1x
```

### Cloud Deployment

Supported platforms:
- AWS EC2 (CPU/GPU instances)
- Google Cloud Run (serverless)
- Azure Container Instances
- DigitalOcean

Update `config/deployment.yaml` for cloud settings.

## 📞 Getting Help

### Documentation
- `README.md` - Full documentation
- `STRUCTURE.md` - Project structure details
- Code docstrings - Inline documentation

### Common Issues
- Check `requirements.txt` for dependencies
- Verify config files in `config/`
- Check logs in `logs/morph1x.log`

### Community
- GitHub Issues for bugs
- GitHub Discussions for questions
- Pull Requests for contributions

## 🎉 You're Ready!

Now you can:
- ✅ Run real-time detection
- ✅ Process videos
- ✅ Use REST API
- ✅ Customize detections
- ✅ Extend functionality

**Happy detecting! 🚀**

---

**Need help?** Check the full README.md or STRUCTURE.md for detailed information.