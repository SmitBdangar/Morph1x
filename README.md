# Morph1x

An object detection and tracking system powered by YOLOv8 with real-time visualization, API server, and video processing capabilities.

## Features

-  **Real-time Object Detection** - YOLOv8 with 80 COCO classes
-  **Object Tracking** - ByteTrack integration for persistent tracking
-  **Live Visualization** - HUD panel with detection info
-  **REST API** - FastAPI with streaming and detection endpoints
-  **Video Processing** - Batch video processing with output saving
-  **Configuration Management** - YAML-based centralized config
-  **Enterprise Structure** - Production-ready project layout
-  **Logging & Monitoring** - Comprehensive logging system
-  **GPU Support** - CUDA acceleration available

## Project Structure

```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/morph1x.git
cd morph1x

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Download Model

```bash
# YOLOv8 model will auto-download on first run
# Or manually download and place in models/current/
mkdir -p models/current
# Model will be cached after first use
```

### 3. Configure

Edit `config/model_config.yaml`:

```yaml
model:
  path: "models/current/yolov8n.pt"

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"

video:
  source: "C:\\path\\to\\video.mp4"
```

### 4. Run Detection

**CLI Runner (with HUD):**
```bash
python src/scripts/run_detection.py "path/to/video.mp4"
python src/scripts/run_detection.py 
```

**Process Video (save output):**
```bash
python src/scripts/process_video.py input.mp4 -o output.mp4
python src/scripts/process_video.py input.mp4 --no-display
```

**REST API Server:**
```bash
python src/api/main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Deployment Configuration (`config/deployment.yaml`)

```yaml

## API Endpoints

### Health Check
```
GET /health
```

### Detect Objects in Image
```
POST /detect
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "total_detections": 3,
  "detections": [
    {
      "id": "123-P",
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300],
      "track_id": 123
    }
  ],
  "fps": 24.5
}
```
```bash
# Run on video with HUD
python src/scripts/run_detection.py "C:\video.mp4"

# Process and save
python src/scripts/process_video.py input.mp4 -o output.mp4

# Use custom config
python src/scripts/run_detection.py video.mp4 -c config/custom.yaml
```

### REST API

```bash
# Start server
python src/api/main.py

# Test detection
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"

# Stream video
curl "http://localhost:8000/stream" > video.mpg
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/unit/test_detector.py -v
```

## Performance Tips

1. **Frame Size**: Lower `MAX_FRAME_SIZE` for faster processing
   ```yaml
   visualization:
     frame_resize:
       max_width: 640
       max_height: 480
   ```

2. **Model Selection**: Use smaller models for faster inference
   - `yolov8n.pt` - Nano (fastest)
   - `yolov8s.pt` - Small
   - `yolov8m.pt` - Medium
   - `yolov8l.pt` - Large

3. **Batch Processing**: Process multiple frames efficiently
4. **GPU Acceleration**: Ensure CUDA is installed for faster inference
5. **Confidence Threshold**: Increase for fewer detections
   ```yaml
   inference:
     confidence_threshold: 0.6
   ```

## Troubleshooting

### ModuleNotFoundError
```bash
# Add src to Python path or install package
pip install -e .
```

### Model Not Found
```bash
# Download YOLOv8 model
mkdir -p models/current
# Will auto-download on first run
```

### Low FPS
- Reduce frame resolution
- Use smaller model (yolov8n)
- Enable GPU support
- Close other applications

### CUDA Not Available
```bash
# Install GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## License

This project is licensed under the MIT License.

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Morph1x** - Intelligent Vision System for Enterprise Applications
