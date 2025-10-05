# Data Directory

This directory contains sample data, examples, and resources for the Morph1x project.

## Directory Structure

```
data/
├── README.md              # This file
├── sample_images/         # Sample images for testing
├── sample_videos/         # Sample videos for testing
├── models/               # Additional model files
├── configs/              # Configuration examples
└── examples/             # Example scripts and usage
```

## Sample Data

### Images
- `sample_images/` - Contains test images with people and animals
- Use these for testing detection accuracy
- Format: JPG, PNG, BMP

### Videos
- `sample_videos/` - Contains sample video files
- Use these for testing video processing
- Format: MP4, AVI, MOV

## Usage Examples

### Testing with Sample Images
```python
import cv2
from src.detection import create_detector

detector = create_detector()
image = cv2.imread("data/sample_images/test_image.jpg")
detections = detector.detect_living_beings(image)
```

### Testing with Sample Videos
```bash
python -m src.main --source "data/sample_videos/test_video.mp4" --output "output.mp4"
```

## Adding Your Own Data

1. **Images**: Place your test images in `sample_images/`
2. **Videos**: Place your test videos in `sample_videos/`
3. **Models**: Place custom models in `models/`
4. **Configs**: Place configuration files in `configs/`

## File Formats Supported

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

### Models
- PyTorch (.pt)
- ONNX (.onnx)
- TensorFlow (.pb)

## Notes

- Keep file sizes reasonable for testing
- Use descriptive filenames
- Include metadata when possible
- Respect copyright and licensing for sample data
