# tests/integration/test_end_to_end.py
# Full end-to-end pipeline test: image -> detection -> output

import io
from PIL import Image
from ultralytics import YOLO

def test_full_pipeline(tmp_path):
    """Test the complete YOLOv8 pipeline."""
    # Create dummy image
    img = Image.new("RGB", (640, 480), color=(0, 255, 0))
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    # Load a pretrained YOLO model (smallest for speed)
    model = YOLO("yolov8n.pt")

    # Run inference
    results = model.predict(source=str(img_path))

    # Check predictions
    assert results is not None
    assert len(results) > 0
    assert hasattr(results[0], "boxes")

    # Save output
    output_path = tmp_path / "result.jpg"
    results[0].save(filename=str(output_path))
    assert output_path.exists()
