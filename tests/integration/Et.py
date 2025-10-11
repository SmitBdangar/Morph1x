from PIL import Image
from ultralytics import YOLO

def test_full_pipeline(tmp_path):
    img = Image.new("RGB", (640, 480), color=(0, 255, 0))
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    model = YOLO("yolov8n.pt")

    results = model.predict(source=str(img_path))

    assert results is not None
    assert len(results) > 0
    assert hasattr(results[0], "boxes")

    output_path = tmp_path / "result.jpg"
    results[0].save(filename=str(output_path))
    assert output_path.exists()
