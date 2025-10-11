import io
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "welcome" in response.text.lower()

def test_predict_endpoint():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    result = response.json()
    assert "detections" in result
    assert isinstance(result["detections"], list)
