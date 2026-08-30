import io
import tempfile
from pathlib import Path
from PIL import Image
import torch
from fastapi.testclient import TestClient

from src.model import SimpleCNN
from src.serve import app, load_model_checkpoint


def create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint_unloaded():
    client = TestClient(app)
    # Ensure not loaded
    import src.serve as serve_module

    serve_module.model = None
    serve_module.is_model_loaded = False
    serve_module.loaded_model_path = None

    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_health_and_predict_with_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        model = SimpleCNN(in_channels=3, num_classes=10)
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "architecture": "cnn",
                "num_classes": 10,
            },
            ckpt_path,
        )

        loaded = load_model_checkpoint(ckpt_path)
        assert loaded is True

        client = TestClient(app)

        # Test health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True

        # Test predict
        img_bytes = create_test_image_bytes()
        files = {"image": ("test.png", img_bytes, "image/png")}
        predict_resp = client.post("/predict", files=files)
        assert predict_resp.status_code == 200
        predict_data = predict_resp.json()
        assert predict_data["status"] == "success"
        assert "prediction" in predict_data
        assert "probabilities" in predict_data
        assert len(predict_data["probabilities"]) == 10
        assert "confidence" in predict_data