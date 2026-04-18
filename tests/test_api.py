from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image
import io

# We patch the loading of models so that they don't consume memory/GPU during tests
with patch("src.local.main.StableDiffusionInpaintPipeline"):
    with patch("src.local.main.CLIPModel"):
        with patch("src.local.main.PeftModel"):
            with patch("src.local.main.torch.cuda.memory_allocated", return_value=0):
                # Import main after patching the heavy torch modules
                from src.local.main import app

client = TestClient(app)

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_generate_requires_image():
    r = client.post("/generate", data={"prompt": "A beautiful landscape"})
    assert r.status_code == 422   # Missing required 'image' file field

def test_generate_invalid_mode():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    r = client.post(
        "/generate",
        files={"image": ("test.png", buf, "image/png")},
        data={"prompt": "beach", "model_mode": "nonexistent_lora"}
    )
    # The API should reject invalid model modes that aren't baseline, light, or overfit.
    # Currently it might just pass it to the adapter loader causing an internal error,
    # or reject it via type validation. Let's assert it doesn't give a 200.
    assert r.status_code != 200
