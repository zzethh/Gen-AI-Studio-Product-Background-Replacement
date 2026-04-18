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
    assert r.status_code != 200

def test_generate_invalid_pdf():
    # Attempting to upload a PDF byte stream instead of an image
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<\n/Title (Test)\n>>\nendobj\n"
    r = client.post(
        "/generate",
        files={"image": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"prompt": "beach"}
    )
    # PIL.Image.open will fail to read PDF bytes, which should gracefully return an error (400 or 500 level)
    # but strictly NOT 200 OK.
    assert r.status_code != 200

def test_generate_null_prompt():
    # Valid image but missing the required prompt string
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    r = client.post(
        "/generate",
        files={"image": ("test.png", buf, "image/png")},
        # Omitting 'prompt' data
    )
    assert r.status_code == 422  # Missing required field form data

def test_generate_invalid_data_types():
    # Attempting to send non-integer steps strings
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    r = client.post(
        "/generate",
        files={"image": ("test.png", buf, "image/png")},
        data={
            "prompt": "beach",
            "num_inference_steps": "twenty_steps", # Invalid
            "guidance_scale": "very_high" # Invalid
        }
    )
    # FastAPI internal Pydantic strict-type parsing should cleanly trap this before execution
    assert r.status_code == 422

def test_extract_mask_requires_image():
    # Test that the secondary /extract_mask endpoint also correctly enforces image dependencies
    r = client.post("/extract_mask")
    # Body is missing entirely
    assert r.status_code == 422 

def test_upload_blocks_without_hf_token():
    # We patch HF_TOKEN to None to simulate missing environment variables
    with patch("src.local.main.HF_TOKEN", None):
        r = client.post("/upload", data={"repo_id": "test/model"})
        # Should gracefully return a 400 Bad Request regarding the token instead of 500
        assert r.status_code == 400
        assert "HF_TOKEN" in r.json()["detail"]
