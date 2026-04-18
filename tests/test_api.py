from fastapi.testclient import TestClient
from unittest.mock import patch
from PIL import Image
import io

with patch("src.local.main.StableDiffusionInpaintPipeline"):
    with patch("src.local.main.CLIPModel"):
        with patch("src.local.main.PeftModel"):
            with patch("src.local.main.torch.cuda.memory_allocated", return_value=0):
                from src.local.main import app

client = TestClient(app)

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_generate_requires_image():
    r = client.post("/generate", data={"prompt": "A beautiful landscape"})
    assert r.status_code == 422

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
    assert r.status_code != 200

def test_generate_invalid_pdf():
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<\n/Title (Test)\n>>\nendobj\n"
    r = client.post(
        "/generate",
        files={"image": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"prompt": "beach"}
    )
    assert r.status_code != 200

def test_generate_null_prompt():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    r = client.post(
        "/generate",
        files={"image": ("test.png", buf, "image/png")}
    )
    assert r.status_code == 422

def test_generate_invalid_data_types():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    
    r = client.post(
        "/generate",
        files={"image": ("test.png", buf, "image/png")},
        data={
            "prompt": "beach",
            "num_inference_steps": "twenty_steps",
            "guidance_scale": "very_high"
        }
    )
    assert r.status_code == 422

def test_extract_mask_requires_image():
    r = client.post("/extract_mask")
    assert r.status_code == 422 

def test_upload_blocks_without_hf_token():
    with patch("src.local.main.HF_TOKEN", None):
        r = client.post("/upload", data={"repo_id": "test/model"})
        assert r.status_code == 400
        assert "HF_TOKEN" in r.json()["detail"]
