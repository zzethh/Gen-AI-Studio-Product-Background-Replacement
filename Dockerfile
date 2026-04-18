FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Pre-cache Hugging Face Models during Docker build to speed up inference startups
RUN python3 -c "import torch; \
    from diffusers import StableDiffusionInpaintPipeline; \
    from transformers import CLIPModel, CLIPProcessor; \
    StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', torch_dtype=torch.float32); \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Default command (overridden in docker-compose)
CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
