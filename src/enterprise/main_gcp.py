
import os
import io
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from PIL import Image, ImageFilter
from rembg import remove
from huggingface_hub import HfApi
import time
import csv
from transformers import CLIPProcessor, CLIPModel
from src.tracking_gcp import GCPObservability
from src.storage_gcp import download_model_from_gcs, upload_image_to_gcs
import threading
app = FastAPI(title='DLOps Gen-AI Studio — Background Replacement API (GCP Native)')
gcp_obs = GCPObservability(project_id='project-c5eebb76-bcc6-4730-840', location='us-central1', experiment_name='sd-inpainting')

def poll_vram():
    while True:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0)
            gcp_obs.stream_custom_metric_to_cloud_monitoring('gpu_vram_usage_bytes', used)
        time.sleep(5)
threading.Thread(target=poll_vram, daemon=True).start()
pipe = None
clip_model = None
clip_processor = None
MODEL_ID = os.environ.get('MODEL_ID', 'runwayml/stable-diffusion-inpainting')
LORA_DIR = os.environ.get('LORA_DIR', 'models/fashion-lora')
HF_TOKEN = os.environ.get('HF_TOKEN', None)

@app.on_event('startup')
def load_model():
    global pipe, clip_model, clip_processor
    print(f'Loading inpainting model: {MODEL_ID}')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, safety_checker=None, use_auth_token=HF_TOKEN).to('cuda')
    print('Loading CLIP model for quantitative evaluation...')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda')
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    if (os.path.exists(LORA_DIR) and os.path.exists(os.path.join(LORA_DIR, 'adapter_model.safetensors'))):
        print(f'Loading LoRA adapter from: {LORA_DIR}')
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR, adapter_name='overfit')
            print("LoRA adapter 'overfit' loaded successfully using PEFT!")
            light_lora_dir = 'models/fashion-lora-light'
            if (os.path.exists(light_lora_dir) and os.path.exists(os.path.join(light_lora_dir, 'adapter_model.safetensors'))):
                pipe.unet.load_adapter(light_lora_dir, adapter_name='light')
                print("LoRA adapter 'light' loaded successfully using PEFT!")
            if hasattr(pipe.unet, 'set_adapter'):
                pipe.unet.set_adapter('overfit')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'LoRA loading failed: {e}. Using base inpainting model.')
    else:
        print(f'No local adapter found. Triggering GCS Download Phase...')
        download_model_from_gcs('genai-models', 'fashion-lora/adapter_model.safetensors', LORA_DIR)
    print('Model loaded and ready for inference!')
    os.makedirs('logs', exist_ok=True)
    log_file = 'logs/inference_log.csv'
    if (not os.path.exists(log_file)):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'prompt', 'model_mode', 'inference_steps', 'latency_sec', 'clip_score'])

def resize_to_512(image: Image.Image) -> tuple:
    original_size = image.size
    resized = image.resize((512, 512), Image.LANCZOS)
    return (resized, original_size)

def resize_to_original(image: Image.Image, original_size: tuple) -> Image.Image:
    return image.resize(original_size, Image.LANCZOS)

def create_background_mask(product_image: Image.Image) -> Image.Image:
    fg_rgba = remove(product_image)
    alpha = fg_rgba.split()[(- 1)]
    alpha_eroded = alpha.filter(ImageFilter.MinFilter(size=3))
    mask_array = np.array(alpha_eroded)
    mask_array = np.where((mask_array > 128), 255, 0).astype(np.uint8)
    inverted_mask = (255 - mask_array)
    mask = Image.fromarray(inverted_mask).convert('L')
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    return mask

@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_ID, 'lora': LORA_DIR}
QUALITY_SUFFIX = ', high quality, sharp focus, professional photography, well-lit, detailed background, 4k'
NEGATIVE_PROMPT = 'blurry, low quality, distorted, deformed, disfigured, bad anatomy, watermark, text, duplicate, ghosting, out of focus, pixelated, overexposed, underexposed, cropped'

@app.post('/extract_mask')
async def extract_mask(image: UploadFile=File(..., description='Product image to process')):
    image_bytes = (await image.read())
    product_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    (resized_image, original_size) = resize_to_512(product_image)
    mask = create_background_mask(resized_image)
    buf = io.BytesIO()
    mask.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')

@app.post('/generate')
async def generate(image: UploadFile=File(..., description='Product image to keep'), prompt: str=Form(..., description='Text prompt for the new background'), num_inference_steps: int=Form(50, description='Number of denoising steps'), guidance_scale: float=Form(9.0, description='Guidance scale for CFG'), strength: float=Form(1.0, description='How much to transform the masked area (0-1)'), model_mode: str=Form('baseline', description="Which model version to use: 'baseline', 'overfit', 'light'")):
    if (pipe is None):
        raise HTTPException(status_code=503, detail='Model not loaded yet.')
    try:
        image_bytes = (await image.read())
        product_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        (resized_image, original_size) = resize_to_512(product_image)
        print('Creating background mask with rembg...')
        mask = create_background_mask(resized_image)
        enhanced_prompt = (prompt + QUALITY_SUFFIX)
        print(f"Running inpainting in mode '{model_mode}' with prompt: {enhanced_prompt}")
        start_time = time.time()
        if ((model_mode == 'baseline') and hasattr(pipe.unet, 'disable_adapter')):
            with pipe.unet.disable_adapter():
                result = pipe(prompt=enhanced_prompt, negative_prompt=NEGATIVE_PROMPT, image=resized_image, mask_image=mask, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength)
        else:
            if (hasattr(pipe.unet, 'set_adapter') and (model_mode in ['overfit', 'light'])):
                try:
                    pipe.unet.set_adapter(model_mode)
                except ValueError:
                    print(f"Adapter '{model_mode}' not found. Using default adapter.")
            result = pipe(prompt=enhanced_prompt, negative_prompt=NEGATIVE_PROMPT, image=resized_image, mask_image=mask, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, strength=strength)
        output_image = result.images[0]
        inference_latency = (time.time() - start_time)
        clip_score = 0.0
        if (clip_model and clip_processor):
            inputs = clip_processor(text=[prompt], images=output_image, return_tensors='pt', padding=True).to('cuda')
            with torch.no_grad():
                outputs = clip_model(**inputs)
            clip_score = outputs.logits_per_image[0][0].item()
        print(f'Latency: {inference_latency:.2f}s | CLIP Score: {clip_score:.2f}')
        gcp_obs.log_inference_metrics(prompt=enhanced_prompt, model_mode=model_mode, num_steps=num_inference_steps, latency=inference_latency, clip_score=clip_score)
        output_gcs_uri = upload_image_to_gcs('genai-studio-outputs', 'output_buffer.png', f'inference-{int(time.time())}.png')
        print(f'Saved artifact to {output_gcs_uri}')
        buf = io.BytesIO()
        output_image.save(buf, format='PNG')
        buf.seek(0)
        headers = {'X-Inference-Latency': str(round(inference_latency, 2)), 'X-CLIP-Score': str(round(clip_score, 2))}
        return StreamingResponse(buf, media_type='image/png', headers=headers)
    except torch.cuda.OutOfMemoryError:
        gcp_obs.stream_custom_metric_to_cloud_monitoring('api_errors_total', 1, labels={'type': 'oom'})
        raise HTTPException(status_code=503, detail='GPU out of memory')
    except Exception as e:
        gcp_obs.stream_custom_metric_to_cloud_monitoring('api_errors_total', 1, labels={'type': 'process_failed'})
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/upload')
async def upload_to_hub(repo_id: str=Form('Zenith754/genai-bg-replacement-lora')):
    if (not HF_TOKEN):
        raise HTTPException(status_code=400, detail='HF_TOKEN not found in environment.')
    if (not os.path.exists(LORA_DIR)):
        raise HTTPException(status_code=404, detail='LoRA weights not found locally.')
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True, repo_type='model')
        api.upload_folder(folder_path=LORA_DIR, repo_id=repo_id, repo_type='model', token=HF_TOKEN)
        url = f'https://huggingface.co/{repo_id}'
        return {'status': 'success', 'url': url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if (__name__ == '__main__'):
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
