# Gen-AI Studio — Product Background Replacement

| Model Mode       | CLIP Score | Latency (Inference) | System Status |
|------------------|------------|---------------------|---------------|
| **Baseline**     | ~28.4      | 4.1s                | Active        |
| **Light LoRA**   | ~27.1      | 4.4s                | Active        |
| **Overfit LoRA** | ~14.2      | 4.4s                | Failing (Gate)|

AI-powered product background replacement using Stable Diffusion Inpainting + LoRA fine-tuning, with a full MLOps pipeline. The project demonstrates end-to-end ML deployment — from data preparation and model fine-tuning to experiment tracking, containerization, and a live interactive UI — while showcasing a real-world failure mode (Catastrophic Forgetting) and its mitigation.

## Evaluating Results Live

The system streams formal generative metrics (`CLIP Score`, `Latency`) for every completed inference natively. 

**If you are looking at the code locally:**
Just open `eval/eval_results.jsonl` in your standard text editor or IDE (VS Code, PyCharm, etc.) to view the metric records.

**If you want to observe it live on the Google Cloud VM (Demonstration Mode):**
Since the backend is running live on our server (`34.45.215.233`), you can actually watch the metric evaluations stream into the file in real-time while you actuate the Streamlit UI.

If you SSH into the VM, type this command:
```bash
tail -f eval/eval_results.jsonl
```

## Architecture

> [!IMPORTANT]
> **GPU Compatibility**: This project runs on NVIDIA GTX 1080 Ti (11GB VRAM, compute capability 6.1) using **Float32** for both training and inference.

```
┌─────────────────┐     ┌──────────────────────────────────────┐     ┌──────────────┐
│   Streamlit UI  │────▶│          FastAPI Backend              │────▶│  GPU (CUDA)  │
│  (Image Upload  │     │                                      │     │              │
│  + 3-Way Model  │◀────│  1. rembg → extract product mask     │◀────│  SD Inpaint  │
│    Selector)    │     │  2. Invert → background mask         │     │  + PEFT LoRA │
│                 │     │  3. SD Inpainting → new background   │     └──────────────┘
│  Before/After   │     │  4. Return composite image           │
└─────────────────┘     │  5. Expose Telemetry Metrics         │
                        └──────────────────────────────────────┘

### The Dual-Path MLOps Evolution
As a final phase, this repository was restructured to demonstrate the difference between **Local IaaS Deployments** and **Google Cloud Native (PaaS/SaaS) Enterprise Deployments**:
* **`deployments/local/`**: Runs locally using standalone Docker containers for `MLflow` (Tracking) and `Prometheus` (Monitoring).
* **`deployments/gcp/`**: Demonstrates the Enterprise PaaS evolution, actively decoupling the monolith by utilizing `Vertex AI Experiments` for model tracking and `Google Cloud Monitoring` for system telemetry natively without managing containers.
```

---

## Input / Output Specification

### What Input Does It Expect?

| Input | Description |
|---|---|
| **Product Image** | A photo of a single product (PNG, JPG, JPEG, or WebP). |
| **Text Prompt** | A natural-language description of the desired new background. |
| **Model Mode** | One of three modes: `Baseline`, `Light LoRA`, or `Overfit LoRA`. |
| **Inference Steps** | Number of denoising iterations (default: 50). Higher = sharper but slower. |
| **Guidance Scale** | How strictly the model follows the text prompt (default: 9.0). |
| **Strength** | How much of the masked background to replace (default: 1.0 = full replacement). |

**Best results with:**
- A product photographed on a **white or solid-color background** (standard e-commerce studio shots).
- The product should be a **single, clearly visible object** — the AI needs a clean boundary between product and background.

**Also works with:**
- Product close-ups with sharp focus and studio lighting.
- Products on simple, uncluttered backgrounds (e.g., a plain wooden table).

**Will struggle with:**
- Products in busy, complex scenes with many overlapping objects.
- Transparent or highly reflective products (glass bottles, mirrors) where `rembg` cannot distinguish foreground from background.

### What Does It Output?

The system returns a **single composite image** (PNG) at the original input resolution where:
- The **product is preserved pixel-perfectly** in its exact position and shape.
- The **background is entirely replaced** with an AI-generated scene matching the text prompt.

### What to Expect in the Output?

| Model Mode | Expected Output Quality |
|---|---|
| **Baseline (No LoRA)** | High-quality, diverse scene generation. The base Stable Diffusion model produces realistic backgrounds (beaches, forests, studios) with natural lighting and sharp details. This is the strongest generalist. |
| **Light LoRA (350 steps)** | Subtle style adaptation. The model learns basic product-lighting cues from the fashion dataset while still preserving its ability to generate diverse, natural backgrounds. Differences from Baseline are intentionally subtle. |
| **Overfit LoRA (44,000 steps)** | Deliberately degraded output demonstrating **Catastrophic Forgetting**. The model was trained for too many steps on close-up product images and has forgotten how to generate natural scenes. Backgrounds appear blurry, zoomed-in, and washed out. This is kept intentionally as a teaching demonstration. |

---

## System Generalization & Evaluation Metrics
To explicitly evaluate how well the system accomplishes its core goal (**preserve the product, generate a realistic background**), we enforce a dual-metric evaluation pipeline built natively into our custom API headers:

1. **Deterministic Structure Preservation (rembg)**: To guarantee generalization across products regardless of the model's hallucinations, we decouple product protection from the generative model. By using `rembg` (U2Net) to create a perfect alpha-mask layer, the Stable Diffusion latent representation is strictly mathematically prevented from altering the foreground subject.
2. **Semantic Verification (CLIP Scoring)**: We evaluate the generalization of the generated background using `openai/clip-vit-base-patch32`. After generation, the composite output is passed alongside the user's text prompt into the CLIP vision-language model. This produces a quantitative **CLIP Score** (typically 20-30). This allows us to statistically track if our generic prompts are successfully represented in the visual generation over time.

---

## How It Works

1. **Upload** a product photo (e.g., sneaker on white background).
2. **Describe** the new background: *"A sunny beach with palm trees and blue sky"*.
3. **Select Model Version** to compare fine-tuning behavior.
4. **AI processes**:
   - `rembg` (U2Net) auto-detects the product and creates a foreground mask.
   - The mask is inverted to identify the background region.
   - Stable Diffusion Inpainting replaces **only** the background using dynamically loaded PEFT adapters.
5. **Result**: Product preserved perfectly on a new AI-generated background.

---

## The MLOps Narrative: Catastrophic Forgetting

A central theme of this project is observing and mitigating **fine-tuning-induced distribution collapse**—commonly known as catastrophic forgetting. By utilizing three parallel deployments, the system allows for real-time tracking of training dynamics:

- **The Base Model**: Trained on billions of diverse images, demonstrating excellent zero-shot generalization across any background prompt.
- **The Light LoRA Adapter**: Fine-tuned carefully for just 350 steps. This adapter successfully integrates domain-specific lighting and contrast cues from the fashion dataset while safely preserving the model's structural generalization.
- **The Overfit LoRA Adapter**: Deliberately over-trained for 44,000 steps on narrow fashion close-ups. This adapter explicitly demonstrates the collapse of the UNet's latent distribution; it entirely forgets how to synthesize natural backgrounds (e.g., beaches, forests), reverting to producing blurry, zoomed-in artifacts.

This juxtaposition actively demonstrates why continuous experiment validation (via MLflow) and robust fine-tuning guardrails are indispensable in enterprise MLOps.

---

## Local Setup & Quick Start

The complete pipeline can be fully reproduced locally.

### Environment Requirements
- Python 3.10 via `conda`
- NVIDIA GPU with ≥11GB VRAM (Float32 precision)

### Quick Execution
```bash
conda activate genai
pip install -r requirements.txt
source commands.sh
start_all 
```

### Start Services
```bash
source commands.sh
start_all        # Starts MLflow + FastAPI + Streamlit
start_ngrok      # Creates public tunnel URL
```

### Testing & Validation
The project includes a robust, offline Pytest suite (`tests/test_api.py`) that strictly validates API logic, payload enforcement, and data casting without relying on heavy GPU models.

```bash
pip install pytest httpx
PYTHONPATH=. pytest tests/ -v
```
**Test Coverage Includes:**
1. Health Endpoint validation (`/health`)
2. Missing Form/Payload logic parsing (`422 Unprocessable`)
3. Model Parameter Guardrails (Rejecting non-registered LoRA endpoints)
4. Input Extension Spoofing (Safely rejecting `.pdf` parsing)
5. Pydantic Null Value handling
6. Strict Type Casting enforcement
7. Sub-Endpoint Payload drops (`/extract_mask`)
8. Authentication drops cleanly bypassing 500s (`/upload`)

---

## Live Google Cloud GPU Access

This project is currently deployed on a **Google Cloud n1-standard-4 Spot VM** with an attached **NVIDIA Tesla T4 GPU**.

| Service | Access URL | Description |
|---|---|---|
| **Streamlit UI** | [http://34.45.215.233:8501](http://34.45.215.233:8501) | Interactive frontend Gen-AI UI |
| **FastAPI Backend** | [http://34.45.215.233:8000/docs](http://34.45.215.233:8000/docs) | Raw Swagger inference API |
| **MLflow Tracking** | [http://34.45.215.233:5000](http://34.45.215.233:5000) | Live SQLite-backed experiment tracking (DNS Rebinding Check Disabled) |
| **Grafana Dashboard** | [http://34.45.215.233:3000](http://34.45.215.233:3000) | Prometheus live hardware telemetry (Login: admin / admin) |

*If local scripts are required, refer to `start_gcp.sh`!*

### Train LoRA Models
```bash
source commands.sh
train_light      # 350-step balanced fine-tune → models/fashion-lora-light
train_full       # Full 1-epoch fine-tune → models/fashion-lora (overfit demo)
```

### Stop Services
```bash
source commands.sh
stop_all
```

---

## Project Structure
```
dlops_project/
├── deployments/
│   ├── gcp/                   # Native GCP Cloud (Vertex/Storage) deployment configs
│   │   ├── cloudbuild.yaml
│   │   └── gcp-architecture.txt
│   └── local/                 # IaaS Local Dockerized deployment configs
│       ├── docker-compose.yml 
│       └── prometheus.yml
├── src/
│   ├── local/                 
│   │   ├── main.py            # Local FastAPI Backend (Prometheus, CSV)
│   │   └── app.py             # Local Streamlit UI (SQLite Fetching)
│   ├── enterprise/
│   │   ├── main_gcp.py        # Cloud FastAPI Backend (GCS, Managed Monitoring)
│   │   ├── app_gcp.py         # Cloud Streamlit UI (Vertex Fetching)
│   │   ├── tracking_gcp.py    # Vertex AI telemetry & alerting abstractions
│   │   └── storage_gcp.py     # Cloud Storage (GCS) weight loading mechanisms
│   ├── prepare_dataset.py
│   └── train_lora.py
├── models/                    # Fine-tuned adapter weights
├── start_gcp.sh               # Active Remote Run configuration
```

---

## API Endpoints

### `GET /health`
Returns model load status.

### `POST /generate`
| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | File | required | Product image (PNG/JPG/JPEG/WebP) |
| `prompt` | string | required | Background description in natural language |
| `num_inference_steps` | int | 50 | Denoising steps (higher = sharper, slower) |
| `guidance_scale` | float | 9.0 | Classifier-Free Guidance strength |
| `strength` | float | 1.0 | How much of the background to replace (0–1) |
| `model_mode` | string | "baseline" | `baseline`, `light`, or `overfit` |

### `POST /upload`
Push trained LoRA weights to Hugging Face Hub.

---

## Tech Stack
| Component | Technology |
|---|---|
| **Base Model** | Stable Diffusion Inpainting v1.5 (`runwayml/stable-diffusion-inpainting`) |
| **Fine-Tuning** | PEFT LoRA (Low-Rank Adaptation of Large Language Models, applied to UNet attention layers) |
| **Background Removal** | rembg (U2Net neural network) |
| **Training Framework** | PyTorch 2.1.2 + xFormers Memory Efficient Attention |
| **Experiment Tracking** | MLflow |
| **System Monitoring** | Prometheus + Grafana (real-time telemetry: latency, GPU VRAM, request counts) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Public Exposure** | Ngrok |
| **Model Registry** | Hugging Face Hub |

---

## Dataset
- **Name**: [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (Hugging Face)
- **Contents**: ~44,000 fashion product images with captions
- **Usage**: Used to fine-tune LoRA adapters on the UNet attention layers of the Stable Diffusion Inpainting model

## Hugging Face Hub
- **Trained LoRA Weights**: [Zenith754/genai-bg-replacement-lora](https://huggingface.co/Zenith754/genai-bg-replacement-lora)
- The Streamlit sidebar has a **"Push to Hugging Face Hub"** button that uploads the locally trained LoRA adapter to this repository via the `huggingface_hub` API.

---

## How Each Component Works

### 1. Stable Diffusion Inpainting (`runwayml/stable-diffusion-inpainting`)
The base generative model. It takes three inputs — an image, a binary mask, and a text prompt — and regenerates **only the masked region** while preserving everything else. Internally it works by:
- Encoding the image into a 64×64 latent space using a VAE (Variational Autoencoder).
- Adding Gaussian noise to the latent, then iteratively denoising it over N steps (controlled by `num_inference_steps`), guided by a CLIP text embedding of your prompt.
- The mask tells the model which latent pixels to regenerate and which to copy from the original.

### 2. LoRA / PEFT (Parameter-Efficient Fine-Tuning)
Instead of fine-tuning all 860M parameters of the UNet, LoRA **freezes the base weights** and injects small trainable rank-decomposition matrices into the attention layers (`to_q`, `to_k`, `to_v`, `to_out`). This means:
- Only ~0.5% of parameters are trained (~4M out of 860M).
- The adapter weights are tiny files (~17MB each in `models/fashion-lora/`).
- Multiple adapters can be loaded simultaneously and **hot-swapped at runtime** using `pipe.unet.set_adapter("light")` or `pipe.unet.disable_adapter()` — no model reload needed.

### 3. rembg (Background Removal)
A pre-trained U2Net neural network that segments the foreground (product) from the background. It outputs an RGBA image where the alpha channel = foreground mask. We then:
- Erode the mask by 3px to tighten product boundaries.
- Binarize at threshold 128 to remove semi-transparent edge artifacts.
- Invert: White = background (repaint), Black = product (keep).
- Apply Gaussian blur (radius=2) for smooth mask edges.

### 4. FastAPI Backend (`src/main.py`)
A REST API server running on port 8000. On startup it:
- Downloads and loads the Stable Diffusion Inpainting pipeline onto the GPU.
- Loads both LoRA adapters (`overfit` and `light`) into the UNet using PEFT's multi-adapter system.
- Exposes `/generate` (inference), `/health` (status), and `/upload` (push to HuggingFace) endpoints.

### 5. Streamlit Frontend (`src/local/app.py`)
A highly optimized dark-mode Streamlit dashboard running natively on port 8501. It handles dynamic image uploads, form state management across inference runs, and visually structures the resulting generation data using Streamlit metrics and columns.

### 6. Containerization Elements
While currently deployed via raw Compute Engine VMs for optimal GPU scheduling, the project includes fully functional `docker-compose.yml` blueprints. These blueprints orchestrate isolated instances of MLflow, Prometheus, Grafana, and the FastAPI application to ensure platform-agnostic reproducibility.

### 7. Ngrok Integration
Since the underlying GPU server is hosted on a secure private university network, Ngrok creates an authenticated HTTPS tunnel pointing to the local Streamlit port. This allows external validators (invigilators) seamless dashboard access from any device without requiring tedious SSH or VPN configurations.


## MLflow: Experiment Tracking in Detail

### What Is MLflow Doing in This Project?
Every time `train_lora.py` runs, it:
1. **Creates a new MLflow Run** under the experiment `LoRA-Fashion-Inpainting`.
2. **Logs all hyperparameters** as key-value pairs: `learning_rate`, `lora_rank`, `batch_size`, `max_steps`, `resolution`, `model_id`, `dataset_name`, etc.
3. **Logs the training loss** at every single step as a time-series metric (`loss` vs. `global_step`).
4. **Logs the final trained LoRA adapter** as an artifact (the saved weight files).

### What Will the MLflow Dashboard Show?

When you open **http://localhost:5000** in a browser:

- **Experiments Tab**: Lists the `LoRA-Fashion-Inpainting` experiment with all training runs (both the full 44K-step overfit run and the 350-step light run).
- **Run Comparison View**: Click any run to see:
  - **Parameters Panel**: A table showing every hyperparameter used for that training job.
  - **Metrics Panel**: An interactive **loss curve chart** — you can see the training loss decreasing over steps. For the overfit run you can see the loss dropping to near-zero (memorization). For the light run the loss stabilizes at a healthy range (generalization).
  - **Artifacts Panel**: The saved LoRA weight files (`adapter_config.json`, `adapter_model.safetensors`).
- **Compare Runs**: Select multiple runs and click "Compare" to see hyperparameters and loss curves side-by-side — illustrating why the overfit model failed and the light model succeeded.

MLflow is accessible at `http://localhost:5000`, or via SSH tunnel for remote access.

---

## Full Project Workflow

### Phase 1: Before Going Live (Training & Preparation)

```
Dataset (HuggingFace)              Training Script                MLflow Server
     │                                  │                              │
     │  ashraq/fashion-product-         │  src/train_lora.py           │  localhost:5000
     │  images-small                    │                              │
     └──────────────────────────────────▶  1. Load SD Inpainting       │
                                        │  2. Freeze base UNet         │
                                        │  3. Inject LoRA adapters     │
                                        │  4. Train on fashion data    │
                                        │     └─► Log loss/params ─────▶ Stored in mlflow.db
                                        │  5. Save adapter weights     │
                                        │     └─► models/fashion-lora/ │
                                        │     └─► models/fashion-      │
                                        │         lora-light/          │
```

**What happens**:
1. `source commands.sh && train_full` — Downloads the fashion dataset from HuggingFace, loads the base SD Inpainting model, injects LoRA into UNet attention layers, and trains for 1 full epoch (~44,000 steps). Every 10 steps it prints the loss and logs it to MLflow. Saves weights to `models/fashion-lora/`.
2. `train_light` — Same process but capped at 350 steps. Saves weights to `models/fashion-lora-light/`.
3. Both training runs appear in MLflow with full hyperparameters, loss curves, and artifact links.
4. Trained weights can be pushed to the Hugging Face Hub via the UI for versioned model registry.

### Phase 2: Going Live (Inference & Demo)

```
User's Device                 Ngrok Tunnel              Streamlit (8501)
     │                             │                         │
     │  https://xyz.ngrok.app      │                         │
     └─────────────────────────────▶  Forwards to ───────────▶
                                                             │
                                                     FastAPI Backend (8000)
                                                             │
                                                      ┌──────┴──────┐
                                                      │             │
                                                   rembg         SD Inpainting
                                                (U2Net mask)    (+ LoRA adapters)
                                                      │             │
                                                      └──────┬──────┘
                                                             │
                                                      Composite Image
                                                      returned to user
```

**What happens**:
1. `source commands.sh && start_all` — Starts MLflow (port 5000), FastAPI (port 8000), and Streamlit (port 8501).
2. `start_ngrok` — Creates a public HTTPS URL pointing to port 8501.
3. Users open the Ngrok URL on any device (phone, laptop, projector).
4. Upload a product image and enter a background prompt.
5. Select a model mode (Baseline / Light LoRA / Overfit LoRA).
6. Backend receives the request:
   - rembg extracts the product → creates background mask.
   - The correct LoRA adapter is hot-swapped (or disabled for Baseline).
   - Stable Diffusion runs inpainting with the given parameters.
   - Output image is returned and displayed side-by-side with the original.
7. MLflow tracks all experiment metadata and is accessible at `http://localhost:5000`.

### Phase 3: After the Demo (Model Registry & Reproducibility)

```
Local LoRA Weights          Hugging Face Hub              Docker Image
     │                           │                             │
     │  models/fashion-lora/     │  Zenith754/genai-bg-        │  nvidia/cuda
     │  models/fashion-lora-     │  replacement-lora           │  + requirements.txt
     │  light/                   │                             │  + src/
     │                           │                             │
     └──► Push via UI button ────▶  Versioned model registry   │
                                                               │
     docker-compose.yml ───────────────────────────────────────▶  Reproducible
                                                                  deployment on
                                                                  any GPU server
```

**What happens**:
1. **Model Registry**: LoRA weights are uploaded to [Zenith754/genai-bg-replacement-lora](https://huggingface.co/Zenith754/genai-bg-replacement-lora) via the Streamlit sidebar. Anyone can download and use them.
2. **Experiment Archive**: MLflow stores all training metadata in `mlflow.db`. Loss curves, hyperparameters, and artifact paths are permanently queryable.
3. **Reproducibility**: `docker compose up` rebuilds the entire stack from scratch on any NVIDIA GPU machine — zero manual setup.
4. **Teardown**: `source commands.sh && stop_all` kills all services cleanly.