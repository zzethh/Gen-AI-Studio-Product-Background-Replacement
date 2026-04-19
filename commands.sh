#!/bin/bash
# ============================================================
# Gen-AI Studio — Quick Reference Commands
# ============================================================

# --- Environment Setup ---
# conda activate genai
# export $(cat .env | xargs)

# --- Start All Services ---
start_all() {
    echo "Starting MLflow..."
    nohup mlflow server --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
    
    echo "Starting FastAPI Backend..."
    nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
    
    echo "Starting Streamlit Frontend..."
    nohup streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false > logs/streamlit.log 2>&1 &
    
    echo "All services started!"
}

# --- Start Ngrok Tunnel ---
start_ngrok() {
    echo "Starting Ngrok tunnel..."
    pkill -f ngrok 2>/dev/null
    sleep 1
    nohup ./ngrok http 8501 > logs/ngrok.log 2>&1 &
    sleep 3
    echo "--- Ngrok Global URL ---"
    curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
}

# --- Stop All Services ---
stop_all() {
    pkill -f mlflow
    pkill -f uvicorn
    pkill -f streamlit
    echo "All services stopped."
}

# --- Train LoRA (Demo - 50 samples) ---
train() {
    python src/train_lora.py --max_steps 50 --max_train_samples 50
}

# --- Train LoRA (Full Dataset - Background) ---
train_full() {
    echo "Starting full dataset training in the background."
    source /DATA/anikde/miniconda3/etc/profile.d/conda.sh
    conda activate genai
    export $(cat .env | xargs)
    
    # Ensure MLflow is running first
    nohup mlflow server --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
    sleep 3
    
    # Run the script with Python unbuffered (-u) so logs write immediately
    nohup python -u src/train_lora.py --num_epochs 1 > logs/train_full.log 2>&1 &
    
    echo "Training started! Check progress with: tail -f logs/train_full.log"
}

train_light() {
    echo "Starting light dataset training (500 steps) in the background."
    source /DATA/anikde/miniconda3/etc/profile.d/conda.sh
    conda activate genai
    export $(cat .env | xargs)
    
    # Ensure MLflow is running first
    nohup mlflow server --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
    sleep 3
    
    # Run the script with max 350 steps
    nohup python -u src/train_lora.py --max_steps 350 --output_dir models/fashion-lora-light > logs/train_light.log 2>&1 &
    
    echo "Light training started! Check progress with: tail -f logs/train_light.log"
}

# --- Test API ---
test_api() {
    echo "Health check..."
    curl -s http://localhost:8000/health | python -m json.tool
    
    echo ""
    echo "Testing background replacement..."
    curl -X POST http://localhost:8000/generate \
        -F "image=@test_product.png" \
        -F "prompt=A product on a wooden table in a cozy cafe, warm lighting" \
        -F "num_inference_steps=20" \
        -o test_output.png
    echo "Output saved to test_output.png"
}

# --- SSH Tunnel (run from YOUR computer) ---
# ssh -L 8501:localhost:8501 -L 8000:localhost:8000 -L 5000:localhost:5000 anikde@172.25.0.208

# --- Check Logs ---
check_logs() {
    echo "=== Backend ===" && tail -n 5 logs/backend.log
    echo "=== Streamlit ===" && tail -n 5 logs/streamlit.log
    echo "=== MLflow ===" && tail -n 5 logs/mlflow.log
}

echo "Usage: source commands.sh && start_all | stop_all | start_ngrok | train | train_full | test_api | check_logs"
