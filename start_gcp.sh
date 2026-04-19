#!/bin/bash
export PATH=/home/anikde/.local/bin:$PATH
export HF_TOKEN="your_hf_token_here" # Set this in your environment or a .env file
cd /home/anikde/dlops_project

# Kill old processes
pkill -f uvicorn 2>/dev/null
pkill -f 'mlflow server' 2>/dev/null
pkill -f streamlit 2>/dev/null
sleep 2

# Start MLflow (with allowed hosts for reverse proxy)
nohup python3 -m mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts 34.45.215.233,genai-studio.duckdns.org,localhost \
  > logs/mlflow.log 2>&1 &

# Start Backend (with root-path for Nginx /api/ proxy)
nohup python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --root-path /api > logs/backend.log 2>&1 &

# Start Streamlit (with CORS/XSRF disabled for reverse proxy)
nohup python3 -m streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false > logs/streamlit.log 2>&1 &

# Restart Docker containers (Grafana + Prometheus)
docker-compose up -d prometheus grafana 2>/dev/null

echo "Services started. PIDs:"
ps aux | grep -E 'uvicorn|mlflow|streamlit' | grep -v grep
