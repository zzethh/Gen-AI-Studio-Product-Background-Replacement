#!/bin/bash
export PATH=/home/anikde/.local/bin:$PATH
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Export it or add it to .env before running."
  exit 1
fi
cd /home/anikde/dlops_project

# Kill old processes
pkill -f uvicorn 2>/dev/null
pkill -f 'mlflow server' 2>/dev/null
sleep 2

# Start MLflow
nohup python3 -m mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts > logs/mlflow.log 2>&1 &

# Start Backend
nohup python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &

echo "Services started. PIDs:"
ps aux | grep -E 'uvicorn|mlflow' | grep -v grep
