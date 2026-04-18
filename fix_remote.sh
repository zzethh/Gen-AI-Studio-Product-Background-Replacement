#!/bin/bash
export PATH=/home/anikde/miniconda3/bin:$PATH
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

# Kill broken streamlit
pkill -f 'streamlit' || true
sleep 1

# Start the clean streamlit
nohup python3 -m streamlit run src/local/app.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
echo 'Streamlit successfully restarted'

sleep 2
echo "=== Streamlit HTTP ==="
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/
