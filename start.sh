#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Hugging Face cache location (inside the pod)
if [ -d "/workspace" ]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
else
  export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
fi

# Create venv but keep access to system site-packages (torch is preinstalled in the RunPod PyTorch image)
if [ ! -d ".venv" ]; then
  python3 -m venv --system-site-packages .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# Debug info (shows up in logs once)
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("is_cuda_available:", torch.cuda.is_available())
PY

exec uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}"