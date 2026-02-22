#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Hugging Face cache location (inside the pod)
if [ -d "/workspace" ]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
else
  export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

exec uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}"