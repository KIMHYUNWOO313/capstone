#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/capstone_forecast/final_handoff
UV_CACHE_DIR=/home/ubuntu/capstone_forecast/.uv-cache \
  uv pip install --python .venv/bin/python 'huggingface-hub>=0.34,<1.0'
rm -rf /home/ubuntu/capstone_forecast/.uv-cache
.venv/bin/python -c 'import huggingface_hub; print(huggingface_hub.__version__)'
sudo systemctl restart forecast-web
