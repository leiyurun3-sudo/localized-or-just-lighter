#!/usr/bin/env bash
# Optional, for a new AutoDL image. Reuse the CUDA PyTorch from the server.
set -eo pipefail
cd -- "$(dirname -- "$0")"
if [[ -f /etc/network_turbo ]]; then source /etc/network_turbo; fi
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check --progress-bar off --no-deps -r requirements.txt
USE_TF=0 USE_FLAX=0 .venv/bin/python -c 'import torch, torchvision, scipy; from transformers import CLIPTextModel, T5EncoderModel; from diffusers import FluxPipeline; assert torch.cuda.is_available(); print("CUDA ready:", torch.cuda.get_device_name(0))'

