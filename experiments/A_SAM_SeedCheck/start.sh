#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/local-models/FLUX.1-dev}"
if [[ -z "${PYTHON:-}" && -x .venv/bin/python ]]; then PYTHON="$PWD/.venv/bin/python"; fi
exec "${PYTHON:-python}" -u study/launch.py "${1:-full}" "${2:-1,2}"

