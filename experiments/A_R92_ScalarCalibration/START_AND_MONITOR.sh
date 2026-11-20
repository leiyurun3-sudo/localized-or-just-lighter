#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ -z "${FLUX_MODEL_PATH:-}" && -f /root/autodl-tmp/local-models/FLUX.1-dev/model_index.json ]]; then
    export FLUX_MODEL_PATH=/root/autodl-tmp/local-models/FLUX.1-dev
fi
# Only the small missing generation helper is installed; existing model/framework versions are retained.
if ! "$PYTHON" -c 'import einops' >/dev/null 2>&1; then
    "$PYTHON" -m pip install einops==0.8.1
fi
"$PYTHON" start.py all
"$PYTHON" -u status.py
