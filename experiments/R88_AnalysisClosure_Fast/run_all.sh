#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export R88_SERVER_PROJECT="${R88_SERVER_PROJECT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready}"
PYTHON="${PYTHON:-$R88_SERVER_PROJECT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Python missing: $PYTHON"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export R88_REFERENCE_CACHE="${R88_REFERENCE_CACHE:-on}"
export R88_SOURCE_KV_CACHE="${R88_SOURCE_KV_CACHE:-on}"
export R88_KV_CACHE_MAX_MB="${R88_KV_CACHE_MAX_MB:-3072}"
export PYTHONDONTWRITEBYTECODE=1
STAGE=initializing
trap 'code=$?; if ((code)); then printf "Stopped in %s, code=%s. Existing outputs kept; rerun start.sh to resume.\n" "$STAGE" "$code"; fi' EXIT
STAGE=exact_equivalence
"$PYTHON" -u equivalence.py
STAGE=generation
"$PYTHON" -u run.py --generate
STAGE=CPU_evaluation
"$PYTHON" -u run.py --evaluate
STAGE=source_statistics
"$PYTHON" -u analyze.py --results "$ROOT/runtime_base/results"
STAGE=final_package
"$PYTHON" -u run.py --package
DELIVERY="${R88_DELIVERY_DIR:-/root/autodl-tmp}"
mkdir -p -- "$DELIVERY"
ARCHIVE="$ROOT/R88_AnalysisClosure_Fast_results.zip"
[[ -s "$ARCHIVE" ]] || { echo "Missing final package: $ARCHIVE"; exit 2; }
if [[ "$ARCHIVE" != "$DELIVERY/R88_AnalysisClosure_Fast_results.zip" ]]; then
  cp -- "$ARCHIVE" "$DELIVERY/R88_AnalysisClosure_Fast_results.zip.tmp"
  mv -- "$DELIVERY/R88_AnalysisClosure_Fast_results.zip.tmp" "$DELIVERY/R88_AnalysisClosure_Fast_results.zip"
fi
printf 'GENERATION + METRICS COMPLETE: %s/R88_AnalysisClosure_Fast_results.zip\n' "$DELIVERY"
STAGE=complete
date -Is > "$ROOT/complete.txt"
printf 'COMPLETE. Results are in %s\n' "$DELIVERY"
"$PYTHON" -u run.py --download-url "$DELIVERY/R88_AnalysisClosure_Fast_results.zip"

