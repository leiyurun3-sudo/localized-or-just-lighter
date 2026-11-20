#!/usr/bin/env bash
set -euo pipefail
# Optional: analysis-only packages. Does not install/upgrade torch or model tools.
"${PYTHON:-python}" -m pip install pandas scipy statsmodels matplotlib
"${PYTHON:-python}" -m pip install --no-deps einops
