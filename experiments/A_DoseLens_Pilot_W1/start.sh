#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
stage="${1:-core}"
if (( $# > 0 )); then shift; fi
case "$stage" in
  core|scan|all) ;;
  *) printf 'Usage: bash start.sh [core|scan|all] [run.py options]\n' >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-python}"
# Read the same optional output argument for the monitor; preserve all run options.
output="outputs"
argv=("$@")
for (( i=0; i<${#argv[@]}; i++ )); do
  case "${argv[i]}" in
    --output-dir=*) output="${argv[i]#*=}" ;;
    --output-dir)
      if (( i+1 >= ${#argv[@]} )); then printf 'Missing --output-dir value\n' >&2; exit 2; fi
      output="${argv[i+1]}"
      ;;
  esac
done
# Python acquires flock BEFORE spawning, then passes its locked FD to the child.
# The detached child has a separate session; terminal Ctrl+C only stops status.py.
launch_status=0
"$PYTHON" -u run.py run --stage "$stage" --background "$@" || launch_status=$?
if (( launch_status == 3 )); then
  printf 'Runner already active; monitoring %s. Use its --output-dir if different.\n' "$output"
elif (( launch_status != 0 )); then
  exit "$launch_status"
fi
printf 'Monitoring every 3 seconds. Ctrl+C stops only the monitor.\n'
"$PYTHON" -u status.py --output-dir "$output" --watch --interval 3
