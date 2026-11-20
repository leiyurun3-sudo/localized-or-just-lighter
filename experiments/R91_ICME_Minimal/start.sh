#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
mkdir -p logs
# Driver owns a nonblocking process lock; repeat starts do not duplicate inference.
log="logs/run_$(date +%Y%m%d_%H%M%S).log"
nohup "$PYTHON" -u run.py all "$@" >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > last_pid.txt
printf '%s\n' "$log" > last_log.txt
echo "Started PID=$pid; log=$log"
echo 'Progress: python status.py'
echo 'Pause after current image: touch runtime_base/PAUSE'
