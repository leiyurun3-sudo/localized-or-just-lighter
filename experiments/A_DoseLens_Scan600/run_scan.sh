#!/usr/bin/env bash
# Run on the core300 server, from the same Python environment.
set -euo pipefail
base=/root/autodl-tmp
cd "$base/A_DoseLens_Pilot_W1"
export MODEL_PATH="$base/local-models/FLUX.1-dev"
export PYTHON="${PYTHON:-python}"
export CORE_HASH=855b1e69403e05951aaf23804b926d5f6d52d17410a6271b6a6e5f89e1fb3459

# Check the original runner inputs and effective device before it can retire old jobs.
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import run

output = Path("outputs")
expected = os.environ["CORE_HASH"]
args = SimpleNamespace(cases="tokens_manifest.json", jobs="jobs.json", stage="scan",
                       dtype="bfloat16", offload="auto")
cases, jobs, case_doc, job_doc = run.load_inputs(args)
model = run.validate_model(Path(os.environ["MODEL_PATH"]).resolve())
_, identity = run.fingerprint(args, cases, case_doc, job_doc, model, run.check_dependencies())
saved = run.read_json(output / "_protocols" / expected / "protocol.json")
device = torch.cuda.get_device_properties(0)
env = dict(saved["resolved_environment"])
import diffusers
import transformers
env.update(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
           gpu=torch.cuda.get_device_name(0),
           offload="model" if device.total_memory >= 26 * 1024**3 else "sequential")
identity["resolved_environment"] = env
raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("Core fingerprint differs. Use the original core300 installation, model and GPU.")
if not run.completed(output / "_engineering", expected, engineering=True):
    raise SystemExit("Restore the core300 engineering record in the original outputs directory.")
for job in jobs:
    if job["stage"] == "core" and not run.completed(output / job["job_id"], expected):
        raise SystemExit("Restore the original core output for " + job["job_id"])
PY

# Same manifest, job IDs, cache, output directory and generation entry point as core300.
bash start.sh scan

# Ctrl+C in status.py or attaching to a different stage can also return zero.
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from run import read_json

out = Path("outputs")
state = read_json(out / "state.json")
manifest = read_json(out / "reports_manifest.json")
expected = {f"{j['job_id']}/report.json" for j in read_json(Path("jobs.json"))["jobs"]}
listed = [r["report_path"] for r in manifest["reports"]]
if not (state.get("status") == "completed" and state.get("requested_stage") == "scan"
        and manifest.get("stage") == "scan"
        and state.get("protocol_hash") == manifest.get("protocol_hash") == os.environ["CORE_HASH"]
        and len(listed) == len(expected) == 900 and set(listed) == expected):
    raise SystemExit("Scan incomplete. Run this script again to monitor or resume before collecting.")
PY

prefix="$base/A_DoseLens_scan900_$(date -u +%Y%m%dT%H%M%SZ)_$$"
"$PYTHON" collect.py --out "${prefix}_reports.zip" --native --native-out "${prefix}_images.zip"
