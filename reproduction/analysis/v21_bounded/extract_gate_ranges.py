"""Export native SAM scan gate ranges without unpacking the reports ZIP.

Usage: python extract_gate_ranges.py REPORTS.zip [--output gate_ranges.csv]
Spatial peaks use *_factor_max (already scaled). Uniform uses its applied
scalar *_gain. Each count is over steps, with a strict threshold of 1 + 1e-6.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from zipfile import ZipFile


ARMS = ("sam_native", "uniform")
SCALES = (0.25, 0.5, 0.75, 1.0, 1.25)
BRANCHES = ("model", "visible")
STEPS = tuple(range(2, 28))
THRESHOLD = 1 + 1e-6


def extract(path):
    rows = {}
    with ZipFile(path) as archive:
        for member in archive.namelist():
            parts = member.split("/")
            if len(parts) != 3 or parts[0] != "outputs" or parts[2] != "job.json":
                continue
            job = json.loads(archive.read(member))["job"]
            arm = job["arm"]
            if arm not in ARMS:
                continue
            source, scale = job["case"], float(job["scale"])
            job_id = parts[1]
            if job["job_id"] != job_id or not job_id.startswith(source + "__") or scale not in SCALES:
                raise ValueError(f"Unexpected scan identity: {job_id}")
            profile = json.loads(archive.read(f"outputs/{job_id}/profile.json"))
            if tuple(r["step"] for r in profile) != STEPS:
                raise ValueError(f"Incomplete active-step profile: {job_id}")
            for branch in BRANCHES:
                field = branch + ("_factor_max" if arm == "sam_native" else "_gain")
                peaks = [float(r[field]) for r in profile]
                if not all(math.isfinite(p) for p in peaks):
                    raise ValueError(f"Nonfinite {field}: {job_id}")
                if arm == "uniform" and any(p != scale for p in peaks):
                    raise ValueError(f"Uniform gain disagrees with scale: {job_id}")
                key = (source, arm, scale, branch)
                if key in rows:
                    raise ValueError(f"Duplicate profile: {key}")
                rows[key] = (*key, max(peaks), len(peaks), sum(p > THRESHOLD for p in peaks))
    sources = {key[0] for key in rows}
    # All keys are unique members of the expected grid, so cardinality establishes coverage.
    if len(sources) != 50 or len(rows) != 50 * len(ARMS) * len(SCALES) * len(BRANCHES):
        raise ValueError(f"Incomplete scan: {len(sources)} sources, {len(rows)} branch profiles")
    return [rows[k] for k in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gate_ranges.csv"))
    args = parser.parse_args()
    rows = extract(args.reports_zip)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("source", "arm", "scale", "branch", "effective_peak", "active_steps",
                         "steps_gt_1_plus_1e_minus_6"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
