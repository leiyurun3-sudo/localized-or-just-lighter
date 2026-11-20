#!/usr/bin/env python3
"""Read runner PID identity and heartbeat; Ctrl+C exits only this monitor."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from run import ROOT, now, process_identity, read_json, resolve_path


def lock_held():
    if sys.platform != "linux":
        return None
    import fcntl
    try:
        fd = os.open(ROOT / ".run.lock", os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def snapshot(output):
    errors = []
    records = {}
    for name in ("state", "launch"):
        try:
            records[name] = read_json(output / f"{name}.json")
        except FileNotFoundError:
            records[name] = {}
        except (ValueError, OSError) as exc:
            records[name] = {}
            errors.append(f"{name}.json: {exc}")
    state, launch = records["state"], records["launch"]
    launching = launch.get("launched_at", "") > state.get("started_at", "")
    if launch.get("process") != state.get("process") and process_identity(launch.get("pid", -1)) == launch.get("process") and launch.get("process"):
        launching = True
    source = launch if launching else state
    identity = source.get("process")
    alive = bool(identity and process_identity(identity.get("pid", -1)) == identity)
    held = lock_held()
    result = dict(state) if not launching else {"status": "launching", "stage": "starting", **launch}
    result.update(observed_at=now(), process_alive=alive, lock_held=held, read_errors=errors)
    if not source:
        result.update(status="unknown" if held else "not_started", stage="unknown")
        return result
    if not alive and result.get("status") not in {"completed", "failed"}:
        result.update(status="failed", detected_by="monitor", error={"type": "ProcessExited",
                      "message": "Runner identity is absent (exit, reboot, SIGKILL or stale PID); outputs preserved"})
    heartbeat = result.get("heartbeat_at")
    if heartbeat:
        try:
            age = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)).total_seconds())
            result["heartbeat_age_seconds"] = round(age, 1)
            if alive and age > 15 and result.get("status") == "running":
                result["health"] = "heartbeat_stale"
        except (ValueError, TypeError):
            result["health"] = "invalid_heartbeat"
    return result


def render(result):
    progress = result.get("progress", {})
    pieces = [f"{now()} PID={result.get('pid', '?')}",
              f"status={result.get('status', 'unknown')}", f"alive={result['process_alive']}",
              f"stage={result.get('stage', '?')}",
              f"jobs={result.get('completed', 0)}/{result.get('total', '?')}"]
    for key in ("current_job", "heartbeat_age_seconds", "health"):
        if result.get(key) is not None:
            pieces.append(f"{key}={result[key]}")
    if progress:
        pieces.append("progress=" + json.dumps(progress, ensure_ascii=False, separators=(",", ":")))
    if result.get("error"):
        pieces.append("error=" + json.dumps(result["error"], ensure_ascii=False))
    if result.get("read_errors"):
        pieces.append("read_errors=" + repr(result["read_errors"]))
    return " | ".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--watch", "--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    output = resolve_path(args.output_dir)
    try:
        while True:
            result = snapshot(output)
            print(json.dumps(result, ensure_ascii=False) if args.json else render(result), flush=True)
            terminal = result.get("status") in {"completed", "failed", "not_started"}
            if not args.watch or (terminal and not result["process_alive"]):
                return 1 if result.get("status") == "failed" else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Monitor stopped; background runner continues.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
