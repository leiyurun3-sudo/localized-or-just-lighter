"""Launch and monitor the detached Linux worker; one archive per completed seed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study"
WORK = ROOT / "runs"
STATE = WORK / "batch.json"

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def persist(data):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf8")
    temp.replace(STATE)

def package(seed, mode, directory):
    manifest = read(directory / "reports_manifest.json")
    jobs = [j for j in read(STUDY / "jobs.json")["jobs"] if mode == "full" or j["stage"] == "core"]
    expected = {j["job_id"] for j in jobs}
    paths = [r["report_path"] for r in manifest["reports"]]
    if len(paths) != len(expected) or set(paths) != {j+"/report.json" for j in expected}:
        raise ValueError("Runner manifest does not cover the complete selected grid")
    protocol = manifest["protocol_hash"]
    outfile = ROOT.parent / f"A_SAM_seed{seed}_{mode}.zip"
    temp = outfile.with_suffix(".zip.part")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=4) as z:
        z.writestr("batch.json", json.dumps(dict(seed=seed, mode=mode, jobs=len(jobs), protocol=protocol)))
        for name in ("reports_manifest.json", "_engineering/report.json", "_engineering/parity.json",
                     f"_protocols/{protocol}/protocol.json"):
            z.write(directory / name, name)
        for j in jobs:
            name = j["job_id"]
            report = read(directory / name / "report.json")
            if (report["seed"], report["case"], report["arm"], report["status"], report["_runner"]["protocol_hash"]) != (
                    seed, j["case"], j["arm"], "completed", protocol):
                raise ValueError("Output identity mismatch: "+name)
            for file in ("report.json","completed.json","profile.json","job.json","native.png"):
                member = name+"/"+file
                z.write(directory / member, member, compress_type=ZIP_STORED if file.endswith(".png") else ZIP_DEFLATED)
    temp.replace(outfile)
    return str(outfile)

def work(args):
    mode, seeds = args.mode, [int(s) for s in args.seeds.split(",")]
    stage = "core" if mode == "light" else "all"
    batch = dict(status="running", pid=os.getpid(), mode=mode, seeds=seeds, archives=[], started=time.time())
    persist(batch)
    try:
        model = Path(os.environ["MODEL_PATH"]).expanduser()
        if not (model / "model_index.json").is_file():
            raise FileNotFoundError(f"Set MODEL_PATH to the existing FLUX.1-dev diffusers directory: {model}")
        for seed in seeds:
            batch.update(seed=seed, phase="generation", child=None)
            persist(batch)
            output = WORK / f"seed{seed}"
            cmd = [sys.executable, "-u", "run.py", "run", "--cases", f"study/cases{seed}.json",
                   "--jobs", "study/jobs.json", "--output-dir", str(output), "--stage", stage,
                   "--model-path", str(model), "--dtype","bfloat16","--offload","model"]
            with subprocess.Popen(cmd, cwd=ROOT) as proc:
                batch["child"] = proc.pid
                persist(batch)
                rc = proc.wait()
                if rc:
                    raise RuntimeError(f"seed {seed}: run.py exited with code {rc}. See runs/batch.log")
            batch.update(phase="packing", child=None)
            persist(batch)
            batch["archives"].append(package(seed, mode, output))
            persist(batch)
        batch.update(status="completed", phase="completed")
    except BaseException as error:
        batch.update(status="failed", error=str(error))
        raise
    finally:
        persist(batch)

def monitor(pid):
    try:
        while True:
            data = read(STATE)
            if data.get("pid") == pid:
                seed = data.get("seed")
                line = f"{data['mode']} | {data['status']} | seed {seed} | {data.get('phase','starting')}"
                state = WORK / f"seed{seed}" / "state.json"
                if state.exists() and data.get("child"):
                    live = read(state)
                    if live.get("pid") == data["child"]:
                        line += f" | {live['completed']}/{live['total']} | {live['stage']} {live.get('current_job') or ''}"
                print("\r\033[2K"+line, end="", flush=True)
                if data["status"] in ("completed","failed"):
                    print()
                    for path in data["archives"]:
                        print(path)
                    if data.get("error"):
                        print(data["error"])
                        log = WORK / "batch.log"
                        if log.exists():
                            with log.open("rb") as inp:
                                inp.seek(max(0, log.stat().st_size-4000))
                                print(inp.read().decode(errors="replace"))
                        return 1
                    return 0
            try:
                os.kill(pid, 0)
                stat = Path(f"/proc/{pid}/stat").read_text()
                if stat.rsplit(")", 1)[1].split()[0] == "Z":
                    raise ProcessLookupError(pid)
            except (FileNotFoundError, ProcessLookupError):
                print("\nWorker stopped. Re-run the same start command to resume.")
                return 1
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nMonitor closed. Background worker continues.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["light","full"])
    parser.add_argument("seeds", choices=["1","2","1,2"])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.worker:
        work(args)
        return 0
    with (WORK / "batch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for _ in range(30):
                if STATE.is_file():
                    data = read(STATE)
                    if data.get("status") == "running":
                        print("Active batch:", data["mode"], data["seeds"])
                        return monitor(data["pid"])
                time.sleep(.1)
            raise RuntimeError("A worker is starting. Run start.sh again to monitor.")
        with (WORK / "batch.log").open("ab") as log:
            proc = subprocess.Popen([sys.executable,"-u",__file__,args.mode,args.seeds,"--worker"],
                                    cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
        for _ in range(200):
            if STATE.exists() and read(STATE).get("pid")==proc.pid:
                break
            if proc.poll() is not None:
                if STATE.exists() and read(STATE).get("pid")==proc.pid:
                    break
                raise RuntimeError("Worker failed to start; see runs/batch.log")
            time.sleep(.05)
    print("Monitor refresh: 3 s. Ctrl+C closes the monitor only.")
    return monitor(proc.pid)

if __name__ == "__main__":
    sys.exit(main())

