#!/usr/bin/env python3
"""Offline, locked, resumable DoseLens pilot runner (no evaluation)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = 1
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
os.environ.update(OFFLINE)
LOG = logging.getLogger("doselens")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def process_identity(pid):
    """Linux boot ID + process start ticks defeats stale/reused PID files."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] in {"Z", "X"}:
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return {"pid": int(pid), "start_ticks": fields[19], "boot_id": boot}
    except (OSError, ValueError, IndexError):
        return None


class AlreadyRunning(RuntimeError):
    pass


def acquire_lock():
    if sys.platform != "linux":
        raise RuntimeError("Execution requires Linux flock and /proc; no task was launched.")
    import fcntl
    fd = os.open(ROOT / ".run.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise AlreadyRunning("Another runner owns .run.lock; no duplicate launched.")
    return fd


class Progress:
    def __init__(self, output, args):
        self.path = output / "state.json"
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.error = None
        self.value = {"schema_version": 1, "status": "running", "stage": "starting",
                      "requested_stage": args.stage, "command": args.command,
                      "pid": os.getpid(), "process": process_identity(os.getpid()),
                      "started_at": now(), "heartbeat_at": now(), "progress_at": now(),
                      "output_dir": str(output), "completed": 0, "total": 0,
                      "log_path": args.log_path}
        self.update()
        self.thread = threading.Thread(target=self._pulse, daemon=True)
        self.thread.start()

    def update(self, **fields):
        with self.mutex:
            self.value.update(fields)
            self.value["heartbeat_at"] = now()
            atomic_json(self.path, self.value)

    def _pulse(self):
        while not self.stop.wait(3):
            try:
                self.update()
            except BaseException as exc:
                self.error = exc
                return

    def callback(self, event=None, **kwargs):
        if self.error:
            raise RuntimeError("Heartbeat write failed") from self.error
        if event is None:
            event = {}
        elif isinstance(event, str):
            event = {"stage": event}
        elif not isinstance(event, dict):
            raise TypeError("progress_callback expects a JSON dictionary")
        event = {**event, **kwargs}
        # Adapter events cannot overwrite runner PID, status, job or protocol identity.
        self.update(progress=event, progress_at=now(),
                    stage=str(event.get("stage", self.value["stage"])))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)


def check_dependencies():
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        raise RuntimeError("Dependencies missing. Install manually: python -m pip install --no-deps -r requirements.txt (keeps installed torch/torchvision)") from exc
    versions = {}
    errors = []
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    for line in ["torch>=2.1", "torchvision>=0.16"] + lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        try:
            version = importlib.metadata.version(req.name)
            versions[req.name] = version
            if version not in req.specifier:
                errors.append(f"{req.name}: installed {version}, required {req.specifier}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {req.name}")
    if errors:
        raise RuntimeError("Dependency preflight failed (nothing installed automatically):\n" + "\n".join(errors)
                           + "\nInstall manually: python -m pip install --no-deps -r requirements.txt\nKeep your existing matched CUDA torch/torchvision installation.")
    return versions


def validate_model(model):
    if not model.is_dir():
        raise FileNotFoundError(f"Local Diffusers model directory missing: {model}")
    index = read_json(model / "model_index.json")
    if index.get("_class_name") != "FluxPipeline":
        raise ValueError("model_index.json must describe a FluxPipeline")
    files = {model / "model_index.json"}
    shard_counts = {}
    for component in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        directory = model / component
        config = directory / "config.json"
        if not config.is_file():
            raise FileNotFoundError(config)
        read_json(config)
        files.add(config)
        indexes = sorted(directory.glob("*.index.json"))
        weights = set()
        for index_path in indexes:
            mapping = read_json(index_path).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Missing/nonempty weight_map required: {index_path}")
            # Values are shard filenames; tensor names (keys) are NOT filenames.
            for shard in set(mapping.values()):
                if not isinstance(shard, str) or not shard:
                    raise ValueError(f"Invalid shard filename in {index_path}")
                shard_path = Path(shard)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    raise ValueError(f"Shard escapes component: {shard}")
                # HF snapshot files may symlink to an existing local blobs directory.
                candidate = directory / shard_path
                weights.add(candidate)
            files.add(index_path)
        if not indexes:
            # from_pretrained without a variant requires the canonical filename.
            names = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin") if component in {"transformer", "vae"} else ("model.safetensors", "pytorch_model.bin")
            weights = {directory / name for name in names if (directory / name).is_file()}
        if not weights:
            raise FileNotFoundError(f"No local model weights in {directory}")
        for path in weights:
            if not path.is_file() or path.stat().st_size <= 8:
                raise FileNotFoundError(f"Missing/empty shard: {path}")
        files.update(weights)
        shard_counts[component] = len(weights)
    scheduler = model / "scheduler" / "scheduler_config.json"
    read_json(scheduler)
    files.add(scheduler)
    for component in ("tokenizer", "tokenizer_2"):
        directory = model / component
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files.update(p for p in directory.iterdir() if p.is_file())
    signature = []
    for path in sorted(files):
        stat = path.stat()
        item = {"path": str(path.relative_to(model)), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
        if path.suffix not in {".bin", ".safetensors"}:
            item["sha256"] = digest_file(path)
        signature.append(item)
    return {"model_path": str(model), "shards": shard_counts, "files": signature,
            "weight_identity": "resolved local path, size and mtime_ns; configuration/tokenizer content SHA256"}


def load_inputs(args):
    case_document = read_json(resolve_path(args.cases))
    job_document = read_json(resolve_path(args.jobs))
    cases = {}
    for raw in case_document["cases"]:
        case = dict(raw)
        key = str(case["case"])
        if key in cases:
            raise ValueError(f"Duplicate case: {key}")
        for field in ("source_prompt", "target_prompt", "source_mask_tokens", "target_mask_tokens", "unchanged_tokens", "seed", "image_path"):
            if field not in case:
                raise ValueError(f"Case {key} missing {field}")
        image = resolve_path(case["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        case["image_path"] = str(image)
        cases[key] = case
    jobs = {}
    for raw in job_document["jobs"]:
        job = dict(raw)
        key = job["job_id"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError(f"Unsafe job_id: {key!r}")
        if key in jobs or str(job["case"]) not in cases:
            raise ValueError(f"Duplicate job or unknown case: {key}")
        if job.get("stage") not in {"core", "scan"}:
            raise ValueError(f"Invalid job stage: {key}")
        job.setdefault("scale", 1.0)
        if not job.get("arm") or not isinstance(job.get("scale"), (int, float)) or not math.isfinite(job["scale"]):
            raise ValueError(f"Invalid arm/scale: {key}")
        deps = job.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ValueError(f"depends_on must be a list of job IDs: {key}")
        if job.get("anchor_job_id") and job["anchor_job_id"] not in deps:
            raise ValueError(f"anchor_job_id must also appear in depends_on: {key}")
        job["depends_on"] = deps
        jobs[key] = job
    if not cases or not jobs:
        raise ValueError("Cases and jobs must be nonempty")
    ordered, visiting, seen = [], set(), set()

    def visit(key):
        if key not in jobs:
            raise ValueError(f"Unknown dependency: {key}")
        if key in visiting:
            raise ValueError(f"Dependency cycle at {key}")
        if key in seen:
            return
        visiting.add(key)
        for dependency in jobs[key]["depends_on"]:
            visit(dependency)
        visiting.remove(key)
        seen.add(key)
        ordered.append(jobs[key])

    # Check the whole DAG even when only core is selected.
    for key in sorted(jobs, key=lambda key: jobs[key]["stage"] != "core"):
        visit(key)
    if args.stage == "core":
        selected = {key for key, job in jobs.items() if job["stage"] == "core"}
        todo = list(selected)
        while todo:
            for key in jobs[todo.pop()]["depends_on"]:
                if key not in selected:
                    selected.add(key)
                    todo.append(key)
        ordered = [job for job in ordered if job["job_id"] in selected]
    if not ordered:
        raise ValueError("Selected stage contains no jobs")
    return cases, ordered, case_document, job_document


def token_precheck(model, cases):
    from transformers import AutoTokenizer
    # Load both components locally; spatial attention indices belong to T5 only.
    tokenizers = [AutoTokenizer.from_pretrained(str(model / "tokenizer_2"), local_files_only=True)]
    AutoTokenizer.from_pretrained(str(model / "tokenizer"), local_files_only=True)

    def has_token(prompt, token):
        for tokenizer in tokenizers:
            ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
            for word in str(token).replace("_", " ").split():
                needle = tokenizer(word, add_special_tokens=False)["input_ids"]
                if needle and any(ids[i:i + len(needle)] == needle for i in range(len(ids) - len(needle) + 1)):
                    return True
        return False

    for key, case in cases.items():
        for field, prompt in (("source_mask_tokens", "source_prompt"), ("target_mask_tokens", "target_prompt")):
            tokens = case[field]
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"Empty/invalid {field}: {key}")
            for token in tokens:
                if not has_token(case[prompt], token):
                    raise ValueError(f"Token not found after tokenization: {key}/{field}/{token!r}")
        for token in case["unchanged_tokens"]:
            if not all(has_token(case[p], token) for p in ("source_prompt", "target_prompt")):
                raise ValueError(f"Unchanged token absent from one prompt: {key}/{token!r}")
    return {"cases_checked": len(cases), "local_files_only": True}


def fingerprint(args, cases, case_document, job_document, model_report, versions):
    code = {}
    paths = [p for p in ROOT.glob("*.py") if p.name not in {"status.py", "collect.py", "evaluate.py", "analyze.py", "prepare_tokens.py"}]
    paths += list((ROOT / "sam_flow").rglob("*.py"))
    paths += [p for p in (ROOT / "configs").rglob("*") if p.is_file()]
    paths += [ROOT / "requirements.txt"]
    paths += [ROOT / name for name in ("protocol.json", "UPSTREAM.json") if (ROOT / name).is_file()]
    for path in sorted(paths):
        code[str(path.relative_to(ROOT))] = digest_file(path)
    identity = {"schema_version": PROTOCOL_VERSION, "cases": case_document, "jobs": job_document,
                "code": code, "images": {key: digest_file(case["image_path"]) for key, case in cases.items()},
                "model": model_report, "versions": versions, "dtype": args.dtype, "offload": args.offload}
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest(), identity


def completed(directory, protocol, engineering=False):
    try:
        marker = read_json(directory / "completed.json")
        report = read_json(directory / "report.json")
        if not engineering:
            if report.get("status") != "completed" or not (directory / "native.png").is_file():
                return False
            if report.get("arm") != "flowedit":
                read_json(directory / "profile.json")
        return (marker["status"] == "completed" and marker["protocol_hash"] == protocol
                and marker["report_sha256"] == digest_file(directory / "report.json")
                and report["_runner"]["protocol_hash"] == protocol
                and (not engineering or report.get("passed") is True))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def fresh_directory(directory, output):
    if directory.exists():
        archive = output / "_attempts" / directory.name / (stamp() + f"_{os.getpid()}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
    directory.mkdir(parents=True)


def save_report(directory, report, protocol, job_id):
    if not isinstance(report, dict):
        raise TypeError("Adapter must return a JSON dictionary")
    report = dict(report)
    report["_runner"] = {"protocol_hash": protocol, "job_id": job_id,
                         "finished_at": now(), "pid": os.getpid()}
    atomic_json(directory / "report.json", report)
    return report


def mark_completed(directory, protocol, job_id):
    atomic_json(directory / "completed.json", {"status": "completed", "protocol_hash": protocol,
                "job_id": job_id, "completed_at": now(), "report_sha256": digest_file(directory / "report.json")})


def execute(args, output, progress):
    progress.update(stage="preflight.dependencies")
    versions = check_dependencies()
    if not args.model_path:
        raise ValueError("Set MODEL_PATH or --model-path to an existing local FLUX Diffusers directory")
    model = resolve_path(args.model_path)
    progress.update(stage="preflight.model")
    model_report = validate_model(model)
    cases, jobs, case_document, job_document = load_inputs(args)
    progress.update(stage="preflight.tokenizers", total=len(jobs))
    tokens = token_precheck(model, cases)
    protocol, identity = fingerprint(args, cases, case_document, job_document, model_report, versions)
    progress.update(protocol_hash=protocol)
    protocol_dir = output / "_protocols" / protocol
    atomic_json(protocol_dir / "protocol.json", identity)
    atomic_json(protocol_dir / "preflight.json", {"passed": True, "time": now(), "model": model_report,
                "tokens": tokens, "versions": versions, "selected_jobs": len(jobs), "stage": args.stage})
    if args.command == "preflight":
        progress.update(status="completed", stage="preflight.completed", finished_at=now())
        LOG.info("Preflight passed; %d selected jobs, no GPU generation", len(jobs))
        return
    progress.update(stage="loading_runtime")
    from adapter import load_runtime
    runtime = load_runtime(str(model), offload=args.offload, dtype=args.dtype)
    identity["resolved_environment"] = getattr(runtime, "environment", {})
    protocol = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    progress.update(protocol_hash=protocol)
    atomic_json(output / "_protocols" / protocol / "protocol.json", identity)
    # Keep adapter scouts separate when generation code, inputs or protocol change.
    if hasattr(runtime, "cache_identity"):
        runtime.cache_identity = protocol
    progress.update(stage="validating_cases")
    for key, case in cases.items():
        result = runtime.validate_case(case)
        if result is False or (isinstance(result, dict) and result.get("passed") is False):
            raise RuntimeError(f"Adapter tokenizer precheck failed: {key}: {result}")
    engineering = output / "_engineering"
    if not completed(engineering, protocol, engineering=True):
        fresh_directory(engineering, output)
        progress.update(stage="engineering", current_job="_engineering", progress={})
        report = runtime.engineering_check(cases[str(jobs[0]["case"])], engineering, progress.callback)
        report = save_report(engineering, report, protocol, "_engineering")
        if report.get("passed") is not True:
            raise RuntimeError("Real native/observe engineering parity failed; see _engineering/report.json")
        mark_completed(engineering, protocol, "_engineering")
    current_reports = [{"report_path": f"{job['job_id']}/report.json"} for job in jobs
                       if completed(output / job["job_id"], protocol)]
    def publish_reports():
        atomic_json(output / "reports_manifest.json", {"protocol_hash": protocol,
                    "stage": args.stage, "reports": current_reports})
    publish_reports()
    count = len(current_reports)
    progress.update(completed=count)
    for job in jobs:
        job_id = job["job_id"]
        directory = output / job_id
        if completed(directory, protocol):
            LOG.info("Resume completed %s", job_id)
            continue
        for dependency in job["depends_on"]:
            if not completed(output / dependency, protocol):
                raise RuntimeError(f"Dependency {dependency} is not completed for this protocol")
        fresh_directory(directory, output)
        atomic_json(directory / "job.json", {"job": job, "protocol_hash": protocol, "started_at": now()})
        progress.update(stage="job.starting", current_job=job_id, current_case=str(job["case"]), progress={})
        LOG.info("Starting %s (%d/%d completed)", job_id, count, len(jobs))
        report = runtime.run_job(cases[str(job["case"])], dict(job), directory, progress.callback)
        report = save_report(directory, report, protocol, job_id)
        if report.get("passed") is False or report.get("status") != "completed":
            raise RuntimeError(f"Adapter reported failure for {job_id}; outputs preserved")
        if progress.error:
            raise RuntimeError("Heartbeat write failed") from progress.error
        mark_completed(directory, protocol, job_id)
        current_reports.append({"report_path": f"{job_id}/report.json"})
        publish_reports()
        count += 1
        progress.update(completed=count, stage="job.completed")
        LOG.info("Completed %s (%d/%d)", job_id, count, len(jobs))
    progress.update(status="completed", stage="completed", current_job=None, finished_at=now())


class StopRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Stopped by signal {signal.Signals(signum).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("preflight", "run"), default="run")
    parser.add_argument("--stage", choices=("core", "scan", "all"), default="core")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--cases", default="tokens_manifest.json")
    parser.add_argument("--jobs", default="jobs.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--offload", choices=("auto", "model", "sequential", "none"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--background", action="store_true", help="Launch detached, keeping the acquired lock")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--log-path", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s PID=%(process)d %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S%z")
    try:
        if args.lock_fd is None:
            lock_fd = acquire_lock()
        else:
            import fcntl
            lock_fd = args.lock_fd
            if os.fstat(lock_fd).st_ino != os.stat(ROOT / ".run.lock").st_ino:
                raise ValueError("Inherited lock descriptor does not match .run.lock")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except AlreadyRunning as exc:
        LOG.error("%s", exc)
        return 3
    except Exception as exc:
        LOG.error("Cannot acquire runner lock: %s", exc)
        return 1
    output = resolve_path(args.output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
        if args.background:
            logs = output / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"run_{stamp()}_launcher{os.getpid()}.log"
            child_args = [arg for arg in sys.argv[1:] if arg != "--background"]
            command = [sys.executable, "-u", str(ROOT / "run.py"), *child_args,
                       "--lock-fd", str(lock_fd), "--log-path", str(log_path)]
            with log_path.open("ab", buffering=0) as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, start_new_session=True, pass_fds=(lock_fd,))
            # Closing our descriptor does not release the child's inherited flock.
            atomic_json(output / "launch.json", {"pid": proc.pid, "process": process_identity(proc.pid),
                        "launched_at": now(), "log_path": str(log_path), "stage": args.stage})
            print(f"Started runner PID={proc.pid}; log={log_path}", flush=True)
            return 0
        progress = None

        def stop(signum, frame):
            raise StopRequested(signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        try:
            progress = Progress(output, args)
            LOG.info("Runner started: stage=%s offload=%s dtype=%s", args.stage, args.offload, args.dtype)
            execute(args, output, progress)
            return 0
        except BaseException as exc:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            is_oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
            LOG.error("Runner failed: %s\n%s", exc, traceback.format_exc())
            if progress is not None:
                progress.close()
                progress.update(status="failed", failure_stage=progress.value.get("stage"), stage="failed",
                                error={"type": type(exc).__name__, "message": str(exc), "oom": is_oom},
                                finished_at=now())
            return 128 + exc.signum if isinstance(exc, StopRequested) else 1
        finally:
            if progress is not None:
                progress.close()
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
