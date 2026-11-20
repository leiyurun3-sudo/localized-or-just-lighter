"""Offline preflight, six-arm exact GPU equivalence, then permit the 480-job run.

GPU work happens only in --worker children, through each runtime's normal queue.
Latents and full images stay under equivalence/run; delivery uses only report.json.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'equivalence'
RUN = BASE / 'run'
REFERENCE = ROOT / 'reference_runtime'
OPTIMIZED = ROOT / 'runtime_base'
PACKAGES = ('torch', 'diffusers', 'transformers', 'accelerate', 'safetensors',
            'numpy', 'Pillow', 'lpips', 'torchvision', 'scikit-image', 'einops')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUBLAS_WORKSPACE_CONFIG',
            'PYTORCH_CUDA_ALLOC_CONF', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM',
            'R88_REFERENCE_CACHE', 'R88_SOURCE_KV_CACHE', 'R88_KV_CACHE_MAX_MB',
            'HF_HOME', 'FLUX_MODEL_PATH', 'CLIP_PATH', 'DINO_PATH',
            'LPIPS_PATH', 'LPIPS_TRUNK_PATH', 'R88_SERVER_PROJECT')
STEP_COUNTERS = {'nfe_step', 'nfe_edit_cumulative', 'source_capture_saved_nfe_step'}
ARMS = ['REQ', 'GLB_R', 'U25', 'REQ_U', 'SHUF_R', 'REV_R']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
    os.environ.setdefault('R88_REFERENCE_CACHE', 'on')
    os.environ.setdefault('R88_SOURCE_KV_CACHE', 'on')
    os.environ.setdefault('R88_KV_CACHE_MAX_MB', '3072')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def code_files(folder):
    return {p.name: sha(p) for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix in ('.py', '.json') and p.name != 'plan.json'}


def verify_reference():
    expected = read(ROOT / 'reference_runtime_sha256.json')
    actual = {p.name: sha(p) for p in sorted(REFERENCE.iterdir()) if p.is_file()}
    if actual != expected:
        raise RuntimeError('Frozen reference runtime missing or changed. Restore reference_runtime from the delivered ZIP.')


def subset_plan(formal):
    case = formal['tasks'][0]['case']
    jobs = [copy.deepcopy(j) for j in formal['jobs'] if j['case'] == case and j['seed'] == 0]
    if [j['arm'] for j in jobs] != ARMS:
        raise ValueError('Equivalence requires the first source, seed 0, complete six-arm order.')
    result = copy.deepcopy(formal)
    result.update(planned=6, tasks=[copy.deepcopy(formal['tasks'][0])], jobs=jobs,
                  input_sha256={k: v for k, v in formal['input_sha256'].items() if k.startswith('data/' + case + '/')})
    # Job and numerical settings are unchanged; only the execution cohort is restricted.
    return result


def weight_identity(path):
    """Header/config hashes plus file identity; large weight bodies are not rehashed."""
    path = Path(path).resolve()
    st = path.stat()
    value = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
    if path.suffix == '.safetensors':
        with path.open('rb') as handle:
            raw = handle.read(8)
            n = int.from_bytes(raw, 'little')
            if len(raw) != 8 or not 2 < n < 100_000_000:
                raise RuntimeError('Invalid safetensors header: ' + str(path))
            header = handle.read(n)
        obj = json.loads(header)
        entries = [x for k, x in obj.items() if k != '__metadata__']
        if not entries or st.st_size != 8 + n + max(x['data_offsets'][1] for x in entries):
            raise RuntimeError('Incomplete safetensors: ' + str(path))
        value['header_sha256'] = hashlib.sha256(header).hexdigest()
    elif path.suffix in ('.json', '.txt') or st.st_size < 4 * 1024 * 1024:
        value['sha256'] = sha(path)
    return value


def environment_identity(formal):
    import run as launcher
    launcher.RUNTIME = OPTIMIZED
    ready = launcher.preflight_generation(formal)
    import model_loader
    flux = Path(model_loader.preflight_runtime(formal))
    files = set()
    for folder in (flux, Path(os.environ['CLIP_PATH']), Path(os.environ['DINO_PATH'])):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.txt', '.safetensors', '.bin', '.model'):
                files.add(p.resolve())
    files.update(Path(os.environ[k]).resolve() for k in ('LPIPS_PATH', 'LPIPS_TRUNK_PATH'))
    inventory = [weight_identity(p) for p in sorted(files)]
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits']
    gpu = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    if not gpu:
        raise RuntimeError('nvidia-smi returned no GPU identity; equivalence is not approved.')
    import torch
    return dict(packages={p: importlib.metadata.version(p) for p in PACKAGES},
                python=sys.version, executable=sys.executable, torch_cuda=torch.version.cuda,
                torch_build=torch.__config__.show(), gpu=gpu,
                environment={k: os.environ.get(k) for k in ENV_KEYS}, weights=inventory,
                weight_identity_scope='resolved path, size, mtime, inode; configs/header SHA256; no full large-weight body hashing',
                evaluation_paths=ready['paths'])


def fingerprint(formal):
    verify_reference()
    value = dict(reference=code_files(REFERENCE), optimized=code_files(OPTIMIZED),
                 launcher={n: sha(ROOT / n) for n in ('run.py', 'equivalence.py', 'run_all.sh', 'start.sh')},
                 scientific_files={n: sha(ROOT / n) for n in ('manifest.json', 'protocol.json', 'stats_plan.json')},
                 formal_plan=formal, subset=subset_plan(formal), environment=environment_identity(formal),
                 variants={'reference': {'R88_REFERENCE_CACHE': 'off', 'R88_SOURCE_KV_CACHE': 'off'},
                           'optimized': {k: os.environ[k] for k in ('R88_REFERENCE_CACHE', 'R88_SOURCE_KV_CACHE', 'R88_KV_CACHE_MAX_MB')}})
    # Check actual scientific files rather than trusting the plan's declared hashes.
    for name, expected in formal['input_sha256'].items():
        path = OPTIMIZED / name
        if sha(path) != expected:
            raise RuntimeError('Scientific input changed: ' + str(path))
    return dict(fingerprint=digest(value), identity=value)


def deep_cpu_preflight(identity):
    """Load lazy evaluator dependencies before either child can enter the GPU queue."""
    path = BASE / 'cpu_preflight.json'
    key = digest(dict(environment=identity['identity']['environment'],
                      evaluator=identity['identity']['optimized'].get('evaluate.py')))
    if path.is_file() and read(path).get('fingerprint') == key and read(path).get('status') == 'passed':
        print('CPU evaluator preflight reused: same software and weight identities.', flush=True)
        return
    from transformers import AutoConfig, CLIPConfig, CLIPProcessor, AutoImageProcessor, CLIPModel, AutoModel
    import torch
    import evaluate
    CLIPConfig.from_pretrained(os.environ['CLIP_PATH'], local_files_only=True)
    AutoConfig.from_pretrained(os.environ['DINO_PATH'], local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(os.environ['CLIP_PATH'], local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(os.environ['DINO_PATH'], local_files_only=True)
    lpips_model = evaluate.load_lpips(os.environ['LPIPS_PATH'], os.environ['LPIPS_TRUNK_PATH'], 'cpu')
    with torch.inference_mode():
        sample = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        check = lpips_model(sample, sample)
    if not torch.isfinite(check).all().item():
        raise RuntimeError('Offline LPIPS CPU forward returned non-finite values.')
    del lpips_model, clip_processor, dino_processor
    save(path, dict(status='passed', fingerprint=key, checked=['LPIPS calibration/trunk strict CPU load and forward',
        'CLIP/DINO local processors and model configs', 'CLIPModel/AutoModel lazy import',
        'FLUX/CLIP/DINO safetensor headers and complete file lengths'],
        scope='CPU preflight only; not GPU equivalence or full evaluator inference'))
    print('CPU evaluator preflight passed; no CUDA model/context created.', flush=True)


def stage_run(identity):
    manifest = RUN / 'identity.json'
    if not manifest.exists() and RUN.exists() and any(RUN.glob('*/results/**/record.json')):
        raise RuntimeError('Equivalence identity is missing beside existing results. Preserve/remove equivalence/run and rerun both variants.')
    if manifest.exists() and read(manifest)['fingerprint'] != identity['fingerprint']:
        archive = BASE / 'archive' / (str(time.time_ns()) + '_' + read(manifest)['fingerprint'][:12])
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not RUN.resolve().is_relative_to(BASE.resolve()) or not archive.resolve().is_relative_to(BASE.resolve()):
            raise RuntimeError('Invalid equivalence archive path')
        shutil.move(str(RUN), str(archive))
        print('Changed execution identity: previous equivalence preserved at ' + str(archive), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    plan = identity['identity']['subset']
    for variant, source in [('reference', REFERENCE), ('optimized', OPTIMIZED)]:
        target = RUN / variant
        target.mkdir(parents=True, exist_ok=True)
        for name in code_files(source):
            destination = target / name
            if destination.exists() and sha(destination) != sha(source / name):
                raise RuntimeError('Staged runtime code changed; archive equivalence/run and rerun: ' + str(destination))
            if not destination.exists():
                shutil.copy2(source / name, destination)
        for name, expected in plan['input_sha256'].items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_file = OPTIMIZED / name
            if destination.exists():
                if sha(destination) != expected:
                    raise RuntimeError('Equivalence input changed: ' + str(destination))
            else:
                try:
                    os.link(source_file, destination)
                except OSError:
                    shutil.copy2(source_file, destination)
        if (target / 'plan.json').exists() and read(target / 'plan.json') != plan:
            raise RuntimeError('Staged six-arm plan differs; preserve then remove equivalence/run to rerun.')
        save(target / 'plan.json', plan)
    save(manifest, identity)


def worker(variant, preflight_only=False):
    offline()
    runtime = RUN / variant
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    sys.path.insert(0, str(runtime))
    os.environ.update(OUTPUT_NAME='R88_equivalence_' + variant,
                      R88_QUEUE_PROJECT=os.environ.get('R88_SERVER_PROJECT', '/root/autodl-tmp/MINT_Flow_Jupyter_Ready'))
    if variant == 'reference':
        os.environ.update(R88_REFERENCE_CACHE='off', R88_SOURCE_KV_CACHE='off')
    import run as launcher
    launcher.RUNTIME = runtime
    plan = read(runtime / 'plan.json')
    launcher.preflight_generation(plan)
    os.environ['R88_PREFLIGHT_PLAN_SHA256'] = sha(runtime / 'plan.json')
    if preflight_only:
        print(variant + ': offline imports and pinned weights ready; GPU not initialized.', flush=True)
        return
    import torch
    import engine
    original = engine.execute
    signature = inspect.signature(original)

    def captured(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        fields = bound.arguments
        job = fields['job']
        capture = {k: fields[k].detach().cpu().clone() for k in ('y', 'ids')}
        capture['budget_profile'] = copy.deepcopy(fields.get('budget_profile'))
        result = original(*args, **kwargs)
        capture['z'] = result[0].detach().cpu().clone()
        for key in ('y', 'ids', 'z'):
            if not torch.isfinite(capture[key]).all().item():
                raise RuntimeError('Non-finite equivalence tensor: ' + key + ' / ' + job['id'])
        destination = runtime / 'results' / job['id'] / 'equivalence_tensors.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(capture, temporary)
        os.replace(temporary, destination)
        return result

    engine.execute = captured
    # Completed records without captures must never silently pass a resumed run.
    for job in plan['jobs']:
        folder = runtime / 'results' / job['id']
        if (folder / 'record.json').exists() and read(folder / 'record.json').get('status') == 'completed' and not (folder / 'equivalence_tensors.pt').is_file():
            raise RuntimeError('Completed equivalence capture missing: preserve/remove equivalence/run before rerun.')
    import runner
    runner.work(runtime)
    progress = read(runtime / 'results' / 'progress.json')
    if progress.get('status') != 'completed' or progress.get('completed') != 6:
        raise RuntimeError(variant + ': six-arm equivalence run is incomplete.')


def finite_exact(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError(path + ': field mismatch ' + str(set(a) ^ set(b)))
        for k in a:
            finite_exact(a[k], b[k], path + '/' + str(k))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(path + ': length mismatch')
        for i, (x, y) in enumerate(zip(a, b)):
            finite_exact(x, y, path + '/' + str(i))
    else:
        for value in (a, b):
            if isinstance(value, float) and not math.isfinite(value):
                raise AssertionError(path + ': non-finite value')
        if type(a) is not type(b) or a != b:
            raise AssertionError(path + ': exact value mismatch: ' + repr(a) + ' != ' + repr(b))


def scientific_rows(rows):
    return [{k: v for k, v in row.items() if k not in STEP_COUNTERS} for row in rows]


def compare(identity):
    import torch
    import numpy as np
    from PIL import Image
    checks, artifacts = [], {}
    results = [RUN / variant / 'results' for variant in ('reference', 'optimized')]
    for job in identity['identity']['subset']['jobs']:
        folders = [root / job['id'] for root in results]
        records = [read(p / 'record.json') for p in folders]
        for folder, record in zip(folders, records):
            if record.get('status') != 'completed':
                raise AssertionError(job['id'] + ': incomplete record')
            for name, expected in record['artifact_sha256'].items():
                actual = folder.parent / name if name.startswith('sources/') else folder / name
                if sha(actual) != expected:
                    raise AssertionError('Committed artifact changed: ' + str(actual))
        captures = [torch.load(p / 'equivalence_tensors.pt', map_location='cpu', weights_only=True) for p in folders]
        for key in ('y', 'ids', 'z'):
            a, b = [c[key] for c in captures]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item() or not torch.equal(a, b):
                raise AssertionError(job['id'] + ': tensor not exactly equal/finite: ' + key)
        profiles = [c['budget_profile'] for c in captures]
        if all(isinstance(p, list) for p in profiles):
            profiles = [scientific_rows(p) for p in profiles]
        finite_exact(*profiles, job['id'] + '/input_budget_profile')
        pngs = ['raw.png', 'pattern.png']
        if any((p / 'final.png').exists() for p in folders):
            pngs.append('final.png')
        for name in pngs:
            arrays = []
            for folder in folders:
                with Image.open(folder / name) as im:
                    arrays.append(np.asarray(im).copy())
            if arrays[0].dtype != arrays[1].dtype or not np.array_equal(*arrays):
                raise AssertionError(job['id'] + ': PNG pixels differ: ' + name)
        maps = [p / 'request_maps.npz' for p in folders]
        map_keys = []
        if job['arm'] == 'U25':
            if any(p.exists() for p in maps):
                raise AssertionError('U25 must not create an E/P request map.')
        else:
            with np.load(maps[0], allow_pickle=False) as a, np.load(maps[1], allow_pickle=False) as b:
                if set(a.files) != set(b.files) or not {'E_original', 'P_original'}.issubset(a.files):
                    raise AssertionError(job['id'] + ': E/P map key mismatch')
                map_keys = sorted(a.files)
                for key in map_keys:
                    x, y = a[key], b[key]
                    if x.dtype != y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.array_equal(x, y):
                        raise AssertionError(job['id'] + ': E/P map differs/non-finite: ' + key)
        edits = [read(p / 'steps.json')['edit'] for p in folders]
        rows = [e['steps'] for e in edits]
        for steps in rows:
            if len(steps) != 28 or any(not {'step_index', 'sigma', 'h', 'applied_correction_rms'}.issubset(s) for s in steps):
                raise AssertionError(job['id'] + ': missing full 28-step RMS profile')
        finite_exact(*[scientific_rows(r) for r in rows], job['id'] + '/28_steps')
        checks.append(dict(job=job['id'], arm=job['arm'], status='exact_equal', tensor_keys=['y', 'ids', 'z'],
                           pngs=pngs, map_keys=map_keys, steps=28,
                           reference={k: records[0].get(k) for k in ('nfe_standalone', 'nfe_incremental', 'nfe_total', 'nfe_edit', 'timings', 'runtime_seconds_synchronized')},
                           optimized={k: records[1].get(k) for k in ('nfe_standalone', 'nfe_incremental', 'nfe_total', 'nfe_edit', 'timings', 'runtime_seconds_synchronized')}))
        for folder in folders:
            for name in pngs + ['steps.json', 'record.json', 'equivalence_tensors.pt'] + ([] if job['arm'] == 'U25' else ['request_maps.npz']):
                path = folder / name
                artifacts[path.relative_to(BASE).as_posix()] = sha(path)
    return dict(status='passed', fingerprint=identity['fingerprint'], checks=checks, artifact_sha256=artifacts,
                tolerance='zero: torch.equal / numpy.array_equal / exact finite scalar values',
                raw_definition='Frozen runtime raw.png is the final decoded output; compare separate final.png only when actually emitted.',
                step_fields_excluded=sorted(STEP_COUNTERS),
                timing_scope='Measured six-arm worker records include capture overhead; not a 480-job speed claim',
                source_code_sha256={k: identity['identity'][k] for k in ('reference', 'optimized', 'launcher')},
                scientific_fingerprint=digest({k: identity['identity'][k] for k in ('scientific_files', 'formal_plan', 'subset')}),
                environment_fingerprint=digest(identity['identity']['environment']),
                identity_file='run/identity.json', checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def valid_pass(identity):
    report = BASE / 'report.json'
    if not report.is_file():
        return False
    value = read(report)
    if value.get('status') != 'passed' or value.get('fingerprint') != identity['fingerprint']:
        return False
    if not value.get('artifact_sha256'):
        return False
    for relative, expected in value['artifact_sha256'].items():
        path = (BASE / relative).resolve()
        if not path.is_relative_to(BASE.resolve()) or not path.is_file() or sha(path) != expected:
            raise RuntimeError('Approved equivalence artifact missing/changed: ' + relative + '. Preserve/remove equivalence/run and report.json, then rerun.')
    return True


def ensure(formal=None, require_only=False):
    offline()
    if formal is None:
        import run as launcher
        formal = launcher.materialize(ROOT / 'manifest.json', ROOT / 'data')
    identity = fingerprint(formal)
    if valid_pass(identity):
        print('EXACT EQUIVALENCE PASS reused: code, scientific inputs, environment and artifacts unchanged.', flush=True)
        return
    if require_only:
        raise RuntimeError('Formal generation requires a current exact equivalence PASS. Run python equivalence.py first.')
    deep_cpu_preflight(identity)
    stage_run(identity)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for variant in ('reference', 'optimized'):
        subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', variant, '--preflight-only'], env=env, check=True)
    start = time.monotonic()
    try:
        for variant in ('reference', 'optimized'):
            print('EQUIVALENCE ' + variant + ': entering shared GPU queue for six arms.', flush=True)
            subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', variant], env=env, check=True)
        result = compare(identity)
        if fingerprint(formal)['fingerprint'] != identity['fingerprint']:
            raise RuntimeError('Code/input/environment changed during equivalence; no pass issued.')
        result['total_wall_seconds_including_queue_and_load'] = time.monotonic() - start
        save(BASE / 'report.json', result)
    except Exception as exc:
        save(BASE / 'report.json', dict(status='failed', fingerprint=identity['fingerprint'], error=str(exc)))
        raise
    print('EXACT EQUIVALENCE PASS: six arms; tensors, image pixels, E/P maps and 28-step profiles identical.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference', 'optimized'])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.preflight_only)
    else:
        ensure()


if __name__ == '__main__':
    main()
