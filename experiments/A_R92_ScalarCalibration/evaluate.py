"""Offline CPU evaluation. Run after generation: python evaluate.py [--check-only]."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
METRICS = ('lpips_preserved', 'clip_target_similarity', 'clip_directional_similarity',
           'clip_image_similarity', 'edit_mae', 'preservation_mae')
FIELDS = ['id', 'case', 'seed', 'arm', 'eval_status', 'eval_device', 'lpips_status',
          'lpips_preserved_status', *METRICS, 'raw_sha256', 'source_sha256',
          'outer_sha256', 'job_sha256', 'metric_signature']


class Pending(RuntimeError):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Progress:
    def __init__(self, path, check_only, threads):
        self.path = path
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.time()
        self.data = dict(status='checking', eval_status='not_run', phase='preflight',
                         evaluation_pending=True, check_only=check_only, device='cpu',
                         threads=threads, pid=os.getpid(), total=0, completed=0,
                         evaluated_this_run=0, resumed=0, current_job=None, error=None)
        self.write()
        self.worker = threading.Thread(target=self.heartbeat, daemon=True)
        self.worker.start()

    def write(self, **updates):
        with self.lock:
            self.data.update(updates)
            now = time.time()
            self.data.update(updated_at_unix=now, heartbeat_unix=now,
                             updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),
                             elapsed_seconds=round(now - self.started, 3))
            atomic_json(self.path, self.data)

    def heartbeat(self):
        while not self.stop.wait(2.0):
            try:
                self.write()
            except Exception as exc:
                print('Status heartbeat write failed: %s' % exc, file=sys.stderr, flush=True)

    def close(self):
        self.stop.set()
        self.worker.join()


def configure(threads):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', CUDA_VISIBLE_DEVICES='')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    packages = Path('D:/DevCache/tracememory_metrics/packages')
    if os.name == 'nt' and packages.is_dir():
        sys.path.insert(0, str(packages))


def unique(paths):
    return list(dict.fromkeys(p.expanduser().resolve() for p in paths))


def clip_files(path):
    """Validate local inventory/config only; never deserialize model weights here."""
    config = read_json(path / 'config.json')
    vision = config.get('vision_config', {})
    if (config.get('model_type') != 'clip' or vision.get('patch_size') != 14
            or vision.get('hidden_size') != 1024 or vision.get('num_hidden_layers') != 24
            or vision.get('image_size', 224) != 224):
        raise ValueError('expected CLIP ViT-L/14 (224px) config')
    files = [path / 'config.json', path / 'preprocessor_config.json']
    if (path / 'tokenizer.json').is_file():
        files.append(path / 'tokenizer.json')
    else:
        files.extend([path / 'vocab.json', path / 'merges.txt'])
    for optional in ('tokenizer_config.json', 'special_tokens_map.json'):
        if (path / optional).is_file():
            files.append(path / optional)
    for name in ('model.safetensors', 'pytorch_model.bin',
                 'model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        weight = path / name
        if weight.is_file():
            files.append(weight)
            if name.endswith('.index.json'):
                files.extend(path / n for n in sorted(set(read_json(weight)['weight_map'].values())))
            break
    else:
        raise ValueError('missing model.safetensors or pytorch_model.bin (or shard index)')
    missing = [str(p) for p in files if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError('missing/empty files: ' + ', '.join(missing))
    return files


def discover():
    specs, problems = {}, []
    for name in ('numpy', 'PIL', 'torch', 'torchvision', 'transformers', 'lpips'):
        try:
            specs[name] = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            specs[name] = None
        if specs[name] is None:
            problems.append('Missing dependency: %s (Python: %s)' % (name, sys.executable))
    home_cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
    local = Path('/root/autodl-tmp/local-models')
    roots = [local, ROOT / 'models', ROOT / 'hf_cache/hub', Path.cwd() / 'hf_cache/hub',
             local / 'hf_cache/hub', Path('/root/autodl-tmp/hf_cache/hub'),
             home_cache / 'huggingface/hub']
    if os.name == 'nt':
        roots.append(Path('D:/DevCache/tracememory_metrics/models'))
    for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    if os.environ.get('HF_HOME'):
        roots.extend([Path(os.environ['HF_HOME']), Path(os.environ['HF_HOME']) / 'hub'])
    torch_home = Path(os.environ.get('TORCH_HOME', str(home_cache / 'torch')))
    roots.extend([torch_home / 'hub/checkpoints', local / 'torch/hub/checkpoints',
                  local / 'checkpoints', Path('/root/autodl-tmp/torch/hub/checkpoints')])
    roots = unique(roots)
    trunks, calibrations, clips = [], [], []
    for base in roots:
        trunks.extend([base / 'alexnet-owt-7be5be79.pth', base / 'hub/checkpoints/alexnet-owt-7be5be79.pth'])
        calibrations.extend([base / 'lpips/weights/v0.1/alex.pth', base / 'weights/v0.1/alex.pth',
                             base / 'lpips/v0.1/alex.pth', base / 'alex.pth'])
        if base.is_dir():
            clips.append(base)
            clips.extend(sorted(base.glob('*clip*')))
            clips.extend(sorted(base.glob('models--openai--clip-vit-large-patch14/snapshots/*')))
            clips.extend(sorted(base.glob('snapshots/*')))
            trunks.extend(sorted(base.glob('*/alexnet-owt-7be5be79.pth')))
    if specs['lpips']:
        for location in specs['lpips'].submodule_search_locations or []:
            calibrations.append(Path(location) / 'weights/v0.1/alex.pth')
    models, inventory = {}, []
    for key, candidates in (('LPIPS_TRUNK_PATH', trunks), ('LPIPS_CALIBRATION_PATH', calibrations),
                            ('CLIP_MODEL_PATH', clips)):
        explicit = os.environ.get(key)
        candidates = unique([Path(explicit)] if explicit else candidates)
        invalid = []
        for path in candidates:
            if key == 'CLIP_MODEL_PATH':
                if not path.is_dir() or not (path / 'config.json').is_file():
                    continue
                try:
                    files = clip_files(path)
                except (ValueError, KeyError, OSError) as exc:
                    invalid.append('%s: %s' % (path, exc))
                    continue
            else:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                files = [path]
            models[key] = str(path)
            inventory.extend(files)
            break
        if key not in models:
            problems.append('Missing/invalid %s%s. Set %s to an existing local %s. Searched: %s%s' % (
                key, '='+explicit if explicit else '', key,
                'CLIP ViT-L/14 model directory' if key == 'CLIP_MODEL_PATH' else 'weight file',
                ', '.join(map(str, candidates)), '; ' + '; '.join(invalid) if invalid else ''))
    if problems:
        raise Pending('\n'.join(problems))
    # File identities invalidate resume if local weights/config are replaced.
    identity = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in unique(inventory)]
    signature = json_digest(dict(definition='R91-R89-spatial-alex01-normalized-direction-v1',
                                 files=identity, helper=digest(ROOT / 'metric_helpers.py'),
                                 evaluator=digest(Path(__file__))))
    return dict(models=models, metric_signature=signature, python=sys.executable,
                dependencies={k: v.origin for k, v in specs.items()},
                check_scope='module discovery and local model inventory; imports and weight loading not run')


def save_csv(path, jobs, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: done[j['id']].get(k, '') for k in FIELDS}
                    for j in jobs if j['id'] in done)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def finite(row):
    try:
        return all(math.isfinite(float(row[k])) for k in METRICS)
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def inputs(rt, job):
    jid, case = job['id'], str(job['case'])
    raw = rt / 'results' / jid / 'raw.png'
    source = rt / 'results/sources' / (case + '_source_eval.png')
    mask = rt / 'data' / case / 'outer.png'
    record = read_json(rt / 'results' / jid / 'record.json')
    hashes = dict(raw_sha256=digest(raw), source_sha256=digest(source), outer_sha256=digest(mask))
    for field, record_field in (('raw_sha256', 'raw_sha256'), ('source_sha256', 'source_eval_sha256'),
                                ('outer_sha256', 'outer_sha256')):
        if hashes[field] != record.get(record_field):
            raise Pending('%s: %s mismatch/missing in record.json; native input: %s' %
                          (jid, record_field, {'raw_sha256': raw, 'source_sha256': source,
                                              'outer_sha256': mask}[field]), 3)
    hashes['job_sha256'] = json_digest(job)
    return raw, source, mask, hashes


def evaluate(rt, progress, threads, preflight):
    jobs = read_json(rt / 'plan.json')['jobs']
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise Pending('plan.json jobs must be nonempty with unique IDs', 3)
    for j in jobs:
        for field in ('id', 'case', 'seed', 'arm', 'source_prompt', 'target_prompt'):
            if field not in j:
                raise Pending('plan job missing field: ' + field, 3)
        for field in ('id', 'case'):
            value = str(j[field])
            if not value or value in ('.', '..') or '/' in value or '\\' in value or ':' in value:
                raise Pending('invalid path component %s=%r' % (field, value), 3)
    out = rt / 'results/evaluation.csv'
    progress.write(total=len(jobs), phase='validating_resume', output=str(out))
    previous = {}
    if out.is_file():
        with out.open(encoding='utf-8-sig', newline='') as f:
            previous = {r['id']: r for r in csv.DictReader(f)}
    done, prepared, errors = {}, {}, []
    signature = preflight['metric_signature']
    for j in jobs:
        jid = j['id']
        progress.write(current_job=jid)
        try:
            prepared[jid] = inputs(rt, j)
            hashes = prepared[jid][3]
            row = previous.get(jid, {})
            if (row.get('eval_status') == 'ok' and row.get('eval_device') == 'cpu'
                    and row.get('metric_signature') == signature and finite(row)
                    and all(row.get(k) == v for k, v in hashes.items())):
                done[jid] = row
        except (OSError, ValueError, KeyError, Pending) as exc:
            errors.append('%s: %s' % (jid, exc))
    save_csv(out, jobs, done)
    resumed = len(done)
    progress.write(completed=resumed, resumed=resumed, current_job=None)
    if errors:
        raise Pending('Native input validation failed:\n' + '\n'.join(errors), 3)
    if len(done) == len(jobs):
        return
    progress.write(status='running', eval_status='running', phase='loading_models')
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        from metric_helpers import load_lpips, lpips_tensor, image, cosine, clip_features
    except Exception as exc:
        raise Pending('Dependency import failed in %s: %s: %s' %
                      (sys.executable, type(exc).__name__, exc)) from exc
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(2, threads))
    model_paths = preflight['models']
    try:
        lp = load_lpips(model_paths['LPIPS_CALIBRATION_PATH'], model_paths['LPIPS_TRUNK_PATH'], 'cpu')
        lp.spatial = True
        clip = CLIPModel.from_pretrained(model_paths['CLIP_MODEL_PATH'], local_files_only=True).to('cpu').eval()
        cp = CLIPProcessor.from_pretrained(model_paths['CLIP_MODEL_PATH'], local_files_only=True)
    except Exception as exc:
        raise Pending('Offline CPU model load failed: %s: %s; paths=%s' %
                      (type(exc).__name__, exc, json.dumps(model_paths))) from exc
    environment = dict(preflight, torch=torch.__version__, device='cpu', threads=threads,
                       models_loaded=True, runtime_base=str(rt), metric_names=list(METRICS),
                       lpips_calibration_sha256=digest(model_paths['LPIPS_CALIBRATION_PATH']),
                       sources_used={str(j['case']): str(prepared[j['id']][1]) for j in jobs},
                       masks_used={str(j['case']): str(prepared[j['id']][2]) for j in jobs})
    environment.pop('check_scope', None)
    atomic_json(progress.path.parent / 'local_evaluation_environment.json', environment)
    cache = {}
    with torch.inference_mode():
        for j in jobs:
            jid, case = j['id'], j['case']
            if jid in done:
                continue
            progress.write(phase='evaluating', current_job=jid)
            raw, src, maskpath, hashes = inputs(rt, j)
            if hashes != prepared[jid][3]:
                raise Pending('Native input changed during evaluation: ' + jid, 3)
            key = (hashes['source_sha256'], hashes['outer_sha256'], j['source_prompt'], j['target_prompt'])
            if key not in cache:
                source = image(src)
                a = np.asarray(source, dtype=np.float64) / 255
                with Image.open(maskpath) as original:
                    mask = np.asarray(original.convert('L').resize((512, 512), Image.Resampling.NEAREST),
                                      dtype=np.float64) / 255
                if not np.isfinite(mask).all() or mask.sum() <= 0 or (1-mask).sum() <= 0:
                    raise Pending('Empty edit/preservation region or nonfinite original mask: ' + str(maskpath), 3)
                sf, tf = clip_features(clip, cp, [source], [j['source_prompt'], j['target_prompt']], 'cpu')
                cache[key] = (a, mask, sf, tf, lpips_tensor(source, 'cpu'))
            a, mask, sf, tf, st = cache[key]
            edited = image(raw)
            spatial = lp(st, lpips_tensor(edited, 'cpu'))
            keep = torch.from_numpy(1-mask).float()[None, None]
            keep = torch.nn.functional.interpolate(keep, size=spatial.shape[-2:], mode='area')
            if keep.sum().item() <= 0:
                raise Pending('Empty LPIPS preservation region: ' + jid, 3)
            ef = clip.get_image_features(**cp(images=[edited], return_tensors='pt')).float()
            norm = lambda x: torch.nn.functional.normalize(x, dim=-1)
            diff = np.abs(a-np.asarray(edited, dtype=np.float64)/255).mean(2)
            row = dict(id=jid, case=case, seed=j['seed'], arm=j['arm'], eval_status='ok', eval_device='cpu',
                       lpips_status='ok', lpips_preserved_status='ok_receptive_field_spill',
                       lpips_preserved=float((spatial*keep).sum()/keep.sum()),
                       clip_target_similarity=cosine(ef[0], tf[1]),
                       clip_directional_similarity=cosine((norm(ef)-norm(sf))[0], (norm(tf)[1]-norm(tf)[0])),
                       clip_image_similarity=cosine(ef[0], sf[0]),
                       edit_mae=float((diff*mask).sum()/mask.sum()),
                       preservation_mae=float((diff*(1-mask)).sum()/(1-mask).sum()),
                       **hashes, metric_signature=signature)
            if not finite(row):
                raise Pending('Nonfinite/undefined metric for %s: %s' % (jid, {k: row[k] for k in METRICS}), 3)
            # Recheck before committing if generation or another writer touched native inputs.
            if inputs(rt, j)[3] != hashes:
                raise Pending('Native input changed before metric commit: ' + jid, 3)
            done[jid] = row
            save_csv(out, jobs, done)
            progress.write(completed=len(done), evaluated_this_run=len(done)-resumed)
            print('%s/%s %s LPIPS=%.6f' % (len(done), len(jobs), jid, row['lpips_preserved']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-base', type=Path, default=ROOT / 'runtime_base')
    parser.add_argument('--check-only', action='store_true', help='Check local dependency/model inventory, without loading weights or evaluating')
    parser.add_argument('--threads', type=int, default=None, help='1..6 CPU threads; default EVAL_CPU_THREADS or 6')
    args = parser.parse_args()
    rt = args.runtime_base.expanduser().resolve()
    progress = Progress(rt.parent / 'status_evaluation.json', args.check_only, args.threads)
    try:
        threads = args.threads if args.threads is not None else int(os.environ.get('EVAL_CPU_THREADS', '6'))
        if not 1 <= threads <= 6:
            raise Pending('--threads / EVAL_CPU_THREADS must be an integer in 1..6', 3)
        progress.write(threads=threads)
        configure(threads)
        preflight = discover()
        progress.write(models=preflight['models'])
        if args.check_only:
            progress.write(status='preflight_ready', eval_status='not_run', phase='preflight_complete',
                           preflight=preflight, evaluation_pending=True)
            print(json.dumps(dict(status='preflight_ready', evaluation_pending=True, **preflight),
                             ensure_ascii=False, indent=2), flush=True)
        else:
            evaluate(rt, progress, threads, preflight)
            progress.write(status='complete', eval_status='ok', phase='complete',
                           evaluation_pending=False, current_job=None)
        return 0
    except KeyboardInterrupt:
        progress.write(status='evaluation_pending', eval_status='pending', phase='interrupted', error='Interrupted')
        return 130
    except Exception as exc:
        code = exc.code if isinstance(exc, Pending) else 1
        message = '%s: %s' % (type(exc).__name__, exc)
        progress.write(status='evaluation_pending', eval_status='pending', evaluation_pending=True,
                       phase='failed', error=message, exit_code=code)
        print('evaluation_pending: ' + message, file=sys.stderr, flush=True)
        return code
    finally:
        progress.close()


if __name__ == '__main__':
    sys.exit(main())
