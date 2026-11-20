"""CPU-only evaluation boundary; keep per-job resume and source-feature caches."""
import csv
import math
import os
from pathlib import Path
import subprocess
import sys


def valid_row(row):
    try:
        if row.get('eval_status') != 'ok' or row.get('eval_device') != 'cpu' or row.get('lpips_status') != 'ok':
            return False
        fields = ('clip_target_similarity', 'clip_image_similarity', 'dino_full_feature_similarity',
                  'dino_pattern_crop_feature_similarity', 'lpips_full')
        if not all(math.isfinite(float(row[k])) for k in fields):
            return False
        if row.get('clip_directional_status') != 'zero_direction' or row.get('clip_directional_similarity') not in ('', None):
            if not math.isfinite(float(row['clip_directional_similarity'])):
                return False
        if row.get('lpips_preserved_status') == 'ok_receptive_field_spill':
            if not math.isfinite(float(row['lpips_preserved'])):
                return False
        elif row.get('lpips_preserved_status') != 'empty_region':
            return False
        if row.get('pixel_status') == 'ok':
            psnr = float(row['pixel_psnr_db'])
            return not math.isnan(psnr) and psnr != -math.inf
        return row.get('pixel_status') == 'empty_region'
    except (ValueError, TypeError, KeyError):
        return False


def cache_matches(row, *, raw_sha, source_sha, record_sha, mask_sha, mask_path, data_dir):
    """Per-job content identity: a changed mask invalidates only that job's cached metrics."""
    expected = dict(eval_raw_sha256=raw_sha, eval_source_sha256=source_sha,
                    eval_record_sha256=record_sha, eval_mask_sha256=mask_sha,
                    eval_mask_path=str(mask_path), eval_data_dir=str(data_dir))
    return bool(mask_sha) and valid_row(row) and all(row.get(k) == value for k,value in expected.items())


def command(runtime_root, *, clip_path=None, dino_path=None, lpips_path=None, lpips_trunk_path=None, device='cpu'):
    """Pure command builder for the parent; never clears caches or touches CUDA."""
    if device != 'cpu':
        raise ValueError('R89 evaluate_all preserves the original CPU evaluation contract')
    root = Path(runtime_root).resolve()
    paths = dict(CLIP_PATH=clip_path, DINO_PATH=dino_path, LPIPS_PATH=lpips_path, LPIPS_TRUNK_PATH=lpips_trunk_path)
    paths = {name:Path(value or os.environ.get(name, '')).expanduser().resolve()
             for name,value in paths.items() if value or os.environ.get(name)}
    missing = set(('CLIP_PATH','DINO_PATH','LPIPS_PATH','LPIPS_TRUNK_PATH')) - paths.keys()
    if missing:
        raise ValueError('Explicit local evaluator paths required: '+', '.join(sorted(missing)))
    for name,path in paths.items():
        if not (path.is_dir() if name in ('CLIP_PATH','DINO_PATH') else path.is_file()):
            raise FileNotFoundError(name+': '+str(path))
    argv = [sys.executable, '-B', '-u', str(Path(__file__).with_name('evaluate.py')),
            '--from-results', str(root/'results'), '--data-dir', str(root/'data'), '--device', 'cpu',
            '--clip-path', str(paths['CLIP_PATH']), '--dino-path', str(paths['DINO_PATH']),
            '--lpips-path', str(paths['LPIPS_PATH']), '--lpips-trunk-path', str(paths['LPIPS_TRUNK_PATH'])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    return argv, env


def evaluate_all(runtime_root, **options):
    import reporting as report
    root = Path(runtime_root).resolve()
    argv, env = command(root, **options)
    plan, _ = report.load_plan(root)
    provenance = report.fingerprint(root)
    report.guard_existing(root, provenance)
    out = root/'results'
    if report.read(out/'plan.json') != plan:
        raise ValueError('Evaluation results plan differs from generation plan')
    if any(report.completed(root, job, provenance['fingerprint']) is None for job in plan['jobs']):
        raise RuntimeError('evaluate_all requires all 640 committed native results')
    # Existing evaluation.csv stays in place. Evaluator invalidates only changed inputs/configuration.
    # No generation caches, original R88 outputs, statistics, ZIPs or GPU queue locks are touched.
    with report.file_lock(root/'.evaluation.lock'):
        subprocess.run(argv, env=env, cwd=root, check=True)
        with (out/'evaluation.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if (len(rows) != len(plan['jobs']) or {r['id'] for r in rows} != {j['id'] for j in plan['jobs']}
                or not all(valid_row(r) for r in rows)):
            raise RuntimeError('Evaluation incomplete; per-job cache retained for retry')
    return dict(status='completed', completed=len(rows), device='cpu', evaluation_csv=str(out/'evaluation.csv'),
                data_dir=str(root/'data'), cache_policy='existing CSV resumed by configuration and input hashes; bounded source/text feature cache',
                generation_nfe=0, statistics_owner='parent')
