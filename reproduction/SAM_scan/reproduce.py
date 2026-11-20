"""Statistical reproduction from the bundled scores and selected core profile fields."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from overlap import COMPARISONS, FROZEN_METRICS, X, Y, integrate, pair_summary, stats

ROOT = Path(__file__).resolve().parent
CORE_METRICS = (Y, X, 'mae_edit', 'clip_target')
PAIRS = [('sam_native', 'uniform'), ('two_constant', 'uniform'), ('sam_native', 'two_constant'),
         ('sam_native', 'uniform_at_local'), ('spatial_at_uniform', 'uniform'), ('flowedit_24', 'uniform')]
CROSSED_ARMS = ['sam_native', 'uniform_at_local', 'spatial_at_uniform', 'uniform']
CONTRASTS = {'native_spatial_minus_uniform': [1, 0, 0, -1],
             'allocation_at_spatial_profile': [1, -1, 0, 0], 'allocation_at_uniform_profile': [0, 0, 1, -1],
             'change_profile_with_uniform_allocation': [0, 1, 0, -1],
             'change_profile_with_spatial_allocation': [1, 0, -1, 0],
             'interaction_spatial_minus_uniform_profile': [1, -1, -1, 1]}


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def observed():
    data = ROOT / 'data'
    spec = json.loads((data/'protocol.json').read_text(encoding='utf-8'))
    sources = read_csv(data/'sources.csv')
    rules = {r['method']: r for r in spec['settings']}
    source = {r['case']: r for r in sources}
    cases = sorted(source)
    eligible = {c for c in cases if int(source[c]['preservation_pixels']) > 0}
    if len(cases) != len(sources) or len(cases) != 50 or len(eligible) != 42 or len(rules) != 18:
        raise ValueError('Unexpected source panel or settings')
    scores, curves = {}, defaultdict(list)
    for r in read_csv(data/'scores.csv'):
        c, method = r['case'], r['method']
        s, rule = source[c], rules[method]
        if ((c,method) in scores or r['id'] != c+'__'+method or r['category'] != s['category']
                or r['seed'] != s['seed'] or r['arm'] != rule['arm'] or r['stage'] != rule['stage']
                or float(r['scale']) != rule['scale']
                or (int(r['n_max']) if r['n_max'] else None) != rule['n_max']):
            raise ValueError('Inconsistent job '+r['id'])
        values = {m: float(r[m]) if r[m] else None for m in FROZEN_METRICS}
        for m, v in values.items():
            empty = (m.endswith('_preservation') and c not in eligible)
            if (v is None) != empty or (v is not None and not math.isfinite(v)):
                raise ValueError('Invalid metric '+r['id']+' '+m)
        scores[c,method] = values
        if rule['curve_rank'] is not None:
            curves[c,r['arm']].append(dict(case=c, category=s['category'], rank=rule['curve_rank'], **values))
    if len(scores) != len(cases)*len(rules):
        raise ValueError('Incomplete score panel')
    for c in cases:
        for arm in spec['grid']:
            curve = curves[c,arm]
            curve.sort(key=lambda p: p['rank'])
            if [p['rank'] for p in curve] != [1,2,3,4,5]:
                raise ValueError('Incomplete curve '+c+' '+arm)
    return cases, eligible, spec, scores, curves


def scan_results(cases, eligible, spec, curves):
    summaries, rows, points = {}, [], []
    for name, (a,b) in COMPARISONS.items():
        records = [integrate(c, eligible, name, curves[c,a], curves[c,b])[0] for c in cases]
        rows.extend(records)
        summaries[name] = pair_summary(records)
    for arm, grid in spec['grid'].items():
        for i, knob in enumerate(grid):
            cells = [curves[c,arm][i] for c in cases]
            points.append(dict(arm=arm, rank=i+1, knob=knob, is_default=knob==spec['curve_defaults'][arm],
                               response_denominator=len(cases), preservation_denominator=len(eligible),
                               metrics={m:stats([p[m] for p in cells]) for m in FROZEN_METRICS}))
    return dict(pairs=summaries, operating_points=points), rows


def core_results(cases, eligible, scores):
    # The original core script consumes one stream in contrast-major, metric-minor order.
    rng = np.random.default_rng(20261118)
    def delta(c, left, right, metric):
        return scores[c,left][metric] - scores[c,right][metric]
    def estimate(values):
        a = np.asarray(values)
        draws = rng.integers(len(a), size=(20000,len(a)))
        return dict(n=len(a), mean=float(a.mean()), ci=np.quantile(a[draws].mean(1), [.025,.975]).tolist(),
                    positive=int((a>0).sum()), negative=int((a<0).sum()))
    contrasts = {}
    for left,right in PAIRS:
        contrasts[left+' - '+right] = {
            m:estimate([delta(c,left,right,m) for c in cases if scores[c,left][m] is not None])
            for m in CORE_METRICS}
    regional = [c for c in cases if c in eligible]
    native = np.array([delta(c,'uniform','sam_native',Y) for c in regional])
    constant = np.array([delta(c,'uniform','two_constant',Y) for c in regional])
    draws = rng.integers(len(native), size=(20000,len(native)))
    den = native[draws].mean(1)
    recovery = dict(n=len(native), native_gain=float(native.mean()), constant_gain=float(constant.mean()),
                    ratio=float(constant.mean()/native.mean()),
                    ci=np.quantile(constant[draws].mean(1)/den,[.025,.975]).tolist() if (den>0).all() else None)
    common = {l+' - '+r:estimate([delta(c,l,r,X) for c in regional]) for l,r in PAIRS}
    source_rows = [dict(case=c, native_gain=n, constant_gain=k) for c,n,k in zip(regional,native,constant)]
    return dict(contrasts=contrasts, recovery=recovery, common42=common), source_rows


def crossed_results(cases, scores):
    summary, rows = {}, []
    for m in (Y,X,'mae_edit'):
        ids = [c for c in cases if all(scores[c,a][m] is not None for a in CROSSED_ARMS)]
        matrix = np.array([[scores[c,a][m] for a in CROSSED_ARMS] for c in ids])
        draws = np.random.default_rng(1208).integers(len(ids),size=(20000,len(ids)))
        summary[m] = {}
        for name, coeff in CONTRASTS.items():
            d = matrix @ np.array(coeff)
            summary[m][name] = dict(n=len(ids), mean=float(d.mean()), positive=int((d>0).sum()),
                                    ci=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
            rows.extend(dict(case=c, metric=m, contrast=name, difference=v) for c,v in zip(ids,d))
    return summary, rows


def profile_results(cases):
    steps = defaultdict(dict)
    for r in read_csv(ROOT/'data/core_steps.csv'):
        group = steps[r['case'],r['method']]
        step = int(r['step'])
        if step in group:
            raise ValueError('Duplicate active step')
        group[step] = r
    anchors = {'two_constant':'sam_native', 'uniform_at_local':'sam_native', 'spatial_at_uniform':'uniform'}
    rows, passed = [], Counter()
    for c in cases:
        for method, anchor in anchors.items():
            a, t = steps[c,method], steps[c,anchor]
            if len(a)!=26 or a.keys()!=t.keys() or any((a[k]['sigma'],a[k]['next_sigma']) !=
                    (t[k]['sigma'],t[k]['next_sigma']) for k in a):
                raise ValueError('Unequal active time grids')
            for stage, suffix, rtol in [('fp32','_rms',1e-4),('bf16','_state_rms',.01)]:
                both = True
                for branch in ('model','visible'):
                    col = branch+suffix
                    target = np.array([float(t[k][col]) for k in sorted(a)])
                    actual = np.array([float(a[k][col]) for k in sorted(a)])
                    error = actual-target
                    failures = int((abs(error)>1e-6+rtol*target).sum())
                    gain = None
                    if method=='two_constant':
                        gain = sum(float(t[k][branch+'_rms']) for k in sorted(t))/sum(float(t[k]['residual_rms']) for k in sorted(t))
                    both = both and failures==0
                    rows.append(dict(case=c,method=method,anchor=anchor,branch=branch,stage=stage,steps=len(a),
                                     native_calibrated_gain=gain, signed_total_relative=float(error.sum()/target.sum()),
                                     integrated_abs_relative=float(abs(error).sum()/target.sum()),
                                     failed_steps=failures if method!='two_constant' else None))
                if method!='two_constant':
                    passed[method+'_'+stage] += int(both)
    return dict(core_profile_jobs=len(steps), step_rows=sum(map(len,steps.values())),
                both_branches_all_steps_pass=dict(passed)), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    cases, eligible, spec, scores, curves = observed()
    scan, scan_rows = scan_results(cases,eligible,spec,curves)
    core, gains = core_results(cases,eligible,scores)
    crossed, crossed_rows = crossed_results(cases,scores)
    profile, profile_rows = profile_results(cases)
    counts = dict(scores=len(scores), sources=len(cases), preservation_sources=len(eligible),
                  core_jobs=sum(s['stage']=='core' for s in spec['settings'])*len(cases),
                  scan_points=sum(map(len,curves.values())), valid_preservation_scores=sum(v[Y] is not None for v in scores.values()))
    summary = dict(counts=counts, scan=scan, core=core, crossed=crossed, profiles=profile)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, rows in [('scan_per_source',scan_rows),('core_gains',gains),('crossed_per_source',crossed_rows),
                       ('profile_per_source',profile_rows)]:
        write_csv(args.out/(name+'.csv'),rows)
    match = scan['pairs']['SU']
    print(f"{len(scores)} scores | {len(cases)} sources | {len(eligible)} preservation-eligible | {counts['scan_points']} curve points")
    print(f"Two-constant recovery: {core['recovery']['ratio']:.1%} (n={core['recovery']['n']})")
    print(f"Uniform minus native at matched response: {match['mean']:.4f} "
          f"[{match['lo']:.4f}, {match['hi']:.4f}], {match['valid']}/{match['eligible']} positive-width overlaps")


if __name__=='__main__':
    main()

