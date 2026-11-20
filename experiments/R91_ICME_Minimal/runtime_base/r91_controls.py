"""R91 changes allocation only. No model calls or evaluation-mask inputs."""
import hashlib,json,os
from pathlib import Path

def permutation(case,seed,draw):
    # Hash sorting specifies the permutation independently of NumPy/PyTorch RNG versions.
    prefix=f'R91-cell-v1:{case}:{seed}:{draw}:'
    return sorted(range(4096),key=lambda i:(hashlib.sha256((prefix+str(i)).encode()).digest(),i))

def install(controls,job):
    import torch
    from controls import pack
    original=controls['r86_P_original'];E=controls['r86_E_original']
    if original.shape!=(1,1,64,64):raise ValueError('R91 requires 64x64 original P')
    mode=job['r91_map_mode']
    info=dict(mode=mode,base_floor=.25,extra_nfe=0)
    if mode=='cell_permutation':
        indices=permutation(job['case'],job['seed'],job['r91_permutation_draw'])
        P=original.flatten()[torch.tensor(indices,device=original.device)].reshape_as(original)
        if not torch.equal(torch.sort(P.flatten()).values,torch.sort(original.flatten()).values):raise ValueError('Histogram mismatch')
        info.update(draw=job['r91_permutation_draw'],histogram_preserved=True,permutation_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest())
    elif mode=='floor':
        floor=float(job['r91_floor'])
        if floor not in (0.,.5):raise ValueError('Unexpected floor')
        P=floor+(1-floor)*(1-E)
        info.update(actual_floor=floor,histogram_preserved=False)
    else:raise ValueError('Unknown R91 allocation')
    if not torch.isfinite(P).all() or P.min()<0 or P.max()>1:raise ValueError('Invalid allocation')
    channels=controls['route_mask'].shape[-1]//4
    controls['r88_P_correction']=P
    controls['effective_route_mask64']=P
    controls['r88_P_correction_packed']=pack(P.expand(-1,channels,-1,-1))
    # Legacy route_floor=.25 denotes the base probe. The exact packed P bypasses
    # re-flooring in spectral_candidate; actual_floor is saved separately.
    controls['route_mask64'].copy_((P-.25)/.75)
    controls['route_mask'].copy_(pack(controls['route_mask64'].expand(-1,channels,-1,-1)))
    controls['r91_allocation']=info
    return dict(r91_allocation=info)

_checked=set()
def barrier(root,job):
    """Four held-out historical reproductions before any new scientific output."""
    phases=['sentinel','maps','seed1','seed2','seed3']
    if phases.index(job['r91_phase'])>phases.index(os.environ.get('R91_THROUGH','seed3')):
        (root/'PAUSE').touch()
        return
    if job.get('r91_phase')=='sentinel' or str(root) in _checked:return
    import numpy as np
    from PIL import Image
    import reporting
    ids=reporting.read(root/'historical/sentinel_jobs.json');rows=[]
    for jid in ids:
        a=np.asarray(Image.open(root/'historical/sentinels'/jid/'raw.png').convert('RGB'),dtype=float)
        b=np.asarray(Image.open(root/'results'/jid/'raw.png').convert('RGB'),dtype=float)
        delta=np.abs(a-b)
        rows.append(dict(id=jid,rgb_mae_8bit=float(delta.mean()),p99_8bit=float(np.quantile(delta,.99)),max_8bit=float(delta.max()),passed=bool(delta.mean()<=.25 and np.quantile(delta,.99)<=2)))
    passed=all(r['passed'] for r in rows)
    reporting.save(root/'results/sentinel_check.json',dict(passed=passed,rows=rows,criterion='RGB MAE <=0.25/255 and 99th percentile <=2/255. Numerical compatibility check, not a perceptual equivalence test.'))
    if not passed:raise RuntimeError('Historical reproduction differs: inspect sentinel_check.json; no automatic pooling or tolerance override.')
    _checked.add(str(root))

def validate_plan(root):
    import reporting
    from r86_compat import expected_anchor_arm
    plan,tasks=reporting.load_plan(root);archive=reporting.read(root/'historical/anchors.json')
    seen={};counts={}
    for j in plan['jobs']:
        phase=j['r91_phase'];counts[phase]=counts.get(phase,0)+1
        if j.get('r86_anchor'):
            aid=j['r86_anchor'];a=archive[aid]['job'] if j.get('r91_archived_anchor') else seen[aid]
            assert a['arm']==expected_anchor_arm(j)
            for k in ('case','seed','source_prompt','target_prompt','guidance','steps','src_steps','attention_blocks','memory_pairing'):
                assert a.get(k)==j.get(k),(j['id'],k)
        assert j['guidance']==3.5 and j['steps']==28 and j['corr_lambda']==.25
        assert j['resolution']==512 and j['query_selection']=='all_image_queries'
        seen[j['id']]=j
    assert counts==dict(sentinel=4,maps=240,seed1=160,seed2=160,seed3=160),counts
    return counts
