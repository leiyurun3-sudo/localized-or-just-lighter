"""CPU-only checks of the new transformations and frozen dependency graph."""
from pathlib import Path
import sys,json,ast
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime_base';sys.path.insert(0,str(RT))
import reporting
from r91_controls import permutation,install,validate_plan
from controls import pack
import math
# Evaluate the unchanged tensor-only engine functions without importing the
# diffusion model dependency; no backend/model behavior is simulated.
tree=ast.parse((RT/'engine.py').read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('spectral_candidate','rms')]
scope={'torch':torch,'math':math,'F':torch.nn.functional}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'engine_tensor_functions','exec'),scope)
spectral_candidate,rms=scope['spectral_candidate'],scope['rms']
from r86_compat import validate_profile

def main():
    for p in list(ROOT.glob('*.py'))+list(RT.glob('*.py')):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    counts=validate_plan(RT);plan=reporting.read(RT/'plan.json');archive=reporting.read(RT/'historical/anchors.json')
    reporting.fingerprint(RT)
    torch.manual_seed(91);E=torch.rand(1,1,64,64);P=.25+.75*(1-E);correction=torch.randn(1,1024,64)
    for mode in ['cell_permutation','floor']:
        jobs=[j for j in plan['jobs'] if j.get('r91_map_mode')==mode]
        for j in [jobs[0],jobs[1]]:
            c=dict(r86_P_original=P.clone(),r86_E_original=E.clone(),route_mask64=torch.zeros_like(P),route_mask=torch.zeros_like(correction))
            install(c,j);actual=c['r88_P_correction']
            assert torch.isfinite(actual).all()
            if mode=='cell_permutation':
                assert torch.equal(torch.sort(P.flatten()).values,torch.sort(actual.flatten()).values)
                assert sorted(permutation(j['case'],j['seed'],j['r91_permutation_draw']))==list(range(4096))
            else:assert torch.equal(actual,j['r91_floor']+(1-j['r91_floor'])*(1-E))
            candidate=spectral_candidate(correction,j,c['route_mask'],c['r88_P_correction_packed'])[0]
            assert torch.equal(candidate,pack(actual.expand(-1,16,-1,-1))*correction)
            for target in [.1,1.,.001]:assert abs(rms(candidate*(target/rms(candidate)))-target)<1e-6*target
            a=archive[j['r86_anchor']]['profile'];sigmas=[r['sigma'] for r in a]+[a[-1]['sigma']-a[-1]['h']];validate_profile(j,a,sigmas)
    assert permutation('x',0,9001)!=permutation('x',0,9002)
    # Independent simple statistics check: source aggregation and Holm ordering.
    sys.path.insert(0,str(ROOT))
    from analyze import holm,interval
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    lo,hi=interval(np.ones(40)*-.01,['a']*20+['b']*20,np.random.default_rng(1),100)
    assert abs(lo+.01)<1e-12 and abs(hi+.01)<1e-12
    result=dict(status='passed',checks=['all Python syntax','724 frozen jobs and dependencies','source/profile input integrity','4096-cell bijection and exact histogram','both floor formulas','exact packed candidate and RMS normalization','archived profile schedules','source-stratified interval and Holm arithmetic'],counts=counts,gpu_generation='not run locally')
    (ROOT/'cpu_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
