"""Contract and measured deviations for the fixed one-shot comparator."""
import math

def validate_scalar_job(job):
    from r86_compat import validate_r88_job
    if job.get('arm') != 'REQ_SC':raise ValueError('Only REQ_SC is new in A-R92')
    expected=.25*float(job['r92_I_U'])/float(job['r92_I_R'])
    if not math.isfinite(expected) or expected<=0 or abs(float(job['corr_lambda'])-expected)>1e-14:
        raise ValueError('Scalar coefficient differs from the frozen per-pair rule')
    native=dict(job,arm='REQ',corr_lambda=.25)
    validate_r88_job(native)

def validate_plan(root):
    import reporting as rep
    plan,tasks=rep.load_plan(root)
    archive=rep.read(root/'historical/anchors.json')
    if len(plan['jobs'])!=80 or len(tasks)!=40:raise ValueError('Expected exactly 40 sources / 80 runs')
    pairs=set()
    for job in plan['jobs']:
        validate_scalar_job(job)
        pair=(job['case'],job['seed'])
        if pair in pairs or job['seed'] not in [0,17]:raise ValueError('Duplicate or unexpected source/seed')
        pairs.add(pair)
        for key,name in [('r92_native_req_id','REQ'),('r92_native_uniform_id','U25')]:
            a=archive[job[key]]
            if a['job']['arm']!=name or len(a['profile'])!=28:raise ValueError('Wrong native anchor')
            for field in ['case','seed','source_prompt','target_prompt','steps','resolution','gamma','guidance','eta','src_steps','attention_blocks','query_selection']:
                if a['job'].get(field)!=job.get(field):raise ValueError('Anchor mismatch: '+field)
            integral=sum(float(x['h'])*float(x['applied_correction_rms']) for x in a['profile'])
            expected=job['r92_I_R' if name=='REQ' else 'r92_I_U']
            if abs(integral-expected)>1e-13:raise ValueError('Native integral mismatch')
        # Every inherited algorithm field must remain unchanged apart from the scalar.
        native=archive[job['r92_native_req_id']]['job']
        for field,value in native.items():
            if field not in ['id','label','arm','corr_lambda'] and job.get(field)!=value:
                raise ValueError('Changed native REQ setting: '+field)
    if any(sum(c==case for c,s in pairs)!=2 for case in {c for c,s in pairs}):raise ValueError('Incomplete pairing')
    return plan

def inspect_generated(root,job,stats,archive):
    validate_scalar_job(job)
    actual=stats['steps'];ref=archive[job['r92_native_uniform_id']]['profile']
    if len(actual)!=28:raise ValueError('Incomplete scalar trajectory')
    measured=0.;weighted_abs=0.;max_relative=0.;step_differences=[]
    for x,y in zip(actual,ref):
        if x['step_index']!=y['step_index'] or abs(x['h']-y['h'])>1e-8 or abs(x['sigma']-y['sigma'])>1e-8:
            raise ValueError('Scalar/native schedule mismatch')
        if abs(x['corr_lambda']-job['corr_lambda'])>1e-14 or x.get('r86_target_rms') is not None:
            raise ValueError('Expected constant scalar without stepwise replay')
        b=float(x['applied_correction_rms']);target=float(y['applied_correction_rms']);h=float(y['h'])
        if not math.isfinite(b) or b<0 or target<=0:raise ValueError('Nonfinite RMS')
        measured+=h*b;weighted_abs+=h*abs(b-target)
        max_relative=max(max_relative,abs(b-target)/target)
        step_differences.append(dict(step_index=x['step_index'],sigma=y['sigma'],h=h,
            actual_rms=b,target_U25_rms=target,signed_difference=b-target,relative_difference=(b-target)/target))
    iu=float(job['r92_I_U'])
    shape=sum(abs(x['h']*x['applied_correction_rms']/measured-y['h']*y['applied_correction_rms']/iu)
              for x,y in zip(actual,ref)) if measured>0 else None
    return dict(lambda_SC=job['corr_lambda'],native_I_R=job['r92_I_R'],native_I_U=iu,
                actual_I_SC=measured,cumulative_relative_error=(measured-iu)/iu,
                integrated_absolute_relative_error=weighted_abs/iu,max_step_relative_error=max_relative,
                normalized_temporal_shape_L1=shape,steps=step_differences,
                interpretation='Measured deviations, not constraints: one fixed coefficient does not guarantee matched actual RMS.')
