"""Frozen A-R92 paired analysis; source is the sampling unit, not output/seed."""
from pathlib import Path
import argparse,json,math,csv
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime_base';OUT=ROOT/'analysis'
OUT.mkdir(exist_ok=True)
args=argparse.ArgumentParser();args.add_argument('--diagnostics-only',action='store_true');args=args.parse_args()
plan=json.loads((RT/'plan.json').read_text());jobs=plan['jobs'];tasks={t['case']:t for t in plan['tasks']}
if len(jobs)!=80 or len(tasks)!=40:raise ValueError('Expected the frozen 80-output / 40-source plan')
rows=[];step_rows=[]
for job in jobs:
    folder=RT/'results'/job['id']
    if not (folder/'record.json').is_file():continue
    record=json.loads((folder/'record.json').read_text())
    if record.get('status')!='completed':continue
    stored=record['job']
    if any(stored[k]!=job[k] for k in ['id','case','seed','arm','corr_lambda']):raise ValueError('Recorded job differs from plan')
    summary=json.loads((folder/'steps.json').read_text())['edit']['r92_scalar_diagnostics']
    r=dict(id=job['id'],case=job['case'],seed=job['seed'],category=int(tasks[job['case']]['target_category_id']))
    r.update({k:v for k,v in summary.items() if k not in ['steps','interpretation']})
    rows.append(r)
    step_rows.extend([dict(case=job['case'],seed=job['seed'],**x) for x in summary['steps']])
if not rows:
    (OUT/'analysis_status.json').write_text(json.dumps({'status':'generation_incomplete','completed':0,'expected':80},indent=2))
    raise SystemExit('No completed trajectories; no inference produced')
diag=pd.DataFrame(rows);diag.to_csv(OUT/'scalar_diagnostics_per_output.csv',index=False)
pd.DataFrame(step_rows).to_csv(OUT/'scalar_diagnostics_per_step.csv',index=False)
if len(rows)!=80:
    (OUT/'analysis_status.json').write_text(json.dumps({'status':'generation_incomplete','completed':len(rows),'expected':80},indent=2))
    raise SystemExit('Partial diagnostics saved; complete all 80 before inference')
if diag.duplicated(['case','seed']).any():raise ValueError('Duplicate source/seed')
cases=sorted(tasks);cat=np.array([int(tasks[c]['target_category_id']) for c in cases])
rng=np.random.default_rng(92013)
indices=np.empty((20000,len(cases)),dtype=np.int32)
for category in sorted(set(cat)):
    ix=np.flatnonzero(cat==category)
    indices[:,ix]=rng.choice(ix,size=(20000,len(ix)),replace=True)
def summarize(vector):
    v=vector.reindex(cases).to_numpy(dtype=float)
    if len(v)!=40 or not np.isfinite(v).all():raise ValueError('Missing/nonfinite paired source values')
    boot=v[indices].mean(axis=1)
    return dict(n_sources=40,mean=float(v.mean()),ci_low=float(np.quantile(boot,.025)),
                ci_high=float(np.quantile(boot,.975)),negative_sources=int((v<0).sum()),positive_sources=int((v>0).sum()))
diagnostic_metrics=['native_I_R','native_I_U','actual_I_SC','lambda_SC','cumulative_relative_error',
                    'integrated_absolute_relative_error','max_step_relative_error','normalized_temporal_shape_L1']
ds=diag.groupby('case')[diagnostic_metrics].mean().reindex(cases)
ds.to_csv(OUT/'scalar_diagnostics_per_source.csv')
pd.DataFrame([dict(metric=m,**summarize(ds[m])) for m in diagnostic_metrics]).to_csv(OUT/'scalar_diagnostics_summary.csv',index=False)
if args.diagnostics_only:
    (OUT/'analysis_status.json').write_text(json.dumps({'status':'diagnostics_complete_outcomes_pending','completed':80},indent=2))
    print('All 80 trajectory diagnostics saved; image outcome analysis pending.');raise SystemExit(0)
new=pd.read_csv(RT/'results/evaluation.csv',dtype={'case':str})
if new.id.duplicated().any() or set(new.id)!=set(j['id'] for j in jobs):raise ValueError('Metric rows must cover exactly the 80 jobs')
if not new.eval_status.eq('ok').all():raise ValueError('Unsuccessful metric rows')
new['seed']=new.seed.astype(int)
old=pd.read_csv(RT/'historical/scores.csv',dtype={'case':str})
old=old[old.case.isin(cases)&old.seed.isin([0,17])&old.arm.isin(['REQ','REQ_U','U25'])].copy()
edit=pd.read_csv(ROOT/'historical_edit_mae.csv',dtype={'case':str})
old=old.merge(edit[['case','seed','arm','edit_mae']],on=['case','seed','arm'],how='left',validate='one_to_one')
if len(old)!=240:raise ValueError('Missing original comparator rows')
metrics=['lpips_preserved','clip_directional_similarity','edit_mae']
combined=pd.concat([old[['id','case','seed','arm']+metrics],new[['id','case','seed','arm']+metrics]],ignore_index=True)
combined.to_csv(OUT/'combined_outcomes.csv',index=False)
means=combined.groupby(['case','arm'])[metrics].mean().groupby('arm').mean()
means.to_csv(OUT/'operating_points.csv')
contrasts=[];vectors={};source_rows=[]
for metric in metrics:
    paired=combined.pivot(index=['case','seed'],columns='arm',values=metric)
    for a,b in [('REQ_SC','REQ_U'),('REQ_SC','U25'),('REQ_SC','REQ'),('REQ_U','U25')]:
        if metric=='edit_mae' and b=='REQ':continue
        values=paired[a]-paired[b]
        if len(values)!=80 or not np.isfinite(values).all():raise ValueError('Missing comparator value')
        v=values.groupby(level='case').mean().reindex(cases)
        key=a+'-'+b
        primary=metric=='lpips_preserved' and a=='REQ_SC' and b in ['REQ_U','U25']
        contrasts.append(dict(contrast=key,metric=metric,procedure='Holm2' if primary else 'descriptive',**summarize(v)))
        vectors[(key,metric)]=v.to_numpy()
        source_rows.extend(dict(case=c,category=int(cat[i]),contrast=key,metric=metric,difference=float(v.loc[c])) for i,c in enumerate(cases))
table=pd.DataFrame(contrasts);table['p_raw']=np.nan;table['p_holm2']=np.nan
which=table.index[table.procedure.eq('Holm2')].tolist()
D=np.column_stack([vectors[(table.loc[i,'contrast'],'lpips_preserved')] for i in which])
observed=np.abs(D.mean(axis=0));extreme=np.zeros(2,dtype=np.int64)
for _ in range(100):
    signs=rng.integers(0,2,size=(10000,40),dtype=np.int8)*2-1
    extreme+=(np.abs(signs@D/40)>=observed-1e-15).sum(axis=0)
p=(extreme+1)/1000001;order=np.argsort(p,kind='stable');adjust=np.empty(2)
adjust[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.array([2,1])))
table.loc[which,'p_raw']=p;table.loc[which,'p_holm2']=adjust
table.to_csv(OUT/'paired_contrasts.csv',index=False)
pd.DataFrame(source_rows).to_csv(OUT/'paired_source_differences.csv',index=False)
pd.DataFrame([dict(case=c,seed=int(seed),difference=float(value))
 for (c,seed),value in (combined.pivot(index=['case','seed'],columns='arm',values='lpips_preserved')['REQ_SC']-
                       combined.pivot(index=['case','seed'],columns='arm',values='lpips_preserved')['REQ_U']).items()
]).to_csv(OUT/'scalar_vs_stepwise_each_seed.csv',index=False)
# Diagnostics and outcomes are associated descriptions, not a causal partition.
joint=ds.copy();joint['SC_minus_REQ_U_lpips']=vectors[('REQ_SC-REQ_U','lpips_preserved')]
joint['SC_minus_U25_lpips']=vectors[('REQ_SC-U25','lpips_preserved')]
joint.to_csv(OUT/'diagnostics_and_outcomes_per_source.csv')
summary=dict(status='complete',sources=40,seeds=[0,17],new_outputs=80,
    primary_family='Holm2: SC-REQ_U and SC-U25 preservation LPIPS',
    actual_cumulative_error_mean=float(ds.cumulative_relative_error.mean()),
    temporal_shape_L1_mean=float(ds.normalized_temporal_shape_L1.mean()),
    no_equivalence_claim='Intervals containing zero do not establish equivalence; no margin prespecified',
    stopping='Fixed 80 complete; no automatic follow-up generation or coefficient search',
    missing_old_outcome='Native REQ edit MAE unavailable; omitted without imputation')
(OUT/'analysis_status.json').write_text(json.dumps(summary,indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,2.6))
    q=table[table.metric.eq('lpips_preserved')].reset_index(drop=True)
    for i,row in q.iterrows():
        ax.errorbar(row['mean'],i,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],fmt='o',color='#3B6EA5',capsize=3)
    ax.axvline(0,color='grey',lw=.7);ax.set(yticks=range(len(q)),yticklabels=q.contrast,xlabel='Preservation LPIPS difference')
    bx.scatter(ds.cumulative_relative_error*100,joint.SC_minus_REQ_U_lpips,s=14,color='#3B6EA5')
    bx.axhline(0,color='grey',lw=.7);bx.axvline(0,color='grey',lw=.7)
    bx.set(xlabel='Actual cumulative magnitude error (%)',ylabel='SC minus stepwise LPIPS')
    fig.tight_layout();fig.savefig(OUT/'scalar_comparison.pdf');fig.savefig(OUT/'scalar_comparison.png',dpi=180);plt.close(fig)
except ImportError:
    print('Matplotlib unavailable; all numeric analysis saved, plot omitted.')
print(table.to_string(index=False));print('Fixed experiment complete: interpret outcomes and magnitude deviations jointly.')
