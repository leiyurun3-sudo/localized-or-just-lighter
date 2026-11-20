"""Prespecified source-paired summaries. New seeds and old seeds stay identifiable."""
from pathlib import Path
import json,warnings,itertools
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime_base';OUT=ROOT/'analysis'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def interval(values,categories,rng,reps=20000):
    x=np.asarray(values,float);cat=np.asarray(categories);boot=np.zeros(reps)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Nonfinite or empty paired contrast')
    for c in sorted(set(cat)):
        a=x[cat==c];boot+=a[rng.integers(0,len(a),(reps,len(a)))].sum(1)
    return np.quantile(boot/len(x),[.025,.975])

def signflip(x,rng,n=1000000):
    x=np.asarray(x,float);nex=0;obs=abs(x.mean())
    for start in range(0,n,10000):
        z=rng.integers(0,2,(min(10000,n-start),len(x)),dtype=np.int8)*2-1
        nex+=np.count_nonzero(np.abs(z@x/len(x))>=obs-1e-15)
    return (nex+1)/(n+1)

def holm(ps):
    ps=np.asarray(ps);order=np.argsort(ps);out=np.empty(len(ps));out[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))));return out

def main():
    OUT.mkdir(exist_ok=True);plan=read(RT/'plan.json');protocol=read(ROOT/'protocol.json')
    new=pd.read_csv(RT/'results/evaluation.csv',dtype={'case':str});old=pd.read_csv(RT/'historical/scores.csv',dtype={'case':str})
    scientific=[j for j in plan['jobs'] if j['r91_phase']!='sentinel'];want={j['id'] for j in scientific}
    new=new[new.id.isin(want)].copy()
    if set(new.id)!=want or len(new)!=720:raise RuntimeError('Analysis needs all 720 scientific outputs with unique evaluator rows')
    meta={j['id']:j for j in scientific}
    for k in ['case','seed','arm']:new[k]=new.id.map(lambda jid:meta[jid][k])
    needed=['lpips_preserved','clip_directional_similarity','clip_target_similarity','edit_mae']
    for k in needed:new[k]=pd.to_numeric(new[k],errors='coerce')
    if not np.isfinite(new[needed].to_numpy()).all():raise RuntimeError('Missing/undefined metrics: retain statuses and resolve before full-cohort inference')
    new['origin']='R91';old['origin']='historical'
    categories={t['case']:t['target_category_id'] for t in plan['tasks']}
    # Recover historical edit MAE from the preserved native REQ-U/U25 records only.
    hmae=ROOT/'historical_edit_mae.csv'
    if hmae.exists():old=old.merge(pd.read_csv(hmae,dtype={'case':str}),on=['case','seed','arm'],how='left',validate='one_to_one')
    d=pd.concat([old,new],ignore_index=True);d.to_csv(OUT/'combined_scores.csv',index=False)
    rng=np.random.default_rng(91013);rows=[];perseed=[];mixed=[];signs=[];contrasts={}
    def summarize(name,series,family,metric='lpips_preserved',test=True):
        per_source=series.groupby(level='case').mean()
        cat=[categories[c] for c in per_source.index];ci=interval(per_source.values,cat,rng)
        row=dict(contrast=name,metric=metric,family=family,n_sources=len(per_source),mean=float(per_source.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),p_raw=signflip(per_source.values,rng) if test else np.nan)
        rows.append(row)
        return row
    for a,b in [('REQ_U','U25'),('REQ','GLB_R')]:
        p=d[d.arm.isin([a,b])].pivot(index=['case','seed'],columns='arm',values='lpips_preserved');delta=(p[a]-p[b]).dropna();wide=delta.unstack('seed')
        assert wide.shape==(40,5) and not wide.isna().any().any()
        contrasts[a+'-'+b]=delta
        for label,seeds,family,test in [('original_two',[0,17],'historical_descriptive',False),('new_three',[1,2,3],'new_seed_lpips',True),('pooled_five',[0,17,1,2,3],'updated_sensitivity',False)]:
            summarize(a+'-'+b+':'+label,delta[delta.index.get_level_values('seed').isin(seeds)],family,test=test)
        for s in [0,17,1,2,3]:
            x=wide[s];ci=interval(x,[categories[c] for c in x.index],rng)
            perseed.append(dict(contrast=a+'-'+b,seed=s,mean=float(x.mean()),ci_low=ci[0],ci_high=ci[1],negative_sources=int((x<0).sum())))
        for s1,s2 in itertools.combinations([0,17,1,2,3],2):
            signs.append(dict(contrast=a+'-'+b,seed1=s1,seed2=s2,opposite_sign_sources=int(((wide[s1]*wide[s2])<0).sum()),n=40))
        wide['mean']=wide.mean(1);wide['negative_seeds']=(wide[[0,17,1,2,3]]<0).sum(1);wide.to_csv(OUT/(a+'-'+b+'_source_seed.csv'))
        # Paired differences remove source baseline difficulty; crossed random
        # intercepts here represent variability of the allocation effect.
        # https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html
        import statsmodels.formula.api as smf
        md=delta.rename('difference').reset_index();md['one']=1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model=smf.mixedlm('difference ~ 1',md,groups='one',re_formula='0',vc_formula={'source':'0+C(case)','seed':'0+C(seed)'})
                fit=model.fit(reml=True,method='lbfgs')
                m=dict(contrast=a+'-'+b,converged=bool(fit.converged),fixed_effect=float(fit.fe_params.iloc[0]),wald_ci=[float(x) for x in fit.conf_int().loc['Intercept']],residual_variance=float(fit.scale),variance_components=dict(zip(model.exog_vc.names,map(float,fit.vcomp))),interpretation='secondary crossed-effects model; only five seed levels; boundary/nonconvergence warnings retained')
            except Exception as exc:m=dict(contrast=a+'-'+b,status='fit_failed',error=str(exc))
            m['warnings']=[str(w.message) for w in caught];mixed.append(m)
    for metric in ['lpips_preserved','clip_directional_similarity']:
        p=d[d.seed.isin([0,17])].pivot(index=['case','seed'],columns='arm',values=metric)
        rand=(p.RAND_A+p.RAND_B)/2
        for arm in ['REQ','GLB_R','SOURCE_R']:
            if arm not in p:raise ValueError('Missing historical mechanism arm '+arm)
            summarize(arm+'-RAND',p[arm]-rand,'permutation_lpips' if metric=='lpips_preserved' else 'permutation_clip',metric)
        ((p.RAND_A-p.RAND_B).abs()/np.sqrt(2)).rename('two_draw_sample_sd').to_csv(OUT/f'permutation_sd_{metric}.csv')
        controls=pd.DataFrame({arm:p[arm]-p.GLB_R for arm in ['REQ','SOURCE_R']});controls['RAND']=rand-p.GLB_R
        controls.to_csv(OUT/f'mechanism_vs_GLB_R_{metric}.csv')
        if metric=='lpips_preserved':
            means=controls.groupby(level='case').mean().mean()
            recovery={arm:float(means[arm]/means.REQ) for arm in ['SOURCE_R','RAND']}
            (OUT/'recovery_ratios.json').write_text(json.dumps(dict(ratios=recovery,denominator_REQ_minus_GLB_R=float(means.REQ),interpretation='Ratios of cohort-mean gains; descriptive, not a mechanistic partition.'),indent=2),encoding='utf-8')
        for arm in ['FLOOR_000','FLOOR_050']:
            sub=(p[arm]-p.REQ_U).dropna();assert len(sub)==40
            summarize(arm+'-REQ_U',sub,'floor_lpips_clip',metric)
        subset=protocol['floor_cases']
        for arm in ['FLOOR_000','REQ_U','FLOOR_050']:
            sub=(p[arm]-p.U25).loc[p.index.get_level_values('case').isin(subset)].dropna()
            summarize(arm+'-U25:floor_subset',sub,'floor_vs_uniform_descriptive',metric,False)
    # Edit MAE is descriptive. A large pixel change need not fulfill the request.
    p=d[d.seed.isin([0,17])].pivot(index=['case','seed'],columns='arm',values='edit_mae')
    for arm in ['FLOOR_000','FLOOR_050']:summarize(arm+'-REQ_U',(p[arm]-p.REQ_U).dropna(),'floor_mae_descriptive','edit_mae',False)
    table=pd.DataFrame(rows);table['p_holm']=np.nan
    for family,group in table[table.p_raw.notna()].groupby('family'):table.loc[group.index,'p_holm']=holm(group.p_raw)
    table.to_csv(OUT/'planned_contrasts.csv',index=False)
    pd.DataFrame(perseed).to_csv(OUT/'per_seed.csv',index=False);pd.DataFrame(signs).to_csv(OUT/'pairwise_seed_signs.csv',index=False)
    (OUT/'mixed_models.json').write_text(json.dumps(mixed,indent=2,allow_nan=False),encoding='utf-8')
    # Show tested floor points only; never label interpolated segments Pareto optimal.
    floors=protocol['floor_cases'];ops=[]
    for arm,f in [('FLOOR_000',0),('REQ_U',.25),('FLOOR_050',.5),('U25',None)]:
        q=d[d.case.isin(floors)&d.seed.isin([0,17])&(d.arm==arm)]
        assert len(q)==40
        ops.append(dict(arm=arm,floor=f,**{m:float(q[m].mean()) for m in ['lpips_preserved','clip_directional_similarity','edit_mae']}))
    pd.DataFrame(ops).to_csv(OUT/'floor_operating_points.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(3.4,2.6));pts=pd.DataFrame(ops)
    ax.plot(pts.lpips_preserved[:3],pts.clip_directional_similarity[:3],'o-',color='#3B6EA5',lw=.8,ms=4)
    for i in range(3):ax.annotate('floor '+str(ops[i]['floor']),(pts.lpips_preserved[i],pts.clip_directional_similarity[i]),xytext=(4,5),textcoords='offset points',fontsize=7)
    ax.plot(pts.lpips_preserved[3],pts.clip_directional_similarity[3],'s',color='#B96335',label='U25');ax.legend(frameon=False,fontsize=7)
    ax.set(xlabel='Preservation LPIPS',ylabel='Directional CLIP');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'floor_points.pdf');plt.close(fig)
    (OUT/'interpretation.txt').write_text('New three-seed replication and pooled five-seed sensitivity are separate. Source is the inference unit. All pairwise seed sign disagreements remain reported; the probability of any sign change over five seeds is not comparable with the original two-seed fraction. Mixed-model seed variance is secondary. Nonsignificance does not establish equivalence. Random maps retain target-derived histograms and anchor profiles. Floor segments connect tested points only. Human perceptual meaning remains unmeasured.\n',encoding='utf-8')
    print(table.to_string(index=False))
if __name__=='__main__':main()
