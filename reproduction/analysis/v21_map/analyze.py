"""Reproduce all v21 map diagnostics on CPU from the anonymous compact inputs."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
N_BOOT = 10000
ARMS = ("REQ","SHUF_R","SHUF2_R","SOURCE_R","REV_HIST_R","RAND_A","RAND_B")
INPUT_RECORDS = ("observed_source_seed.csv","static_controls.csv","native_step_summary.csv")


def save(path, rows):
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def auc(score,pos,neg):
    """Pixel-level weighted rank AUC, with half credit for ties on the 64x64 grid."""
    order=np.argsort(score,axis=None,kind="stable")
    val=score.ravel()[order]
    groups=np.r_[0,np.nonzero(np.diff(val))[0]+1]
    p=np.add.reduceat(pos.ravel()[order],groups).astype(float)
    n=np.add.reduceat(neg.ravel()[order],groups).astype(float)
    return np.sum(p*(np.cumsum(n)-.5*n))/(p.sum()*n.sum())


def tv(field):
    return (np.abs(np.diff(field,axis=0)).mean()+np.abs(np.diff(field,axis=1)).mean())/2


def contrast_saliency(rgb):
    """Fixed image-only local colour contrast at two scales in gamma-encoded source RGB."""
    rgb=rgb.astype(float)/(64*255)
    small=gaussian_filter(rgb,sigma=(1,1,0),mode="reflect")
    scales=[np.sqrt(np.square(small-gaussian_filter(rgb,sigma=(s,s,0),mode="reflect")).mean(axis=2))
            for s in (4,12)]
    return np.mean(scales,axis=0)


def corr(a,b):
    a=rankdata(a).astype(float); b=rankdata(b).astype(float)
    return np.corrcoef(a,b)[0,1]


def boot_summary(values,idx):
    values=np.asarray(values)
    draws=values[idx].mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(sources=len(values),mean=values.mean(),ci_low=lo,ci_high=hi,
                median=np.median(values),positive=np.count_nonzero(values>0),
                negative=np.count_nonzero(values<0),zero=np.count_nonzero(values==0))


def effects_fit(data,idx):
    """Within-source and within-map-family slopes, recomputing means in each source resample."""
    # Dimensions: source, map family, [AUC, TV, LPIPS benefit].
    data=data*np.array([10.,100.,1.])
    centered=data-data.mean(axis=1,keepdims=True)
    def fit(v):
        v=(v-v.mean(axis=0,keepdims=True)).reshape(-1,3)
        b,_,rank,_=np.linalg.lstsq(v[:,:2],v[:,2],rcond=None)
        if rank!=2:
            raise ValueError("Rank-deficient alignment/smoothness fit")
        return b
    coeff=fit(centered)
    samples=np.stack([fit(centered[ids]) for ids in idx])
    return coeff,np.quantile(samples,[.025,.975],axis=0)


def measure(inputs):
    with np.load(inputs/"inputs.npz",allow_pickle=False) as f:
        d={k:f[k] for k in f.files}
    arms=d["arms"].tolist()
    n,s=d["native_map"].shape[:2]
    valid=np.ones((512,512),dtype=np.uint8)
    valid[[0,-1],:]=0; valid[:,[0,-1]]=0
    pixels=valid.reshape(64,8,64,8).sum(axis=(1,3)).astype(int)
    obs=[]; ctrl=[]; steps=[]
    # Random draws enter condition means BEFORE source resampling.
    for i in range(n):
        ident=f"S{i+1:02}"
        pos=d["preserve"][i].astype(int); neg=pixels-pos
        sal=contrast_saliency(d["rgb_sum"][i])
        sal_auc=auc(-sal,pos,neg)
        strata=np.unique(np.stack([pos.ravel(),neg.ravel()],axis=1),axis=0,return_inverse=True)[1].ravel()
        groups=[np.flatnonzero(strata==g) for g in np.unique(strata)]
        for j,seed in enumerate(d["seeds"]):
            common=dict(source=ident,seed=int(seed))
            p=d["native_map"][i,j].astype(float)
            order=np.sort(p.ravel())
            uniform_mix=np.mean(p**2)
            st=d["native_steps"][i,j]
            q=(st[:,2]/st[:,1])**2
            h=st[:,0]
            steps.append(dict(**common,steps=len(st),mix_sq=uniform_mix,step0_q=q[0],
                              h_weighted_q=np.average(q,weights=h),
                              h_weighted_q_minus_mix=np.average(q-uniform_mix,weights=h),
                              below_mix_steps=int(np.count_nonzero(q<uniform_mix))))
            for k,arm in enumerate(arms):
                field=p if k==0 else order[d["hist_ranks"][i,j,k-1]]
                c,b,applied=d["initial"][i,j,k]
                obs.append(dict(**common,arm=arm,map_auc=auc(field,pos,neg),tv=tv(field),
                                edit_weighted_mean=np.average(field,weights=neg),
                                bg_weighted_mean=np.average(field,weights=pos),
                                release_saliency_rho=corr(-field,sal),shared_initial_raw_rms=c,
                                candidate_rms=b,applied_rms=applied,initial_q=(b/c)**2,
                                histogram_q=uniform_mix,q_minus_histogram=(b/c)**2-uniform_mix,
                                bg_lpips=d["lpips"][i,j,k],uniform_at_s_bg_lpips=d["shared_s_uniform_lpips"][i,j],
                                lpips_gain=d["shared_s_uniform_lpips"][i,j]-d["lpips"][i,j,k]))
            orientations=[np.rot90(p,k) for k in range(1,4)]+[np.rot90(np.fliplr(p),k) for k in range(4)]
            base_auc=auc(p,pos,neg); base_tv=tv(p)
            assert all(abs(tv(f)-base_tv)<1e-12 for f in orientations)
            rotated_auc=np.mean([auc(f,pos,neg) for f in orientations])
            rotated_rho=np.mean([corr(-f,sal) for f in orientations])
            perm_tv=[]
            rng=np.random.default_rng([2718,i,int(seed)])
            for draw in range(16):
                shuffled=p.ravel().copy()
                for ids in groups:
                    shuffled[ids]=rng.permutation(shuffled[ids])
                shuffled=shuffled.reshape(p.shape)
                if draw==0:
                    assert abs(auc(shuffled,pos,neg)-base_auc)<1e-12
                    assert abs(np.average(shuffled,weights=pos)-np.average(p,weights=pos))<1e-12
                perm_tv.append(tv(shuffled))
            ctrl.append(dict(**common,source_saliency_edit_auc=sal_auc,native_auc=base_auc,
                             native_minus_saliency_auc=base_auc-sal_auc,
                             orientation_auc=rotated_auc,auc_at_equal_tv=base_auc-rotated_auc,
                             native_release_saliency_rho=corr(-p,sal),orientation_rho=rotated_rho,
                             rho_at_equal_tv=corr(-p,sal)-rotated_rho,
                             native_tv=base_tv,within_mask_permuted_tv=np.mean(perm_tv),
                             delta_tv_at_equal_auc=np.mean(perm_tv)-base_tv))
    return obs,ctrl,steps


def read_records(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return [{k:v if k in ("source","arm") else int(v) if k=="seed" else float(v)
                 for k,v in r.items()} for r in csv.DictReader(f)]


def ordered(records,columns,keys):
    index={tuple(r[k] for k in columns):r for r in records}
    if len(index)!=len(records) or index.keys()!=set(keys):
        raise ValueError(f"Incomplete or duplicate records for {columns}")
    return [index[key] for key in keys]


def summarize_records(obs,ctrl,steps,output):
    sources=sorted({r["source"] for r in obs})
    seeds=(0,17)
    arms=list(ARMS)
    n,s=len(sources),len(seeds)
    if n!=40:
        raise ValueError(f"Expected 40 sources, found {n}")
    source_seeds=[(source,seed) for source in sources for seed in seeds]
    obs=ordered(obs,("source","seed","arm"),[(*k,a) for k in source_seeds for a in arms])
    ctrl=ordered(ctrl,("source","seed"),source_seeds)
    steps=ordered(steps,("source","seed"),source_seeds)
    idx=np.random.default_rng(0).integers(n,size=(N_BOOT,n))
    results=[]
    def summarize(group,metric,values):
        results.append(dict(comparison=group,metric=metric,**boot_summary(values,idx)))
    def native_by_source(records,key):
        return np.array([r[key] for r in records]).reshape(n,s).mean(axis=1)
    for key in (k for k in ctrl[0] if k not in ("source","seed")):
        summarize("static",key,native_by_source(ctrl,key))
    for key in ("step0_q","mix_sq","h_weighted_q","h_weighted_q_minus_mix","below_mix_steps"):
        summarize("native_steps",key,native_by_source(steps,key))
    def by_arm(key):
        a=np.array([r[key] for r in obs]).reshape(n,s,len(arms)).mean(axis=1)
        return np.column_stack((a[:,:-2],a[:,-2:].mean(axis=1)))
    families=arms[:-2]+["RAND_AB"]
    for key in ("map_auc","tv","release_saliency_rho","initial_q","q_minus_histogram","lpips_gain"):
        array=by_arm(key)
        for k,arm in enumerate(families):
            summarize(arm,key,array[:,k])
        for k in (1,2,3,4,5):
            summarize("REQ_minus_"+families[k],key,array[:,0]-array[:,k])
    data=np.stack([by_arm(x) for x in ("map_auc","tv","lpips_gain")],axis=2)
    threshold=[]
    for i,vals in enumerate(data):
        wins=vals[:,2]>0; losses=vals[:,2]<0
        mixed=bool(wins.any() and losses.any())
        lower_win=float(vals[wins,0].min()) if wins.any() else None
        higher_loss=float(vals[losses,0].max()) if losses.any() else None
        threshold.append(dict(source=sources[i],positive_families=int(wins.sum()),negative_families=int(losses.sum()),
                              min_auc_among_positive=lower_win,max_auc_among_negative=higher_loss,
                              mixed_signs=mixed,lower_auc_win_higher_auc_loss=mixed and lower_win<higher_loss))
    save(output/"threshold_ordering.csv",threshold)
    for key in ("mixed_signs","lower_auc_win_higher_auc_loss"):
        summarize("P4_ordering",key,np.array([r[key] for r in threshold],dtype=int))
    assoc=[]
    for name,subset in (("six_families",data),("four_nonreversed_nonrandom",data[:,:4])):
        rs=np.array([corr(v[:,0],v[:,2]) for v in subset])
        summarize(name,"within_source_spearman_auc_gain",rs)
        coeff,limits=effects_fit(subset,idx)
        for j,term in enumerate(("gain_per_0.10_AUC","gain_per_0.01_TV")):
            assoc.append(dict(model=name,term=term,sources=n,map_families=subset.shape[1],
                              slope=coeff[j],ci_low=limits[0,j],ci_high=limits[1,j]))
    save(output/"summary.csv",results)
    save(output/"conditional_associations.csv",assoc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-records",action="store_true",
                        help="Use the three anonymous source/seed CSVs as the only data inputs.")
    parser.add_argument("--input-dir",type=Path,default=HERE,
                        help="Input directory (default: beside this script). Prefer inputs.npz when present, otherwise use CSVs.")
    parser.add_argument("--output-dir",type=Path,default=HERE)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.from_records or not (args.input_dir / "inputs.npz").exists():
        records=[read_records(args.input_dir/name) for name in INPUT_RECORDS]
    else:
        records=measure(args.input_dir)
        for name,table in zip(INPUT_RECORDS,records,strict=True):
            save(args.output_dir/name,table)
    summarize_records(*records,args.output_dir)


if __name__=="__main__":
    main()
