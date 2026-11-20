"""CPU prediction from source_inputs.csv. No original images or archives are needed."""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ("intercept", "area", "area_attenuation")
METRICS = ("rmse", "mae")
DRAWS, SEED = 5000, 20261119


def fit_predict(x, y, weights, x_test):
    active = weights > 0
    root = np.sqrt(weights[active])
    predictions = []
    for k in (1,2,3):
        beta, _, rank, _ = np.linalg.lstsq(x[active,:k]*root[:,None],y[active]*root,rcond=None)
        if rank < k:
            raise ValueError("bootstrap training design lacks full rank")
        predictions.append(x_test[:,:k] @ beta)
    return np.stack(predictions,axis=1)


def heldout(x, y, group, weights):
    pred = np.zeros((len(y),3))
    for g in np.unique(group[weights>0]):
        test = group == g
        pred[test] = fit_predict(x,y,weights*(~test),x[test])
    return pred


def errors(y, pred, weights):
    diff = pred - y[:,None]
    rmse = np.sqrt(np.average(diff**2, axis=0, weights=weights))
    mae = np.average(np.abs(diff),axis=0,weights=weights)
    measures = np.column_stack((rmse,mae))
    # Positive is a reduction in error when attenuation is added to mask area.
    return np.vstack((measures,measures[1]-measures[2]))


def report(point, draws):
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return {name:{metric:dict(value=point[i,j],lo=lo[i,j],hi=hi[i,j]) for j,metric in enumerate(METRICS)}
            for i,name in enumerate((*MODELS,"area_minus_full"))}


def extract(rows, original=False):
    x=np.array([[1,float(r["background_fraction"]),
                 1-float(r["spatial_magnitude"])/float(r["uniform_magnitude"])] for r in rows])
    suffix = "_masked_input_lpips" if original else "_lpips"
    y=np.array([float(r["uniform"+suffix])-float(r["spatial"+suffix]) for r in rows])
    return x,y


def records(output, spec, rows, x, y, pred):
    for r,features,target,values in zip(rows,x,y,pred,strict=True):
        output.append(dict(design=spec,operator=r["operator"],source=r["source"],category=r["category"],
                           background_fraction=features[1],attenuation=features[2],observed_U_minus_S=target,
                           **dict(zip(MODELS,values,strict=True))))


def stratified_weights(rng, rows):
    cats=np.array([r["category"] for r in rows])
    counts=np.zeros(len(rows),dtype=int)
    for c in np.unique(cats):
        ids=np.flatnonzero(cats==c)
        counts[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
    return counts


def transfer(data, predictions):
    direct={r["source"]:r for r in data if r["operator"]=="directedit"}
    train=[r for r in data if r["operator"]=="correction" and r["source"] not in direct]
    control=[r for r in data if r["operator"]=="correction" and r["source"] in direct]
    test=[direct[r["source"]] for r in control]
    assert len(train)==len(test)==len(control)==20
    x,y=extract(train)
    targets=[extract(control),extract(test)]
    unit=np.ones(20)
    points=[]
    summary={}
    for name, rows, (xt,yt) in zip(("heldout_correction","heldout_directedit"),(control,test),targets,strict=True):
        pred=fit_predict(x,y,unit,xt)
        points.append(errors(yt,pred,unit))
        records(predictions,name,rows,xt,yt,pred)
        summary[name]=dict(n=20,mean_target=float(yt.mean()),mean_full_prediction=float(pred[:,2].mean()),
                           attenuation_range=[min(xt[:,2]),max(xt[:,2])],
                           outside_training_attenuation=int(((xt[:,2]<x[:,2].min()) | (xt[:,2]>x[:,2].max())).sum()))
    rng=np.random.default_rng(SEED)
    distribution=[]
    for _ in range(DRAWS):
        a=stratified_weights(rng,train)
        b=stratified_weights(rng,test)  # same held-out image weights for both operators
        distribution.append([errors(yt,fit_predict(x,y,a,xt),b) for xt,yt in targets])
    samples=np.asarray(distribution)
    for i,name in enumerate(("heldout_correction","heldout_directedit")):
        summary[name]["errors"]=report(points[i],samples[:,i])
    summary["training"]=dict(n=len(train),mean_target=y.mean(),attenuation_range=[min(x[:,2]),max(x[:,2])],
                             full_coefficients=np.linalg.lstsq(x,y,rcond=None)[0].tolist())
    return summary


def sam_analysis(rows,predictions):
    categories=np.array([int(r["category"]) for r in rows])
    ids,group=np.unique(categories,return_inverse=True)
    unit=np.ones(len(rows))
    targets=[extract(rows,original) for original in (False,True)]
    points,summary={},{}
    for name,(x,y) in zip(("feature_map","masked_input"),targets,strict=True):
        summary[name]={}
        for split,g in (("source",np.arange(len(y))),("category",group)):
            pred=heldout(x,y,g,unit)
            records(predictions,f"sam_{name}_{split}",rows,x,y,pred)
            point=errors(y,pred,unit)
            if split=="source":
                summary[name]["leave_one_source"]={k:dict(zip(METRICS,a,strict=True)) for k,a in zip((*MODELS,"area_minus_full"),point,strict=True)}
            else:
                points[name]=point
                summary[name]["category_folds"]={str(c):dict(n=int((categories==c).sum()),
                        area_minus_full_mae=errors(y[categories==c],pred[categories==c],unit[categories==c])[3,1])
                        for c in ids}
    # Resample whole edit categories; all copies of a held-out category are kept out of training.
    rng=np.random.default_rng(SEED+1)
    distributions=[[],[]]
    retries=0
    while len(distributions[0])<DRAWS:
        counts=rng.multinomial(len(ids), np.full(len(ids),1/len(ids)))
        weights=counts[group]
        try:
            out=[errors(y,heldout(x,y,group,weights),weights) for x,y in targets]
        except ValueError:
            retries+=1
            continue
        for d,a in zip(distributions,out,strict=True):d.append(a)
    for name,dist in zip(("feature_map","masked_input"),distributions,strict=True):
        summary[name]["leave_one_category"]=report(points[name],dist)
    summary["bootstrap_rank_deficient_redraws"]=retries
    return summary


if __name__=="__main__":
    with (HERE/"source_inputs.csv").open(encoding="utf-8",newline="") as f:
        data=list(csv.DictReader(f))
    predictions=[]
    results=dict(bootstrap_draws=DRAWS,seed=SEED,transfer=transfer(data,predictions),
                 sam=sam_analysis([r for r in data if r["operator"]=="sam"],predictions))
    (HERE/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (HERE/"predictions.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=predictions[0]);w.writeheader();w.writerows(predictions)
