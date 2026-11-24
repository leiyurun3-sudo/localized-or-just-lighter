"""Paired source bootstrap; report default outcomes and bounded response matching."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v21_bounded"))
from bounded_mrc import integrate

SCALES = (0.25, 0.5, 0.75, 1.0)
STUDY = Path(__file__).resolve().parent

def estimate(values, denominator=None):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n":0, "mean":None,"ci95":None}
    draws = np.random.default_rng(0).integers(len(x), size=(10000,len(x)))
    sampled = x[draws].mean(axis=1)
    point = float(x.mean())
    if denominator is not None:
        y = np.asarray(denominator,dtype=float)
        div = y[draws].mean(axis=1)
        if abs(y.mean()) < 1e-12:
            return {"n":len(x),"mean":None,"ci95":None}
        point /= float(y.mean())
        if div.min() <= 0 <= div.max():
            return {"n":len(x),"mean":point,"ci95":None}
        sampled /= div
    return {"n":len(x),"mean":point,"ci95":np.quantile(sampled,[.025,.975]).tolist()}

def summarize(path, save=None):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows=list(csv.DictReader(f))
    jobs={j["job_id"]:j for j in json.loads((STUDY/"jobs.json").read_text())["jobs"]}
    ids=[r["id"] for r in rows]
    full=len(rows)==len(jobs)
    expected=set(jobs) if full else {k for k,v in jobs.items() if v["stage"]=="core"}
    seeds={int(r["seed"]) for r in rows}
    if len(set(ids))!=len(ids) or set(ids)!=expected or len(seeds)!=1:
        raise ValueError("Expected one complete paired light or full seed grid")
    images=defaultdict(dict)
    for r in rows:
        j=jobs[r["id"]]; scale=float(r["scale"]); arm=r["arm"]
        if (r["case"],arm,scale)!=(j["case"],j["arm"],j.get("scale",1)):
            raise ValueError("Identity mismatch: "+r["id"])
        metrics={k:float(r[k]) for k in ["bg_fm","clip_directional","mae_edit","mae_preservation"]}
        if not np.isfinite(list(metrics.values())).all() or metrics["mae_preservation"]<=0:
            raise ValueError("Undefined source outcome at "+r["id"])
        images[r["case"]][arm,scale]=metrics
    def series(arm,key):
        return np.array([im[arm,1.0][key] for im in images.values()])
    def gap(a,b,key="bg_fm"):
        return series(a,key)-series(b,key)
    arms=["uniform","sam_native","two_constant","uniform_replay"]
    result={"seed":seeds.pop(),"sources":len(images),
            "constant_recovery_fraction":estimate(gap("uniform","two_constant"),gap("uniform","sam_native")),
            "cpr_spatial_minus_uniform_at_spatial":estimate(gap("sam_native","uniform_replay")),
            "default_clip_constants_minus_spatial":estimate(gap("two_constant","sam_native","clip_directional")),
            "defaults":{arm:{"bg_fm":estimate(series(arm,"bg_fm")),"directional_clip":estimate(series(arm,"clip_directional")),
                             "selectivity":estimate(series(arm,"mae_edit")/series(arm,"mae_preservation"))} for arm in arms}}
    if full:
        source_rows=[]
        for case,im in sorted(images.items()):
            curves=[{s:(im[arm,s]["clip_directional"],{"fm":im[arm,s]["bg_fm"]}) for s in SCALES} for arm in ("sam_native","uniform")]
            shared=integrate(curves,"fm",SCALES,True)
            positive=integrate(curves,"fm",SCALES,True,bounds=(0.,float("inf")))
            source_rows.append(dict(case=case,**shared,positive_only=positive["matched"]))
        valid=[r for r in source_rows if r["status"]=="valid"]
        pos=[r for r in valid if r["hi"]>0]
        def summarize_support(s):
            return {"gap":estimate([r["matched"] for r in s]),
                    "default_gap":estimate([r["default"] for r in s]),
                    "fraction_of_default":estimate([r["matched"] for r in s],[r["default"] for r in s])}
        result["mrc"]={"eligible":len(images),"supported":len(valid),"entire_interval_nonpositive":len(valid)-len(pos),
                       "all_shared":summarize_support(valid),"exclude_nonpositive":summarize_support(pos),
                       "positive_only_gap":estimate([r["positive_only"] for r in pos])}
        if save:
            with (save/"mrc_sources.csv").open("w",newline="",encoding="utf8") as out:
                w=csv.DictWriter(out,fieldnames=list(source_rows[0]));w.writeheader();w.writerows(source_rows)
    if save:
        (save/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    return result

if __name__=="__main__":
    print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv)>1 else STUDY/"seed0.csv"),indent=2))

