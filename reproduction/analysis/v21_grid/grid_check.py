"""Post hoc grid validation on observed source/setting scores. Requires Python and NumPy."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SETTINGS = {"S": (.25, .5, .75, 1., 1.25), "U": (.25, .5, .75, 1., 1.25),
            "F": (12., 16., 20., 24., 28.)}
FULL = (0, 1, 2, 3, 4)
THREE = (0, 2, 4)
EXAMPLES = {"114000000004": "mountain reflection", "511000000000": "dog posture",
            "513000000001": "jar lid", "824000000007": "rice-field background"}


def write(folder, name, rows):
    with (folder / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None or isinstance(v, float) and not math.isfinite(v) else v
                             for k, v in row.items()})


def read(path):
    grouped, eligibility = defaultdict(dict), {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            c, a, k = r["source"], r["arm"], float(r["setting"])
            eligible = r["background_eligible"] == "True"
            if eligibility.setdefault(c, eligible) != eligible or k in grouped[c, a]:
                raise ValueError(f"Inconsistent or duplicate observation: {c} {a} {k}")
            x, y = float(r["directional_clip"]), float(r["fm_lpips"] or "nan")
            if not np.isfinite(x) or np.isfinite(y) != eligible:
                raise ValueError(f"Invalid observation: {c} {a} {k}")
            grouped[c, a][k] = (k, x, y)
    curves = {}
    for c in sorted(eligibility):
        for a, settings in SETTINGS.items():
            if tuple(sorted(grouped[c, a])) != settings:
                raise ValueError(f"Incomplete grid: {c} {a}")
            curves[c, a] = [grouped[c, a][s] for s in settings]
    return curves, eligibility


def crossings(curve, query):
    """Ordered segment predictions at an observed response, retaining original knob endpoints."""
    hits = []
    for (ka, xa, ya), (kb, xb, yb) in zip(curve, curve[1:]):
        if min(xa, xb) <= query <= max(xa, xb):
            first = ya if xa == xb else ya + (query-xa)/(xb-xa)*(yb-ya)
            last = yb if xa == xb else first
            hits.append((ka, kb, first, last))
    return hits


def leave_one(curves, eligibility):
    rows = []
    for (case, arm), full in curves.items():
        for i in (1, 2, 3):
            k, x, y = full[i]
            ka, xa, ya = full[i-1]
            kb, xb, yb = full[i+1]
            t = (k-ka)/(kb-ka)
            xhat, yhat = xa+t*(xb-xa), ya+t*(yb-ya)
            reduced = full[:i] + full[i+1:]
            support = [p[1] for p in reduced]
            hits = crossings(reduced, x)
            local = min(xa, xb) <= x <= max(xa, xb)
            local_pred = crossings([full[i-1], full[i+1]], x)
            first = hits[0][2] if hits else np.nan
            last = hits[-1][3] if hits else np.nan
            rows.append(dict(source=case, arm=arm, eligible=eligibility[case], held_rank=i+1, held_knob=k,
                             neighbor_lo_knob=ka, neighbor_hi_knob=kb, observed_clip=x, observed_lpips=y,
                             knob_pred_clip=xhat, knob_pred_lpips=yhat, knob_clip_error=xhat-x,
                             knob_lpips_error=yhat-y, response_lo=min(support), response_hi=max(support),
                             inside_neighbor_response=local, inside_retained_response=bool(hits),
                             local_reversal=(x-xa)*(xb-x)<0, retained_segments_at_response=len(hits),
                             first_segment_start_knob=hits[0][0] if hits else None,
                             first_segment_end_knob=hits[0][1] if hits else None,
                             last_segment_start_knob=hits[-1][0] if hits else None,
                             last_segment_end_knob=hits[-1][1] if hits else None,
                             first_at_observed_clip=first, last_at_observed_clip=last,
                             first_lpips_error=first-y, last_lpips_error=last-y,
                             neighbor_at_observed_clip=local_pred[0][2] if local_pred else None))
    return rows


def mean_ci(values):
    """Equal-source bootstrap; callers pass one finite scalar per source in sorted ID order."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return (None, None, None)
    if len(v) == 1:
        return (float(v[0]), None, None)
    ix = np.random.default_rng(0).integers(len(v), size=(10000, len(v)))
    low, high = np.quantile(v[ix].mean(axis=1), [.025, .975])
    return float(v.mean()), low, high


def summarize_errors(rows):
    per_source, summary = [], []
    for arm in SETTINGS:
        for cohort in ("background_eligible", "all"):
            sub = [r for r in rows if r["arm"]==arm and (cohort=="all" or r["eligible"])]
            metrics = ["knob_clip_error"] if cohort=="all" else [
                "knob_clip_error", "knob_lpips_error", "first_lpips_error", "last_lpips_error"]
            for field in metrics:
                by_case = defaultdict(list)
                for r in sub:
                    if np.isfinite(r[field]):
                        by_case[r["source"]].append(r[field])
                local = []
                for c, errors in sorted(by_case.items()):
                    local.append(dict(arm=arm, cohort=cohort, quantity=field, source=c, n=len(errors),
                                      mae=float(np.mean(np.abs(errors))), bias=float(np.mean(errors))))
                per_source.extend(local)
                mae, lo, hi = mean_ci([r["mae"] for r in local])
                bias, blo, bhi = mean_ci([r["bias"] for r in local])
                summary.append(dict(arm=arm, cohort=cohort, quantity=field, sources=len(local),
                                    n_possible=len(sub), n_scored=sum(r["n"] for r in local),
                                    n_local_reversals=sum(r["local_reversal"] for r in sub),
                                    n_outside_retained=sum(not r["inside_retained_response"] for r in sub),
                                    n_multiple_segments=sum(r["retained_segments_at_response"]>1 for r in sub),
                                    mean_abs_error=mae, mae_lo=lo, mae_hi=hi,
                                    signed_bias=bias, bias_lo=blo, bias_hi=bhi))
    return per_source, summary


def integral(a, b, bounds=None):
    lo = max(min(p[1] for p in a), min(p[1] for p in b))
    hi = min(max(p[1] for p in a), max(p[1] for p in b))
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    result = dict(lo=lo, hi=hi, width=max(0., hi-lo), first=None, last=None,
                  folded_width=None, status="valid" if hi>lo else "touch" if hi==lo else "disjoint")
    if hi <= lo:
        return result
    knots = sorted({lo, hi} | {p[1] for p in a+b if lo < p[1] < hi})
    first_areas, last_areas, folded = [], [], []
    for l, h in zip(knots, knots[1:]):
        ac, bc = crossings(a, l+(h-l)/2), crossings(b, l+(h-l)/2)
        first_areas.append((h-l)*(bc[0][2]-ac[0][2]))
        last_areas.append((h-l)*(bc[-1][3]-ac[-1][3]))
        if len(ac)>1 or len(bc)>1:
            folded.append(h-l)
    result.update(first=math.fsum(first_areas)/(hi-lo), last=math.fsum(last_areas)/(hi-lo),
                  folded_width=math.fsum(folded))
    return result


def grids(pair):
    yield "five", FULL, FULL
    yield "three", THREE, THREE
    for i in (1, 2, 3):
        four = tuple(k for k in FULL if k!=i)
        yield f"drop_both_{i+1}", four, four
        yield f"drop_{pair[0]}_{i+1}", four, FULL
        yield f"drop_{pair[1]}_{i+1}", FULL, four


def compare_grids(curves, eligibility):
    rows = []
    for c in sorted(eligibility):
        if not eligibility[c]:
            continue
        for pair in ("SU","SF","UF"):
            a,b = (curves[c,arm] for arm in pair)
            full = integral(a,b)
            for label,ia,ib in grids(pair):
                reduced = integral([a[i] for i in ia], [b[i] for i in ib])
                fixed = integral(a,b,(reduced["lo"],reduced["hi"])) if reduced["width"]>0 else None
                if fixed and abs(fixed["width"]-reduced["width"])>1e-14:
                    raise ValueError("A reduced grid must have support contained in the full grid")
                rows.append(dict(source=c,pair=pair,grid=label,ranks_a="".join(str(i+1) for i in ia),
                                 ranks_b="".join(str(i+1) for i in ib),**reduced,full_width=full["width"],
                                 full_first=full["first"],full_last=full["last"],
                                 full_first_on_reduced=fixed["first"] if fixed else None,
                                 full_last_on_reduced=fixed["last"] if fixed else None))
    return rows


def summarize_grids(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair"],row["grid"]].append(row)
    summary=[]
    for (pair,grid),group in groups.items():
        supported=[r for r in group if r["status"]=="valid"]
        mean, lo, hi = mean_ci([r["first"] for r in supported])
        own, olo, ohi=mean_ci([r["first"]-r["full_first"] for r in supported])
        fixed, flo, fhi=mean_ci([r["first"]-r["full_first_on_reduced"] for r in supported])
        summary.append(dict(pair=pair,grid=grid,n_eligible=len(group),n_supported=len(supported),
                            lost_full_support=sum(r["full_width"]>0 and r["width"]==0 for r in group),
                            narrower=sum(r["width"] < r["full_width"]-1e-14 for r in group),
                            median_width=np.median([r["width"] for r in supported]),
                            positives=sum(r["first"]>0 for r in supported),negatives=sum(r["first"]<0 for r in supported),
                            mean=mean,ci_low=lo,ci_high=hi,
                            full_mean_on_same_sources=np.mean([r["full_first"] for r in supported]),
                            own_interval_shift=own,own_lo=olo,own_hi=ohi,
                            fixed_interval_shift=fixed,fixed_lo=flo,fixed_hi=fhi,
                            sign_flips_own=sum(r["first"]*r["full_first"]<0 for r in supported),
                            sign_flips_fixed=sum(r["first"]*r["full_first_on_reduced"]<0 for r in supported)))
    return summary


def examples(rows):
    full_cases={r["source"] for r in rows if r["pair"]=="SU" and r["grid"]=="five"
                and r["first"] is not None and r["first"]<0}
    if full_cases != set(EXAMPLES):
        raise ValueError("Five-point counterexample membership differs from archived source IDs")
    result=[]
    for c,label in EXAMPLES.items():
        gr={r["grid"]:r for r in rows if r["source"]==c and r["pair"]=="SU"}
        out=dict(source=c,edit=label)
        for g in ("five","three","drop_both_2","drop_both_3","drop_both_4"):
            for field in ("lo","hi","width","first","last","folded_width"):
                out[g+"_"+field]=gr[g][field]
        for prefix,gs in [("joint",[f"drop_both_{i}" for i in (2,3,4)]),
                          ("one_arm",[f"drop_{a}_{i}" for a in "SU" for i in (2,3,4)])]:
            ds=[gr[g]["first"] for g in gs if gr[g]["first"] is not None]
            out.update({prefix+"_supported":len(ds),prefix+"_negative":sum(d<0 for d in ds),
                        prefix+"_min":min(ds) if ds else None,prefix+"_max":max(ds) if ds else None})
        result.append(out)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=HERE/"observations.csv")
    parser.add_argument("--output",type=Path,default=HERE)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    curves,eligibility=read(args.input)
    held=leave_one(curves,eligibility)
    source,errors=summarize_errors(held)
    detail=compare_grids(curves,eligibility)
    for filename,records in [("held_points.csv",held),("held_source_errors.csv",source),
                             ("held_summary.csv",errors),("grid_sources.csv",detail),
                             ("grid_summary.csv",summarize_grids(detail)),("counterexamples.csv",examples(detail))]:
        write(args.output,filename,records)


if __name__=="__main__":
    main()
