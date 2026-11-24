"""Paired four- versus five-setting SAM MRC. Run with Python 3 and NumPy."""
import argparse
import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
SCALES = (0.25, 0.5, 0.75, 1.0, 1.25)
ARMS = ("sam_native", "uniform")
SCORERS = ("spatial", "black")
GROUPS = ("full", "bounded", "full_common_sources", "full_on_bounded_intervals")
DRAWS = 10000


def observations(root):
    """Join the original scan and harmonized image scores by archived job ID."""
    if root.is_file():
        with ZipFile(root) as z:
            def read(name):
                with z.open(name) as f:
                    return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")))
            return join(read, "SAM_scan/data/scores.csv", "cpu/harmonized_lpips")

    def read(name):
        with (root / name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if (root / "SAM_scan/data/scores.csv").exists():
        return join(read, "SAM_scan/data/scores.csv", "cpu/harmonized_lpips")
    return join(read, "analysis/metrics/metrics.csv", "analysis/harmonized_lpips")


def join(read, scan_path, score_dir):
    scores = {}
    for part in ("core", "scan"):
        for row in read(f"{score_dir}/sam_{part}_scores.csv"):
            if row["id"] in scores:
                raise ValueError("Duplicate harmonized score: " + row["id"])
            scores[row["id"]] = row
    curves = defaultdict(lambda: defaultdict(dict))
    eligibility = {}
    for row in read(scan_path):
        arm = row["arm"]
        if arm not in ARMS:
            continue
        job, case, scale = row["id"], row["case"], float(row["scale"])
        data = scores[job]
        if ((data["case"], data["seed"], data["arm"]) != (case, "0", row["method"])
                or row["seed"] != "0" or scale in curves[case][arm]):
            raise ValueError("Inconsistent or duplicate SAM setting: " + job)
        keep = int(data["preservation_pixels"]) > 0
        if eligibility.setdefault(case, keep) != keep:
            raise ValueError("Inconsistent background eligibility: " + case)
        clip = float(row["clip_directional"])
        outcomes = {k: float(data[k]) if data[k] else None for k in SCORERS}
        if not math.isfinite(clip) or any((v is not None and math.isfinite(v)) != keep for v in outcomes.values()):
            raise ValueError("Inconsistent response or outcome: " + job)
        curves[case][arm][scale] = (clip, outcomes)
    if len(curves) != 50 or sum(eligibility.values()) != 42:
        raise ValueError("Expected the frozen 50-source cohort with 42 backgrounds")
    for case, arms in curves.items():
        if set(arms) != set(ARMS) or any(tuple(sorted(points)) != SCALES for points in arms.values()):
            raise ValueError("Incomplete five-setting grid: " + case)
    return curves, eligibility


def first_value(curve, response):
    for (xa, ya), (xb, yb) in zip(curve, curve[1:]):
        if min(xa, xb) <= response <= max(xa, xb):
            if response == xa or xa == xb:
                return ya
            if response == xb:
                return yb
            return ya + (response-xa)/(xb-xa)*(yb-ya)
    raise ValueError("Interpolation outside observed support")


def integrate(curves, scorer, scales, eligible, bounds=None):
    a, b = [[(arm[s][0], arm[s][1][scorer]) for s in scales] for arm in curves]
    lo = max(min(p[0] for p in a), min(p[0] for p in b))
    hi = min(max(p[0] for p in a), max(p[0] for p in b))
    if bounds is not None:
        lo, hi = max(lo,bounds[0]), min(hi,bounds[1])
    default = curves[1][1.0][1][scorer] - curves[0][1.0][1][scorer] if eligible else None
    if not eligible:
        return dict(status="empty_background",lo=None,hi=None,width=None,default=None,matched=None)
    if hi <= lo:
        return dict(status="touch" if hi==lo else "disjoint",lo=lo,hi=hi,width=0.,default=default,matched=None)
    knots = sorted({lo, hi} | {x for x,_ in a+b if lo < x < hi})
    areas = []
    for left,right in zip(knots,knots[1:]):
        mid = left+(right-left)/2
        areas.append((first_value(b,mid)-first_value(a,mid))*(right-left))
    return dict(status="valid",lo=lo,hi=hi,width=hi-lo,default=default,matched=math.fsum(areas)/(hi-lo))


def bootstrap_index(n):
    return np.random.default_rng(0).integers(n, size=(DRAWS,n))


def limits(sample):
    return np.quantile(sample, (.025,.975))


def signs(vals):
    return int(sum(vals>0)), int(sum(vals==0)), int(sum(vals<0))


def summarize(rows):
    status = Counter(r["status"] for r in rows)
    valid = [r for r in rows if r["status"]=="valid"]
    d = np.array([r["default"] for r in valid])
    m = np.array([r["matched"] for r in valid])
    ix = bootstrap_index(len(d))
    dm, mm = d[ix].mean(axis=1), m[ix].mean(axis=1)
    if (dm <= 0).any() or d.mean() <= 0:
        raise ValueError("Retained-fraction denominator crosses zero")
    positive, zero, negative = signs(m)
    dp, dz, dn = signs(d)
    summary = dict(n_roster=len(rows), n_eligible=len(rows)-status["empty_background"], n_valid=len(valid),
                   empty_background=status["empty_background"], disjoint=status["disjoint"],
                   touch=status["touch"], outside_common=status["outside_common_sources"],
                   default_positive=dp, default_zero=dz, default_negative=dn,
                   matched_positive=positive, matched_zero=zero, matched_negative=negative,
                   strict_reversals=int(sum(d*m<0)), median_width=np.median([r["width"] for r in valid]))
    for name, estimate, dist in (("default",d.mean(),dm), ("matched",m.mean(),mm),
                                 ("matched_minus_default",(m-d).mean(),mm-dm),
                                 ("retained_fraction",m.mean()/d.mean(),mm/dm)):
        summary[name] = estimate
        summary[name+"_lo"],summary[name+"_hi"] = limits(dist)
    return summary


def comparison(groups):
    bounded = {r["source"]:r for r in groups["bounded"] if r["status"]=="valid"}
    full = {r["source"]:r for r in groups["full_common_sources"] if r["status"]=="valid"}
    ids = sorted(bounded)
    assert set(ids)==set(full)
    d = np.array([bounded[c]["default"] for c in ids])
    b = np.array([bounded[c]["matched"] for c in ids])
    f = np.array([full[c]["matched"] for c in ids])
    diff = b-f
    ix = bootstrap_index(len(ids))
    dist = diff[ix].mean(axis=1)
    dmean = d[ix].mean(axis=1)
    result = dict(n=len(ids),bounded_minus_full=diff.mean(),strict_sign_flips=int(sum(b*f<0)),
                  flip_ids=" ".join(c for c,x,y in zip(ids,b,f) if x*y<0),
                  ratio_change=diff.mean()/d.mean(),
                  narrower_widths=sum(bounded[c]["width"] < full[c]["width"] for c in ids))
    result["difference_lo"],result["difference_hi"]=limits(dist)
    result["ratio_change_lo"],result["ratio_change_hi"]=limits(dist/dmean)
    fixed = {r["source"]:r for r in groups["full_on_bounded_intervals"] if r["status"]=="valid"}
    result["max_abs_fixed_interval_difference"]=max(abs(bounded[c]["matched"]-fixed[c]["matched"]) for c in ids)
    assert result["max_abs_fixed_interval_difference"] < 1e-12
    return result


def write(root, name, rows):
    with (root/name).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,default=HERE.parents[1],
                        help="Review supplement ZIP, extracted packet root or revision directory")
    parser.add_argument("--output",type=Path,default=HERE)
    args = parser.parse_args()
    curves, eligibility = observations(args.records)
    details, summaries, comparisons = [], [], []
    for scorer in SCORERS:
        groups = {name:[] for name in GROUPS}
        for case in sorted(curves):
            arms = [curves[case][k] for k in ARMS]
            eligible = eligibility[case]
            full = integrate(arms,scorer,SCALES,eligible)
            bounded = integrate(arms,scorer,SCALES[:-1],eligible)
            if bounded["status"] == "valid":
                assert full["status"]=="valid" and bounded["default"]==full["default"]
                common = full.copy()
                fixed = integrate(arms,scorer,SCALES,eligible,(bounded["lo"],bounded["hi"]))
            else:
                common = full.copy()
                if full["status"]=="valid":
                    common.update(status="outside_common_sources",matched=None)
                fixed = bounded.copy()
            for name, rec in zip(GROUPS,(full,bounded,common,fixed)):
                item=dict(scorer=scorer,group=name,source=case,**rec)
                groups[name].append(item)
                details.append(item)
        for name, rows in groups.items():
            summaries.append(dict(scorer=scorer,group=name,**summarize(rows)))
        comparisons.append(dict(scorer=scorer,**comparison(groups)))
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output,"per_source.csv",details)
    write(args.output,"summary.csv",summaries)
    write(args.output,"common_set_change.csv",comparisons)


if __name__ == "__main__":
    main()
