"""Validated source-level overlap integration and source bootstrap."""
from collections import Counter
import math

import numpy as np

COMPARISONS = {"SU": ("sam_native", "uniform"), "SF": ("sam_native", "flowedit"),
               "UF": ("uniform", "flowedit")}


X, Y = "clip_directional", "lpips_preservation"


FROZEN_METRICS = (X, Y, "mae_edit", "clip_target", "mae_full", "mae_preservation",
                  "lpips_full", "lpips_edit", "clip_image")


def first_branch(curve, response):
    """Interpolate on the earliest knob-ordered segment containing response."""
    for i, (a, b) in enumerate(zip(curve, curve[1:])):
        xa, xb = a[X], b[X]
        if min(xa, xb) <= response <= max(xa, xb):
            if xa == xb or response == xa:
                return i + 1, a[Y]
            if response == xb:
                return i + 1, b[Y]
            t = (response - xa) / (xb - xa)
            return i + 1, a[Y] + t * (b[Y] - a[Y])
    raise ValueError(f"Response {response} outside first-branch support")


def integrate(case, eligible, key, left, right):
    a, b = COMPARISONS[key]
    rec = dict(case=case, category=left[0]["category"], pair=key, subtracted_arm=a,
               minuend_arm=b, eligible=case in eligible, status=None)
    for label, curve in (("a", left), ("b", right)):
        responses = [p[X] for p in curve if p[X] is not None]
        rec[f"{label}_finite_xy"] = sum(p[X] is not None and p[Y] is not None for p in curve)
        rec[f"{label}_min_response"] = min(responses) if len(responses) == 5 else None
        rec[f"{label}_max_response"] = max(responses) if len(responses) == 5 else None
    rec.update(overlap_lo=None, overlap_hi=None, overlap_width=None, area_a=None, area_b=None,
               signed_area=None, average_lpips_a=None, average_lpips_b=None, difference=None,
               nonpositive=None, segments=0)
    pieces = []
    if not rec["eligible"]:
        rec["status"] = "no_preservation_region"
    elif min(rec["a_finite_xy"], rec["b_finite_xy"]) != 5:
        rec["status"] = "missing"
    else:
        lo = max(rec["a_min_response"], rec["b_min_response"])
        hi = min(rec["a_max_response"], rec["b_max_response"])
        rec.update(overlap_lo=lo, overlap_hi=hi, overlap_width=max(0., hi-lo))
        if hi < lo:
            rec["status"] = "disjoint"
        elif hi == lo:
            rec["status"] = "touch"
        else:
            breaks = sorted({lo, hi} | {p[X] for p in left + right if lo < p[X] < hi})
            for x0, x1 in zip(breaks, breaks[1:]):
                mid = x0 + (x1-x0)/2
                ia, ya = first_branch(left, mid)
                ib, yb = first_branch(right, mid)
                pieces.append(dict(lo=x0, hi=x1, mid=mid, a_segment=ia, b_segment=ib,
                                   a_lpips=ya, b_lpips=yb, signed_area=(yb-ya)*(x1-x0)))
            aa = math.fsum(p["a_lpips"]*(p["hi"]-p["lo"]) for p in pieces)
            ab = math.fsum(p["b_lpips"]*(p["hi"]-p["lo"]) for p in pieces)
            area = math.fsum(p["signed_area"] for p in pieces)
            diff = area/(hi-lo)
            rec.update(status="valid", area_a=aa, area_b=ab, signed_area=area,
                       average_lpips_a=aa/(hi-lo), average_lpips_b=ab/(hi-lo),
                       difference=diff, nonpositive=diff <= 0., segments=len(pieces))
    return rec, pieces


def stats(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None)


def pair_summary(records):
    status = Counter(r["status"] for r in records)
    valid = [r for r in records if r["status"] == "valid"]
    diff = np.array([r["difference"] for r in valid])
    a, b = records[0]["subtracted_arm"], records[0]["minuend_arm"]
    lo = hi = None
    if len(valid):
        draws = np.random.default_rng(0).integers(len(diff), size=(10000, len(diff)))
        lo, hi = np.quantile(diff[draws].mean(axis=1), [.025, .975], method="linear").tolist()
    return dict(subtracted_arm=a, minuend_arm=b, sources=len(records),
                eligible=sum(r["eligible"] for r in records), missing=status["missing"],
                disjoint=status["disjoint"], touch=status["touch"],
                no_preservation_region=status["no_preservation_region"], valid=len(valid),
                mean=stats(diff)["mean"], median=stats(diff)["median"], lo=lo, hi=hi,
                width_median=stats([r["overlap_width"] for r in valid])["median"],
                positive=int((diff>0).sum()), negative=int((diff<0).sum()), zero=int((diff==0).sum()),
                nonpositive=int((diff<=0).sum()),
                nonpositive_cases=[r["case"] for r in valid if r["nonpositive"]],
                cases_by_status={s: [r["case"] for r in records if r["status"] == s]
                                 for s in ("valid", "missing", "disjoint", "touch", "no_preservation_region")})
