"""Compare saved SAM outputs at each point's own response. Python 3 and NumPy."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


GRID = (0.25, 0.5, 0.75, 1.0, 1.25)
NATIVE = {"spatial": "sam_native", "uniform": "uniform"}
POINTS = ("two_constant", "spatial_at_uniform")


def load_by_source(path):
    cases = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case = cases[row["case"]]
            if row["method"] in case:
                raise ValueError("Duplicate source and arm")
            case[row["method"]] = row
    return cases


def crossing(nodes, target, which):
    """Evaluate each segment in measured setting order, without extrapolating."""
    segments = list(zip(nodes, nodes[1:]))
    if which == "last":
        segments.reverse()
    for (x0, y0), (x1, y1) in segments:
        if min(x0, x1) <= target <= max(x0, x1):
            if x1 == x0:
                return y0 if which == "first" else y1
            return y0 + (target - x0) * (y1 - y0) / (x1 - x0)
    return None


def masked_contrasts(cases):
    out = []
    for case, arms in sorted(cases.items()):
        for point in POINTS:
            row = arms[point]
            target = float(row["directional_clip"])
            observed = float(row["lpips_spatial"])
            for base, prefix in NATIVE.items():
                nodes = []
                for knob in GRID:
                    name = prefix if knob == 1 else f"{prefix}_scale{knob:g}"
                    value = arms[name]
                    nodes.append((float(value["directional_clip"]), float(value["lpips_spatial"])))
                for which in ("first", "last"):
                    fitted = crossing(nodes, target, which)
                    out.append(dict(source=case, category=row["category"], point=point, sweep=base,
                                    crossing=which, response=target, observed_loss=observed,
                                    sweep_loss=fitted, sweep_minus_point=None if fitted is None else fitted-observed))
    return out


def summarize(rows):
    output = []
    for point in POINTS:
        for sweep in NATIVE:
            for which in ("first","last"):
                group = [r for r in rows if (r["point"],r["sweep"],r["crossing"]) == (point,sweep,which)]
                supported = [r for r in group if r["sweep_minus_point"] is not None]
                categories = defaultdict(list)
                for row in supported:
                    categories[row["category"]].append(row["sweep_minus_point"])
                vals = np.asarray([r["sweep_minus_point"] for r in supported])
                rng = np.random.default_rng(20261119)
                sampled = np.zeros(10000)
                for cat in sorted(categories):
                    a = np.asarray(categories[cat])
                    sampled += rng.choice(a, size=(len(sampled),len(a)),replace=True).sum(axis=1)
                if len(vals):
                    lower,upper = np.quantile(sampled/len(vals), [0.025,0.975])
                    mean = vals.mean()
                else:
                    mean = lower = upper = None
                output.append(dict(point=point, sweep=sweep, crossing=which, eligible=len(group),
                                   supported=len(vals), positive=int((vals>0).sum()), negative=int((vals<0).sum()),
                                   mean=mean, low=lower, high=upper))
    return output


def write_csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    cases=load_by_source(args.points)
    rows=masked_contrasts(cases)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"point_contrasts.csv",rows)
    write_csv(args.output/"point_summary.csv",summarize(rows))
