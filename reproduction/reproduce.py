"""Recompute the headline point estimates from the extracted review supplement."""

import argparse
import csv
import json
from collections import defaultdict
from math import fsum, isfinite
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
SCALES = (0.25, 0.5, 0.75, 1.0, 1.25)


def read_rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def method(arm, scale):
    return arm if scale == 1 else f"{arm}_scale{scale:g}"


def operating_sources():
    """Read one measured row per configuration and eligible source."""
    fields = ("lpips_spatial", "lpips_black", "directional_clip", "edit_mae", "background_mae")
    sources = defaultdict(dict)
    for row in read_rows("cpu/selectivity/operating_sources.csv"):
        source = sources[row["case"]]
        name = row["method"]
        if name in source or int(row["background_pixels"]) <= 0:
            raise ValueError(f"Invalid operating row for {row['case']} / {name}")
        values = {field: float(row[field]) for field in fields}
        if not all(map(isfinite, values.values())) or values["background_mae"] <= 0:
            raise ValueError(f"Invalid measurements for {row['case']} / {name}")
        source[name] = values
    return [sources[case] for case in sorted(sources)]


def first_crossing(curve, response):
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if min(x0, x1) <= response <= max(x0, x1):
            if response == x0:
                return y0
            return y0 + (response - x0) / (x1 - x0) * (y1 - y0)
    raise ValueError("Response outside observed support")


def matched_gap(source, metric, scales):
    """Mean uniform-minus-spatial loss across this source's CLIP overlap."""
    curves = []
    for arm in ("sam_native", "uniform"):
        points = [source[method(arm, scale)] for scale in scales]
        curves.append([(p["directional_clip"], p[metric]) for p in points])
    spatial, uniform = curves
    low = max(min(x for x, _ in curve) for curve in curves)
    high = min(max(x for x, _ in curve) for curve in curves)
    if high <= low:
        return None
    breaks = sorted({low, high} | {x for curve in curves for x, _ in curve if low < x < high})
    areas = []
    for left, right in zip(breaks, breaks[1:]):
        # On each open interval both first-crossing branches are linear.
        mid = left + (right - left) / 2
        areas.append((right - left) * (first_crossing(uniform, mid) - first_crossing(spatial, mid)))
    return fsum(areas) / (high - low)


def estimate(values, denominator=None, *, ci, seed, draws):
    result = {"mean" if denominator is None else "ratio_of_means":
              mean(values) if denominator is None else mean(values) / mean(denominator)}
    if ci:
        import numpy as np

        n = len(values)
        indices = np.random.default_rng(seed).integers(n, size=(draws, n))
        samples = np.asarray(values)[indices].mean(axis=1)
        if denominator is not None:
            samples /= np.asarray(denominator)[indices].mean(axis=1)
        result["ci95"] = np.quantile(samples, [0.025, 0.975]).tolist()
    return result


def matched_summary(sources, metric, scales, ci):
    matched, same_native = [], []
    for source in sources:
        gap = matched_gap(source, metric, scales)
        if gap is not None:
            matched.append(gap)
            same_native.append(source["uniform"][metric] - source["sam_native"][metric])
    return {
        "overlap_sources": len(matched),
        "default_gap_on_overlap": mean(same_native),
        "matched_gap": estimate(matched, ci=ci, seed=0, draws=10000),
        "matched_positive": sum(x > 0 for x in matched),
        "matched_negative": sum(x < 0 for x in matched),
        "matched_zero": matched.count(0),
        "retained_fraction": estimate(matched, same_native, ci=ci, seed=0, draws=10000),
    }


def projection(sources, metric, ci):
    native, constants = [], []
    for source in sources:
        uniform = source["uniform"][metric]
        default = uniform - source["sam_native"][metric]
        native.append(default)
        constants.append(uniform - source["two_constant"][metric])
    return {
        "background_sources": len(native),
        "default_gap": mean(native),
        "two_constant_gain": mean(constants),
        "two_constant_recovery": estimate(constants, native, ci=ci, seed=1208, draws=20000),
        **matched_summary(sources, metric, SCALES, ci),
        "scales_at_or_below_default": matched_summary(sources, metric, SCALES[:-1], ci),
    }


def selectivity(sources):
    results = []
    for scale in SCALES:
        point = {"scale": scale}
        for label, arm in (("spatial", "sam_native"), ("uniform", "uniform")):
            rows = [source[method(arm, scale)] for source in sources]
            point[label] = mean(r["edit_mae"] / r["background_mae"] for r in rows)
        results.append(point)
    return results


def restoring():
    arms = ("REQ", "U25", "GLB_R", "REQ_U")
    cases = defaultdict(dict)
    for row in read_rows("source_correction_and_blending/two_seed_per_output.csv"):
        key = row["arm"], row["seed"]
        if key[0] not in arms or key[1] not in ("0", "17"):
            continue
        case = cases[row["case"]]
        if key in case:
            raise ValueError(f"Duplicate restoring row for {row['case']} / {key}")
        case[key] = float(row["lpips_preserved"])
    result = {}
    for name, spatial, uniform in (("default", "REQ", "U25"),
                                  ("spatial_profile", "REQ", "GLB_R"),
                                  ("uniform_profile", "REQ_U", "U25")):
        gaps = [mean(case[spatial, s] - case[uniform, s] for s in ("0", "17"))
                for case in cases.values()]
        result[name] = {"sources": len(gaps), "spatial_minus_uniform": mean(gaps),
                        "uniform_better": sum(x > 0 for x in gaps)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", action="store_true", help="Include paired SAM bootstrap intervals (requires NumPy).")
    args = parser.parse_args()
    if args.ci:
        try:
            import numpy
        except ImportError:
            parser.error("--ci requires NumPy. Omit --ci for standard-library point estimates.")
    sources = operating_sources()
    print(json.dumps({
        "restoring": restoring(),
        "sam": {m: projection(sources, "lpips_" + m, args.ci) for m in ("spatial", "black")},
        "selectivity_mean_of_source_ratios": selectivity(sources),
    }, indent=2, allow_nan=False))
