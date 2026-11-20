"""Reproduce source- and cohort-curve MRC from observations.csv (NumPy only)."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
B = 10000
ARMS = {"SAM": ("S", "U", "F"), "Restoring": ("S", "U")}
PAIRS = {"SAM": ("SU", "SF", "UF"), "Restoring": ("SU",)}


def load(path, cohort):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["cohort"] == cohort]
    cases = sorted({r["source"] for r in rows})
    grids = {"S": (.25, .5, .75, 1., 1.25), "U": (.25, .5, .75, 1., 1.25),
             "F": (12., 16., 20., 24., 28.)} if cohort == "SAM" else {"S": (.7, 1., 1.3), "U": (.7, 1., 1.3)}
    data = np.full((len(cases), len(grids), len(grids["S"]), 3), np.nan)
    seen, info = set(), {}
    for r in rows:
        c, a, knob = r["source"], r["arm"], float(r["setting"])
        key = c, a, knob
        if key in seen:
            raise ValueError(f"Duplicate point {key}")
        seen.add(key)
        flags = (r["background_eligible"] == "True", r["ambiguous_prompt"] == "True", r["category"])
        if info.setdefault(c, flags) != flags:
            raise ValueError(f"Inconsistent source flags for {c}")
        node = [float(r[k]) if r[k] else np.nan for k in ("directional_clip", "edit_mae", "fm_lpips")]
        if not np.isfinite(node[:2]).all() or np.isfinite(node[2]) != flags[0]:
            raise ValueError(f"Incomplete coordinates for {key}")
        data[cases.index(c), ARMS[cohort].index(a), grids[a].index(knob)] = node
    if len(seen) != np.prod(data.shape[:3]):
        raise ValueError("Incomplete source/arm grid")
    eligible = np.array([info[c][0] for c in cases])
    ambiguous = np.array([info[c][1] for c in cases])
    categories = np.array([info[c][2] for c in cases])
    return cases, eligible, ambiguous, categories, data


def branch_values(x, y, mid, last):
    if last:
        x, y = x[:, ::-1], y[:, ::-1]
    left, right = x[:, None, :-1], x[:, None, 1:]
    covered = (np.minimum(left, right) <= mid[..., None]) & (mid[..., None] <= np.maximum(left, right))
    i = covered.argmax(axis=-1)
    src = np.arange(len(x))[:, None]
    x0, x1 = x[src, i], x[src, i+1]
    frac = np.divide(mid-x0, x1-x0, out=np.zeros_like(mid), where=x0 != x1)
    return y[src, i] + frac * (y[src, i+1] - y[src, i])


def integrate(nodes, pair_ix, coord=0, last=False, shared=False):
    """Integrate over cells partitioned by response knots; supports a batch of sources or mean curves."""
    x, y = nodes[..., coord], nodes[..., 2]
    a, b = pair_ix
    support = x if shared else x[:, [a, b]]
    lo = support.min(axis=2).max(axis=1)
    hi = support.max(axis=2).min(axis=1)
    # Clipping knots before sorting introduces zero-length cells at the endpoints.
    flat = support.reshape(len(nodes), -1)
    knots = np.sort(np.clip(flat, lo[:, None], np.maximum(lo, hi)[:, None]), axis=1)
    dx = np.diff(knots, axis=1)
    mid = knots[:, :-1] + dx * .5
    diff = branch_values(x[:, b], y[:, b], mid, last) - branch_values(x[:, a], y[:, a], mid, last)
    delta = np.divide((diff * dx).sum(axis=1), hi-lo,
                      out=np.full(len(nodes), np.nan), where=hi > lo)
    return lo, hi, delta


@dataclass
class Variant:
    name: str
    nodes: np.ndarray
    pair_ix: tuple
    coord: int
    mask: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    mean_curve: bool = False

    def estimate(self, ix):
        """Row-wise source resampling; ix contains indices into the complete source roster."""
        keep = self.mask[ix]
        if self.mean_curve:
            ans = []
            for start in range(0, len(ix), 400):
                block = ix[start:start+400]
                w = keep[start:start+400, :, None, None, None]
                count = w.sum(axis=1)
                sums = np.where(w, self.nodes[block], 0.).sum(axis=1)
                mean_nodes = np.divide(sums, count, out=np.full_like(sums, np.nan), where=count != 0)
                ans.append(integrate(mean_nodes, self.pair_ix, self.coord)[2])
            return np.concatenate(ans)
        weights = np.where(keep, self.weights[ix], 0.)
        area = (np.where(keep, self.values[ix], 0.) * weights).sum(axis=1)
        denom = weights.sum(axis=1)
        return np.divide(area, denom, out=np.full_like(area, np.nan), where=denom > 0)


def draws(mask):
    ix = np.flatnonzero(mask)
    return ix[np.random.default_rng(0).integers(len(ix), size=(B, len(ix)))]


def interval(vals):
    count = int(np.isfinite(vals).sum())
    if count < B:
        return "", "", count
    low, high = np.quantile(vals, [.025, .975])
    return low, high, count


def result(v, primary, eligible):
    ids = np.flatnonzero(v.mask)[None, :]
    estimate = v.estimate(ids)[0]
    low, high, complete = interval(v.estimate(draws(v.mask)))
    union = v.mask | primary.mask
    common = v.mask & primary.mask
    row = dict(n_eligible=int(eligible.sum()), n=int(v.mask.sum()),
               estimate_status="valid" if np.isfinite(estimate) else "no_cohort_curve_overlap",
               mean=estimate,
               ci_low=low, ci_high=high, finite_draws=complete)
    for label, mask in (("vs_primary", union), ("vs_primary_on_shared_sources", common)):
        one = np.flatnonzero(mask)[None, :]
        resampled = draws(mask)
        diff = v.estimate(one)[0] - primary.estimate(one)[0]
        dlow, dhigh, dcount = interval(v.estimate(resampled) - primary.estimate(resampled))
        row.update({label: diff, label+"_lo": dlow, label+"_hi": dhigh, label+"_finite_draws": dcount})
    row["n_shared_with_primary"] = int(common.sum())
    row["n_union_with_primary"] = int(union.sum())
    if v.mean_curve:
        lo, hi, _ = integrate(v.nodes[v.mask].mean(axis=0, keepdims=True), v.pair_ix, v.coord)
        row.update(n_positive="", n_negative="", n_tied="", median_source_width="",
                   cohort_curve_lo=lo[0], cohort_curve_hi=hi[0])
    else:
        values = v.values[v.mask]
        row.update(n_positive=int((values>0).sum()), n_negative=int((values<0).sum()),
                   n_tied=int((values==0).sum()), median_source_width=np.median((v.hi-v.lo)[v.mask]),
                   cohort_curve_lo="", cohort_curve_hi="")
    return row


def analyze(path, cohort):
    cases, eligible, ambiguous, _, nodes = load(path, cohort)
    primary_valid = {}
    for pair in PAIRS[cohort]:
        ix = tuple(ARMS[cohort].index(a) for a in pair)
        lo, hi, _ = integrate(nodes, ix)
        primary_valid[pair] = eligible & (hi > lo)
    common_sources = np.logical_and.reduce(list(primary_valid.values()))
    summaries, source_rows = [], []
    for pair in PAIRS[cohort]:
        ix = tuple(ARMS[cohort].index(a) for a in pair)
        specs = ["primary", "last_crossing", "width_weighted", "edit_mae"]
        if cohort == "SAM":
            specs += ["exclude_ambiguous", "common_sources", "common_interval", "common_interval_width_weighted"]
        specs += ["cohort_curve_pair_sources", "cohort_curve_eligible"]
        for name in specs:
            coord = int(name == "edit_mae")
            shared_interval = name.startswith("common_interval")
            lo, hi, values = integrate(nodes, ix, coord, name == "last_crossing", shared_interval)
            mask = eligible.copy()
            if name == "exclude_ambiguous":
                mask &= ~ambiguous
            if name == "common_sources" or shared_interval:
                mask &= common_sources
            retained = mask.copy()
            mean_curve = name.startswith("cohort_curve")
            if name != "cohort_curve_eligible":
                mask &= hi > lo
            if shared_interval:
                assert np.array_equal(mask, common_sources)
            weighted = name.endswith("width_weighted")
            weights = np.maximum(hi-lo, 0.) if weighted else np.ones(len(cases))
            v = Variant(name, nodes, ix, coord, mask, lo, hi, values, weights, mean_curve)
            if name == "primary":
                primary = v
            desc = dict(cohort=cohort, pair=pair, variant=name, response=["directional_clip","edit_mae"][coord])
            row = result(v, primary, eligible)
            row.update(n_total=len(cases), n_retained_eligible=int(retained.sum()),
                       n_pair_positive_overlap=int((retained & (hi > lo)).sum()),
                       n_pair_touch=int((retained & (hi == lo)).sum()),
                       n_pair_disjoint=int((retained & (hi < lo)).sum()))
            summaries.append({**desc, **row})
            for i, case in enumerate(cases):
                if not eligible[i]:
                    status = "empty_background"
                elif not retained[i]:
                    status = "excluded_ambiguous" if name == "exclude_ambiguous" else "outside_common_source_set"
                elif mask[i]:
                    status = "in_cohort_curve" if mean_curve else "valid"
                else:
                    status = "disjoint" if hi[i] < lo[i] else "touch"
                source_rows.append(dict(**desc, source=case, status=status, included=int(mask[i]),
                                        support_lo=lo[i] if eligible[i] else "",
                                        support_hi=hi[i] if eligible[i] else "",
                                        source_width=max(0., hi[i]-lo[i]) if eligible[i] else "",
                                        difference=values[i] if mask[i] and not mean_curve else "",
                                        weight=weights[i] if mask[i] and not mean_curve else ""))
    return summaries, source_rows


def historical_inverse(path):
    """Archived inverse endpoint: integrate CLIP over mean-curve LPIPS overlap."""
    cases, _, _, categories, nodes = load(path, "Restoring")
    # integrate() takes the response in coordinate 0 and the outcome in coordinate 2.
    curves = nodes[..., [2, 1, 0]]
    mean_curve = curves.mean(axis=0, keepdims=True)
    lo, hi, estimate = integrate(mean_curve, (0, 1))
    rng = np.random.default_rng(20261106)
    groups = [np.flatnonzero(categories == k) for k in sorted(set(categories))]
    draws_by_source = np.concatenate([rng.choice(g, (20000, len(g))) for g in groups], axis=1)
    samples = []
    for start in range(0, len(draws_by_source), 400):
        curve = curves[draws_by_source[start:start+400]].mean(axis=1)
        # The historical endpoint requires strict ordering in both cohort LPIPS curves.
        _, _, delta = integrate(curve, (0, 1))
        ordered = (np.diff(curve[..., 0], axis=2) < 0).all(axis=(1, 2))
        samples.extend(np.where(ordered, delta, np.nan))
    finite = int(np.isfinite(samples).sum())
    limits = np.quantile(samples, [.025, .975]) if finite == len(samples) else ("", "")
    return dict(cohort="Restoring", sources=len(cases), response="fm_lpips",
                outcome="directional_clip", pair="SU", order="average_sources_then_match",
                support_lo=lo[0], support_hi=hi[0], mean=estimate[0],
                ci_low=limits[0], ci_high=limits[1], finite_draws=finite, draws=20000, seed=20261106)


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "observations.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, sources = [], []
    for cohort in ARMS:
        rows, details = analyze(args.input, cohort)
        summaries.extend(rows)
        sources.extend(details)
    write(args.output / "summary.csv", summaries)
    write(args.output / "per_source.csv", sources)
    write(args.output / "historical_inverse.csv", [historical_inverse(args.input)])


if __name__ == "__main__":
    main()
