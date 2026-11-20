"""Thin the bounded SAM sweep at either interior scale. Uses the frozen first-crossing estimator."""
import argparse
from pathlib import Path

import numpy as np

from bounded_mrc import ARMS, SCORERS, bootstrap_index, integrate, limits, observations, summarize, write


HERE = Path(__file__).resolve().parent
FOUR = (0.25, 0.50, 0.75, 1.00)
GRIDS = {"bounded_four": FOUR, "keep_050": (0.25, 0.50, 1.00), "keep_075": (0.25, 0.75, 1.00)}


def measure(curves, eligibility, scorer):
    rows = []
    for source in sorted(curves):
        arms = [curves[source][key] for key in ARMS]
        eligible = eligibility[source]
        results = {grid: integrate(arms, scorer, scales, eligible) for grid, scales in GRIDS.items()}
        baseline = results["bounded_four"]
        in_all = all(record["status"] == "valid" for record in results.values())
        for grid, record in results.items():
            fixed = None
            if record["status"] == "valid":
                if baseline["status"] != "valid" or baseline["default"] != record["default"]:
                    raise ValueError("Inconsistent nested support or scale-1 default: " + source)
                bounds = record["lo"], record["hi"]
                if bounds[0] < baseline["lo"] or bounds[1] > baseline["hi"]:
                    raise ValueError("Thinned interval exceeds bounded four-point support: " + source)
                fixed = integrate(arms, scorer, FOUR, eligible, bounds)["matched"]
            rows.append(dict(scorer=scorer, grid=grid, source=source, **record,
                             common_all_sources=in_all, bounded_status=baseline["status"],
                             bounded_width=baseline["width"], bounded_matched=baseline["matched"],
                             bounded_matched_on_grid_interval=fixed))
    return rows


def change(rows):
    """Rows are paired sources supported by both the three- and four-setting grids."""
    valid = [row for row in rows if row["status"] == "valid"]
    thin = np.array([row["matched"] for row in valid])
    four = np.array([row["bounded_matched"] for row in valid])
    restricted = np.array([row["bounded_matched_on_grid_interval"] for row in valid])
    default = np.array([row["default"] for row in valid])
    ix = bootstrap_index(len(valid))
    mean_default = default[ix].mean(axis=1)
    if np.any(mean_default <= 0):
        raise ValueError("Nonpositive resampled default denominator")
    result = dict(n_paired=len(valid), default=default.mean(),
                  n_narrower=sum(row["width"] < row["bounded_width"] for row in valid),
                  bounded_positive=int((four > 0).sum()), bounded_zero=int((four == 0).sum()),
                  bounded_negative=int((four < 0).sum()), thin_positive=int((thin > 0).sum()),
                  thin_zero=int((thin == 0).sum()), thin_negative=int((thin < 0).sum()),
                  sign_flips=int((thin*four < 0).sum()),
                  flip_ids=" ".join(r["source"] for r in valid if r["matched"]*r["bounded_matched"] < 0))
    for interval, b in (("own_intervals", four), ("identical_interval", restricted)):
        diff = thin - b
        result[interval+"_bounded_mean"] = b.mean()
        result[interval+"_bounded_retained"] = b.mean()/default.mean()
        result[interval+"_delta"] = diff.mean()
        result[interval+"_retained_delta"] = diff.mean()/default.mean()
        dist = diff[ix].mean(axis=1)
        result[interval+"_delta_lo"], result[interval+"_delta_hi"] = limits(dist)
        result[interval+"_retained_delta_lo"], result[interval+"_retained_delta_hi"] = limits(dist/mean_default)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=HERE.parents[1],
                        help="Revision directory, extracted review supplement, or review supplement ZIP")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    curves, eligible = observations(args.records)
    details, totals, contrasts = [], [], []
    for scorer in SCORERS:
        per_source = measure(curves, eligible, scorer)
        details.extend(per_source)
        for grid in GRIDS:
            rows = [row for row in per_source if row["grid"] == grid]
            for cohort in ("grid_support", "common_sources"):
                # summarize() takes the full roster to audit empty, disjoint and outside-common counts.
                subset = [dict(row, status="outside_common_sources") if
                          cohort == "common_sources" and row["status"] == "valid" and not row["common_all_sources"]
                          else row for row in rows]
                meta = dict(scorer=scorer, grid=grid, cohort=cohort)
                totals.append(dict(**meta, **summarize(subset)))
                if grid != "bounded_four":
                    contrasts.append(dict(**meta, **change(subset)))
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output, "thinning_sources.csv", details)
    write(args.output, "thinning_summary.csv", totals)
    write(args.output, "thinning_paired.csv", contrasts)


if __name__ == "__main__":
    main()
