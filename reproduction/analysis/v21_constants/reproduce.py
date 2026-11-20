"""Audit SAM projection constants on recorded trajectories (Python 3 and NumPy).

Run python reproduce.py to print the results, or append --out DIRECTORY for CSVs.
--steps also accepts the unmodified SAM_scan/data/core_steps.csv from the supplement.
The adjacent steps.csv selects two core arms and eight columns without rounding.
The scan900 reports archive (SHA-256
01a2b55591a7f0b8c55c6ebc474afd19a5dec2a0b918d247c99b446040e2eb54)
stores outputs/<case>__<method>/profile.json for all 100 runs. The retained
values agree with the corresponding raw fields at every active step (2--27).
residual_rms is measured before either branch gate. model_rms and visible_rms
are FP32 displacements from the anchor. The state_rms columns measure those
displacements after the BF16 state cast. All main errors use the FP32 target.
CSV errors are fractions. The recorded row uses observed projections. All
other rows fit only the archived constant-run residuals, with no image scores.
The full core_steps.csv input has SHA-256
041c268c58108a3549ddb06ab45ad8a6038f2fc87bdc1c6b4ced330f3cd76f1a.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ARMS = ("sam_native", "two_constant")
FIELDS = ("residual_rms", "model_rms", "visible_rms", "model_state_rms", "visible_state_rms")
FITS = ("recorded", "source_total", "source_min_step", "shared_leave_one_out")


def read_steps(path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in ARMS:
                key = row["case"], row["method"], int(row["step"])
                if key in rows:
                    raise ValueError(f"Duplicate step {key}")
                rows[key] = [float(row[f]) for f in FIELDS]
    cases = sorted({case for case, arm, step in rows})
    expected = {(case, arm, step) for case in cases for arm in ARMS for step in range(2, 28)}
    if len(cases) != 50 or rows.keys() != expected:
        raise ValueError("Expected 50 matched sources and 26 active steps per arm.")
    arrays = [np.array([[rows[case, arm, step] for step in range(2, 28)] for case in cases])
              for arm in ARMS]
    if any(not np.isfinite(a).all() or (a <= 0).any() for a in arrays):
        raise ValueError("All archived profile RMS values must be finite and positive.")
    return cases, *arrays


def best_gain(ratios, weights):
    """A minimizer of weighted absolute error over scalar gains in [0, 1]."""
    x, w = ratios.ravel(), weights.ravel()
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    return np.clip(x[order[np.searchsorted(cumulative, cumulative[-1] / 2)]], 0, 1)


def analyze(cases, native, constant):
    sources, summaries, minima = [], [], {}
    residual = constant[:, :, 0]
    for column, branch in ((1, "model"), (2, "visible")):
        target = native[:, :, column]
        total = target.sum(axis=1)
        gains = {
            "recorded": total / native[:, :, 0].sum(axis=1),
            "source_total": total / residual.sum(axis=1),
        }
        ratios = target / residual
        # Normalization gives each training source one vote in the mean step loss.
        weights = residual / total[:, None]
        gains["source_min_step"] = np.array([best_gain(x, w) for x, w in zip(ratios, weights)])
        loo = []
        for i in range(len(cases)):
            train = np.arange(len(cases)) != i
            loo.append(best_gain(ratios[train], weights[train]))
        gains["shared_leave_one_out"] = np.array(loo)
        np.testing.assert_allclose(gains["recorded"][:, None] * residual, constant[:, :, column],
                                   rtol=0, atol=4e-8)
        cast_target = native[:, :, column + 2]
        cast_loss = np.abs(constant[:, :, column + 2] - cast_target).sum(axis=1) / cast_target.sum(axis=1)
        for fit in FITS:
            gain = gains[fit]
            if np.any((gain < 0) | (gain > 1)):
                raise ValueError("Fitted gain outside the editor's admissible range")
            profile = constant[:, :, column] if fit == "recorded" else gain[:, None] * residual
            delta = profile - target
            total_bias = delta.sum(axis=1) / total
            step_loss = np.abs(delta).sum(axis=1) / total
            if fit == "source_min_step":
                minima[branch] = step_loss
            summaries.append(dict(branch=branch, fit=fit, n=len(cases), mean_gain=gain.mean(),
                                  mean_total_bias=total_bias.mean(), mean_abs_total_error=np.abs(total_bias).mean(),
                                  mean_step_error=step_loss.mean(),
                                  mean_observed_postcast_step_error=cast_loss.mean() if fit == "recorded" else ""))
            for i, case in enumerate(cases):
                sources.append(dict(case=case, branch=branch, fit=fit, gain=gain[i],
                                    signed_total_error=total_bias[i], step_error=step_loss[i],
                                    observed_postcast_step_error=cast_loss[i] if fit == "recorded" else ""))
    return summaries, sources, minima["visible"] - minima["model"]


def write_rows(handle, rows):
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=Path, default=Path(__file__).with_name("steps.csv"))
    parser.add_argument("--out", type=Path, help="Directory for summary.csv and per_source.csv")
    args = parser.parse_args()
    summaries, sources, difference = analyze(*read_steps(args.steps))
    write_rows(sys.stdout, summaries)
    # Both branch minima are fitted separately within the same source.
    draws = np.random.default_rng(0).integers(len(difference), size=(10000, len(difference)))
    low, high = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
    print(f"Paired visible minus model minimum step error: {difference.mean():.8f} "
          f"[{low:.8f}, {high:.8f}], {sum(difference > 0)}/{len(difference)} positive.",
          file=sys.stderr)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, records in (("summary.csv", summaries), ("per_source.csv", sources)):
            with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
                write_rows(handle, records)
