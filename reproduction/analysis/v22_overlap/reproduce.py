"""Run python -B reproduction/analysis/v22_overlap/reproduce.py (requires NumPy).

Reads the adjacent reproduction packet and writes three small files beside this script.
FM and MI denote the v21 spatial feature-map and masked-input LPIPS scorers.
All SAM gaps in summary.csv are uniform minus spatial. Fractions use units of one.
"""
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
from v21_bounded import bounded_mrc as v21


def read(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique(rows, key):
    indexed = {r[key]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} in input")
    return indexed


def overlap(curves, eligible):
    reference = {(r["scorer"], r["group"]): r for r in read("analysis/v21_bounded/summary.csv")}
    details, summaries = [], []
    for scales, grid in ((v21.SCALES[:-1], "bounded"), (v21.SCALES, "full")):
        for scorer, label in (("spatial", "FM"), ("black", "MI")):
            all_supported, retained, clipped = [], [], []
            nonpositive = straddling = 0
            for case in sorted(curves):
                arms = [curves[case][a] for a in v21.ARMS]
                row = v21.integrate(arms, scorer, scales, eligible[case])
                status = row["status"]
                positive = None
                if status == "valid":
                    all_supported.append(row)
                    if row["hi"] <= 0:
                        status = "wholly_nonpositive"
                        nonpositive += 1
                    else:
                        retained.append(row)
                        positive = row
                        if row["lo"] < 0:
                            status = "straddles_zero"
                            straddling += 1
                            positive = v21.integrate(arms, scorer, scales, True, (0., row["hi"]))
                        else:
                            status = "nonnegative_interval"
                        clipped.append(positive)
                details.append(dict(settings=len(scales), scorer=label, source=case, status=status,
                                    lo=row["lo"], hi=row["hi"], default=row["default"],
                                    matched_full_interval=row["matched"],
                                    matched_positive_interval=positive["matched"] if positive else None))

            for rule, rows in (("all_supported", all_supported),
                               ("exclude_wholly_nonpositive", retained),
                               ("clip_interval_at_zero", clipped)):
                # Input order is ascending source ID. v21 pairs both means on each draw.
                result = v21.summarize(rows)
                stats = {k: result[k] for k in (
                    "matched_positive", "matched_zero", "matched_negative", "default", "default_lo",
                    "default_hi", "matched", "matched_lo", "matched_hi", "retained_fraction",
                    "retained_fraction_lo", "retained_fraction_hi")}
                if rule == "all_supported":
                    archived = reference[scorer, grid]
                    if (len(rows) != int(archived["n_valid"]) or
                            any(not math.isclose(v, float(archived[k]), abs_tol=1e-12, rel_tol=0)
                                for k, v in stats.items())):
                        raise ValueError(f"Failed to reproduce archived {grid} {label} summary")
                summaries.append(dict(settings=len(scales), scorer=label, rule=rule,
                                      n_total=len(curves), n_background=sum(eligible.values()),
                                      n_overlap=len(all_supported), n_hi_le_zero=nonpositive,
                                      n_straddling_zero=straddling, n_analyzed=len(rows), **stats))
    return details, summaries


def sam_checks(eligible):
    roster = unique(read("SAM_scan/data/sources.csv"), "case")
    if {case: int(r["preservation_pixels"]) > 0 for case, r in roster.items()} != eligible:
        raise ValueError("Source roster disagrees with harmonized score eligibility")
    measures = unique(read("cpu/harmonized_lpips/sam_core_scores.csv"), "id")
    core = [r for r in read("SAM_scan/data/scores.csv") if r["stage"] == "core"]
    all_ids = sorted(eligible)
    bg_ids = [i for i in all_ids if eligible[i]]
    cohorts = {"background": {}, "all": {}}
    losses = {}
    for arm in ("sam_native", "uniform", "two_constant", "uniform_at_local", "spatial_at_uniform"):
        cases = unique([r for r in core if r["method"] == arm], "case")
        if cases.keys() != roster.keys():
            raise ValueError(f"Incomplete core records for {arm}")
        for cohort, ids in (("background", bg_ids), ("all", all_ids)):
            rows = [cases[i] for i in ids]
            entry = dict(n=len(rows), directional_clip=mean(float(r["clip_directional"]) for r in rows),
                         edit_mae=mean(float(r["mae_edit"]) for r in rows))
            if cohort == "background":
                losses[arm] = np.array([float(measures[r["id"]]["spatial"]) for r in rows])
                entry.update(background_fm=float(losses[arm].mean()),
                             edit_to_background_mae=mean(float(r["mae_edit"])/float(r["mae_preservation"])
                                                         for r in rows))
            cohorts[cohort][arm] = entry
    default = losses["uniform"] - losses["sam_native"]
    gain = losses["uniform"] - losses["two_constant"]
    # Native-default recovery has its own archived bootstrap in reproduction/reproduce.py.
    ix = np.random.default_rng(1208).integers(len(bg_ids), size=(20000, len(bg_ids)))
    draws = gain[ix].mean(axis=1) / default[ix].mean(axis=1)
    return dict(operating_points=cohorts, background_fm_default_U_minus_S=default.mean(),
                background_fm_U_minus_two_constants=gain.mean(),
                background_fm_two_constant_recovery=gain.mean()/default.mean(),
                recovery_ci95=np.quantile(draws, (.025, .975)).tolist())


def restoring_checks():
    folder = Path("source_correction_and_blending")
    amplitudes = unique(read(folder / "scalar_profile_diagnostics.csv"), "case")
    scores = unique(read(folder / "two_seed_per_output.csv"), "id")
    cases = sorted(amplitudes)
    if len(cases) != 40 or set(cases) != {r["case"] for r in scores.values()}:
        raise ValueError("Inconsistent restoring source roster")
    comparisons = {}
    for condition, spatial, uniform in (("default", "REQ", "U25"), ("at_U", "REQ_U", "U25"),
                                        ("at_S", "REQ", "GLB_R")):
        gaps = []
        for case in cases:
            def loss(seed, arm):
                return float(scores[f"R88_pie_{case}_s{seed}_{arm}"]["lpips_preserved"])
            gaps.append(mean(loss(seed, spatial) - loss(seed, uniform) for seed in (0, 17)))
        comparisons[condition] = dict(mean_S_minus_U=mean(gaps), spatial_wins=sum(g < 0 for g in gaps),
                                      uniform_wins=sum(g > 0 for g in gaps), ties=gaps.count(0))
    # These diagnostics already average the two native trajectory integrals within source.
    s = mean(float(r["native_I_R"]) for r in amplitudes.values())
    u = mean(float(r["native_I_U"]) for r in amplitudes.values())
    flower, = [r for r in read("analysis/v21_transfer/source_inputs.csv")
               if (r["operator"], r["source"], r["seed"]) == ("correction", "000000000117", "0")]
    fs, fu = float(flower["spatial_magnitude"]), float(flower["uniform_magnitude"])
    return dict(sources=len(cases), seeds=[0, 17], background_fm=comparisons,
                cohort_native_integrals=dict(S=s, U=u, U_minus_S=u-s, relative_increase=u/s-1),
                flower_seed0_native_integrals=dict(source=flower["source"], S=fs, U=fu,
                                                   relative_increase=fu/fs-1))


def main():
    curves, eligible = v21.observations(ROOT)
    details, summaries = overlap(curves, eligible)
    checks = dict(sam=sam_checks(eligible), restoring=restoring_checks())
    v21.write(HERE, "per_source.csv", details)
    v21.write(HERE, "summary.csv", summaries)
    (HERE / "cross_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
