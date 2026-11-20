"""Paired source analysis; no models, GPU, graphs or network.

  python analyze.py --metrics outputs/evaluation/metrics.csv --baseline sam_native

Match method and baseline on exact (case, seed). For each metric, subtract
baseline from method within each matched seed, then average differences within
source case. Bootstrap independent SOURCE cases, not seeds/jobs, with replacement
(default 20,000 resamples). The 95% percentile CI is emitted only for >=2 sources.
All effects are method-minus-baseline: lower MAE/LPIPS, higher CLIP is favorable.
MAE/LPIPS measure change from source, so edit-region values are change diagnostics.
Missing/empty metrics are omitted pairwise and counted; never impute zero.
Methods, baseline, eligible counts and unmatched coverage are exported explicitly.
All eligible effects include failed BF16 realized-state matches. FP32 projection
matching (1e-6+1e-4*target) and parent-reported realized-state matching after BF16
cast (1e-6+0.01*target) have separate pass/fail/missing coverage. Matching flags
never remove observations. Projection mean-RMS and Euler velocity dose remain
separate diagnostics, not interchangeable measures or fabricated CPU effects.
Outputs: analysis.json, summary.csv, paired_differences.csv beside input by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEFAULT_METRICS = ("mae_full", "mae_edit", "mae_preservation", "lpips_full", "lpips_edit",
                   "lpips_preservation", "clip_directional", "clip_target", "clip_image")
REPLAY_ARMS = {"uniform_replay", "spatial_replay"}


def flag(row, name):
    raw = row.get(name)
    if raw is True or str(raw).lower() == "true":
        return True
    if raw is False or str(raw).lower() == "false":
        return False
    return None


def match_status(rows, stage):
    relevant = [row for row in rows if row.get("arm") in REPLAY_ARMS]
    if not relevant:
        return "not_applicable"
    suffix = "fp32_match_passed" if stage == "fp32" else "realized_state_match_passed"
    flags = [flag(row, f"profile_{branch}_{suffix}") for row in relevant for branch in ("model", "visible")]
    return "fail" if False in flags else "missing" if None in flags else "pass"


def matching_coverage(rows):
    records = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        record = dict(method=method, n_jobs=len(subset), n_sources=len({r["case"] for r in subset}),
                      evaluated_jobs=sum(row.get("eval_status") == "ok" for row in subset))
        for stage in ("fp32", "realized_state"):
            states = [match_status([row], stage) for row in subset]
            for state in ("pass", "fail", "missing", "not_applicable"):
                record[f"{stage}_{state}_jobs"] = states.count(state)
            applicable = sum(s != "not_applicable" for s in states)
            measured = states.count("pass") + states.count("fail")
            record[f"{stage}_applicable_jobs"] = applicable
            record[f"{stage}_measured_jobs"] = measured
            record[f"{stage}_pass_fraction_measured"] = states.count("pass") / measured if measured else None
            record[f"{stage}_measured_fraction_applicable"] = measured / applicable if applicable else None
            suffix = "fp32_match_passed" if stage == "fp32" else "realized_state_match_passed"
            replay = [row for row in subset if row.get("arm") in REPLAY_ARMS]
            for branch in ("model", "visible"):
                flags = [flag(row, f"profile_{branch}_{suffix}") for row in replay]
                record[f"{stage}_{branch}_pass_jobs"] = flags.count(True)
                record[f"{stage}_{branch}_fail_jobs"] = flags.count(False)
                record[f"{stage}_{branch}_missing_jobs"] = flags.count(None)
        records.append(record)
    return records


def profile_diagnostics(rows):
    """Export reported branch means and dose separately, without inventing estimates."""
    out = []
    for row in rows:
        data = {key: row.get(key) for key in ("id", "case", "seed", "method", "arm", "scale", "n_max", "eval_status")}
        # Keep all parent profile flags, shape errors and quantization diagnostics.
        data.update({key: val for key, val in row.items() if key.startswith("profile_")})
        out.append(data)
    return out


def value(row, metric):
    raw = row.get(metric)
    if raw is None or raw == "":
        return None
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {metric} for {row.get('id')}; expected empty/null")
    return result


def load_rows(path):
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload["rows"] if isinstance(payload, dict) else payload
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No metric rows")
    seen = set()
    categories = {}
    for row in rows:
        if not isinstance(row.get("case"), str) or not row["case"] or not row.get("method"):
            raise ValueError("Every metric row requires string case and method")
        raw_seed = row.get("seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, str)):
            raise ValueError("Every row requires an integer seed")
        seed = int(raw_seed)
        if str(seed) != str(raw_seed).strip():
            raise ValueError("Seed must be an exact integer")
        row["seed"] = seed
        key = (row["case"], seed, row["method"])
        if key in seen:
            raise ValueError(f"Duplicate paired cell: {key}")
        seen.add(key)
        category = str(row.get("category", ""))
        if row["case"] in categories and categories[row["case"]] != category:
            raise ValueError("Category differs between methods for source " + row["case"])
        categories[row["case"]] = category
    return rows


def bootstrap_ci(differences, resamples, seed):
    if len(differences) < 2:
        return None, None
    import numpy as np
    data = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    # Bounded memory even when a caller requests many resamples.
    batch_size = max(1, min(1024, 1_000_000 // len(data)))
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        means[start:start + count] = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def paired_summary(rows, baseline, method, metric, group, resamples, rng_seed):
    selected = [row for row in rows if group == "all" or str(row.get("category", "")) == group]
    ref = {(r["case"], r["seed"]): r for r in selected if r["method"] == baseline}
    alt = {(r["case"], r["seed"]): r for r in selected if r["method"] == method}
    source_pairs = {}
    excluded = []
    for key in sorted(ref.keys() & alt.keys()):
        a, b = ref[key], alt[key]
        if a.get("eval_status") != "ok" or b.get("eval_status") != "ok":
            excluded.append(dict(case=key[0], seed=key[1], reason="evaluation_not_ok"))
            continue
        av, bv = value(a, metric), value(b, metric)
        if av is None or bv is None:
            excluded.append(dict(case=key[0], seed=key[1], reason="null_or_empty_metric"))
            continue
        delta = bv - av
        if not math.isfinite(delta):
            raise ValueError("Non-finite paired difference")
        source_pairs.setdefault(key[0], []).append(dict(seed=key[1], baseline=av, method=bv, difference=delta,
            fp32_matching=match_status([a, b], "fp32"), realized_state_matching=match_status([a, b], "realized_state")))
    details = []
    for case, pairs in sorted(source_pairs.items()):
        count = len(pairs)
        detail = dict(group=group, baseline=baseline, method=method, metric=metric, case=case,
            n_seed_pairs=count, seeds=[pair["seed"] for pair in pairs],
            baseline_mean=math.fsum(pair["baseline"] for pair in pairs) / count,
            method_mean=math.fsum(pair["method"] for pair in pairs) / count,
            difference=math.fsum(pair["difference"] for pair in pairs) / count)
        for stage in ("fp32", "realized_state"):
            for state in ("pass", "fail", "missing", "not_applicable"):
                detail[f"{stage}_{state}_pairs"] = sum(pair[stage + "_matching"] == state for pair in pairs)
        details.append(detail)
    differences = [row["difference"] for row in details]
    # Stable RNG stream regardless of method/metric loop order.
    stream = int.from_bytes(hashlib.sha256(json.dumps([rng_seed, baseline, method, metric, group]).encode()).digest()[:8], "big")
    low, high = bootstrap_ci(differences, resamples, stream)
    n = len(details)
    summary = dict(group=group, baseline=baseline, method=method, metric=metric,
        difference_definition="method_minus_baseline", bootstrap_unit="source_case",
        interpretation="source_change_diagnostic" if metric in {"mae_edit", "lpips_edit"} else (
            "higher_is_better" if metric.startswith("clip_") else "lower_source_difference"),
        n_sources=n, n_seed_pairs=sum(d["n_seed_pairs"] for d in details),
        baseline_source_mean=math.fsum(d["baseline_mean"] for d in details) / n if n else None,
        method_source_mean=math.fsum(d["method_mean"] for d in details) / n if n else None,
        mean_difference=math.fsum(differences) / n if n else None,
        ci95_low=low, ci95_high=high, bootstrap_resamples=resamples if n >= 2 else 0,
        ci_status="ok" if n >= 2 else "insufficient_sources" if n == 1 else "no_eligible_pairs",
        baseline_cells=len(ref), method_cells=len(alt), matched_cells=len(ref.keys() & alt.keys()),
        baseline_only_cells=len(ref.keys() - alt.keys()), method_only_cells=len(alt.keys() - ref.keys()),
        excluded_matched_cells=len(excluded))
    for stage in ("fp32", "realized_state"):
        for state in ("pass", "fail", "missing", "not_applicable"):
            name = f"{stage}_{state}_pairs"
            summary[name] = sum(detail[name] for detail in details)
    coverage = dict(group=group, baseline=baseline, method=method, metric=metric,
        baseline_only=[dict(case=k[0], seed=k[1]) for k in sorted(ref.keys() - alt.keys())],
        method_only=[dict(case=k[0], seed=k[1]) for k in sorted(alt.keys() - ref.keys())], excluded=excluded)
    return summary, details, coverage


def write_csv(path, rows, fallback_fields):
    fields = list(rows[0]) if rows else fallback_fields
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.csv or metrics.json from evaluate.py")
    parser.add_argument("--baseline", "--reference", required=True, help="Exact runner method/arm name")
    parser.add_argument("--methods", nargs="+", help="Default: every method other than baseline")
    parser.add_argument("--metric-names", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap", "--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed, not generation seed")
    parser.add_argument("--by-category", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("Bootstrap resamples must be >=2; default is 20000")
    rows = load_rows(args.metrics)
    available = {row["method"] for row in rows}
    if args.baseline not in available:
        parser.error("Baseline not found; available methods: " + ", ".join(sorted(available)))
    methods = args.methods or sorted(available - {args.baseline})
    if not methods or any(method not in available or method == args.baseline for method in methods):
        parser.error("Select at least one non-baseline method present in the metrics")
    if len(set(methods)) != len(methods) or len(set(args.metric_names)) != len(args.metric_names):
        parser.error("Methods and metric names must be unique")
    for metric in args.metric_names:
        if not any(metric in row for row in rows):
            parser.error("Unknown metric column: " + metric)
    groups = ["all"]
    if args.by_category:
        groups += sorted({str(row.get("category", "")) for row in rows})
    summaries, details, coverage = [], [], []
    for group in groups:
        for method in methods:
            for metric in args.metric_names:
                summary, pairs, excluded = paired_summary(rows, args.baseline, method, metric, group, args.bootstrap, args.seed)
                summaries.append(summary)
                details.extend(pairs)
                coverage.append(excluded)
    output = args.output_dir or args.metrics.resolve().parent
    payload = dict(schema="doselens-paired-source-bootstrap-v1", input=str(args.metrics.resolve()),
        baseline=args.baseline, methods=methods, bootstrap_resamples=args.bootstrap, rng_seed=args.seed,
        confidence=0.95, interval="two-sided percentile; numpy linear quantile", unit="source_case",
        pairing="Exact case+seed; method minus baseline per seed; equal-weight within-source mean, then equal-weight sources",
        missing="Pairwise complete per metric; empty region / zero direction / evaluation failures excluded, never zero-filled",
        ci_requirement="At least two independent source cases; otherwise null CI",
        multiple_comparisons="Intervals are pointwise, with no multiplicity adjustment",
        primary_metric="lpips_preservation; supplied annotations have 42 eligible source cases and 8 empty complements",
        matching_policy="All eligible metric pairs retained, including failed realized-state matching; flags never filter effects",
        controlled_estimand="FP32 source-relative model/visible projection norms; tolerance 1e-6 + 1e-4 * target per step",
        realized_state="Parent-reported BF16-cast state matching flags; diagnostic tolerance 1e-6 + 0.01 * target; distinct from controlled estimand",
        profile_units="model_mean_rms and visible_mean_rms are active-step projection displacement means; euler_dose is summed |delta_sigma|*velocity_RMS",
        matching_coverage=matching_coverage(rows), profile_diagnostics=profile_diagnostics(rows),
        summaries=summaries, coverage=coverage)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(serialized + "\n", encoding="utf-8")
    write_csv(output / "summary.csv", summaries, ["method", "metric", "n_sources"])
    write_csv(output / "paired_differences.csv", details, ["group", "baseline", "method", "metric", "case", "difference"])
    print(f"Wrote {len(summaries)} paired summaries to {output}; no models or graphs loaded.")


if __name__ == "__main__":
    main()
