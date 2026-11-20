"""R88 CPU-only scalar analysis (NumPy); no generation or image scoring.

python analyze.py --results runtime_base/results
Optional: --plan runtime_base/plan.json --output path/to/evidence
The adjacent stats_plan.json fixes inference, cohorts and category allocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ("REQ", "GLB_R", "U25", "REQ_U", "SHUF_R", "REV_R")
METRICS = ("lpips_preserved", "clip_target_similarity", "clip_directional_similarity",
           "dino_full_feature_similarity", "lpips_full", "bg_mae")
SEEDS = (0, 17)
FOUR_ARMS = ("REQ", "GLB_R", "U25", "REQ_U")
STATUS_FIELDS = {
    "lpips_preserved": ("lpips_status", "lpips_preserved_status"),
    "lpips_full": ("lpips_status",),
    "clip_target_similarity": ("clip_status",),
    "clip_directional_similarity": ("clip_status", "clip_directional_status"),
    "dino_full_feature_similarity": ("dino_status",),
    "bg_mae": ("metrics_status", "bg_mae_status"),
}


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def category(task):
    for key in ("target_category_id", "category_id", "category"):
        if task.get(key) is not None and str(task[key]).strip():
            return str(task[key]).strip()
    raise ValueError(f"Source {task.get('case')} has no explicit category")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing job id column: {path}")
        rows = {}
        for row in reader:
            ident = row["id"].strip()
            if not ident or ident in rows:
                raise ValueError(f"Empty or duplicate job id {ident!r}: {path}")
            rows[ident] = row
        return rows, set(reader.fieldnames)


def load_inputs(results, plan_path, spec):
    plan = read_json(plan_path)
    tasks = {}
    for task in plan.get("tasks", []):
        case = str(task["case"])
        if case in tasks:
            raise ValueError(f"Repeated source in generation plan: {case}")
        tasks[case] = task
    if len(tasks) != spec["population"]["sources"]:
        raise ValueError("R88 requires the frozen 40-source plan (not the R87 template)")
    categories = {case: category(task) for case, task in tasks.items()}
    counts = dict(Counter(categories.values()))
    expected = {str(k): int(v) for k, v in spec["population"]["category_counts"].items() if v}
    if counts != expected:
        raise ValueError(f"Category allocation mismatch: observed={counts}, frozen={expected}")
    jobs, index = {}, {}
    for original in plan.get("jobs", []):
        job = dict(original)
        job["case"], job["seed"] = str(job["case"]), int(job["seed"])
        ident = str(job["id"])
        key = (job["arm"], job["case"], job["seed"])
        if Path(ident).name != ident or any(c in ident for c in ("/", "\\")):
            raise ValueError(f"Job id must be a folder basename: {ident}")
        if (ident in jobs or key in index or job["arm"] not in ARMS
                or job["case"] not in tasks or job["seed"] not in SEEDS):
            raise ValueError(f"Unexpected/duplicate R88 job: {ident}")
        if job.get("target_category_id") is not None and category(job) != categories[job["case"]]:
            raise ValueError(f"Job/task category mismatch: {ident}")
        jobs[ident], index[key] = job, ident
    required = {(arm, case, seed) for arm in ARMS for case in tasks for seed in SEEDS}
    if set(index) != required:
        raise ValueError(f"Incomplete generation plan: missing {len(required - set(index))} jobs")
    measured, _ = read_csv(results / "metrics.csv")
    evaluated, evaluation_fields = read_csv(results / "evaluation.csv")
    merged = {}
    for ident in sorted(set(measured) | set(evaluated)):
        left, right = measured.get(ident, {}), evaluated.get(ident, {})
        row = dict(left)
        for key, value in right.items():
            if key in METRICS or value not in (None, ""):
                row[key] = value
        # An explicitly empty evaluation score is missing, not a stale-score fallback.
        if ident in evaluated:
            for metric in METRICS:
                if metric in evaluation_fields:
                    row[metric] = right.get(metric, "")
        if ident in jobs:
            job = jobs[ident]
            for source in (left, right):
                for key in ("arm", "case", "seed"):
                    value = source.get(key)
                    if value in (None, ""):
                        continue
                    actual = str(int(value)) if key == "seed" else str(value)
                    if actual != str(job[key]):
                        raise ValueError(f"CSV/plan {key} mismatch for {ident}")
        merged[ident] = row
    metadata = {
        "plan": str(plan_path), "metrics_rows": len(measured), "evaluation_rows": len(evaluated),
        "merged_rows": len(merged), "evaluation_status_counts": dict(Counter(
            row.get("eval_status", "missing") for row in evaluated.values())),
        "missing_job_ids": sorted(set(jobs) - set(merged)),
        "unexpected_csv_ids": sorted(set(merged) - set(jobs)),
        "category_counts": counts,
        "selection_manifest": "Eligibility and the 90-source exclusion roster are established by selection; this analyzer checks category allocation, not image content.",
    }
    return tasks, categories, counts, jobs, index, merged, metadata


def load_records(results, jobs):
    records, issues = {}, {}
    for ident, job in jobs.items():
        path = results / ident / "record.json"
        problem = []
        try:
            record = read_json(path) if path.is_file() else {}
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except (OSError, ValueError) as exc:
            record = {}
            problem.append(f"record_unreadable: {exc}")
        if not record:
            problem.append("missing_record")
        elif record.get("status") != "completed":
            problem.append("record_not_completed")
        actual_job = record.get("job", {})
        for key in ("id", "case", "arm", "seed"):
            if record and (key not in actual_job or str(actual_job[key]) != str(job[key])):
                problem.append(f"record_job_{key}_mismatch")
        records[ident], issues[ident] = record, problem
    return records, issues


def outcome(row, metric, record):
    if not row:
        return None, "missing_csv_job"
    if row.get("status") not in (None, "", "completed", "ok"):
        return None, "generation_status:" + str(row["status"])
    if record and record.get("status") != "completed":
        return None, "record_not_completed"
    value = number(row.get(metric))
    if value is None:
        return None, "missing_or_nonfinite_metric"
    for field in STATUS_FIELDS[metric]:
        state = str(row.get(field, "")).strip().lower()
        if state and not (state.startswith("ok") or state in ("completed", "valid", "success")):
            return None, field + ":" + state
    if metric != "bg_mae" and str(row.get("eval_status", "")).lower() in ("failed", "error"):
        return None, "evaluation_failed"
    return value, None


def source_values(cases, categories, jobs, index, merged, records, record_issues):
    complete, failures, rows = {}, {}, []
    for arm in ARMS:
        for metric in METRICS:
            vals, missing = {}, {}
            for case in cases:
                seedvals, reasons = [], []
                for seed in SEEDS:
                    ident = index[arm, case, seed]
                    value, reason = outcome(merged.get(ident, {}), metric, records[ident])
                    if any("mismatch" in issue for issue in record_issues[ident]):
                        value, reason = None, "record_identity_mismatch"
                    seedvals.append(value)
                    if reason:
                        reasons.append({"seed": seed, "job_id": ident, "reason": reason})
                value = float(np.mean(seedvals)) if all(x is not None for x in seedvals) else None
                if value is not None:
                    vals[case] = value
                else:
                    missing[case] = reasons
                rows.append(dict(arm=arm, case=case, category=categories[case], metric=metric,
                                 seed0=seedvals[0], seed17=seedvals[1], mean=value, missing=reasons))
            complete[arm, metric], failures[arm, metric] = vals, missing
    return complete, failures, rows


class SourceBootstrap:
    """Fixed planned category mixture, paired source bootstrap with cached weights."""

    def __init__(self, cases, categories, counts, spec):
        self.cases, self.categories, self.counts, self.spec = cases, categories, counts, spec
        self.positions = {case: i for i, case in enumerate(cases)}
        self.total = sum(counts.values())

    def cohort_info(self, cohort):
        observed = Counter(self.categories[case] for case in cohort)
        return {
            "sources": len(cohort), "planned_sources": self.total,
            "category_counts": {cat: observed.get(cat, 0) for cat in sorted(self.counts)},
            "missing_sources": [case for case in self.cases if case not in cohort],
            "incomplete_categories": {cat: {"observed": observed.get(cat, 0), "planned": n}
                                      for cat, n in self.counts.items() if observed.get(cat, 0) != n},
            "empty_categories": [cat for cat in sorted(self.counts) if not observed.get(cat, 0)],
        }

    def point_weights(self, cohort):
        observed = Counter(self.categories[case] for case in cohort)
        if any(not observed.get(cat, 0) for cat in self.counts):
            return None
        return np.array([self.counts[self.categories[case]] / self.total
                         / observed[self.categories[case]] for case in cohort])

    @lru_cache(maxsize=8)
    def resample_weights(self, cohort):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
            self.spec["seed"], len(cohort), *[self.positions[c] + 1 for c in cohort]])))
        draws = self.spec["draws"]
        weights = np.zeros((draws, len(cohort)), dtype=np.float64)
        rows = np.arange(draws)[:, None]
        for cat, planned_n in sorted(self.counts.items()):
            candidates = np.array([i for i, case in enumerate(cohort) if self.categories[case] == cat])
            # A partial stratum retains its design weight, not fictitious extra observations.
            picks = rng.choice(candidates, size=(draws, len(candidates)), replace=True)
            np.add.at(weights, (rows, picks), planned_n / self.total / len(candidates))
        return weights

    def summarize(self, mapping):
        cohort = tuple(sorted(mapping))
        info = self.cohort_info(cohort)
        values = np.array([mapping[c] for c in cohort], dtype=np.float64)
        info["available_source_mean"] = float(values.mean()) if len(cohort) else None
        weights = self.point_weights(cohort)
        if weights is None:
            info.update(difference=None, ci95_low=None, ci95_high=None,
                        estimate_status="unestimable_empty_category")
        else:
            samples = self.resample_weights(cohort) @ values
            low, high = np.quantile(samples, [0.025, 0.975], method="linear")
            info.update(difference=float(weights @ values), ci95_low=float(low), ci95_high=float(high),
                        estimate_status="complete" if len(cohort) == self.total else "incomplete_strata")
        return info


def linear_values(complete, coefficients, metric, cases):
    return {case: sum(coef * complete[arm, metric][case] for arm, coef in coefficients.items())
            for case in cases if all(case in complete[arm, metric] for arm in coefficients)}


def make_contrasts(complete, failures, sampler, spec):
    rows = []
    for contrast in spec["contrasts"]:
        for metric in METRICS:
            values = linear_values(complete, contrast["coefficients"], metric, sampler.cases)
            role = contrast["preservation_role"] if metric == "lpips_preserved" else "sensitivity"
            missing = {case: {arm: failures[arm, metric].get(case, []) for arm in contrast["coefficients"]
                              if case not in complete[arm, metric]}
                       for case in sampler.cases if case not in values}
            rows.append(dict(comparison=contrast["id"], metric=metric, role=role,
                             coefficients=contrast["coefficients"], **sampler.summarize(values),
                             source_differences=values, missing_details=missing))
    return rows


def mc_signflip(rows, sampler, spec):
    usable = [row for row in rows if row["difference"] is not None]
    for row in rows:
        row.update(p_raw=None, p_mc=None, mcse=None, mc_extreme=None, mc_draws=spec["draws"],
                   mc_seed=spec["seed"], p_resolution=1.0 / (spec["draws"] + 1))
    if not usable:
        return
    matrix = np.zeros((len(sampler.cases), len(usable)), dtype=np.float64)
    for j, row in enumerate(usable):
        cohort = tuple(sorted(row["source_differences"]))
        weights = sampler.point_weights(cohort)
        for case, weight in zip(cohort, weights):
            matrix[sampler.positions[case], j] = row["source_differences"][case] * weight
    threshold = np.abs(matrix.sum(axis=0)) - spec["tie_tolerance"]
    extreme = np.zeros(len(usable), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    for start in range(0, spec["draws"], spec["batch_size"]):
        batch = min(spec["batch_size"], spec["draws"] - start)
        signs = rng.integers(0, 2, size=(batch, len(sampler.cases)), dtype=np.int8).astype(np.float64)
        signs *= 2.0
        signs -= 1.0
        extreme += np.count_nonzero(np.abs(signs @ matrix) >= threshold, axis=0)
    for row, count in zip(usable, extreme):
        pval = (int(count) + 1) / (spec["draws"] + 1)
        row.update(p_raw=pval, p_mc=pval, mc_extreme=int(count),
                   mcse=math.sqrt(pval * (1 - pval) / spec["draws"]))


def holm(rows, field, family_size):
    if len(rows) != family_size:
        raise ValueError(f"Expected {family_size} planned tests for {field}, got {len(rows)}")
    ordered = sorted(rows, key=lambda row: 1.0 if row["p_raw"] is None else row["p_raw"])
    previous = 0.0
    for rank, row in enumerate(ordered):
        pval = 1.0 if row["p_raw"] is None else row["p_raw"]
        previous = max(previous, min(1.0, (family_size - rank) * pval))
        row[field] = previous if row["p_raw"] is not None else None


def adjust(rows, spec):
    primary = next(row for row in rows if row["role"] == "primary")
    secondary = [row for row in rows if row["role"] == "secondary"]
    holm(secondary, "p_holm_secondary4", 4)
    holm(rows, "p_holm_all30", 30)
    alpha = spec["primary"]["alpha"]
    gate = primary["p_raw"] is not None and primary["p_raw"] <= alpha
    for row in rows:
        row.setdefault("p_holm_secondary4", None)
        row["primary_gate_open"] = gate
        if row["role"] == "primary":
            row.update(inference_role="primary", p_adjusted=row["p_raw"], adjusted_family_size=1,
                       p_gate_adjusted=row["p_raw"], confirmatory_reject=gate)
        elif row["role"] == "secondary":
            adj = row["p_holm_secondary4"]
            gated = max(primary["p_raw"], adj) if primary["p_raw"] is not None and adj is not None else None
            row.update(inference_role="confirmatory_secondary" if gate else "exploratory_gate_closed",
                       p_adjusted=adj, adjusted_family_size=4, p_gate_adjusted=gated,
                       confirmatory_reject=bool(gate and adj is not None and adj <= alpha))
        else:
            row.update(inference_role="exploratory_sensitivity", p_adjusted=row["p_holm_all30"],
                       adjusted_family_size=30, p_gate_adjusted=None, confirmatory_reject=False)
    return {"open": gate, "criterion": "primary raw two-sided p <= alpha",
            "primary_p": primary["p_raw"], "alpha": alpha,
            "primary_supports_lower_preservation_lpips": bool(gate and primary["difference"] < 0),
            "holm30_changes_gate": False}


def win_rows(rows, spec):
    output = []
    for row in rows:
        vals = np.array(list(row["source_differences"].values()))
        interaction = row["comparison"] == "interaction"
        gain = vals if spec["metric_directions"][row["metric"]] == "higher" else -vals
        for threshold in spec["source_wins"]["thresholds"]:
            wins = int(np.count_nonzero(gain > 0 if threshold == 0 else gain >= threshold))
            output.append(dict(comparison=row["comparison"], metric=row["metric"],
                               interpretation="signed_interaction_threshold_not_arm_win" if interaction else "metric_oriented_improvement",
                               threshold=threshold, comparison_operator=">" if threshold == 0 else ">=",
                               sources=len(vals), planned_sources=40, count=wins,
                               fraction_available=wins / len(vals) if len(vals) else None,
                               fraction_planned=wins / 40,
                               lower_difference_sources=int(np.count_nonzero(vals < 0)),
                               higher_difference_sources=int(np.count_nonzero(vals > 0)),
                               tied_sources=int(np.count_nonzero(vals == 0)),
                               median_difference=float(np.median(vals)) if len(vals) else None,
                               missing_sources=row["missing_sources"]))
    return output


def native_decomposition(complete, sampler):
    definitions = {
        "native_REQ-U25": {"REQ": 1, "U25": -1},
        "path1_direction_REQ-GLB_R": {"REQ": 1, "GLB_R": -1},
        "path1_profile_GLB_R-U25": {"GLB_R": 1, "U25": -1},
        "path2_profile_REQ-REQ_U": {"REQ": 1, "REQ_U": -1},
        "path2_direction_REQ_U-U25": {"REQ_U": 1, "U25": -1},
        "profile_request_UminusR": {"REQ_U": 1, "REQ": -1},
        "profile_global_UminusR": {"U25": 1, "GLB_R": -1},
        "interaction": {"REQ_U": 1, "U25": -1, "REQ": -1, "GLB_R": 1},
    }
    rows, checks = [], []
    for metric in METRICS:
        cohort = [case for case in sampler.cases if all(case in complete[arm, metric] for arm in FOUR_ARMS)]
        components = {}
        for name, coef in definitions.items():
            values = linear_values(complete, coef, metric, cohort)
            components[name] = values
            rows.append(dict(metric=metric, component=name, coefficients=coef,
                             inference="pointwise_CI_only_common_four_arm_cohort",
                             **sampler.summarize(values), source_differences=values))
        for path, left, right in (
            ("path1", "path1_direction_REQ-GLB_R", "path1_profile_GLB_R-U25"),
            ("path2", "path2_profile_REQ-REQ_U", "path2_direction_REQ_U-U25"),
        ):
            total = {case: components[left][case] + components[right][case] for case in cohort}
            rows.append(dict(metric=metric, component=path + "_sum", coefficients=definitions["native_REQ-U25"],
                             inference="pointwise_CI_only_common_four_arm_cohort",
                             **sampler.summarize(total), source_differences=total))
            checks.append(dict(metric=metric, path=path, sources=len(cohort),
                               max_abs_identity_error=max((abs(total[c] - components["native_REQ-U25"][c])
                                                           for c in cohort), default=None)))
    return rows, checks


def extract_steps(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("steps"), list):
        return obj["steps"]
    for key in ("edit", "engine_stats", "diagnostics", "step_audit"):
        child = obj.get(key)
        if isinstance(child, dict) and isinstance(child.get("steps"), list):
            return child["steps"]
    return None


def load_step_data(results, jobs, records, spec):
    loaded = {}
    for ident in jobs:
        steps = extract_steps(records[ident])
        origin, errors = "record.json", []
        if steps is None:
            for name in ("steps.json", "step_summary.json"):
                path = results / ident / name
                if path.is_file():
                    try:
                        steps = extract_steps(read_json(path))
                        origin = name
                    except (OSError, ValueError) as exc:
                        errors.append(f"{name}: {exc}")
                    if steps is not None:
                        break
        steps = [] if steps is None else steps
        if len(steps) != spec["steps"]:
            errors.append(f"step_count:{len(steps)}")
        normalized = []
        for position, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"invalid_step:{position}")
                continue
            actual = number(step.get("actual_rms"))
            if actual is None:
                actual = number(step.get("applied_correction_rms"))
            if actual is None:
                actual = number(step.get("applied_correction_global_rms"))
            h, sigma = number(step.get("h")), number(step.get("sigma"))
            if (step.get("step_index") != position or actual is None or actual < 0
                    or h is None or h <= 0 or sigma is None):
                errors.append(f"invalid_values_or_index:{position}")
            normalized.append(dict(step_index=position, actual_rms=actual, h=h, sigma=sigma,
                                   recorded_target_rms=number(step.get("r86_target_rms"))))
        loaded[ident] = dict(steps=normalized, errors=errors, origin=origin)
    return loaded


def dose_audit(results, jobs, index, records, record_issues, categories, spec):
    loaded = load_step_data(results, jobs, records, spec)
    audits, step_rows, profiles = [], [], {}
    for ident, job in jobs.items():
        record, own = records[ident], loaded[ident]
        expected_arm = spec["anchor_arms"][job["arm"]]
        anchor_id = index[expected_arm, job["case"], job["seed"]]
        anchor = loaded[anchor_id]
        issues = list(record_issues[ident]) + list(own["errors"])
        own_complete = not issues
        if anchor_id != ident:
            if anchor["errors"] or record_issues[anchor_id]:
                issues.append("anchor_incomplete")
            for declared in (job.get("r86_anchor"), record.get("job", {}).get("r86_anchor")):
                if declared != anchor_id:
                    issues.append("anchor_reference_mismatch")
            if (record.get("sigma_sha256") and records[anchor_id].get("sigma_sha256")
                    and record["sigma_sha256"] != records[anchor_id]["sigma_sha256"]):
                issues.append("schedule_fingerprint_mismatch")
        cumulative, target_cumulative = 0.0, 0.0
        prefix_complete = True
        abs_errors, rel_errors, weighted_errors = [], [], []
        for i, step in enumerate(own["steps"]):
            ref = anchor["steps"][i] if i < len(anchor["steps"]) else {}
            actual, target, h = step["actual_rms"], ref.get("actual_rms"), step["h"]
            valid = actual is not None and actual >= 0 and h is not None and h > 0
            prefix_complete = prefix_complete and valid
            contribution = h * actual if valid else None
            if valid:
                cumulative += contribution
            target_valid = target is not None and ref.get("h") is not None and ref["h"] > 0
            if target_valid:
                target_cumulative += ref["h"] * target
            error = abs(actual - target) if actual is not None and target is not None else None
            relative = error / target if error is not None and target > 0 else None
            if error is not None:
                abs_errors.append(error)
                if relative is not None:
                    rel_errors.append(relative)
                if h is not None:
                    weighted_errors.append(abs(h) * error)
                if error > spec["absolute_tolerance"] + spec["relative_tolerance"] * abs(target):
                    issues.append(f"rms_mismatch:{i}")
            if h is not None and ref.get("h") is not None and abs(h - ref["h"]) > spec["schedule_tolerance"]:
                issues.append(f"step_size_mismatch:{i}")
            if step["sigma"] is not None and ref.get("sigma") is not None and abs(step["sigma"] - ref["sigma"]) > spec["schedule_tolerance"]:
                issues.append(f"sigma_mismatch:{i}")
            declared_target = step["recorded_target_rms"]
            if declared_target is not None and target is not None and not math.isclose(
                    declared_target, target, rel_tol=spec["relative_tolerance"], abs_tol=spec["absolute_tolerance"]):
                issues.append(f"recorded_target_mismatch:{i}")
            step_rows.append(dict(id=ident, arm=job["arm"], case=job["case"], category=categories[job["case"]],
                                  seed=job["seed"], anchor_id=anchor_id, profile_origin=own["origin"],
                                  step=i, actual_rms=actual, target_rms=target, h=h, sigma=step["sigma"],
                                  h_weighted_rms=contribution,
                                  cumulative_I=cumulative if prefix_complete else None,
                                  partial_cumulative_I=cumulative,
                                  absolute_error=error, relative_error=relative,
                                  weighted_absolute_error=abs(h) * error if h is not None and error is not None else None))
        full_I = cumulative if own_complete else None
        stored_I = number(record.get("correction_impulse"))
        if full_I is not None and stored_I is not None and not math.isclose(
                full_I, stored_I, rel_tol=spec["relative_tolerance"], abs_tol=spec["absolute_tolerance"]):
            issues.append("stored_impulse_mismatch")
        anchor_complete = not anchor["errors"] and not record_issues[anchor_id]
        anchor_I = target_cumulative if anchor_complete else None
        if own_complete:
            rms = np.array([s["actual_rms"] for s in own["steps"]])
            hs = np.array([s["h"] for s in own["steps"]])
            profiles[job["arm"], job["case"], job["seed"]] = {
                "I": full_I, "rms": rms, "h": hs, "dose": rms * hs, "cumulative": np.cumsum(rms * hs)}
        audits.append(dict(id=ident, arm=job["arm"], case=job["case"], seed=job["seed"],
                           category=categories[job["case"]], anchor_id=anchor_id, steps=len(own["steps"]),
                           status="ok" if not issues else "issue", issues=sorted(set(issues)),
                           I=full_I, partial_I=cumulative, stored_I=stored_I, anchor_I=anchor_I,
                           I_minus_anchor=full_I - anchor_I if full_I is not None and anchor_I is not None else None,
                           relative_I_error=abs(full_I - anchor_I) / abs(anchor_I)
                           if full_I is not None and anchor_I not in (None, 0) else None,
                           max_absolute_error=max(abs_errors, default=None),
                           max_relative_error=max(rel_errors, default=None),
                           max_weighted_absolute_error=max(weighted_errors, default=None)))
    return audits, step_rows, profiles


def dose_summaries(profiles, sampler, spec):
    source_rows, summaries, means = [], [], {}
    for arm in ARMS:
        for case in sampler.cases:
            pair = [profiles.get((arm, case, seed)) for seed in SEEDS]
            if not all(p is not None for p in pair):
                source_rows.append(dict(arm=arm, case=case, category=sampler.categories[case],
                                        quantity="I", step=None, seed0=pair[0]["I"] if pair[0] else None,
                                        seed17=pair[1]["I"] if pair[1] else None, mean=None))
                continue
            mean = {name: (pair[0][name] + pair[1][name]) / 2 for name in pair[0]}
            means[arm, case] = mean
            for quantity in ("I", "rms", "dose", "cumulative"):
                for step in ([None] if quantity == "I" else range(spec["steps"])):
                    a = pair[0][quantity] if step is None else pair[0][quantity][step]
                    b = pair[1][quantity] if step is None else pair[1][quantity][step]
                    source_rows.append(dict(arm=arm, case=case, category=sampler.categories[case],
                                            quantity=quantity, step=step, seed0=float(a), seed17=float(b), mean=float((a + b) / 2)))
    for label in (*ARMS, "U25-REQ"):
        for quantity in ("I", "rms", "dose", "cumulative"):
            for step in ([None] if quantity == "I" else range(spec["steps"])):
                values = {}
                for case in sampler.cases:
                    arms = ("U25", "REQ") if label == "U25-REQ" else (label,)
                    if not all((arm, case) in means for arm in arms):
                        continue
                    vals = [means[arm, case][quantity] if step is None else means[arm, case][quantity][step] for arm in arms]
                    values[case] = float(vals[0] - vals[1]) if len(vals) == 2 else float(vals[0])
                summaries.append(dict(arm_or_comparison=label, quantity=quantity, step=step,
                                      interpretation="actual_measured_dose_pointwise_CI",
                                      **sampler.summarize(values)))
    return source_rows, summaries


def validate_spec(spec):
    expected_contrasts = [
        ("REQ_U-U25", {"REQ_U": 1, "U25": -1}),
        ("REQ-GLB_R", {"REQ": 1, "GLB_R": -1}),
        ("interaction", {"REQ_U": 1, "U25": -1, "REQ": -1, "GLB_R": 1}),
        ("REQ-SHUF_R", {"REQ": 1, "SHUF_R": -1}),
        ("REQ-REV_R", {"REQ": 1, "REV_R": -1}),
    ]
    if (tuple(spec["arms"]) != ARMS or tuple(spec["metrics"]) != METRICS
            or tuple(spec["seeds"]) != SEEDS or spec["signflip"]["draws"] != 1_000_000
            or spec["bootstrap"]["draws"] != 20_000
            or [(r["id"], r["coefficients"]) for r in spec["contrasts"]] != expected_contrasts
            or spec["primary"]["contrast"] != "REQ_U-U25" or spec["primary"]["metric"] != "lpips_preserved"):
        raise ValueError("stats_plan.json does not match the fixed R88 analysis contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    spec = read_json(ROOT / "stats_plan.json")
    validate_spec(spec)
    plan_path = args.plan
    if plan_path is None:
        plan_path = next((p for p in (results / "plan.json", results.parent / "plan.json") if p.is_file()), None)
    if plan_path is None:
        parser.error("No generation plan found; supply --plan path/to/plan.json")
    tasks, categories, counts, jobs, index, merged, inputs = load_inputs(results, plan_path, spec)
    cases = tuple(sorted(tasks))
    records, record_issues = load_records(results, jobs)
    complete, failures, source_rows = source_values(cases, categories, jobs, index, merged, records, record_issues)
    sampler = SourceBootstrap(cases, categories, counts, spec["bootstrap"])
    means = []
    for arm in ARMS:
        for metric in METRICS:
            summary = sampler.summarize(complete[arm, metric])
            summary["mean"] = summary.pop("difference")
            means.append(dict(arm=arm, metric=metric, **summary))
    paired = make_contrasts(complete, failures, sampler, spec)
    print("Running fixed 1,000,000-draw CPU sign flips for the 30 planned tests...", flush=True)
    mc_signflip(paired, sampler, spec["signflip"])
    gate = adjust(paired, spec)
    wins = win_rows(paired, spec)
    decomposition, identities = native_decomposition(complete, sampler)
    audits, step_rows, profiles = dose_audit(results, jobs, index, records, record_issues, categories, spec["dose_audit"])
    dose_source_rows, dose_summary = dose_summaries(profiles, sampler, spec["dose_audit"])
    output = (args.output or results / "evidence").resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_ready = all(row["status"] == "ok" for row in audits)
    result = dict(schema="r88-analysis-closure-results-v1", stats_plan=spec,
                  numpy_version=np.__version__, input_summary=inputs, source_order=list(cases),
                  planned_outputs=len(jobs), planned_sources=len(cases), category_counts=counts,
                  primary_gate=gate, arm_means=means, paired=paired, source_wins=wins,
                  native_decomposition=decomposition, decomposition_identities=identities,
                  dose_audit=audits, dose_summary=dose_summary,
                  implementation_audit_pass=bool(audit_ready),
                  implementation_issue_jobs=[row["id"] for row in audits if row["status"] != "ok"],
                  interpretation="Gate labels describe the frozen statistical family. Implementation issues are reported separately; observed scores are retained. Missingness is contrast/metric-specific; intervals are pointwise; no semantic equivalence or human rating is inferred.")
    save_json(output / "r88_statistics.json", result)
    for name, rows in (("paired", paired), ("source_values", source_rows), ("arm_means", means),
                       ("source_wins", wins), ("native_decomposition", decomposition),
                       ("dose_audit", audits), ("actual_profiles", step_rows),
                       ("dose_source_values", dose_source_rows), ("dose_summary", dose_summary)):
        save_csv(output / ("r88_" + name + ".csv"), rows)
    primary = next(row for row in paired if row["role"] == "primary")
    print(json.dumps({"output": str(output), "primary": {key: primary[key] for key in
                      ("comparison", "metric", "sources", "difference", "ci95_low", "ci95_high", "p_raw", "mcse")},
                      "gate": gate, "implementation_audit_pass": audit_ready}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
