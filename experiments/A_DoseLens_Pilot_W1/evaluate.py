"""Offline CPU evaluation, after runner reports have been written.

Full evaluation (all paths must already exist locally):
  python evaluate.py --results results --clip-model /models/clip-vit-base-patch32 \
    --alexnet-weights /models/alexnet.pth --lpips-weights /models/alex-v0.1.pth
Equivalent environment: CLIP_MODEL_PATH (or CLIP_PATH), LPIPS_TRUNK_PATH,
LPIPS_PATH. --lpips-weights bundled explicitly selects the installed lpips
package's weights/v0.1/alex.pth; the pretrained AlexNet backbone is still required.
  python evaluate.py --results results --metrics mae
The latter explicitly requests only pixel metrics, with no substitute for LPIPS/CLIP.

Report contract: outputs/{job_id}/report.json (or record.json), status completed/ok/success,
case, seed, method (arm/variant/condition aliases supported), optional id/job_id.
Identity fields may be nested in job or in sibling job.json. The parent job_id
suffix after case__ is the condition label (e.g. flowedit_24, sam_native_scale0.5,
uniform_at_local); arm/scale/n_max are also retained. native.png is beside the
report, 512x512. Underscore-prefixed runner directories are excluded.
--reports accepts a JSON list, {reports:[...]}, or JSONL; each entry must have
report_path or job_dir, relative to that list file. Never infer completion from PNGs.

MAE: mean RGB absolute source/output difference in [0,1], divided by region mass.
LPIPS: AlexNet v0.1, whole-image scalar; regional variants mask RGB in [0,1]
to BLACK before conversion to [-1,1]. This is masked-RGB LPIPS, not cropped,
spatially averaged ROI LPIPS or region-area-normalized LPIPS. Receptive fields
cross boundaries. Both pretrained backbone and learned calibration are loaded.
Directional CLIP: cosine of differences of unit-normalized image/text embeddings,
ViT-B/32 only. Its local processor performs the model's normal 224px preprocessing;
source/output files and pixel/LPIPS evaluation remain at native 512px.
GT masks are read exclusively from evaluation_only during evaluation.
Empty regions and zero CLIP directions produce JSON null / empty CSV with status.
Primary preservation LPIPS is lpips_preservation: 42 annotation-eligible sources,
8 empty complements in the supplied 50-case pilot (counts exported from manifest).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

# Must precede torch/transformers imports. The only supported device is CPU.
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", HF_DATASETS_OFFLINE="1")
ROOT = Path(__file__).resolve().parent
METRICS = ("mae_full", "mae_edit", "mae_preservation", "lpips_full", "lpips_edit",
           "lpips_preservation", "clip_directional", "clip_target", "clip_image")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_file(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Metric produced a non-finite value")
    return value


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def reports_from(results, report_list=None):
    """Read only explicit runner reports; never load existing metrics or images here."""
    if report_list:
        report_list = report_list.resolve()
        if report_list.suffix.lower() == ".jsonl":
            entries = [json.loads(line) for line in report_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            entries = read_json(report_list)
            if isinstance(entries, dict):
                entries = entries["reports"]
        for entry in entries:
            if "report_path" in entry:
                path = resolve(report_list.parent, entry["report_path"]).resolve()
                yield path, read_json(path)
            elif "job_dir" in entry:
                path = resolve(report_list.parent, entry["job_dir"]).resolve() / "report.json"
                yield path, entry
            else:
                raise ValueError("Report list entry requires report_path or job_dir")
    else:
        paths = sorted(set(results.rglob("report.json")) | set(results.rglob("record.json")))
        # Prefer report.json when both representations are present for one job.
        for path in paths:
            if any(part.startswith("_") for part in path.relative_to(results).parts[:-1]):
                continue
            if path.name == "record.json" and path.with_name("report.json").exists():
                continue
            yield path, read_json(path)


def condition_label(info):
    case, job_id = str(info.get("case", "")), str(info.get("job_id", ""))
    if case and job_id.startswith(case + "__"):
        return job_id[len(case) + 2:]
    if info.get("method"):
        return info["method"]
    arm = info.get("arm", info.get("variant", info.get("condition")))
    if arm == "flowedit" and info.get("n_max") is not None:
        return f"flowedit_{int(info['n_max'])}"
    if arm and info.get("scale", 1) != 1:
        return f"{arm}_scale{finite(info['scale']):g}"
    return arm


def profile_columns(info):
    """Keep projection displacement, velocity dose and matching errors distinct."""
    result = {}
    invalid = []
    profile = info.get("profile") or {}
    if not isinstance(profile, dict):
        return {"profile_status": "invalid_summary"}
    for key, val in profile.items():
        if isinstance(val, float) and not math.isfinite(val):
            invalid.append(key)
            val = None
        if val is None or isinstance(val, (str, bool, int, float)):
            result["profile_" + key] = val
        elif isinstance(val, list) and all(v is None or isinstance(v, (int, float)) and math.isfinite(v) for v in val):
            result["profile_" + key] = json.dumps(val, allow_nan=False)
    result["profile_status"] = "nonfinite_fields" if invalid else "ok" if profile else "not_applicable"
    result["profile_nonfinite_fields"] = ",".join(invalid)
    return result


def fp32_matching_columns(info, directory):
    """Read measured step logs after generation; BF16 diagnostics stay parent-owned."""
    if info.get("arm") not in {"uniform_replay", "spatial_replay"}:
        return {"profile_fp32_matching_status": "not_applicable"}
    path = directory / "profile.json"
    result = {"profile_fp32_matching_status": "missing_step_profile",
              "profile_model_fp32_match_passed": None, "profile_visible_fp32_match_passed": None}
    if not path.is_file():
        return result
    steps = read_json(path)
    if not isinstance(steps, list) or not steps:
        return result
    for branch in ("model", "visible"):
        flags = []
        for step in steps:
            target, actual = step.get("target_" + branch + "_rms"), step.get(branch + "_rms")
            if target is None or actual is None:
                break
            target, actual = finite(target), finite(actual)
            if target < 0 or actual < 0:
                raise ValueError("Projection RMS must be nonnegative")
            flags.append(abs(actual - target) <= 1e-6 + 1e-4 * target)
        if len(flags) == len(steps):
            result["profile_" + branch + "_fp32_match_passed"] = all(flags)
            result["profile_" + branch + "_fp32_matching_steps"] = len(flags)
            result["profile_" + branch + "_fp32_failed_steps"] = sum(not flag for flag in flags)
    known = [result["profile_" + branch + "_fp32_match_passed"] for branch in ("model", "visible")]
    result["profile_fp32_matching_status"] = "fail" if False in known else "missing" if None in known else "pass"
    return result


def load_lpips(backbone, calibration, torch):
    import lpips
    metric = lpips.LPIPS(net="alex", version="0.1", pnet_rand=True, pretrained=False,
                         spatial=False, eval_mode=True, verbose=False)
    trunk = torch.load(backbone, map_location="cpu", weights_only=True)
    if "state_dict" in trunk:
        trunk = trunk["state_dict"]
    trunk = {k.removeprefix("module."): v for k, v in trunk.items()}
    # Official LPIPS preserves torchvision feature indices in sliceN.<index>.*.
    mapped = {key: trunk["features." + key.split(".", 1)[1]] for key in metric.net.state_dict()}
    metric.net.load_state_dict(mapped, strict=True)
    learned = torch.load(calibration, map_location="cpu", weights_only=True)
    if "state_dict" in learned:
        learned = learned["state_dict"]
    learned = {k.removeprefix("module."): v for k, v in learned.items()}
    for index in range(5):
        prefix = f"lin{index}."
        layer = {k[len(prefix):]: v for k, v in learned.items() if k.startswith(prefix)}
        getattr(metric, f"lin{index}").load_state_dict(layer, strict=True)
    return metric.to("cpu").eval().requires_grad_(False)


def load_models(args, requested):
    provenance = dict(device="cpu", network="offline; local_files_only=True", weights={})
    models = {}
    if requested & {"lpips", "clip"}:
        import torch
        torch.set_num_threads(args.threads)
        models["torch"] = torch
    if "lpips" in requested:
        if args.lpips_weights == "bundled":
            import lpips
            calibration = Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / "alex.pth"
        elif args.lpips_weights:
            calibration = Path(args.lpips_weights).expanduser()
        else:
            raise ValueError("LPIPS requires --lpips-weights / LPIPS_PATH (or explicit bundled)")
        if not args.alexnet_weights:
            raise ValueError("LPIPS requires pretrained torchvision AlexNet --alexnet-weights / LPIPS_TRUNK_PATH")
        backbone = Path(args.alexnet_weights).expanduser()
        if not backbone.is_file() or not calibration.is_file():
            raise FileNotFoundError("Both LPIPS calibration and pretrained AlexNet weights must exist locally")
        provenance["weights"]["lpips"] = dict(model="alex", version="0.1",
            backbone=provenance_file(backbone), calibration=provenance_file(calibration),
            initialization="pnet_rand=True, pretrained=False; then strictly load all backbone features and five calibration layers")
        models["lpips"] = load_lpips(backbone, calibration, models["torch"])
    if "clip" in requested:
        if not args.clip_model or not Path(args.clip_model).expanduser().is_dir():
            raise ValueError("CLIP requires existing local HF directory --clip-model / CLIP_MODEL_PATH")
        from transformers import CLIPModel, CLIPProcessor
        folder = Path(args.clip_model).expanduser().resolve()
        model = CLIPModel.from_pretrained(str(folder), local_files_only=True).to("cpu").eval().requires_grad_(False)
        vision = model.config.vision_config
        if (vision.patch_size, vision.hidden_size, vision.num_hidden_layers, model.config.projection_dim) != (32, 768, 12, 512):
            raise ValueError("Expected CLIP ViT-B/32: patch32, width768, 12 vision layers, projection512")
        processor = CLIPProcessor.from_pretrained(str(folder), local_files_only=True)
        models.update(clip=model, processor=processor)
        provenance["weights"]["clip"] = dict(model="ViT-B/32", path=str(folder),
            files=[dict(relative_path=p.relative_to(folder).as_posix(), **provenance_file(p))
                   for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix in {".json", ".txt", ".bin", ".safetensors", ".model"}])
    packages = ("numpy", "Pillow", "torch", "torchvision", "lpips", "transformers")
    provenance["packages"] = {}
    for package in packages:
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    return models, provenance


def native_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError(f"Expected native 512x512 without resizing: {path}, got {image.size}")
        return image.convert("RGB")


def metric_row(row, source_case, mask_entry, package, models, requested):
    import numpy as np
    from PIL import Image
    source_path = resolve(package, source_case["image_path"])
    source, edited = native_rgb(source_path), native_rgb(row["native_path"])
    row["source_path"] = str(source_path.resolve())
    source_rgb, output_rgb = np.asarray(source, dtype=np.float32) / 255, np.asarray(edited, dtype=np.float32) / 255
    mask_path = resolve(package, mask_entry["edit_mask"]).resolve()
    if not mask_path.is_relative_to((package / "evaluation_only").resolve()):
        raise ValueError("Evaluation mask must be under package/evaluation_only")
    with Image.open(mask_path) as im:
        if im.size != (512, 512):
            raise ValueError("Evaluation mask must be native 512x512")
        raw_mask = np.asarray(im.convert("L"), dtype=np.uint8)
    if not np.isin(raw_mask, [0, 255]).all():
        raise ValueError("Expected official binary evaluation mask (0 or 255)")
    edit = raw_mask.astype(np.float32) / 255
    preserve = 1 - edit
    row.update(edit_pixels=int(edit.sum()), preservation_pixels=int(preserve.sum()),
               mask_path=str(mask_path), edit_region_status="ok" if edit.sum() else "empty_region",
               preservation_region_status="ok" if preserve.sum() else "empty_region")
    row["preservation_metric_eligible"] = bool(preserve.sum())
    regions = {"full": None, "edit": edit, "preservation": preserve}
    if "mae" in requested:
        difference = np.abs(source_rgb.astype(np.float64) - output_rgb.astype(np.float64)).mean(axis=2)
        for label, mask in regions.items():
            row["mae_" + label] = finite(difference.mean()) if mask is None else (
                finite((difference * mask).sum() / mask.sum()) if mask.sum() else None)
        row["mae_status"] = "ok"
    if "lpips" in requested:
        torch = models["torch"]
        def tensor(rgb, mask):
            array = rgb if mask is None else rgb * mask[:, :, None]
            return torch.from_numpy(np.ascontiguousarray(array * 2 - 1)).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            for label, mask in regions.items():
                if mask is not None and not mask.sum():
                    row["lpips_" + label] = None
                else:
                    row["lpips_" + label] = finite(models["lpips"](tensor(source_rgb, mask), tensor(output_rgb, mask)).item())
        row["lpips_status"] = "ok"
    if "clip" in requested:
        torch, model, processor = models["torch"], models["clip"], models["processor"]
        with torch.inference_mode():
            image_inputs = processor(images=[source, edited], return_tensors="pt")
            text_inputs = processor(text=[source_case["source_prompt"], source_case["target_prompt"]],
                                    padding=True, truncation=False, return_tensors="pt")
            if text_inputs["input_ids"].shape[1] > model.config.text_config.max_position_embeddings:
                raise ValueError("Prompt exceeds local CLIP context; no silent prompt truncation")
            images = model.get_image_features(**image_inputs).float()
            texts = model.get_text_features(**text_inputs).float()
            if not torch.isfinite(images).all() or not torch.isfinite(texts).all():
                raise ValueError("Non-finite CLIP embeddings")
            if (images.norm(dim=-1) <= 1e-12).any() or (texts.norm(dim=-1) <= 1e-12).any():
                raise ValueError("Zero CLIP base embedding")
            images = torch.nn.functional.normalize(images, dim=-1)
            texts = torch.nn.functional.normalize(texts, dim=-1)
            image_delta, text_delta = images[1] - images[0], texts[1] - texts[0]
            denominator = image_delta.norm() * text_delta.norm()
            row["clip_directional"] = finite((image_delta @ text_delta / denominator).item()) if denominator.item() > 1e-12 else None
            row["clip_directional_status"] = "ok" if row["clip_directional"] is not None else "zero_direction"
            row["clip_target"] = finite((images[1] @ texts[1]).item())
            row["clip_image"] = finite((images[1] @ images[0]).item())
            row["clip_status"] = "ok"


def persist(output, rows, provenance):
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(dict(schema="doselens-eval-v1", rows=rows,
        provenance=provenance), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "weights_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", "--from-results", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: RESULTS/evaluation")
    parser.add_argument("--metrics", default="mae,lpips,clip", help="Explicit comma-separated metric families")
    parser.add_argument("--clip-model", "--clip-path", default=os.environ.get("CLIP_MODEL_PATH") or os.environ.get("CLIP_PATH"))
    parser.add_argument("--alexnet-weights", "--lpips-trunk-path", default=os.environ.get("LPIPS_TRUNK_PATH"))
    parser.add_argument("--lpips-weights", "--lpips-path", default=os.environ.get("LPIPS_PATH"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}
    if not requested or requested - {"mae", "lpips", "clip"} or args.threads < 1:
        parser.error("Choose mae,lpips,clip and threads >= 1")
    package, results = args.package_root.resolve(), args.results.resolve()
    cases = {c["case"]: c for c in read_json(package / "tokens_manifest.json")["cases"]}
    masks = {m["case"]: m for m in read_json(package / "evaluation_only" / "manifest.json")}
    reports = list(reports_from(results, args.reports))
    if not reports:
        parser.error("No runner reports found; evaluation starts only after runner reports exist")
    try:
        models, provenance = load_models(args, requested)
    except Exception as exc:
        parser.error(f"Local metric initialization failed: {exc}")
    provenance.update(requested_metrics=sorted(requested), evaluator=provenance_file(Path(__file__)),
        tokens_manifest=provenance_file(package / "tokens_manifest.json"),
        evaluation_manifest=provenance_file(package / "evaluation_only" / "manifest.json"),
        primary_metric="lpips_preservation",
        annotation_coverage=dict(total_sources=len(masks),
            preservation_eligible_sources=sum(bool(m["preservation_metric_eligible"]) for m in masks.values()),
            empty_preservation_sources=sum(not m["preservation_metric_eligible"] for m in masks.values())),
        definitions=dict(lpips="Alex v0.1; masked RGB in [0,1] set to black outside region, then [-1,1]; scalar whole-image mean; no area renormalization",
            mae="RGB [0,1] absolute error; region sum divided by region pixel mass",
            clip_directional="ViT-B/32; cosine(normalize(Iout)-normalize(Isrc), normalize(Ttarget)-normalize(Tsource))",
            resolution="Input PNGs and masks must be native 512x512; no resize; CLIP uses its local processor",
            nulls="Empty region / zero CLIP direction / unrequested or failed metric => JSON null, CSV empty; status explains reason"))
    output = args.output_dir or results / "evaluation"
    rows, seen = [], set()
    for report_path, report in reports:
        job_path = report_path.parent / "job.json"
        info = dict(read_json(job_path).get("job", {})) if job_path.is_file() else {}
        info.update(report.get("job", {}))
        info.update(report)
        info.setdefault("job_id", report.get("_runner", {}).get("job_id", report_path.parent.name))
        method = condition_label(info)
        row = dict(id=str(info.get("id", info.get("job_id", report_path.parent.name))),
                   case=str(info.get("case", "")), seed=info.get("seed"), method=method,
                   arm=info.get("arm"), scale=info.get("scale"), n_max=info.get("n_max"), stage=info.get("stage"),
                   anchor_job_id=info.get("anchor_job_id"), matching_roundoff=info.get("matching_roundoff"),
                   reported_native_path=info.get("native_path"),
                   category=None, report_path=str(report_path), native_path=str(report_path.parent / "native.png"),
                   generation_status=str(info.get("status", "unknown")), eval_status="pending", error="", device="cpu",
                   **{name: None for name in METRICS},
                   mae_status="pending" if "mae" in requested else "not_requested",
                   lpips_status="pending" if "lpips" in requested else "not_requested",
                   clip_status="pending" if "clip" in requested else "not_requested",
                   clip_directional_status="pending" if "clip" in requested else "not_requested")
        row.update(profile_columns(info))
        try:
            if row["case"] not in cases or not isinstance(method, str) or not method:
                raise ValueError("Report requires known string case ID and nonempty method/arm/variant/condition")
            if type(row["seed"]) is not int:
                raise ValueError("Report seed must be an explicit integer")
            identity = (row["case"], row["seed"], method)
            if identity in seen:
                raise ValueError("Duplicate case/seed/method report; retain only one job per paired cell")
            seen.add(identity)
            row["category"] = cases[row["case"]]["category"]
            if report.get("_runner"):
                marker_path = report_path.parent / "completed.json"
                marker = read_json(marker_path) if marker_path.exists() else {}
                if (marker.get("status") != "completed" or marker.get("protocol_hash") != report["_runner"].get("protocol_hash")
                        or marker.get("report_sha256") != sha256(report_path)):
                    row["generation_status"] = "runner_completion_not_confirmed"
            if row["generation_status"].lower() not in {"completed", "complete", "ok", "success"}:
                row["eval_status"] = "generation_not_completed"
            else:
                for field in ("source_prompt", "target_prompt"):
                    if field in info and info[field] != cases[row["case"]][field]:
                        raise ValueError(f"Runner {field} differs from the frozen token manifest")
                row.update(fp32_matching_columns(info, report_path.parent))
                metric_row(row, cases[row["case"]], masks[row["case"]], package, models, requested)
                row["eval_status"] = "ok"
        except Exception as exc:
            row.update(eval_status="error", error=f"{type(exc).__name__}: {exc}")
        for family in ("mae", "lpips", "clip"):
            if row[family + "_status"] == "pending":
                row[family + "_status"] = row["eval_status"]
        if row["clip_directional_status"] == "pending":
            row["clip_directional_status"] = row["eval_status"]
        rows.append(row)
        print(f"EVAL {row['id']} {row['eval_status']}", flush=True)
    persist(output, rows, provenance)
    print(f"Wrote {output / 'metrics.csv'} and finite JSON; {sum(r['eval_status'] == 'ok' for r in rows)}/{len(rows)} evaluated")
    return 1 if any(r["eval_status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
