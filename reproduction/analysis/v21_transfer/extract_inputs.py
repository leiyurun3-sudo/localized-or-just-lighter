"""Optional raw-archive extraction. The statistical reproduction only needs source_inputs.csv."""
import argparse
import csv
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
HARM = ANALYSIS / "harmonized_lpips"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def member(z, key):
    return json.loads(z.read(key))


def build(archives):
    sam_report, = archives.glob("A_DoseLens_scan*reports.zip")
    archive_summary, job_sets = {}, []
    for p in [archives/"A_DoseLens_results.zip", archives/"A_DoseLens_results_full.zip", sam_report]:
        with ZipFile(p) as z:
            found = [member(z,n) for n in z.namelist() if re.fullmatch(r"outputs/\d{12}__[^/]+/report.json", n)]
            assert all(r["status"] == "completed" for r in found)
            models = {Path(member(z,n)["model"]["model_path"]).name for n in z.namelist() if n.endswith("/protocol.json")}
            job_sets.append({r["job_id"] for r in found})
            archive_summary[p.name] = dict(completed=len(found), cases=len({r["case"] for r in found}),
                                           seeds=sorted({r["seed"] for r in found}), backbones=sorted(models))
    assert job_sets[0] <= job_sets[1] <= job_sets[2]
    high = {r["case"]:r for r in read_csv(HARM/"R88_scores.csv")}
    direct = read_csv(HARM/"DirectEdit_scores.csv")
    direct_cases = {r["case"]:r for r in direct}
    scores = {(r["case"],r["arm"]):r for r in direct}
    weights = json.loads((HARM/"weights.json").read_bytes())["weights"]["lpips"]
    result = []

    def entry(method, case, category, pixels, mags, losses, old=None):
        assert 0 < pixels < 512**2 and min(mags) > 0
        result.append(dict(operator=method,source=case,category=int(category),seed=0,
                           background_fraction=pixels / 512**2, spatial_magnitude=mags[0], uniform_magnitude=mags[1],
                           spatial_lpips=losses[0], uniform_lpips=losses[1],
                           spatial_masked_input_lpips=old[0] if old else "",
                           uniform_masked_input_lpips=old[1] if old else ""))

    matched = Counter()
    with ZipFile(archives/"R88_AnalysisClosure_Fast_results.zip") as rz, ZipFile(archives/"R89_DirectEdit_Results.zip") as dz:
        tasks = {r["case"]:r for r in member(rz,"source_selection_manifest.json")["tasks"]}
        history = {r["id"]:r for r in csv.DictReader(io.StringIO(rz.read("evaluation.csv").decode("utf-8-sig")))}
        for case, task in sorted(tasks.items()):
            scores_r, mag_r = [], []
            for arm in ("REQ","U25"):
                job = f"R88_pie_{case}_s0_{arm}"
                rec = history[job]
                assert rec["lpips_calibration_sha256"] == weights["calibration"]["sha256"]
                assert rec["lpips_trunk_sha256"] == weights["backbone"]["sha256"]
                series = member(rz,f"{job}/step_summary.json")["steps"]
                assert [r["step_index"] for r in series] == list(range(28)) and all(r["h"] > 0 for r in series)
                mag_r.append(math.fsum(r["h"]*r["actual_rms"] for r in series))
                scores_r.append(float(rec["lpips_preserved"]))
                assert float(rec["lpips_preserved_mask_mass"]) == int(high[case]["preservation_pixels"])
            entry("correction",case,task["target_category_id"],int(high[case]["preservation_pixels"]),mag_r,scores_r)
            if case not in direct_cases:
                continue
            other = direct_cases[case]
            for key,mode in [("source","RGB"),("mask","L")]:
                with Image.open(other[key]) as a, Image.open(high[case][key]) as b:
                    assert np.array_equal(np.asarray(a.convert(mode)),np.asarray(b.convert(mode)))
                matched[key] += 1
            dm, dl = [], []
            for arm in ("LOCAL_025","GLOBAL_025"):
                prefix = f"external_directedit/results/DE_{case}_{arm}"
                job = member(dz,prefix+"/record.json")["job"]
                assert job["seed"]==0 and all(job[k]==task[k] for k in ("source_prompt","target_prompt","target_category_id"))
                series = member(dz,prefix+"/steps.json")["edit"]["steps"]
                assert [r["step_index"] for r in series]==list(range(30))
                assert all(r["unit"]=="latent_displacement" and not r["multiplied_by_step_size"] for r in series)
                dm.append(math.fsum(r["actual_delta_rms"] for r in series))
                dl.append(float(scores[case,arm]["spatial"]))
            matched["prompt_pair"] += 1
            entry("directedit",case,task["target_category_id"],int(other["preservation_pixels"]),dm,dl)
    sam = {(r["case"],r["arm"]):r for r in read_csv(HARM/"sam_core_scores.csv")}
    cats = {r["case"]:r["category"] for r in read_csv(ANALYSIS/"metrics/metrics.csv")}
    assert not (tasks.keys() & cats.keys()) and len(cats)==50
    with ZipFile(sam_report) as z:
        for case in sorted(cats):
            pixels = int(sam[case,"sam_native"]["preservation_pixels"])
            if not pixels:
                continue
            mags, spatial, black = [], [], []
            for arm in ("sam_native","uniform"):
                rec = sam[case,arm]
                assert int(rec["seed"]) == 0 and int(rec["preservation_pixels"]) == pixels
                steps = member(z,f"outputs/{case}__{arm}/profile.json")
                assert [r["step"] for r in steps]==list(range(2,28))
                mags.append(math.fsum(r["visible_rms"] for r in steps)/26)
                spatial.append(float(rec["spatial"]))
                black.append(float(rec["archived"]))  # exact original target of the existing LOO analysis
            entry("sam",case,cats[case],pixels,mags,spatial,black)
    assert Counter(r["operator"] for r in result)=={"sam":42,"correction":40,"directedit":20}
    with (HERE/"source_inputs.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=result[0]);writer.writeheader();writer.writerows(result)
    inventory=dict(sam_snapshots=archive_summary, earlier_sam_job_ids_nested_in_later=True,
                   sam_full_region_exclusions=8, correction_directedit_same_pixels_and_prompts=dict(matched),
                   sam_sources_shared_with_correction_or_directedit=0,
                   observed_arm_seed_counts=read_csv(HARM/"observed_arm_seed_counts.csv"))
    (HERE/"archive_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_directory",type=Path)
    build(parser.parse_args().archive_directory)
