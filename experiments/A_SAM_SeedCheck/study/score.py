"""Offline CPU analysis of one returned seed ZIP, using the seed-0 paper metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile
import numpy as np
from PIL import Image

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
sys.path.insert(0,str(ROOT))
import evaluate
from summarize import summarize

def bg_lpips(model, torch, source, edited, mask_file):
    def tensor(path):
        rgb=np.asarray(evaluate.native_rgb(path),dtype=np.float32)
        return torch.from_numpy(rgb/127.5-1).permute(2,0,1)[None]
    with Image.open(mask_file) as im:
        keep=1-np.asarray(im.convert("L"),dtype=np.float32)/255
    with torch.inference_mode():
        field=model(tensor(source),tensor(edited))
        w=torch.nn.functional.interpolate(torch.from_numpy(keep)[None,None],size=field.shape[-2:],mode="area")
        return float(((field*w).sum()/w.sum()).item())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--alexnet-weights",default=os.getenv("LPIPS_TRUNK_PATH", "D:/DevCache/tracememory_metrics/models/alexnet-owt-7be5be79.pth"))
    p.add_argument("--clip-model",default=os.getenv("CLIP_MODEL_PATH", "D:/DevCache/doselens_metrics/clip-vit-base-patch32"))
    p.add_argument("--lpips-weights",default=os.getenv("LPIPS_PATH","bundled"))
    args=p.parse_args()
    models,provenance=evaluate.load_models(args,{"lpips","clip"})
    weights=provenance["weights"]; reference=json.loads((STUDY/"metric_weights.json").read_text())
    actual={k:v["sha256"] for k,v in weights["lpips"].items() if k in ("backbone","calibration")}
    clip={v["relative_path"]:v["sha256"] for v in weights["clip"]["files"]}
    clip_files={k:v for k,v in reference["clip_files"].items() if k!="provenance.json"}
    if actual != {"backbone":reference["alexnet"],"calibration":reference["calibration"]} or not all(clip.get(k)==v for k,v in clip_files.items()):
        raise ValueError("CPU scoring weights differ from the seed-0 paper reference")
    models["lpips"].spatial=True
    out=args.archive.resolve().with_suffix("")
    out=out.parent/(out.name+"_scores")
    out.mkdir(exist_ok=True)
    masks={m["case"]:m for m in json.loads((ROOT/"evaluation_only/manifest.json").read_text())}
    jobs=json.loads((STUDY/"jobs.json").read_text())["jobs"]
    rows=[]
    with ZipFile(args.archive) as z,tempfile.TemporaryDirectory(dir=out) as scratch:
        batch=json.loads(z.read("batch.json")); seed=batch["seed"];mode=batch["mode"]
        if seed not in (1,2) or mode not in ("light","full","partial"):raise ValueError("Unrecognized seed or mode")
        cases={c["case"]:c for c in json.loads((STUDY/f"cases{seed}.json").read_text(encoding="utf8"))["cases"]}
        if mode == "partial":
            ids = batch["completed_ids"]
            observed = {name.split("/")[0] for name in z.namelist() if name.endswith("/completed.json")}
            if not ids or len(set(ids)) != len(ids) or set(ids) != observed or batch["planned"] != len(jobs):
                raise ValueError("Partial archive manifest does not match saved completions")
            jobs = [j for j in jobs if j["job_id"] in observed]
        else:
            jobs=[j for j in jobs if mode=="full" or j["stage"]=="core"]
        if batch["jobs"]!=len(jobs):raise ValueError("Incomplete batch")
        edited=Path(scratch)/"native.png"
        for i,j in enumerate(jobs,1):
            case=cases[j["case"]]; key=j["job_id"];mask=masks[j["case"]]
            report=json.loads(z.read(key+"/report.json"))
            if (report["seed"],report["case"],report["arm"],report["scale"],report["_runner"]["protocol_hash"])!=(seed,j["case"],j["arm"],j.get("scale",1),batch["protocol"]):
                raise ValueError("Mismatched generation record: "+key)
            edited.write_bytes(z.read(key+"/native.png"))
            data={"native_path":str(edited)}
            evaluate.metric_row(data,case,mask,ROOT,models,{"mae","clip"})
            fm=bg_lpips(models["lpips"],models["torch"],ROOT/case["image_path"],edited,ROOT/mask["edit_mask"])
            rows.append(dict(id=key,case=j["case"],seed=seed,arm=j["arm"],scale=j.get("scale",1),bg_fm=fm,
                             **{k:data[k] for k in ["clip_directional","mae_edit","mae_preservation"]}))
            if i%20==0 or i==len(jobs):print(f"{i}/{len(jobs)}",flush=True)
    scores=out/"scores.csv"
    with scores.with_suffix(".tmp").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    scores.with_suffix(".tmp").replace(scores)
    (out/"score_provenance.json").write_text(json.dumps(dict(batch=batch,metrics=provenance),indent=2),encoding="utf8")
    if mode == "partial":
        print(scores)
        return
    result=summarize(scores,out)
    print(out/"summary.json")
    if "mrc" in result:print(out/"mrc_sources.csv")

if __name__=="__main__":
    main()

