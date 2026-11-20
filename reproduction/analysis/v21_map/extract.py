"""Optional compact-input rebuild from the three results ZIPs and local archived score/source directories."""
import argparse
import csv
import hashlib
import io
import json
import re
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
FILES = {"R88": "R88_AnalysisClosure_Fast_results.zip", "R89": "R89_AnalysisEvidenceClosure_results.zip",
         "R91": "R91_results.zip"}
ARMS = {"REQ":"R88", "SHUF_R":"R88", "SHUF2_R":"R89", "SOURCE_R":"R89",
        "REV_HIST_R":"R89", "RAND_A":"R91", "RAND_B":"R91"}
SEEDS = (0, 17)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def extract(archives, images, scores):
    with scores.open(encoding="utf-8-sig",newline="") as f:
        table = list(csv.DictReader(f))
    index = {(r["case"], int(r["seed"]), r["arm"]): float(r["lpips_preserved"]) for r in table}
    with ExitStack() as stack:
        zipfiles = {k:stack.enter_context(ZipFile(archives/v)) for k,v in FILES.items()}
        jobs = {n.rsplit("/",2)[-2]:(z,n.rsplit("/",1)[0]) for z in zipfiles.values()
                for n in z.namelist() if n.endswith("/step_summary.json")}
        cases = sorted(re.fullmatch(r"R88_pie_(\d+)_s0_REQ", k)[1] for k in jobs
                       if re.fullmatch(r"R88_pie_(\d+)_s0_REQ", k))
        assert len(cases) == 40
        shape = (len(cases),len(SEEDS))
        maps = np.empty((*shape,64,64),np.float32)
        ranks = np.empty((*shape,len(ARMS)-1,64,64),np.uint16)
        preserve = np.empty((len(cases),64,64),np.uint8)
        rgb_sum = np.empty((len(cases),64,64,3),np.uint16)
        loss = np.empty((*shape,len(ARMS)))
        uniform_loss = np.empty(shape)
        initial = np.empty((*shape,len(ARMS),3))
        native = np.empty((*shape,28,4))
        manifest = []
        for i,case in enumerate(cases):
            image_bytes = (images/f"{case}_source_eval.png").read_bytes()
            mask_bytes = zipfiles["R89"].read(f"outputs/r88_native/masks/{case}/official_preserve.png")
            image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            keep = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 127
            assert image.shape == (512,512,3) and keep.shape == (512,512)
            # Use exactly the valid pixel domain of the published map alignment statistic.
            keep[[0,-1],:] = False
            keep[:,[0,-1]] = False
            preserve[i] = keep.reshape(64,8,64,8).sum(axis=(1,3))
            rgb_sum[i] = image.reshape(64,8,64,8,3).sum(axis=(1,3))
            manifest.append(dict(source=f"S{i+1:02}", benchmark_key_sha256=sha(("PIE/"+case).encode()),
                                 archived_rgb_sha256=sha(image_bytes), official_preserve_sha256=sha(mask_bytes)))
            for j,seed in enumerate(SEEDS):
                uniform_loss[i,j] = index[case,seed,"GLB_R"]
                zg, gp = jobs[f"R88_pie_{case}_s{seed}_GLB_R"]
                global_replay = json.loads(zg.read(gp+"/step_summary.json"))["steps"]
                for k,(arm,family) in enumerate(ARMS.items()):
                    ident = f"{family}_pie_{case}_s{seed}_{arm}"
                    z, prefix = jobs[ident]
                    with np.load(io.BytesIO(z.read(prefix+"/request_maps.npz"))) as arrays:
                        field = arrays["P_correction"]
                    assert field.shape == (64,64) and field.dtype.kind == "f"
                    if k == 0:
                        maps[i,j] = field
                        ordered = np.sort(field.ravel())
                    else:
                        permutation = np.argsort(field.ravel(),kind="stable")
                        np.testing.assert_array_equal(field.ravel()[permutation],ordered)
                        order = np.empty(field.size,np.uint16)
                        order[permutation] = np.arange(field.size)
                        ranks[i,j,k-1] = order.reshape(64,64)
                    loss[i,j,k] = index[case,seed,arm]
                    records = json.loads(z.read(prefix+"/step_summary.json"))["steps"]
                    assert [r["step_index"] for r in records] == list(range(28))
                    first = records[0]
                    assert first["sigma"] == 1 and first["corr_lambda"] == .25
                    initial[i,j,k] = [first["current_correction_global_rms"],first["candidate_global_rms"],first["actual_rms"]]
                    if k == 0:
                        assert all(r["corr_lambda"]==.25 for r in records)
                        native[i,j] = [[r["h"],r["current_correction_global_rms"],r["candidate_global_rms"],
                                        r["actual_rms"]] for r in records]
                        np.testing.assert_allclose([r["actual_rms"] for r in global_replay],
                                                   native[i,j,:,3],rtol=1e-5,atol=1e-8)
                    else:
                        np.testing.assert_allclose([r["actual_rms"] for r in records],
                                                   native[i,j,:,3],rtol=1e-5,atol=1e-8)
                    np.testing.assert_allclose([r["h"] for r in records],native[i,j,:,0],rtol=0,atol=1e-9)
        # Confirm the same initial global reference and achieved matched target for all maps.
        assert np.isfinite(native).all() and np.isfinite(initial).all()
        np.testing.assert_allclose(initial[:,:,:,0],np.broadcast_to(initial[:,:,:1,0],initial[:,:,:,0].shape),rtol=1e-6)
        np.testing.assert_allclose(initial[:,:,:,2],np.broadcast_to(initial[:,:,:1,2],initial[:,:,:,2].shape),rtol=1e-5)
        np.testing.assert_allclose(native[:,:,:,3],.25*native[:,:,:,2],rtol=1e-6)
        with (OUT/"sources.csv").open("w",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=manifest[0]);w.writeheader();w.writerows(manifest)
        np.savez_compressed(OUT/"inputs.npz",arms=np.array(list(ARMS)), seeds=np.array(SEEDS),
                            native_map=maps, hist_ranks=ranks, preserve=preserve, rgb_sum=rgb_sum,
                            initial=initial,native_steps=native,lpips=loss,shared_s_uniform_lpips=uniform_loss)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True,help="Recovered R88 source_eval PNG directory")
    p.add_argument("--scores",type=Path,required=True,help="R91 combined_scores.csv")
    args=p.parse_args()
    extract(args.archives,args.images,args.scores)
