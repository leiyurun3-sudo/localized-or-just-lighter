"""List retained numerical payloads and representative step fields in local result ZIPs."""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

ARCHIVES = ("R88_AnalysisClosure_Fast_results.zip","R89_AnalysisEvidenceClosure_results.zip",
            "R91_results.zip","A_R92_results.zip","R89_partial_results.zip","R91_native_images.zip",
            "A_R92_ScalarCalibration.zip")
HERE=Path(__file__).resolve().parent


def inspect(directory):
    results=[]
    for name in ARCHIVES:
        archive=directory/name
        if not archive.exists():
            continue
        with ZipFile(archive) as z:
            names=z.namelist()
            arrays=[p for p in names if p.endswith(".npz")]
            keys=set()
            for p in arrays:
                with ZipFile(io.BytesIO(z.read(p))) as nz:
                    keys.update(Path(k).stem for k in nz.namelist())
            steps=[p for p in names if p.endswith("/step_summary.json")]
            summaries={}
            for path in steps:
                arm=path.rsplit("/",2)[-2].split("_s")[-1].partition("_")[-1]
                summaries.setdefault(arm,path)
            fields=set()
            for p in summaries.values():
                row=json.loads(z.read(p))["steps"][0]
                fields.update(k for k in row if "rms" in k)
            other=[p for p in names if p.lower().endswith((".npy",".pt",".pth",".safetensors",".bin",".h5",".hdf5"))]
            results.append(dict(archive=name,members=len(names),npz_files=len(arrays),
                                npz_basenames="|".join(sorted(set(Path(p).name for p in arrays))),
                                npz_keys="|".join(sorted(keys)),other_tensor_files=len(other),
                                step_files=len(steps),step_arms_sampled=len(summaries),
                                recorded_rms_fields="|".join(sorted(fields))))
    with (HERE/"archive_inventory.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives",type=Path,required=True)
    inspect(parser.parse_args().archives)
