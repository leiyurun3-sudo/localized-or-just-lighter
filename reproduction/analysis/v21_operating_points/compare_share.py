"""Use the published MRC integration rule with within-mask share of change as the coordinate."""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np

from compare_points import GRID, NATIVE, load_by_source, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("mrc", folder.parent/"v20_mrc"/"reproduce_mrc.py")
    mrc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mrc
    spec.loader.exec_module(mrc)
    cases = load_by_source(args.points)
    ids = sorted(cases)
    nodes = np.empty((len(ids),2,len(GRID),3))
    for i, case in enumerate(ids):
        for j, prefix in enumerate(NATIVE.values()):
            for k, setting in enumerate(GRID):
                name = prefix if setting == 1 else f"{prefix}_scale{setting:g}"
                row = cases[case][name]
                nodes[i,j,k] = (float(row["rho"]),float(row["directional_clip"]),float(row["lpips_spatial"]))
    details, summary = [], []
    for last in (False,True):
        lo,hi,delta=mrc.integrate(nodes,(0,1),last=last)
        good=np.isfinite(delta)
        mode = "last" if last else "first"
        for case, l, h, d in zip(ids,lo,hi,delta):
            details.append(dict(source=case,crossing=mode,lower=l,upper=h,uniform_minus_spatial=d if h>l else ""))
        if good.any():
            interval = np.quantile(delta[mrc.draws(good)].mean(axis=1),[.025,.975])
            summary.append(dict(crossing=mode,eligible=len(ids),supported=int(good.sum()),positive=int((delta>0).sum()),
                                median_width=float(np.median((hi-lo)[good])), mean=float(delta[good].mean()),
                                low=float(interval[0]),high=float(interval[1])))
        else:
            summary.append(dict(crossing=mode,eligible=len(ids),supported=0,positive=0,median_width="",mean="",low="",high=""))
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"share_contrasts.csv",details)
    write_csv(args.output/"share_summary.csv",summary)


if __name__=="__main__":
    main()
