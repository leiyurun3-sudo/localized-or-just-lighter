"""Compare complete full-grid seed archives on their common source support."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
from summarize import STUDY, estimate, summarize

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores",type=Path,nargs="+",help="scores.csv file for each new seed")
    parser.add_argument("--out",type=Path,default=Path("seed_comparison.json"))
    args=parser.parse_args()
    by_seed={}
    summaries={}
    with tempfile.TemporaryDirectory() as tmp:
        for path in [STUDY/"seed0.csv",*args.scores]:
            dest=Path(tmp)/str(len(summaries))
            dest.mkdir()
            summary=summarize(path,dest)
            seed=summary["seed"]
            if seed in summaries:raise ValueError(f"Duplicate seed {seed}")
            summaries[seed]=summary
            if "mrc" in summary:
                with (dest/"mrc_sources.csv").open(newline="") as f:
                    by_seed[seed]={r["case"]:r for r in csv.DictReader(f) if r["status"]=="valid"}
    result={"seed_specific":summaries}
    if len(by_seed)>1:
        common=sorted(set.intersection(*(set(rows) for rows in by_seed.values())))
        paired={}
        for seed,rows in by_seed.items():
            matched=[float(rows[c]["matched"]) for c in common]
            native=[float(rows[c]["default"]) for c in common]
            delta=[float(rows[c]["matched"])-float(by_seed[0][c]["matched"]) for c in common]
            paired[seed]=dict(matched_gap=estimate(matched), fraction_of_default=estimate(matched,native),
                              change_in_matched_gap_vs_seed0=estimate(delta))
        result["common_supported_sources"]=common
        result["common_support"]=paired
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print(args.out.resolve())

if __name__=="__main__":
    main()

