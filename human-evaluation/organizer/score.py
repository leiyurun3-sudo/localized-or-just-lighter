"""Score actual offline exports. Python standard library only."""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

ARMS=("uniform","two_constant","sam_native")
DEFAULT_KEY=Path(__file__).with_name("assignment_key.csv")
CONTRASTS = (("sam_native", "uniform"), ("sam_native", "two_constant"), ("two_constant", "uniform"))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0


def divide(num,den):
    return num/den if den else None


def load_key(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    kits={r["kit"] for r in rows}
    if len(kits)!=1:
        raise ValueError("Key must describe exactly one kit.")
    key={}
    for row in rows:
        rater,item=row["rater"],row["item"]
        if item in key.setdefault(rater,{}) or row["arm"] not in ARMS:
            raise ValueError("Duplicate alias or invalid arm in organizer key.")
        key[rater][item]=(row["case"],row["arm"])
    targets=set(next(iter(key.values())).values())
    cases={c for c,_ in targets}
    if (len(cases)!=50 or len(key)!=3 or targets!=set(itertools.product(cases,ARMS)) or
            any(len(k)!=150 or set(k.values())!=targets for k in key.values())):
        raise ValueError("Key does not cover the same 50 sources x 3 arms per rater.")
    return kits.pop(),key,targets


def decode(paths,kit,key):
    scores={}
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8-sig"))
        r=data.get("rater")
        if data.get("kit")!=kit or data.get("version")!=1 or r not in key:
            raise ValueError(f"{path.name}: wrong kit, version or rater.")
        if r in scores:
            raise ValueError(f"Two exports for {r}; use one intended backup per person.")
        if not isinstance(data.get("answers"),list):
            raise ValueError(f"{path.name}: invalid response list.")
        values={}
        for row in data["answers"]:
            alias=row.get("id")
            if alias not in key[r]:
                raise ValueError(f"{path.name}: unknown item {alias}.")
            pair=key[r][alias]
            v=row.get("score")
            if pair in values or type(v) is not int or v not in (0,1) or type(row.get("uncertain")) is not bool:
                raise ValueError(f"{path.name}: invalid or repeated rating for {alias}.")
            values[pair]=(v,row["uncertain"])
        scores[r]=values
    return scores


def adjusted_kappa(observed,expected):
    return None if expected>=1 else (observed-expected)/(1-expected)


def pair_agreement(a,b,pop):
    common=pop.intersection(a,b)
    counts=[[0,0],[0,0]]
    for item in common:
        counts[a[item][0]][b[item][0]]+=1
    n=len(common)
    if n:
        po=(counts[0][0]+counts[1][1])/n
        p=sum(counts[1])/n
        q=(counts[0][1]+counts[1][1])/n
        pe=p*q+(1-p)*(1-q)
        k=adjusted_kappa(po,pe)
    else:
        po=k=None
    return dict(overlap=n,eligible=len(pop),table_rows_first_cols_second_0_1=counts,
                observed_agreement=po,cohen_kappa=k)


def summarize(pop,ratings):
    ids=sorted(ratings)
    per={}
    for r in ids:
        vals=[ratings[r][p] for p in pop.intersection(ratings[r])]
        n=len(vals)
        yes=sum(v[0] for v in vals)
        per[r]=dict(yes=yes,no=n-yes,rated=n,missing=len(pop)-n,success_rate=divide(yes,n),
                    uncertainty_flags=sum(v[1] for v in vals))
    common=pop.intersection(*(set(ratings[r]) for r in ids))
    m=len(ids)
    votes=[sum(ratings[r][p][0] for r in ids) for p in common]
    complete_yes=sum(votes)
    den=len(common)*m
    joint=dict(raters=ids,overlap=len(common),eligible=len(pop),yes_votes=complete_yes,total_votes=den,
               success_rate=divide(complete_yes,den))
    result=dict(per_rater=per,complete_overlap=joint,cohen={})
    for a,b in itertools.combinations(ids,2):
        result["cohen"][a+"_"+b]=pair_agreement(ratings[a],ratings[b],pop)
    if m==3:
        po=pe=k=None
        if votes:
            po=sum(v*(v-1)+(m-v)*(m-v-1) for v in votes)/(len(votes)*m*(m-1))
            prevalence=complete_yes/den
            pe=prevalence**2+(1-prevalence)**2
            k=adjusted_kappa(po,pe)
        result["fleiss"]=dict(overlap=len(common),eligible=len(pop),observed_agreement=po,
                              expected_agreement=pe,fleiss_kappa=k)
    return result


def percentile(sorted_values, probability):
    """Linear interpolation between order statistics at (B-1)*p."""
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[min(lower+1, len(sorted_values)-1)] - sorted_values[lower])


def source_intervals(vote_rows, rater_count):
    """Bootstrap intact source rows; use the same draw for all arm rates and contrasts."""
    arm_draws = {arm: [] for arm in ARMS}
    contrast_draws = {pair: [] for pair in CONTRASTS}
    n = len(vote_rows)
    if n < 2:
        return dict.fromkeys(ARMS), dict.fromkeys(CONTRASTS)

    denominator = n * rater_count
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        sample = rng.choices(vote_rows, k=n)
        totals = dict(zip(ARMS, map(sum, zip(*sample))))
        for arm in ARMS:
            arm_draws[arm].append(totals[arm] / denominator)
        for a, b in CONTRASTS:
            contrast_draws[a, b].append((totals[a] - totals[b]) / denominator)

    def interval(values):
        values.sort()
        return [percentile(values, 0.025), percentile(values, 0.975)]

    return ({a: interval(v) for a, v in arm_draws.items()},
            {pair: interval(v) for pair, v in contrast_draws.items()})


def paired_source_summary(targets, ratings):
    panel = sorted(ratings)
    m = len(panel)
    cases = sorted({case for case, _ in targets})
    complete = [c for c in cases if all((c, arm) in ratings[r] for r in panel for arm in ARMS)]
    complete_set = set(complete)
    result = dict(raters=panel, eligible_sources=len(cases), complete_sources=len(complete),
                  excluded_source_ids=[c for c in cases if c not in complete_set],
                  bootstrap=dict(unit="source", sampling="unstratified with replacement; fixed raters and paired arms",
                                 replicates=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                                 interval="marginal 95% percentile, linear interpolation at (B-1)*p"),
                  by_arm={}, contrasts={}, source_yes_votes={})
    if m < 2:
        result["status"] = "requires_two_or_three_raters"
        return result

    source_votes = {
        c: {a: sum(ratings[r][c, a][0] for r in panel) for a in ARMS}
        for c in complete
    }
    n = len(complete)
    result["status"] = {0: "no_complete_sources", 1: "one_source_no_interval"}.get(n, "ok")
    result["source_yes_votes"] = source_votes
    vote_rows = [tuple(votes[a] for a in ARMS) for votes in source_votes.values()]
    arm_ci, contrast_ci = source_intervals(vote_rows, m)
    for arm in ARMS:
        counts = [v[arm] for v in source_votes.values()]
        yes = sum(counts)
        result["by_arm"][arm] = dict(yes_votes=yes, total_votes=n*m, success_rate=divide(yes, n*m),
                                     ci95=arm_ci[arm], unanimous_yes_sources=counts.count(m),
                                     unanimous_no_sources=counts.count(0),
                                     split_vote_sources=sum(0 < k < m for k in counts),
                                     uncertainty_flags=sum(ratings[r][c, arm][1] for c in complete for r in panel))
    for a, b in CONTRASTS:
        deltas = [v[a] - v[b] for v in source_votes.values()]
        result["contrasts"][f"{a}_minus_{b}"] = dict(
            first_arm=a, second_arm=b, sources=n, difference=divide(sum(deltas), n*m),
            ci95=contrast_ci[a, b], first_higher_sources=sum(d > 0 for d in deltas),
            tied_sources=deltas.count(0), second_higher_sources=sum(d < 0 for d in deltas))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports",nargs="*",type=Path,help="One JSON export per actual rater")
    parser.add_argument("--key",type=Path,default=DEFAULT_KEY)
    parser.add_argument("--out",type=Path,help="Optional summary JSON file (default: stdout)")
    parser.add_argument("--check-key",action="store_true",help="Check allocation coverage only")
    parser.add_argument("--raters", nargs="+", metavar="ID",
                        help="Prespecified panel of two or three assigned IDs; requires one export from each")
    args=parser.parse_args()
    try:
        kit,key,targets=load_key(args.key)
        if args.check_key:
            if args.exports or args.out or args.raters:
                parser.error("--check-key is a design-only check; omit exports, --raters and --out.")
            print(f"Key OK: {len(key)} independent aliases/orders, {len(targets)} required outputs per rater.")
            return
        if not args.exports:
            parser.error("Supply actual rater JSON files, or use --check-key to inspect the design.")
        scores=decode(args.exports,kit,key)
        if args.raters is not None:
            panel = set(args.raters)
            if len(args.raters) not in (2, 3) or len(panel) != len(args.raters) or not panel.issubset(key):
                parser.error("--raters requires two or three distinct rater IDs from this key.")
            if panel != set(scores):
                parser.error("The exports must match the prespecified --raters panel exactly.")
        summary=dict(kit=kit,raters_submitted=sorted(scores),scope="binary visible target change",
                     pooled_agreement_and_coverage=summarize(targets,scores),
                     by_arm={a:summarize({p for p in targets if p[1]==a},scores) for a in ARMS},
                     paired_sources=paired_source_summary(targets,scores))
        content=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)
        if args.out:
            if args.out.resolve() in {p.resolve() for p in [args.key,*args.exports]}:
                parser.error("--out must differ from the key and all raw exports.")
            args.out.write_text(content+"\n",encoding="utf-8")
        else:
            print(content)
    except (ValueError,KeyError,OSError,TypeError,AttributeError) as error:
        parser.exit(2,f"Scoring error: {error}\n")


if __name__=="__main__":
    main()
