"""Build the 50 reviewed prompt-only token cases; standard library, no model/image IO.

Run: python prepare_tokens.py
Only tokens_manifest.json and token_selection.csv are written beside this script.
Raw prompts remain verbatim in sources_50.json and are copied into the manifest.
The table below is a manual, score-blind review of every FULL prompt pair.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULE = {
    "version": "prompt-only-reviewed-50-v1",
    "evidence": "Full source/target prompts only; no images, generated outputs, scores or GT masks used in token selection.",
    "preprocessing": "Remove only literal [ and ] PIE annotation delimiters; preserve spelling, punctuation, case and whitespace. Raw prompts remain available.",
    "selection": "Compare word spans, retain changed content words, and add the affected object noun when the change is an attribute, action, count, material, style or viewpoint. For object replacement the changed noun itself is sufficient. Exclude articles, conjunctions and prepositions from mask tokens.",
    "insert_delete": "Use a common prompt object noun on both sides as context, including the otherwise empty side. Choose the affected object, support surface, or scene noun. Never invent a token absent from that side's prompt.",
    "manual_review": "All 50 pairs reviewed individually in REVIEW. Context choice is semantic and explicit, not a POS-model or image-based inference. Each row documents its decision. manual_context_required flags insertion/deletion/count cases whose changed words alone give an empty or article-only side.",
    "conflicts": "Prompt pair determines the edit when instruction disagrees; keep source spelling including rasing and brwon. Record conflicts without rewriting prompts.",
    "mask_mode": "union",
    "unchanged_tokens": [],
    "generation_inputs": "Only the upstream case fields plus case and seed are required. No evaluation_only paths or GT masks are included.",
    "matching": "Tokens are literal words/phrases from the corresponding cleaned prompt; actual model-tokenizer matching is performed by the runner, not executed here.",
}

# case | source tokens | target tokens | common context | manual context required | decision
# Every row was selected from the complete prompt pair, before generation/evaluation.
REVIEW = """000000000088|lemon,slices|apple,slices|slices|0|Fruit replacement; slices names the affected food.
000000000003|cat|dog||0|Animal replacement; use the changed animal nouns.
000000000103|abstract,painting|digital,painting|painting|0|Painting style change; retain the shared artwork noun.
000000000044|grass|rocks||0|Support/background material replacement; changed nouns identify the region.
000000000014|hanging,down,hands|rasing,hands|hands|0|Hand pose change; preserve the supplied rasing spelling.
122000000009|lamp|cake||0|Table object replacement; changed object nouns suffice.
113000000006|photo|painting||0|Framed artwork replacement; changed object nouns suffice.
123000000007|sofa|bed||0|Furniture replacement; changed object nouns suffice.
112000000009|angel|demon||0|Character replacement; changed character nouns suffice.
114000000004|mountains,reflection|ship,reflection|reflection|0|Replacement concerns the reflection; retain reflection context.
213000000005|couch|girl,couch|couch|1|Girl insertion; couch is the shared support surface on both sides.
211000000000|dog|flowers,mouth,dog|dog|1|Flowers at mouth insertion; dog is the shared affected animal; mouth is absent from source.
224000000002|grass|flowers,grass|grass|1|Flowers insertion following grass; grass is the shared location noun.
221000000003|bird|two,bird|bird|1|Count change from article a to two; exclude article a and anchor both sides to bird.
213000000008|table|hat,table|table|1|Hat insertion; table is the shared support surface.
314000000006|moon,sky|sky|sky|1|Moon deletion; sky is the common scene region retained in target.
322000000002|holding,coconut,woman|woman|woman|1|Held coconut deletion; woman is the shared affected person.
311000000002|laptop,table|table|table|1|Laptop deletion; table is the common support surface.
324000000008|two,boats,docked,shore,lake|lake|lake|1|Boats and shore clause deletion; lake is the sole shared scene noun.
322000000004|glasses,woman|woman|woman|1|Glasses deletion; woman is the shared affected person.
422000000001|long,hair|short,hair|hair|0|Hair length change; curly is unchanged and omitted.
412000000004|calm,woman|laughing,woman|woman|0|Expression/action change; woman anchors the attribute.
413000000002|burning,fire|extinguished,fire|fire|0|Fire state change; fire anchors the attribute.
423000000001|mug|glass||0|Vessel replacement; changed object nouns suffice.
412000000001|colorful,paint|drab,paint|paint|0|Paint appearance change; paint is shared object context.
524000000004|bag|upside,down,bag|bag|1|Orientation insertion; bag is the affected common object.
523000000000|tray|aerial,photography,tray|tray|1|Global viewpoint insertion; tray is the common scene support; token context does not guarantee global attention coverage.
511000000000|laying,down,dog|jumping,up,dog|dog|0|Posture/action replacement; remove prepositions and retain dog context.
523000000003|square,crate|round,crate|crate|0|Crate shape change; crate anchors the attribute.
513000000001|closed,lid|opened,lid|lid|0|Lid state change; use lid rather than the surrounding jar.
611000000001|pink,hearts|green,hearts|hearts|0|Prompt pair changes hearts; instruction mentions raven in error. Follow prompts.
624000000003|brwon,cabin|golden,cabin|cabin|0|Cabin color change; preserve supplied brwon spelling.
622000000003|black,veil|golden,veil|veil|0|Veil color change; veil anchors the attribute.
623000000002|red,lipstick|green,lipstick|lipstick|0|Lipstick color change; source red occurs twice so upstream token matching can also select powder red. Record ambiguity; retain lexical tokens.
622000000002|white,shirt|blue,shirt|shirt|0|Shirt color change; shirt anchors the attribute.
714000000000|leaf|jewels,leaf|leaf|1|Material insertion; leaf is the shared affected object.
712000000004|couple|garage,kits,couple|couple|1|Figurine-style insertion; couple is the common affected subject.
713000000001|lamp|crystal,lamp|lamp|1|Material insertion; lamp is the shared affected object.
712000000001|man|robot||0|Character replacement; changed nouns suffice even though source man has no brackets.
722000000000|shirt|sweater||0|Garment replacement; changed nouns suffice.
812000000002|girl|standing,front,beautiful,house,girl|girl|1|House/background clause insertion; girl is the shared subject. Target places girl in front of house whereas instruction wording reverses relation; follow target.
813000000002|marble,wall|flower,wall|wall|0|Wall material/appearance replacement; wall is common context.
814000000006|wave|yellow,background,wave|wave|1|Background insertion; wave is the shared scene noun, background is absent from source.
824000000007|rice,field|wonder,land||0|Scene replacement; use content words from both changed noun phrases.
811000000004|black,background|red,background|background|0|Background color change; background anchors the attribute.
921000000005|cat|oil,painting,cat|cat|1|Global style insertion; cat is the common subject noun.
911000000006|watercolor,illustration,chicken|anime,illustration,chicken|illustration,chicken|0|Global illustration style replacement; include medium noun and subject context.
912000000007|woman|pen,ink,sketch,woman|woman|1|Global style insertion; woman is the common subject noun.
911000000003|wolf|wall,mural,wolf|wolf|1|Global mural style insertion; wolf is the common subject noun.
911000000007|cartoon,giraffe|watercolor,giraffe|giraffe|0|Global style replacement; giraffe is the common subject noun."""


def clean_prompt(text):
    return text.replace("[", "").replace("]", "")


def changed_spans(source, target):
    a, b = re.findall(r"\w+(?:'\w+)?", source), re.findall(r"\w+(?:'\w+)?", target)
    return [{"operation": op, "source": a[i:j], "target": b[k:l]}
            for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if op != "equal"]


def reviewed_table():
    rows = {}
    for line in REVIEW.splitlines():
        case, src, tgt, context, fallback, note = line.split("|", 5)
        if case in rows:
            raise ValueError("Duplicate reviewed case: " + case)
        rows[case] = dict(source_mask_tokens=src.split(","), target_mask_tokens=tgt.split(","),
                          object_context=context.split(",") if context else [],
                          manual_context_required=fallback == "1", selection_note=note)
    return rows


def build(sources_path=ROOT / "sources_50.json"):
    payload = json.loads(Path(sources_path).read_text(encoding="utf-8-sig"))
    sources = payload["sources"]
    review = reviewed_table()
    ids = [s["case"] for s in sources]
    if len(ids) != 50 or len(set(ids)) != 50 or set(ids) != set(review):
        raise ValueError("sources_50.json must contain exactly the 50 individually reviewed case IDs")
    cases = []
    for index, source in enumerate(sources):
        case = source["case"]
        src, tgt = clean_prompt(source["source_prompt_raw"]), clean_prompt(source["target_prompt_raw"])
        selection = review[case]
        for prompt, key in ((src, "source_mask_tokens"), (tgt, "target_mask_tokens")):
            for token in selection[key]:
                if not re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", prompt):
                    raise ValueError(f"{case}: {key} token {token!r} absent from prompt")
        for context in selection["object_context"]:
            if context not in src or context not in tgt:
                raise ValueError(f"{case}: context must occur in both prompts")
        cases.append(dict(case=case, seed=source["seed"], image_name=case, target_code="edit",
                          image_path=source["source_image"], source_prompt=src, target_prompt=tgt,
                          source_mask_tokens=selection["source_mask_tokens"],
                          target_mask_tokens=selection["target_mask_tokens"], unchanged_tokens=[],
                          category=source["category"], record_index=index, target_index=0,
                          source_prompt_raw=source["source_prompt_raw"], target_prompt_raw=source["target_prompt_raw"],
                          instruction=source["instruction"], mask_mode="union",
                          changed_spans=changed_spans(src, tgt),
                          source_annotated_spans=re.findall(r"\[([^\]]*)\]", source["source_prompt_raw"]),
                          target_annotated_spans=re.findall(r"\[([^\]]*)\]", source["target_prompt_raw"]),
                          object_context=selection["object_context"],
                          manual_context_required=selection["manual_context_required"],
                          selection_note=selection["selection_note"], manually_reviewed=True,
                          review_basis="full_prompt_pair_only"))
    return dict(schema="doselens-tokens-v1", selection=payload["selection"], source_manifest="sources_50.json",
                path_base="package_root", rule=RULE, case_count=len(cases), reviewed_count=len(cases),
                manual_context_count=sum(c["manual_context_required"] for c in cases), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "sources_50.json")
    args = parser.parse_args()
    manifest = build(args.sources)
    (ROOT / "tokens_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case", "seed", "category", "source_prompt_raw", "target_prompt_raw", "source_prompt", "target_prompt",
              "source_mask_tokens", "target_mask_tokens", "unchanged_tokens", "object_context", "changed_spans",
              "mask_mode", "manually_reviewed", "manual_context_required", "selection_note", "review_basis"]
    with (ROOT / "token_selection.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in manifest["cases"]:
            writer.writerow({k: json.dumps(case[k], ensure_ascii=False) if isinstance(case[k], (list, dict)) else case[k] for k in fields})
    print(f"Prepared {manifest['case_count']} reviewed cases; {manifest['manual_context_count']} require explicit prompt-only context. No images or models loaded.")


if __name__ == "__main__":
    main()
