#!/usr/bin/env python
"""
Apply the local/programmatic attack families (A4, A5, A6, A7) to ingested frontier texts (and, as benign controls, to human texts).

  python scripts/corpus/make_attacks.py --gen claude-opus-5-5 --split locked --family A7 --n 40
  python scripts/corpus/make_attacks.py --gen human --split locked --family A5 --n 20

Sources: usable rows of data/corpus/frontier/<gen>/*.jsonl for the chosen split (or matched human docs for --gen human).
Output: data/corpus/attacked/<family>/<gen>__<split>.jsonl (rows follow scripts/common/schema.py; the quality gate drops
attacked texts whose embedding cosine to the parent is < 0.80). LLM families (A1-A3) are produced by subagent batches; see
scripts/corpus/make_llm_attack_batches.py.
"""

import argparse
import datetime
import glob
import json
import os
import random
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402
from scripts.corpus import attacks  # noqa: E402

HARD = {"too_short", "too_long", "refusal", "meta_words", "duplicate_or_missing"}
TODAY = datetime.date.today().isoformat()
OUT = os.path.join(io_utils.DATA, "corpus", "attacked")


def load_sources(gen, split):
    if gen == "human":
        rows = []
        for r in io_utils.read_jsonl(os.path.join(io_utils.DATA, "corpus", "human", "matched_docs.jsonl")):
            if r["split"] == split and r["words"] >= 80:
                rows.append({"id": io_utils.text_id(r["text"]), "text": r["text"], "prompt_id": r["prompt_id"], "domain": r["domain"],
                             "generator_id": f"human:{r['corpus']}", "access_path": f"corpus:{r['corpus']}@{r['revision']}", "task_type": r["task_type"]})
        return rows
    rows = []
    for f in glob.glob(os.path.join(io_utils.DATA, "corpus", "frontier", gen, "*.jsonl")):
        for r in io_utils.read_jsonl(f):
            if r["split"] == split and not (set(r.get("quality_flags", [])) & HARD) and r["words"] >= 100:
                rows.append(r)
    return rows


def donors(split):
    d = {}
    for r in io_utils.read_jsonl(os.path.join(io_utils.DATA, "corpus", "human", "matched_docs.jsonl")):
        if r["split"] == split:
            d[r["prompt_id"]] = r["text"]
    return d


def variants(family):
    return {"A4": ["t5"], "A5": ["zh", "de"], "A6": ["p25", "p50", "p75"], "A7": ["homoglyph", "zwsp", "typo"]}[family]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="claude-opus-5-5 | claude-sonnet-5-5 | human")
    ap.add_argument("--split", required=True, choices=["locked", "dev", "train"])
    ap.add_argument("--family", required=True, choices=["A4", "A5", "A6", "A7"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    rng = random.Random(f"{args.seed}-{args.gen}-{args.split}-{args.family}")
    src = load_sources(args.gen, args.split)
    if args.family == "A6":
        don = donors(args.split)
        src = [r for r in src if r["prompt_id"] in don]
    if not src:
        sys.exit(f"no usable sources for {args.gen}/{args.split}")
    rng.shuffle(src)
    # oversample sources so that quality-gate rejections still leave >= n rows
    pick = src[: min(len(src), int(args.n * 1.5))]
    vs = variants(args.family)
    gate = attacks.QualityGate() if args.family in ("A4", "A5") else None
    t5 = attacks.T5Paraphraser(args.threads) if args.family == "A4" else None
    bts = {v: attacks.BackTranslator(v, args.threads) for v in vs} if args.family == "A5" else {}
    out, rejected = [], Counter()
    t0 = time.time()
    for i, p in enumerate(pick):
        if len(out) >= args.n:
            break
        v = vs[i % len(vs)]
        ai_share = None
        if args.family == "A4":
            child, tool, aid = t5.paraphrase(p["text"]), f"local_model:{attacks.T5Paraphraser.NAME}", "A4_t5"
        elif args.family == "A5":
            child, tool, aid = bts[v].run(p["text"]), f"local_model:{attacks.BackTranslator.PAIRS[v][0]}", f"A5_{v}"
        elif args.family == "A6":
            share = {"p25": 0.25, "p50": 0.5, "p75": 0.75}[v]
            child, ai_share = attacks.interleave(p["text"], don[p["prompt_id"]], share, rng)
            tool, aid = "programmatic:interleave", f"A6_{v}"
        else:
            fn = {"homoglyph": attacks.attack_homoglyph, "zwsp": attacks.attack_zwsp, "typo": attacks.attack_typo}[v]
            child, tool, aid = fn(p["text"], rng), f"programmatic:{v}", f"A7_{v}"
        if not child:
            rejected["empty"] += 1
            continue
        if args.family in ("A4", "A5"):
            ok, why, cos = gate.check(p["text"], child)
            if not ok:
                rejected[why] += 1
                continue
        human_ctrl = args.gen == "human"
        label = "human" if human_ctrl else "ai"
        origin = "human" if human_ctrl else "ai_attacked"
        if args.family == "A6":
            origin, label = "hybrid", ("ai" if ai_share >= 0.5 else "human")
        row = {"id": io_utils.text_id(child), "text": child, "label": label, "origin": origin, "generator_id": p["generator_id"],
               "access_path": tool, "date": TODAY, "prompt_id": p["prompt_id"], "attack_id": aid, "parent_id": p["id"],
               "domain": p["domain"], "esl": False, "words": io_utils.word_count(child), "split": args.split,
               "task_type": p.get("task_type")}
        if ai_share is not None:
            row["ai_share"] = round(ai_share, 3)
        out.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {len(out)}/{args.n} done ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(OUT, args.family, f"{args.gen}__{args.split}.jsonl")
    io_utils.write_jsonl(path, out)
    print(f"{args.gen}/{args.split}/{args.family}: wrote {len(out)} rows (sources tried {i + 1}, rejected {dict(rejected)}) "
          f"in {time.time() - t0:.0f}s -> {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
