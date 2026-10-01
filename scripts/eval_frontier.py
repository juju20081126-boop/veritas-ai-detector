#!/usr/bin/env python
"""
Honest evaluation of AI-text detectors on the dev or locked split.

  python scripts/eval_frontier.py --split dev    --detectors shipped hc3_roberta binoculars [--max-per-cell N]
  python scripts/eval_frontier.py --split locked --detectors shipped hc3_roberta ... --out data/eval/results/locked_baselines.json

Protocol (GOAL_BRIEF section 7)
  * score every text with each detector (cached per detector/split in data/eval/scores/)
  * threshold = the 99th percentile of DEV clean-human scores (1% FPR on dev; never chosen on the evaluated split)
  * positives = label ai with origin ai_raw / ai_attacked; negatives = label human, origin human (clean = attack none; attacked
    controls reported separately); hybrid (mixed authorship) rows are reported separately by ai_share, never pooled
  * report TPR with Wilson 95% CIs: pooled, per generator x attack family, per length bucket, per genre; realized FPR overall,
    ESL vs native and by CEFR band; AUROC; TPR at the 5%-FPR threshold as well (RAID-comparable)
  * locked split: refuses after 3 evaluations and appends each one to data/locked/ACCESS_LOG.md (time, model hash, command)
Scores come from the detector code; no metric is hard-coded anywhere.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import io_utils, metrics, schema  # noqa: E402

D = io_utils.DATA
SCORES = os.path.join(D, "eval", "scores")
MAX_LOCKED = 3


def load_split(split):
    if split == "locked":
        rows = []
        for n in ("locked_ai.jsonl.gz", "locked_human.jsonl.gz"):
            rows += io_utils.read_jsonl(os.path.join(D, "locked", n))
        return rows
    return io_utils.read_jsonl(os.path.join(D, "splits", f"{split}.jsonl.gz"))


def locked_accesses():
    p = os.path.join(D, "locked", "ACCESS_LOG.md")
    if not os.path.exists(p):
        return 0
    return sum(1 for ln in open(p, encoding="utf-8") if ln.startswith("- 20"))


def cached_scores(det_name, split, rows, detector_factory, max_n=None):
    """Return {id: score}; computes (and caches) only the missing ids."""
    os.makedirs(SCORES, exist_ok=True)
    path = os.path.join(SCORES, f"{det_name}__{split}.jsonl")
    cache = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            cache[r["id"]] = r["s"]
    todo = [r for r in rows if r["id"] not in cache]
    if max_n:
        todo = todo[:max_n]
    if todo:
        det = detector_factory()
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), 32):
                chunk = todo[i:i + 32]
                sc = det.score([r["text"] for r in chunk])
                for r, s in zip(chunk, sc):
                    cache[r["id"]] = float(s)
                    f.write(json.dumps({"id": r["id"], "s": float(s)}) + "\n")
                f.flush()
                if (i // 32) % 10 == 0:
                    print(f"   [{det_name}/{split}] {i + len(chunk)}/{len(todo)} scored ({time.time() - t0:.0f}s)", flush=True)
    return cache


def thin(rows, neg_cap, public_cap, seed=0):
    """Seeded thinning of the DEV split for slow detectors: cap clean-human negatives and public-corpus AI rows.
    New frontier / attacked rows are never thinned. (The locked split is never thinned.)"""
    import random
    rng = random.Random(seed)
    keep, neg, pub = [], [], []
    for r in rows:
        if is_clean_neg(r):
            neg.append(r)
        elif r["access_path"].startswith("public_dataset") and r["origin"] != "human":
            pub.append(r)
        else:
            keep.append(r)
    if neg_cap and len(neg) > neg_cap:
        neg = rng.sample(neg, neg_cap)
    if public_cap and len(pub) > public_cap:
        pub = rng.sample(pub, public_cap)
    return keep + neg + pub


def is_pos(r):
    return r["label"] == "ai" and r["origin"] in ("ai_raw", "ai_attacked")


def is_clean_neg(r):
    return r["label"] == "human" and r["origin"] == "human" and r["attack_id"] == "none"


def cell(scores_by_id, rows, t):
    s = [scores_by_id[r["id"]] for r in rows if r["id"] in scores_by_id]
    k, n = metrics.rate_above(s, t)
    p, lo, hi = metrics.wilson(k, n)
    return {"k": k, "n": n, "rate": p, "lo": lo, "hi": hi}


def fmt(c):
    return "   n/a" if c["n"] == 0 else f"{100 * c['rate']:5.1f}% [{100 * c['lo']:4.1f}-{100 * c['hi']:5.1f}] n={c['n']}"


def evaluate(det_name, split, rows, sc, dev_neg_scores):
    t1 = metrics.threshold_at_fpr(dev_neg_scores, 0.01)
    t5 = metrics.threshold_at_fpr(dev_neg_scores, 0.05)
    pos = [r for r in rows if is_pos(r) and r["id"] in sc]
    neg = [r for r in rows if is_clean_neg(r) and r["id"] in sc]
    att_neg = [r for r in rows if r["label"] == "human" and r["origin"] == "human" and r["attack_id"] != "none" and r["id"] in sc]
    hyb = [r for r in rows if r["origin"] == "hybrid" and r["id"] in sc]
    res = {"detector": det_name, "split": split, "threshold_1pct_fpr_dev": t1, "threshold_5pct_fpr_dev": t5,
           "n_dev_neg": len(dev_neg_scores)}
    res["auroc_pooled"] = metrics.auroc([sc[r["id"]] for r in pos], [sc[r["id"]] for r in neg])
    res["fpr_clean_human"] = cell(sc, neg, t1)
    res["fpr_clean_human_at5"] = cell(sc, neg, t5)
    res["fpr_esl"] = cell(sc, [r for r in neg if r.get("esl")], t1)
    res["fpr_native"] = cell(sc, [r for r in neg if not r.get("esl")], t1)
    res["fpr_attacked_human"] = cell(sc, att_neg, t1)
    res["tpr_pooled"] = cell(sc, pos, t1)
    res["tpr_pooled_at5"] = cell(sc, pos, t5)
    res["tpr_raw"] = cell(sc, [r for r in pos if r["attack_id"] == "none"], t1)
    res["tpr_attacked"] = cell(sc, [r for r in pos if r["attack_id"] != "none"], t1)
    by = lambda keyf: {k: cell(sc, v, t1) for k, v in sorted(_group(pos, keyf).items())}  # noqa: E731
    res["tpr_by_generator_family"] = {f"{g}|{f}": c for (g, f), c in
                                      ((k, cell(sc, v, t1)) for k, v in sorted(_group(pos, lambda r: (r["generator_id"], schema.attack_family(r["attack_id"]))).items()))}
    res["tpr_by_attack"] = by(lambda r: r["attack_id"])
    res["tpr_by_length"] = by(lambda r: io_utils.length_bucket(r["words"]))
    res["tpr_by_genre"] = by(lambda r: schema.genre_of(r["domain"]))
    res["fpr_by_length"] = {k: cell(sc, v, t1) for k, v in sorted(_group(neg, lambda r: io_utils.length_bucket(r["words"])).items())}
    res["fpr_by_genre"] = {k: cell(sc, v, t1) for k, v in sorted(_group(neg, lambda r: schema.genre_of(r["domain"])).items())}
    res["fpr_by_cefr"] = {k: cell(sc, v, t1) for k, v in sorted(_group([r for r in neg if r.get("cefr")], lambda r: str(r["cefr"])[0]).items())}
    res["hybrid_flag_rate_by_ai_share"] = {k: cell(sc, v, t1) for k, v in sorted(_group(hyb, lambda r: str(r.get("ai_share", "?"))[:3]).items())}
    return res


def _group(rows, keyf):
    d = defaultdict(list)
    for r in rows:
        d[keyf(r)].append(r)
    return d


def print_report(res):
    print(f"\n=== {res['detector']} on {res['split']}  (threshold from dev clean-human 1% FPR = {res['threshold_1pct_fpr_dev']:.4f}, n_dev_neg={res['n_dev_neg']})")
    print(f"  AUROC pooled (AI vs clean human): {res['auroc_pooled']:.4f}")
    print(f"  realized FPR clean human : {fmt(res['fpr_clean_human'])}   | ESL: {fmt(res['fpr_esl'])}   | native: {fmt(res['fpr_native'])}")
    print(f"  FPR attacked-human controls: {fmt(res['fpr_attacked_human'])}")
    print(f"  TPR pooled @1%FPR-thr    : {fmt(res['tpr_pooled'])}   | @5%FPR-thr: {fmt(res['tpr_pooled_at5'])}")
    print(f"  TPR raw AI               : {fmt(res['tpr_raw'])}   | attacked AI: {fmt(res['tpr_attacked'])}")
    for title, key in (("by generator|family", "tpr_by_generator_family"), ("by attack", "tpr_by_attack"), ("by length bucket", "tpr_by_length"),
                       ("by genre", "tpr_by_genre"), ("FPR by length", "fpr_by_length"), ("FPR by genre", "fpr_by_genre"), ("FPR by CEFR band", "fpr_by_cefr"),
                       ("hybrid flagged by ai_share", "hybrid_flag_rate_by_ai_share")):
        if res.get(key):
            print(f"  -- {title}")
            for k, c in res[key].items():
                print(f"     {str(k):38s} {fmt(c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "locked"], required=True)
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new", type=int, default=None, help="score at most this many NEW texts per detector (speed cap; sampled in file order)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--neg-cap", type=int, default=0, help="DEV only: cap the number of clean-human negatives (slow detectors)")
    ap.add_argument("--public-cap", type=int, default=0, help="DEV only: cap the number of public-corpus AI rows (slow detectors)")
    ap.add_argument("--model-hash", default=None, help="identifier recorded in the locked access log")
    args = ap.parse_args()

    if args.split == "locked":
        used = locked_accesses()
        if used >= MAX_LOCKED:
            sys.exit(f"REFUSED: the locked test was already evaluated {used} times (max {MAX_LOCKED}).")
        with open(os.path.join(D, "locked", "ACCESS_LOG.md"), "a", encoding="utf-8") as f:
            f.write(f"- {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | model={args.model_hash or ','.join(args.detectors)} | "
                    f"cmd=python scripts/eval_frontier.py {' '.join(sys.argv[1:])}\n")
    from scripts.detectors.baselines import get_detector
    dev_rows = thin(load_split("dev"), args.neg_cap, args.public_cap)
    rows = dev_rows if args.split == "dev" else load_split(args.split)
    results = []
    for name in args.detectors:
        fac = lambda n=name: get_detector(n, threads=args.threads)  # noqa: E731
        dev_neg = [r for r in dev_rows if is_clean_neg(r)]
        sc_dev = cached_scores(name, "dev", dev_neg, fac, args.max_new)
        dev_neg_scores = [sc_dev[r["id"]] for r in dev_neg if r["id"] in sc_dev]
        if args.split == "dev":
            sc = cached_scores(name, "dev", rows, fac, args.max_new)
        else:
            sc = cached_scores(name, "locked", rows, fac, args.max_new)
        res = evaluate(name, args.split, rows, sc, dev_neg_scores)
        res["commit"] = os.popen(f'git -C "{REPO}" rev-parse --short HEAD').read().strip()
        print_report(res)
        results.append(res)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"results": results, "args": vars(args)}, open(args.out, "w", encoding="utf-8"), indent=1)
        print("\nsaved", args.out)


if __name__ == "__main__":
    main()
