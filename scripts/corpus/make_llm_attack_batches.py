#!/usr/bin/env python
"""
Create subagent task batches for the LLM attack families A1 (paraphrase), A2 (iterative) and A3 (humanizer prompt).

  python scripts/corpus/make_llm_attack_batches.py --split locked --stage 1 --n1 40 --n2 40 --n3 40
  python scripts/corpus/make_llm_attack_batches.py --split locked --stage 2          # A2 second pass (after pass 1 is ingested)

Stage 1 per generator: A1 (strengths cycled light/medium/heavy) and A2-pass-1 (medium) are rewritten by the OTHER big Claude
model; A3 (humanizer prompt) by Haiku 4.5. Stage 2: A2-pass-2 ("rewrite again") by Haiku on the pass-1 outputs.
Task files: data/corpus/attacked/_tasks/<batch>.json; replies go to data/corpus/attacked/_raw/<batch>.txt using
    === id: <task id> ===
    <rewritten text>
Manifest: data/corpus/attacked/_manifest.json. Ingest with scripts/corpus/ingest_llm_attacks.py.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402

ATT = os.path.join(io_utils.DATA, "corpus", "attacked")
HARD = {"too_short", "too_long", "refusal", "meta_words", "duplicate_or_missing"}
OTHER = {"claude-opus-5-5": "sonnet", "claude-sonnet-5-5": "opus"}
GENS = list(OTHER)
STRENGTHS = ["light", "medium", "heavy"]


def raw_sources(gen, split):
    rows = []
    for f in glob.glob(os.path.join(io_utils.DATA, "corpus", "frontier", gen, "*.jsonl")):
        for r in io_utils.read_jsonl(f):
            if r["split"] == split and not (set(r.get("quality_flags", [])) & HARD) and r["words"] >= 100:
                rows.append(r)
    return sorted(rows, key=lambda r: r["id"])


def load_manifest():
    p = os.path.join(ATT, "_manifest.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"batches": []}


def save_manifest(m):
    os.makedirs(ATT, exist_ok=True)
    with open(os.path.join(ATT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)


def add_batches(m, tasks, gen, split, kind, worker, size):
    os.makedirs(os.path.join(ATT, "_tasks"), exist_ok=True)
    os.makedirs(os.path.join(ATT, "_raw"), exist_ok=True)
    k = sum(1 for b in m["batches"] if b["generator_id"] == gen and b["split"] == split and b["kind"] == kind)
    for i in range(0, len(tasks), size):
        chunk = tasks[i:i + size]
        bid = f"{kind}__{worker}__{gen.split('-')[1]}__{split}__{k:02d}"
        k += 1
        with open(os.path.join(ATT, "_tasks", bid + ".json"), "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=1)
        m["batches"].append({"batch_id": bid, "generator_id": gen, "split": split, "kind": kind, "worker": worker, "n": len(chunk),
                             "tasks_path": f"data/corpus/attacked/_tasks/{bid}.json", "out_path": f"data/corpus/attacked/_raw/{bid}.txt",
                             "status": "pending"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["locked", "dev", "train"])
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n1", type=int, default=40)
    ap.add_argument("--n2", type=int, default=40)
    ap.add_argument("--n3", type=int, default=40)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    m = load_manifest()
    for gen in GENS:
        rng = random.Random(f"{args.seed}-{gen}-{args.split}")
        if args.stage == 1:
            src = raw_sources(gen, args.split)
            need = args.n1 + args.n2 + args.n3
            if len(src) < max(args.n1, args.n2, args.n3):
                print(f"{gen}/{args.split}: only {len(src)} usable raw texts; skipping (generate/ingest first)")
                continue
            idx = list(range(len(src)))
            rng.shuffle(idx)

            def take(n, offset):  # different subsets per family where possible (wrap-around if few sources)
                return [src[idx[(offset + j) % len(idx)]] for j in range(n)]
            a1 = take(args.n1, 0)
            a2 = take(args.n2, args.n1)
            a3 = take(args.n3, args.n1 + args.n2)
            t1 = [{"id": f"{s['id']}.A1_{STRENGTHS[j % 3]}", "source_id": s["id"], "mode": "paraphrase", "strength": STRENGTHS[j % 3],
                   "text": s["text"]} for j, s in enumerate(a1)]
            t2 = [{"id": f"{s['id']}.A2p1", "source_id": s["id"], "mode": "paraphrase", "strength": "medium", "text": s["text"]} for s in a2]
            t3 = [{"id": f"{s['id']}.A3_humanizer", "source_id": s["id"], "mode": "humanize", "strength": "n/a", "text": s["text"]} for s in a3]
            add_batches(m, t1 + t2, gen, args.split, "a1a2p1", OTHER[gen], args.size)
            add_batches(m, t3, gen, args.split, "a3", "haiku", args.size)
        else:
            # stage 2: second pass over ingested pass-1 outputs
            p1 = os.path.join(ATT, "A1", f"{gen}__{args.split}.jsonl")
            if not os.path.exists(p1):
                print(f"{gen}/{args.split}: no ingested A1/A2 pass-1 file yet")
                continue
            rows = [r for r in io_utils.read_jsonl(p1) if r.get("role") == "a2_pass1"]
            tasks = [{"id": f"{r['id']}.A2p2", "source_id": r["id"], "mode": "rewrite_again", "strength": "n/a", "text": r["text"]} for r in rows]
            add_batches(m, tasks, gen, args.split, "a2p2", "haiku", args.size)
    save_manifest(m)
    pend = [b for b in m["batches"] if b["status"] == "pending"]
    print("batches:", len(m["batches"]), "| pending:", len(pend), dict(Counter((b["kind"], b["worker"], b["split"]) for b in pend)))


if __name__ == "__main__":
    main()
