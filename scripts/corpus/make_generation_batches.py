#!/usr/bin/env python
"""
Split data/corpus/prompts.jsonl into generation batches for real frontier-model generation.

  python scripts/corpus/make_generation_batches.py [--batch-size 20] [--round r2]

Writes, per generator and split, task files data/corpus/frontier/_tasks/<batch_id>.json (a list of
{"prompt_id", "instruction"}) and updates the manifest data/corpus/frontier/_manifest.json that tracks each batch
(pending -> written -> ingested). Prompts already covered by an ingested batch for that generator are skipped, so the
script can be re-run with a different batch size. A generation worker reads one task file and writes its replies to
data/corpus/frontier/_raw/<batch_id>.txt using the delimiter format

    === prompt_id: <id> ===
    <reply text>

which avoids JSON-escaping errors in long texts. scripts/corpus/ingest_generations.py then validates and ingests.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402

FRONTIER = os.path.join(io_utils.DATA, "corpus", "frontier")
SUBAGENT_ALIAS = {"claude-opus-5-5": "opus", "claude-sonnet-5-5": "sonnet", "claude-haiku-4-5": "haiku"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--round", default="r2")
    args = ap.parse_args()
    prompts = io_utils.read_jsonl(os.path.join(io_utils.DATA, "corpus", "prompts.jsonl"))
    os.makedirs(os.path.join(FRONTIER, "_tasks"), exist_ok=True)
    os.makedirs(os.path.join(FRONTIER, "_raw"), exist_ok=True)
    manifest_path = os.path.join(FRONTIER, "_manifest.json")
    old = json.load(open(manifest_path, encoding="utf-8"))["batches"] if os.path.exists(manifest_path) else []
    keep = [b for b in old if b.get("status") in ("ingested", "written", "needs_review", "running")]
    kept_ids = {b["batch_id"] for b in keep}
    batches = list(keep)
    for gen, alias in SUBAGENT_ALIAS.items():
        done = set()
        for f in glob.glob(os.path.join(FRONTIER, gen, "*.jsonl")):
            done |= {r["prompt_id"] for r in io_utils.read_jsonl(f) if not (set(r.get("quality_flags", [])) &
                     {"too_short", "too_long", "refusal", "meta_words", "duplicate_or_missing"})}
        for b in keep:   # prompts in a running/written batch are also covered
            if b["generator_id"] == gen:
                done |= {t["prompt_id"] for t in json.load(open(os.path.join(REPO, b["tasks_path"]), encoding="utf-8"))}
        items = [p for p in prompts if gen in p.get("gen_models", []) and p["prompt_id"] not in done]
        k = 0
        for split in ("locked", "dev", "train"):
            sel = [p for p in items if p["split"] == split]
            by_t = defaultdict(list)
            for p in sel:
                by_t[p["task_type"]].append(p)
            mixed = []
            while any(by_t.values()):
                for t in list(by_t):
                    if by_t[t]:
                        mixed.append(by_t[t].pop(0))
            for i in range(0, len(mixed), args.batch_size):
                chunk = mixed[i:i + args.batch_size]
                bid = f"{alias}__{split}__{args.round}_{k:03d}"
                k += 1
                tasks = [{"prompt_id": p["prompt_id"], "instruction": p["instruction"]} for p in chunk]
                with open(os.path.join(FRONTIER, "_tasks", bid + ".json"), "w", encoding="utf-8") as f:
                    json.dump(tasks, f, ensure_ascii=False, indent=1)
                if bid not in kept_ids:
                    batches.append({"batch_id": bid, "generator_id": gen, "split": split, "n": len(chunk),
                                    "tasks_path": f"data/corpus/frontier/_tasks/{bid}.json",
                                    "out_path": f"data/corpus/frontier/_raw/{bid}.txt", "status": "pending"})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"batch_size": args.batch_size, "batches": batches}, f, indent=1)
    pend = [b for b in batches if b["status"] == "pending"]
    print("batches total:", len(batches), "| pending:", len(pend), dict(Counter((b["generator_id"], b["split"]) for b in pend)))
    print("texts still to generate:", sum(b["n"] for b in pend))


if __name__ == "__main__":
    main()
