#!/usr/bin/env python
"""
Validate and ingest the rewrites produced by subagents for the LLM attack families (A1, A2, A3).

  python scripts/corpus/ingest_llm_attacks.py            # every batch whose raw file exists

Quality gate per rewritten text: not empty, no refusal/meta text, word-count ratio 0.6-1.6, sentence-embedding cosine to its
parent >= 0.80, not a near copy (5-gram Jaccard < 0.9). Rejected counts are reported per family.
Output rows: data/corpus/attacked/{A1,A2,A3}/<generator>__<split>.jsonl (schema in scripts/common/schema.py).
"""

import datetime
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402
from scripts.corpus.ingest_generations import _META, _REFUSE, strip_wrapper  # noqa: E402

ATT = os.path.join(io_utils.DATA, "corpus", "attacked")
TODAY = datetime.date.today().isoformat()
ALIAS = {"opus": "claude-code-subagent:opus", "sonnet": "claude-code-subagent:sonnet", "haiku": "claude-code-subagent:haiku"}


def parse_raw(path):
    txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^=== id: (\S+) ===[ \t]*$", txt)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def source_index():
    idx = {}
    for f in glob.glob(os.path.join(io_utils.DATA, "corpus", "frontier", "*", "*.jsonl")):
        for r in io_utils.read_jsonl(f):
            idx[r["id"]] = r
    for f in glob.glob(os.path.join(ATT, "A1", "*.jsonl")):
        for r in io_utils.read_jsonl(f):
            idx[r["id"]] = r
    return idx


def main():
    from scripts.corpus.attacks import QualityGate
    mpath = os.path.join(ATT, "_manifest.json")
    m = json.load(open(mpath, encoding="utf-8"))
    gate = None
    idx = source_index()
    new_rows = defaultdict(list)   # (family, gen, split) -> rows
    stats = defaultdict(Counter)
    for b in m["batches"]:
        raw = os.path.join(REPO, b["out_path"])
        if b["status"] in ("ingested",) or not os.path.exists(raw):
            continue
        gate = gate or QualityGate()
        tasks = {t["id"]: t for t in json.load(open(os.path.join(REPO, b["tasks_path"]), encoding="utf-8"))}
        got = dict(parse_raw(raw))
        for tid, t in tasks.items():
            fam = "A3" if t["mode"] == "humanize" else ("A2" if tid.endswith(".A2p2") else "A1")
            key = (fam, b["generator_id"], b["split"])
            if tid not in got:
                stats[fam]["missing"] += 1
                continue
            text, _ = strip_wrapper(got[tid])
            if _REFUSE.search(text[:400]) or _META.search(text):
                stats[fam]["refusal_or_meta"] += 1
                continue
            ok, why, cos = gate.check(t["text"], text)
            if not ok:
                stats[fam][why] += 1
                continue
            parent = idx[t["source_id"]]
            if fam == "A1":
                aid = f"A1_{t['strength']}"
            elif fam == "A2":
                aid = "A2_x2"
            else:
                aid = "A3_humanizer"
            row = {"id": io_utils.text_id(text), "text": text, "label": "ai", "origin": "ai_attacked", "generator_id": b["generator_id"],
                   "access_path": ALIAS[b["worker"]], "date": TODAY, "prompt_id": parent["prompt_id"], "attack_id": aid,
                   "parent_id": parent["id"], "domain": parent["domain"], "esl": False, "words": io_utils.word_count(text),
                   "split": b["split"], "task_type": parent.get("task_type"), "cosine_to_parent": round(cos, 3)}
            if tid.endswith(".A2p1"):
                row["role"] = "a2_pass1"
            new_rows[key].append(row)
            stats[fam]["ok"] += 1
        b["status"] = "ingested"
    for (fam, gen, split), rows in new_rows.items():
        path = os.path.join(ATT, fam, f"{gen}__{split}.jsonl")
        old = io_utils.read_jsonl(path) if os.path.exists(path) else []
        have = {r["id"] for r in old}
        io_utils.write_jsonl(path, old + [r for r in rows if r["id"] not in have])
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    for fam in sorted(stats):
        print(fam, dict(stats[fam]))
    print("files written:", sorted({f"{k[0]}/{k[1]}__{k[2]}" for k in new_rows}))


if __name__ == "__main__":
    main()
