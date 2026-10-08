#!/usr/bin/env python
"""
Clean, modern HUMAN text (typed by people in 2023) for the TF-IDF detector: Dolly-15k responses (CC BY-SA 3.0) and long English
OASST1 prompter messages (Apache-2.0), from the local Hugging Face cache (downloaded on first use).

  data/corpus/human/modern_human_heldout.jsonl  1,100 texts, never trained on; measures false positives on unseen modern humans
  data/corpus/human/modern_human_train.jsonl    2,600 texts, disjoint from heldout; extra training negatives

Without these negatives the detector learns "scraped-corpus formatting = human" and flags ~3.8% of clean modern human text at a
1%-FPR threshold (scratch experiment 2026-10-07). Seeds and filters are fixed so the files rebuild identically.

  python scripts/build_modern_human.py
"""

import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402

OUT_DIR = os.path.join(io_utils.DATA, "corpus", "human")
HELDOUT = os.path.join(OUT_DIR, "modern_human_heldout.jsonl")
TRAIN = os.path.join(OUT_DIR, "modern_human_train.jsonl")


def _dolly():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    for ln in open(p, encoding="utf-8"):
        d = json.loads(ln)
        yield d["response"].strip(), "dolly:" + d["category"]


def _oasst():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("OpenAssistant/oasst1", "2023-04-12_oasst_prompts.messages.jsonl.gz", repo_type="dataset")
    for ln in gzip.open(p, "rt", encoding="utf-8"):
        d = json.loads(ln)
        if d.get("lang") == "en" and d.get("role") == "prompter":
            yield (d.get("text") or "").strip(), "oasst:prompter"


def _sample(seed, min_words, n_dolly, n_oasst, exclude=frozenset()):
    rng = random.Random(seed)
    dolly = [{"text": t, "domain": dom} for t, dom in _dolly()
             if min_words <= len(t.split()) <= 700 and t.isascii() and t not in exclude]
    rows = rng.sample(dolly, min(n_dolly, len(dolly)))
    oasst = [{"text": t, "domain": dom} for t, dom in _oasst()
             if min_words <= len(t.split()) <= 700 and t not in exclude]
    return rows + rng.sample(oasst, min(n_oasst, len(oasst)))


def _write(path, rows, prefix, split):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r.update({"id": prefix + hashlib.sha1(r["text"].encode()).hexdigest()[:16], "label": "human", "origin": "human",
                      "attack_id": "none", "generator_id": "human:" + r["domain"].split(":")[0], "split": split,
                      "words": len(r["text"].split())})
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}: {len(rows)} rows {dict(Counter(r['domain'].split(':')[0] for r in rows))}")


def main():
    heldout = _sample(20261007, 80, 800, 300)
    _write(HELDOUT, heldout, "ood-", "ood")
    train = _sample(7, 50, 2000, 600, exclude=frozenset(r["text"] for r in heldout))
    _write(TRAIN, train, "xh-", "extra")


if __name__ == "__main__":
    main()
