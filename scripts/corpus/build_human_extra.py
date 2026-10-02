#!/usr/bin/env python
"""
Extra HUMAN texts to balance the genres that the public AI corpora over-represent (RAID abstracts and news):

  python scripts/corpus/build_human_extra.py

  - arXiv abstracts (gfissore/arxiv-abstracts-2021, CC0): 1,000 abstracts of 110-300 words
  - CNN news (abisee/cnn_dailymail): 1,200 article excerpts of 120-450 words

Documents already used as matched prompts (data/corpus/human/matched_docs.jsonl) are excluded. Output (ignored by git):
data/corpus/human/extra_human.jsonl, rows per scripts/common/schema.py with prompt_id = "<corpus>:<doc id>".
"""

import datetime
import gzip
import json
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scripts.common import io_utils, textnorm  # noqa: E402
from scripts.corpus import registry  # noqa: E402
from scripts.corpus.build_prompts import cut_to_words, norm_ws, reservoir  # noqa: E402

SEED = 20261003
TODAY = datetime.date.today().isoformat()


def row(text, corpus, rev, doc_id, domain):
    return {"id": io_utils.text_id(text), "text": text, "label": "human", "origin": "human", "generator_id": f"human:{corpus}",
            "access_path": f"corpus:{corpus}@{rev}", "date": TODAY, "prompt_id": f"{corpus}:{doc_id}", "attack_id": "none",
            "parent_id": None, "domain": domain, "esl": False, "words": io_utils.word_count(text)}


def main():
    rng = random.Random(SEED)
    reg = registry.load()
    used = {r["doc_id"] for r in io_utils.read_jsonl(os.path.join(io_utils.DATA, "corpus", "human", "matched_docs.jsonl"))}
    out = []

    rev = reg["human_corpora"]["arxiv-abstracts-2021"]["revision"]
    path = hf_hub_download("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz", repo_type="dataset")

    def gen():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ab = norm_ws(r.get("abstract", "").replace("\n", " "))
                if r["id"] not in used and 110 <= io_utils.word_count(ab) <= 300:
                    yield {"doc_id": r["id"], "text": ab}
    pool = reservoir(gen(), 1000, rng)
    out += [row(d["text"], "arxiv-abstracts-2021", rev, d["doc_id"], "academic") for d in pool]
    print("arXiv abstracts:", len(pool))

    rev = reg["human_corpora"]["cnn_dailymail"]["revision"]
    items = []
    for f in ("1.0.0/validation-00000-of-00001.parquet", "1.0.0/test-00000-of-00001.parquet", "1.0.0/train-00000-of-00003.parquet"):
        t = pq.read_table(hf_hub_download("abisee/cnn_dailymail", f, repo_type="dataset"), columns=["article", "id"]).to_pylist()
        for r in t:
            m = re.match(r"^.{0,120}?\(CNN\)\s*(?:--|-|—)?\s*", r["article"], flags=re.S)
            if m and r["id"] not in used:
                a = norm_ws(r["article"][m.end():])
                if io_utils.word_count(a) >= 320 and "Follow @" not in a:
                    items.append({"doc_id": r["id"], "text": a})
        if len(items) > 3000:
            break
    rng.shuffle(items)
    n = 0
    for d in items:
        if n >= 1200:
            break
        txt = textnorm.fix_tokenization(cut_to_words(d["text"], rng.choice([120, 180, 250, 320, 450])))
        if io_utils.word_count(txt) >= 100:
            out.append(row(txt, "cnn_dailymail", rev, d["doc_id"], "news"))
            n += 1
    print("CNN excerpts:", n)
    io_utils.write_jsonl(os.path.join(io_utils.DATA, "corpus", "human", "extra_human.jsonl"), out)
    print("wrote", len(out), "rows")


if __name__ == "__main__":
    main()
