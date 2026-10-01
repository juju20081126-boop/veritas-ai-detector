#!/usr/bin/env python
"""
Sample REAL public corpora for training/dev breadth (older generators, real attacks):

  python scripts/corpus/build_public_ai.py [--only raid|mage|hc3]

  RAID  (Dugan et al. 2024, MIT)       10 converted train shards; 12 models incl. GPT-4/ChatGPT; 11 real attacks
  MAGE  (Li et al. 2024, Apache-2.0)   27 generators; plus OOD sets with GPT-4 text and paraphrased GPT-4/human text
  HC3   (Guo et al. 2023, CC-BY-SA-4) ChatGPT vs human answers

Outputs (ignored by git; rebuildable): data/corpus/public_ai/<name>.jsonl and data/corpus/human/<name>-human.jsonl.
Rows follow scripts/common/schema.py; `split` is train or dev (dev = 15% of groups by hash). Public sets never go into
the locked test.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from scripts.common import io_utils  # noqa: E402
from scripts.corpus import registry  # noqa: E402

SEED = 20261002
TODAY = datetime.date.today().isoformat()
OUT_AI = os.path.join(io_utils.DATA, "corpus", "public_ai")
OUT_HU = os.path.join(io_utils.DATA, "corpus", "human")
MIN_WORDS = 60

RAID_ATTACK_ID = {
    "none": "none", "paraphrase": "A4_raid-t5", "homoglyph": "A7_raid-homoglyph", "zero_width_space": "A7_raid-zwsp",
    "whitespace": "A7_raid-whitespace", "synonym": "A9_raid-synonym", "article_deletion": "A9_raid-article",
    "number": "A9_raid-number", "upper_lower": "A9_raid-case", "alternative_spelling": "A9_raid-altspell",
    "perplexity_misspelling": "A9_raid-misspell", "insert_paragraphs": "A9_raid-paragraphs",
}
RAID_AI_QUOTA = {"none": 1800, "paraphrase": 700, "synonym": 350, "homoglyph": 250, "zero_width_space": 150,
                 "whitespace": 150, "insert_paragraphs": 200, "upper_lower": 100, "article_deletion": 200,
                 "alternative_spelling": 200, "perplexity_misspelling": 200, "number": 100}


def split_of(group):
    return "dev" if int(hashlib.sha1(group.encode()).hexdigest(), 16) % 100 < 15 else "train"


def row(text, label, origin, gen, access, group, attack, parent, domain, esl, rev_split=None, **extra):
    r = {"id": io_utils.text_id(text), "text": text, "label": label, "origin": origin, "generator_id": gen,
         "access_path": access, "date": TODAY, "prompt_id": group, "attack_id": attack, "parent_id": parent,
         "domain": domain, "esl": esl, "words": io_utils.word_count(text), "split": rev_split or split_of(group)}
    r.update(extra)
    return r


def balanced(rows, key, n, rng):
    """Round-robin sample of n rows balanced over `key(row)` values."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, keys = [], sorted(buckets)
    while len(out) < n and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


# ----------------------------------------------------------------------------- RAID
def build_raid(rng):
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    rev = HfApi().dataset_info("liamdugan/raid").sha[:10]
    fs = fsspec.filesystem("https")
    cols = ["id", "adv_source_id", "source_id", "model", "decoding", "attack", "domain", "generation"]
    pooled = []
    for i in range(10):
        url = f"https://huggingface.co/datasets/liamdugan/raid/resolve/refs%2Fconvert%2Fparquet/raid/partial-train/{i:04d}.parquet"
        with fs.open(url, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for g in rng.sample(range(pf.num_row_groups), min(10, pf.num_row_groups)):
                pooled.extend(pf.read_row_group(g, columns=cols).to_pylist())
        print(f"  raid shard {i} done, pooled={len(pooled)}", flush=True)
    pooled = [r for r in pooled if io_utils.word_count(r["generation"]) >= MIN_WORDS]
    ai = [r for r in pooled if r["model"] != "human"]
    hu = [r for r in pooled if r["model"] == "human"]
    chosen = []
    for atk, q in RAID_AI_QUOTA.items():
        chosen += balanced([r for r in ai if r["attack"] == atk], lambda r: r["model"], q, rng)
    ai_rows = []
    for r in chosen:
        atk = RAID_ATTACK_ID[r["attack"]]
        ai_rows.append(row(r["generation"], "ai", "ai_raw" if atk == "none" else "ai_attacked", f"raid:{r['model']}",
                           f"public_dataset:raid@{rev}", f"raid:{r['source_id']}", atk,
                           None if atk == "none" else f"raid:{r['adv_source_id']}", r["domain"], False,
                           decoding=r["decoding"]))
    hu_clean = balanced([r for r in hu if r["attack"] == "none"], lambda r: r["domain"], 2500, rng)
    hu_att = balanced([r for r in hu if r["attack"] != "none"], lambda r: r["attack"], 300, rng)
    hu_rows = [row(r["generation"], "human", "human", "human:raid", f"corpus:raid-human@{rev}", f"raid:{r['source_id']}",
                   "none", None, r["domain"], False) for r in hu_clean]
    hu_rows += [row(r["generation"], "human", "human", "human:raid", f"public_dataset:raid@{rev}", f"raid:{r['source_id']}",
                    RAID_ATTACK_ID[r["attack"]], f"raid:{r['adv_source_id']}", r["domain"], False) for r in hu_att]
    registry.register("public_ai_corpora", "raid", {
        "hf_repo": "liamdugan/raid", "revision": rev, "license": "mit",
        "generators": sorted({r["model"] for r in ai}), "attacks": sorted(RAID_ATTACK_ID),
        "note": "sampled from the converted partial-train shards (domains: abstracts, books, news, poetry)"})
    registry.register("human_corpora", "raid-human", {
        "hf_repo": "liamdugan/raid", "revision": rev, "license": "mit", "verified_human": True,
        "published_before": "2022-11-30", "verification": "per RAID paper (human texts from pre-ChatGPT corpora); not independently verified"})
    return ai_rows, hu_rows


# ----------------------------------------------------------------------------- MAGE
def build_mage(rng):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    rev = HfApi().dataset_info("yaful/MAGE").sha[:10]
    df = pd.read_csv(hf_hub_download("yaful/MAGE", "train.csv", repo_type="dataset"), encoding="utf-8-sig")
    df["domain"] = df["src"].str.split("_").str[0]
    df = df[~df["domain"].isin(["squad", "hswag", "roct"])]
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    ai = df[df["label"] == 0].to_dict("records")
    hu = df[df["label"] == 1].to_dict("records")
    ai_s = balanced(ai, lambda r: r["src"].split("_machine_")[-1], 2500, rng)
    hu_s = balanced(hu, lambda r: r["domain"], 2500, rng)
    ai_rows = [row(r["text"], "ai", "ai_raw", "mage:" + r["src"].split("_machine_")[-1], f"public_dataset:mage@{rev}",
                   "mage:" + io_utils.text_id(r["text"]), "none", None, r["domain"], False) for r in ai_s]
    hu_rows = [row(r["text"], "human", "human", "human:mage", f"corpus:mage-human@{rev}", "mage:" + io_utils.text_id(r["text"]),
                   "none", None, r["domain"], False) for r in hu_s]
    # OOD: GPT-4 text and paraphrased GPT-4 / human text (dev only)
    for name in ("test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"):
        q = pd.read_csv(hf_hub_download("yaful/MAGE", name, repo_type="dataset"), encoding="utf-8-sig")
        for r in q.to_dict("records"):
            if io_utils.word_count(r["text"]) < MIN_WORDS:
                continue
            src, dom = r["src"], r["src"].split("_")[0]
            para = src.endswith("_para")
            human = "_human" in src
            atk = "A4_mage-para" if para else "none"
            g = "mage-ood:" + io_utils.text_id(r["text"])
            if human and not para:
                hu_rows.append(row(r["text"], "human", "human", "human:mage-ood", f"corpus:mage-human@{rev}", g, "none", None, dom, False, "dev"))
            elif human and para:
                hu_rows.append(row(r["text"], "human", "human", "human:mage-ood", f"public_dataset:mage@{rev}", g, atk, "mage-ood:unknown", dom, False, "dev"))
            else:
                ai_rows.append(row(r["text"], "ai", "ai_attacked" if para else "ai_raw", "mage:gpt-4", f"public_dataset:mage@{rev}",
                                   g, atk, "mage-ood:unknown" if para else None, dom, False, "dev"))
    registry.register("public_ai_corpora", "mage", {"hf_repo": "yaful/MAGE", "revision": rev, "license": "apache-2.0",
                                                    "note": "train sample + OOD GPT-4 / paraphrased sets (dev only)"})
    registry.register("human_corpora", "mage-human", {"hf_repo": "yaful/MAGE", "revision": rev, "license": "apache-2.0",
                                                      "verified_human": True, "published_before": "2022-11-30",
                                                      "verification": "sources: CMV, Yelp, XSum, TLDR, ELI5, WritingPrompts, SciGen, CNN, IMDB (all pre-ChatGPT)"})
    return ai_rows, hu_rows


# ----------------------------------------------------------------------------- HC3
def build_hc3(rng):
    from huggingface_hub import HfApi, hf_hub_download
    rev = HfApi().dataset_info("Hello-SimpleAI/HC3").sha[:10]
    rows = [json.loads(l) for l in open(hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset"), encoding="utf-8")]
    ai, hu = [], []
    for idx, r in enumerate(rows):
        g = f"hc3:{idx}"
        dom = r.get("source", "hc3")
        for a in r.get("chatgpt_answers", []):
            if io_utils.word_count(a) >= MIN_WORDS:
                ai.append((g, dom, a))
        for a in r.get("human_answers", []):
            if io_utils.word_count(a) >= MIN_WORDS:
                hu.append((g, dom, a))
    ai = balanced([{"g": g, "d": d, "t": t} for g, d, t in ai], lambda r: r["d"], 1500, rng)
    hu = balanced([{"g": g, "d": d, "t": t} for g, d, t in hu], lambda r: r["d"], 1500, rng)
    ai_rows = [row(r["t"], "ai", "ai_raw", "hc3:chatgpt", f"public_dataset:hc3@{rev}", r["g"], "none", None, r["d"], False) for r in ai]
    hu_rows = [row(r["t"], "human", "human", "human:hc3", f"corpus:hc3-human@{rev}", r["g"], "none", None, r["d"], False) for r in hu]
    registry.register("public_ai_corpora", "hc3", {"hf_repo": "Hello-SimpleAI/HC3", "revision": rev, "license": "cc-by-sa-4.0"})
    registry.register("human_corpora", "hc3-human", {"hf_repo": "Hello-SimpleAI/HC3", "revision": rev, "license": "cc-by-sa-4.0",
                                                     "verified_human": True, "published_before": "2022-11-30",
                                                     "verification": "human answers from Reddit ELI5, FiQA, MedDialog, WikiQA, open_qa (collected Dec 2022)"})
    return ai_rows, hu_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["raid", "mage", "hc3"], default=None)
    args = ap.parse_args()
    rng = random.Random(SEED)
    for name, fn in (("raid", build_raid), ("mage", build_mage), ("hc3", build_hc3)):
        if args.only and args.only != name:
            continue
        print(f"== {name}", flush=True)
        ai_rows, hu_rows = fn(rng)
        io_utils.write_jsonl(os.path.join(OUT_AI, f"{name}.jsonl"), ai_rows)
        io_utils.write_jsonl(os.path.join(OUT_HU, f"{name}-human.jsonl"), hu_rows)
        print(f"   AI rows={len(ai_rows)} human rows={len(hu_rows)} | splits AI {dict(Counter(r['split'] for r in ai_rows))} "
              f"human {dict(Counter(r['split'] for r in hu_rows))}")
        print(f"   AI by attack family: {dict(Counter(r['attack_id'].split('_')[0] for r in ai_rows))}")
        print(f"   AI generators: {len({r['generator_id'] for r in ai_rows})}  domains: {dict(Counter(r['domain'] for r in ai_rows).most_common(6))}")


if __name__ == "__main__":
    main()
