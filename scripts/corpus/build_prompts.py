#!/usr/bin/env python
"""
Build the generation prompt set and the matched human documents (seeded, reproducible).

  python scripts/corpus/build_prompts.py

Outputs
  data/corpus/prompts.jsonl                 (tracked)  one row per prompt: id, task, domain, split, instruction, ...
  data/corpus/human/matched_docs.jsonl      (ignored)  the human text that matches each matched prompt (same topic)
  data/corpus/registry.json                 (tracked)  human corpora registered with revision / license / date

Matched prompts (news, abstract, email, story, explain) come from corpora that also contain a HUMAN text for the same
topic, so a frontier model's answer and a human text share a topic and group id (no topic shortcut; same split).
General prompts (oasst1, dolly) are real human-written user requests with no matched human text.
"""

import argparse
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
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from scripts.common import io_utils, textnorm  # noqa: E402
from scripts.corpus import registry  # noqa: E402

SEED = 20261001
GEN_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5"]

# prompts per task type: (total, locked, dev, train)  and generation quota per split (locked, dev, train)
PLAN = {
    "news":    {"n": 110, "split": (24, 16, 70), "gen": (24, 10, 27)},
    "abstract": {"n": 110, "split": (24, 16, 70), "gen": (24, 10, 27)},
    "email":   {"n": 70,  "split": (15, 10, 45), "gen": (15, 6, 17)},
    "story":   {"n": 100, "split": (22, 14, 64), "gen": (22, 8, 24)},
    "explain": {"n": 80,  "split": (18, 10, 52), "gen": (18, 6, 20)},
    "general": {"n": 170, "split": (47, 34, 89), "gen": (47, 20, 35)},
}
DOMAIN = {"news": "news", "abstract": "academic", "email": "email", "story": "creative", "explain": "explain", "general": "general"}
WORDS = {"news": [250, 300, 400, 500], "story": [300, 400, 500, 600], "general": [150, 250, 350, 500, 700]}

TEMPLATES = {
    "news": ["Write a news article of about {w} words based on these key points:\n{points}",
             "Using the following highlights, write a {w}-word news story:\n{points}",
             "Turn these bullet points into a news article (roughly {w} words):\n{points}"],
    "abstract": ["Write an abstract of about {w} words for a research paper titled \"{title}\".",
                 "Draft the abstract (around {w} words) for the paper \"{title}\"."],
    "email": ["Write an email with the subject line \"{subject}\". Aim for about {w} words.",
              "Please draft a work email about: {subject} (around {w} words)."],
    "story": ["{prompt}\n\nWrite a short story of about {w} words.",
              "Write a {w}-word story for this prompt: {prompt}"],
    "explain": ["{question}\n\nExplain it in about {w} words.", "{question} (answer in roughly {w} words)"],
}


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip()) if s]


def cut_to_words(text, target):
    """Leading paragraphs (then sentences, if one paragraph is huge) of `text` totalling about `target` words.
    Paragraph breaks are preserved so structure is not a label leak."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, n = [], 0
    for p in paras:
        w = io_utils.word_count(p)
        if out and n + w > target * 1.2:
            break
        if not out and w > target * 1.2:
            cut, m = [], 0
            for s in sentences(p):
                sw = io_utils.word_count(s)
                if cut and m + sw > target * 1.2:
                    break
                cut.append(s)
                m += sw
                if m >= target * 0.95:
                    break
            out.append(" ".join(cut))
            n += m
            break
        out.append(p)
        n += w
        if n >= target * 0.95:
            break
    return "\n\n".join(out)


def round25(n, lo=60, hi=500):
    return int(min(hi, max(lo, 25 * round(n / 25))))


def dl(repo, fname):
    return hf_hub_download(repo, fname, repo_type="dataset")


def rev(repo):
    return HfApi().dataset_info(repo).sha[:10]


def norm_ws(s):
    return re.sub(r"[ \t]+", " ", s.replace("\r\n", "\n")).strip()


def reservoir(it, k, rng):
    pool = []
    for i, x in enumerate(it):
        if len(pool) < k:
            pool.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                pool[j] = x
    return pool


# ----------------------------------------------------------------------------- matched sources
def load_news(rng, k):
    items = []
    for f in ("1.0.0/validation-00000-of-00001.parquet", "1.0.0/test-00000-of-00001.parquet"):
        for r in pq.read_table(dl("abisee/cnn_dailymail", f), columns=["article", "highlights", "id"]).to_pylist():
            a = r["article"]
            m = re.match(r"^.{0,120}?\(CNN\)\s*(?:--|-|—)?\s*", a, flags=re.S)
            if not m:
                continue
            a = norm_ws(a[m.end():])
            if io_utils.word_count(a) < 320 or "Follow @" in a:
                continue
            # highlights come as one string; sentences are joined by " . " (sometimes newlines)
            parts = [p.strip() for p in re.split(r"\s\.\s+|\n", r["highlights"]) if p.strip()]
            pts = [p.rstrip(" .") + "." for p in parts][:4]
            if len(pts) < 2:
                continue
            items.append({"doc_id": r["id"], "text": a, "points": "\n".join("- " + p for p in pts)})
    return rng.sample(items, k)


def load_abstracts(rng, k):
    path = dl("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz")

    def gen():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ab = norm_ws(r.get("abstract", "").replace("\n", " "))
                ti = norm_ws(r.get("title", "").replace("\n", " "))
                w = io_utils.word_count(ab)
                if 110 <= w <= 300 and len(ti.split()) >= 4 and "$" not in ti:
                    yield {"doc_id": r["id"], "text": ab, "title": ti}
    pool = reservoir(gen(), 1500, rng)
    return rng.sample(pool, k)


def load_emails(rng, k):
    rows = pq.read_table(dl("Yale-LILY/aeslc", "data/train-00000-of-00001.parquet")).to_pylist()
    items = []
    for i, r in enumerate(rows):
        body = norm_ws(r["email_body"])
        subj = norm_ws(r["subject_line"])
        w = io_utils.word_count(body)
        bad = ("Original Message", "Forwarded by", "http", "---", "\n>")
        if 70 <= w <= 350 and 2 <= len(subj.split()) <= 10 and not any(b in body for b in bad):
            items.append({"doc_id": f"aeslc-{i}", "text": body, "subject": subj})
    return rng.sample(items, k)


def _detok(s):
    """WritingPrompts is PTB-tokenized ("You 've", "ca n't", " ."): undo it so style is not a label leak."""
    from sacremoses import MosesDetokenizer
    global _MD
    try:
        _MD
    except NameError:
        _MD = MosesDetokenizer(lang="en")
    return _MD.detokenize(s.split())


def load_stories(rng, k):
    t = pq.read_table(dl("euclaise/writingprompts", "data/test-00000-of-00001-16503b0c26ed00c6.parquet")).to_pylist()
    items = []
    for i, r in enumerate(t):
        paras = [_detok(p) for p in r["story"].split("<newline>") if p.strip()]
        story = "\n\n".join(paras)
        prompt = textnorm.fix_tokenization(_detok(re.sub(r"^\s*\[\s*[A-Za-z ]{1,6}\s*\]\s*", "", r["prompt"])))
        if io_utils.word_count(story) >= 220 and 8 <= len(prompt.split()) <= 60 and "http" not in story:
            items.append({"doc_id": f"wp-{i}", "text": story, "prompt": prompt})
    return rng.sample(items, k)


def load_explain(rng, k):
    pf = pq.ParquetFile(dl("sentence-transformers/eli5", "pair/train-00000-of-00001.parquet"))

    def gen():
        n = 0
        for b in pf.iter_batches(batch_size=8000):
            for r in b.to_pylist():
                q, a = norm_ws(r["question"]), norm_ws(r["answer"])
                if 6 <= len(q.split()) <= 40 and 60 <= io_utils.word_count(a) <= 400 \
                        and "_URL_" not in a and "[deleted]" not in a and "http" not in a:
                    n += 1
                    yield {"doc_id": f"eli5-{n}", "text": a, "question": q}
    pool = reservoir(gen(), 1200, rng)
    return rng.sample(pool, k)


# ----------------------------------------------------------------------------- general prompts
_BAD = re.compile(r"act as|you are a|imagine you are|role.?play|\bDAN\b|translate|python|javascript|\bsql\b|\bcode\b|"
                  r"regex|\bjson\b|script|```|http|chatgpt|open assistant|\bai\b language model", re.I)
_OK = re.compile(r"\bessay|write (a|an|me)|article|blog|\bletter|speech|\bstory|explain|describe|discuss|compare|"
                 r"what (are|is|were) the|how (does|do|did)|why (do|does|did|is)|\breview\b|\bpoem", re.I)


def load_general(rng, k_oasst, k_dolly):
    out = []
    p = dl("OpenAssistant/oasst1", "2023-04-12_oasst_prompts.messages.jsonl.gz")
    seen = set()
    cands = []
    for line in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(line)
        t = norm_ws(r["text"])
        w = len(t.split())
        if r.get("lang") == "en" and not r.get("deleted") and 8 <= w <= 70 and _OK.search(t) and not _BAD.search(t) \
                and t.lower() not in seen:
            seen.add(t.lower())
            cands.append({"doc_id": "oasst-" + r["message_id"][:8], "instruction": t, "source": "oasst1"})
    out += rng.sample(cands, k_oasst)
    p = dl("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")
    cands = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        t = norm_ws(r["instruction"])
        if r["category"] in ("creative_writing", "general_qa", "open_qa", "brainstorming") and not r.get("context") \
                and 6 <= len(t.split()) <= 60 and not _BAD.search(t) and t.lower() not in seen:
            seen.add(t.lower())
            cands.append({"doc_id": "dolly-" + str(len(cands)), "instruction": t, "source": "dolly15k"})
    out += rng.sample(cands, k_dolly)
    return out


def style_wrap(text, rng):
    x = rng.random()
    if x < 0.25:
        return "I'm a college student and need help with an assignment. " + text, "student"
    if x < 0.40:
        return text + " Use a formal, professional tone.", "formal"
    return text, "plain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(io_utils.DATA, "corpus", "prompts.jsonl"))
    args = ap.parse_args()
    rng = random.Random(SEED)

    loaders = {
        "news": ("cnn_dailymail", "abisee/cnn_dailymail", lambda: load_news(rng, PLAN["news"]["n"]),
                 {"license": "apache-2.0 (dataset); article text (c) CNN", "published_before": "2017-01-01", "domain": "news"}),
        "abstract": ("arxiv-abstracts-2021", "gfissore/arxiv-abstracts-2021", lambda: load_abstracts(rng, PLAN["abstract"]["n"]),
                     {"license": "cc0-1.0", "published_before": "2021-12-31", "domain": "academic"}),
        "email": ("aeslc", "Yale-LILY/aeslc", lambda: load_emails(rng, PLAN["email"]["n"]),
                  {"license": "unknown (Enron corpus)", "published_before": "2019-01-01", "domain": "email"}),
        "story": ("writingprompts", "euclaise/writingprompts", lambda: load_stories(rng, PLAN["story"]["n"]),
                  {"license": "mit (dataset); stories (c) Reddit authors", "published_before": "2018-12-31", "domain": "creative"}),
        "explain": ("eli5", "sentence-transformers/eli5", lambda: load_explain(rng, PLAN["explain"]["n"]),
                    {"license": "unknown (Reddit ELI5 answers)", "published_before": "2019-12-31", "domain": "explain"}),
    }
    prompts, matched = [], []
    per_type = {}
    for ttype, (cname, repo, loader, meta) in loaders.items():
        docs = loader()
        r10 = rev(repo)
        registry.register("human_corpora", cname, {"hf_repo": repo, "revision": r10, "verified_human": False, **meta})
        for d in docs:
            w = rng.choice(WORDS[ttype]) if ttype in WORDS else round25(io_utils.word_count(d["text"]))
            tpl = rng.choice(TEMPLATES[ttype])
            instr = tpl.format(w=w, points=d.get("points", ""), title=d.get("title", ""), subject=d.get("subject", ""),
                               prompt=d.get("prompt", ""), question=d.get("question", ""))
            instr, style = style_wrap(instr, rng)
            human = cut_to_words(d["text"], w) if ttype in ("news", "story") else d["text"]
            human = textnorm.fix_tokenization(human)  # undo the source's own tokenization artifacts
            per_type.setdefault(ttype, []).append({
                "task_type": ttype, "domain": DOMAIN[ttype], "instruction": instr, "target_words": w, "style": style,
                "source": f"{cname}@{r10}", "source_doc": d["doc_id"], "matched_human": True,
                "_human": human, "_cname": cname, "_rev": r10})
    gen = load_general(rng, 110, 60)
    r_o, r_d = rev("OpenAssistant/oasst1"), rev("databricks/databricks-dolly-15k")
    for d in gen:
        w = rng.choice(WORDS["general"])
        instr = d["instruction"]
        if rng.random() < 0.6:
            instr = instr.rstrip() + rng.choice([f" Answer in about {w} words.", f" (about {w} words)", f" Keep it around {w} words."])
        instr, style = style_wrap(instr, rng)
        per_type.setdefault("general", []).append({
            "task_type": "general", "domain": "general", "instruction": instr, "target_words": w, "style": style,
            "source": f"{d['source']}@{r_o if d['source'] == 'oasst1' else r_d}", "source_doc": d["doc_id"],
            "matched_human": False})

    # assign splits and generation quotas per task type
    for ttype, items in per_type.items():
        rng.shuffle(items)
        nl, nd, nt = PLAN[ttype]["split"]
        gl, gd, gt = PLAN[ttype]["gen"]
        bounds = [("locked", 0, nl, gl), ("dev", nl, nl + nd, gd), ("train", nl + nd, nl + nd + nt, gt)]
        for split, a, b, q in bounds:
            for j, it in enumerate(items[a:b]):
                it["split"] = split
                it["gen_models"] = list(GEN_MODELS) if j < q else []
    registry.register("human_corpora", "oasst1-prompts", {"hf_repo": "OpenAssistant/oasst1", "revision": r_o,
                      "license": "apache-2.0", "published_before": "2023-04-12", "note": "prompts only; not used as human text"})
    registry.register("human_corpora", "dolly15k-prompts", {"hf_repo": "databricks/databricks-dolly-15k", "revision": r_d,
                      "license": "cc-by-sa-3.0", "published_before": "2023-04-12", "note": "prompts only; not used as human text"})

    n = 0
    for ttype in PLAN:
        for it in per_type[ttype]:
            n += 1
            pid = f"{ttype[:4]}-{n:04d}"
            row = {"prompt_id": pid, **{k: v for k, v in it.items() if not k.startswith("_")}}
            prompts.append(row)
            if it["matched_human"]:
                text = it["_human"]
                matched.append({"prompt_id": pid, "doc_id": it["source_doc"], "text": text, "domain": it["domain"],
                                "split": it["split"], "task_type": ttype, "corpus": it["_cname"], "revision": it["_rev"],
                                "words": io_utils.word_count(text)})
    io_utils.write_jsonl(args.out, prompts)
    io_utils.write_jsonl(os.path.join(io_utils.DATA, "corpus", "human", "matched_docs.jsonl"), matched)

    print(f"prompts: {len(prompts)}  matched human docs: {len(matched)}")
    from collections import Counter
    print("by split:", dict(Counter(p["split"] for p in prompts)))
    print("by type x split:", {t: dict(Counter(p["split"] for p in prompts if p["task_type"] == t)) for t in PLAN})
    print("to generate per model:", dict(Counter(p["split"] for p in prompts if p["gen_models"])))
    print("style mix:", dict(Counter(p["style"] for p in prompts)))
    w = [m["words"] for m in matched]
    print(f"matched human words: min={min(w)} median={sorted(w)[len(w)//2]} max={max(w)}")
    for t in PLAN:
        ex = next(p for p in prompts if p["task_type"] == t)
        print(f"[{t}] {ex['prompt_id']} ({ex['split']}, {ex['target_words']}w, {ex['style']}): {ex['instruction'][:150].encode('ascii', 'replace').decode()!r}")


if __name__ == "__main__":
    main()
