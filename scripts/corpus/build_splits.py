#!/usr/bin/env python
"""
Assemble the final train / dev / locked-test splits from the corpus pieces. Deterministic (seeded, hash-based group splits).

  python scripts/corpus/build_splits.py [--lock]

Group rule: every row carries `prompt_id` (the prompt for AI rows, the source document/author for human rows); attacked rows
inherit their parent's group. All rows of a group land in ONE split. Matched human documents inherit the split of the prompt
they were matched to. Public corpora (RAID, MAGE) contribute to train/dev only; HC3 human answers may enter the locked test.

Train is balanced per genre (AI share kept within 1/3 .. 2/3; student_essay within 1/4 .. 3/4) by dropping surplus
public/legacy rows only: new frontier and attacked rows are never dropped. Outputs:
  data/splits/train.jsonl.gz, data/splits/dev.jsonl.gz, data/splits/MANIFEST.json   (ignored except the manifest)
  with --lock: data/locked/locked_ai.jsonl.gz (tracked), locked_human.jsonl.gz (third-party text: ignored), MANIFEST.json
Without --lock the locked rows are written to data/splits/locked_PREVIEW.jsonl.gz instead (never evaluate on it).
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils, schema  # noqa: E402
from scripts.corpus import registry  # noqa: E402

SEED = 20261010
D = io_utils.DATA
HARD = {"too_short", "too_long", "refusal", "meta_words", "duplicate_or_missing"}
KEEP = ("id", "text", "label", "origin", "generator_id", "access_path", "date", "prompt_id", "attack_id", "parent_id", "domain",
        "esl", "words", "split", "task_type", "ai_share", "cefr", "target_words", "style", "decoding", "cosine_to_parent", "role")


def u_hash(group, salt):
    return int(hashlib.sha1(f"{salt}:{group}".encode()).hexdigest(), 16) % 10000 / 10000.0


def split_by_hash(group, f_locked, f_dev, salt):
    u = u_hash(group, salt)
    return "locked" if u < f_locked else ("dev" if u < f_locked + f_dev else "train")


def human_row(text, corpus, rev, group, domain, split, esl=False, **extra):
    r = {"id": io_utils.text_id(text), "text": text, "label": "human", "origin": "human", "generator_id": f"human:{corpus}",
         "access_path": f"corpus:{corpus}@{rev}", "date": datetime.date.today().isoformat(), "prompt_id": group, "attack_id": "none",
         "parent_id": None, "domain": domain, "esl": esl, "words": io_utils.word_count(text), "split": split}
    r.update(extra)
    return r


def rj(path):
    return io_utils.read_jsonl(path) if os.path.exists(path) else []


def load_all(reg):
    rows = []
    # matched human docs (split inherited from the prompt)
    for r in rj(os.path.join(D, "corpus", "human", "matched_docs.jsonl")):
        rows.append(human_row(r["text"], r["corpus"], r["revision"], r["prompt_id"], r["domain"], r["split"], task_type=r["task_type"]))
    # W&I + LOCNESS (group = author)
    for r in rj(os.path.join(D, "corpus", "human", "wi_locness.jsonl")):
        rows.append(human_row(r["text"], "wi_locness", r["revision"], r["user"], "student_essay",
                              split_by_hash(r["user"], 0.12, 0.08, "wi"), esl=r["esl"], cefr=r["cefr"]))
    # extra human (arXiv abstracts, CNN excerpts)
    for r in rj(os.path.join(D, "corpus", "human", "extra_human.jsonl")):
        r = dict(r)
        r["split"] = split_by_hash(r["prompt_id"], 0.25, 0.10, "extra")
        rows.append(r)
    # public corpora
    for name in ("raid", "mage", "hc3"):
        for r in rj(os.path.join(D, "corpus", "public_ai", f"{name}.jsonl")):
            rows.append(dict(r))
        for r in rj(os.path.join(D, "corpus", "human", f"{name}-human.jsonl")):
            rows.append(dict(r))
    # HC3 groups: re-split by hash; AI rows of locked groups are dropped (public AI never enters the locked test)
    out = []
    for r in rows:
        if r["access_path"].endswith(tuple(f"@{x}" for x in [reg["public_ai_corpora"]["hc3"]["revision"]])) and r["prompt_id"].startswith("hc3:"):
            s = split_by_hash(r["prompt_id"], 0.15, 0.10, "hc3")
            if r["label"] == "ai" and s == "locked":
                continue
            r["split"] = s
        out.append(r)
    rows = out
    # frontier raw generations
    for f in glob.glob(os.path.join(D, "corpus", "frontier", "claude-*", "*.jsonl")) + glob.glob(os.path.join(D, "corpus", "frontier", "gpt-*", "*.jsonl")):
        for r in rj(f):
            if not (set(r.get("quality_flags", [])) & HARD):
                rows.append(dict(r))
    # attacked rows (AI and human controls)
    for f in glob.glob(os.path.join(D, "corpus", "attacked", "A*", "*.jsonl")):
        rows += [dict(r) for r in rj(f)]
    return rows


def drop_dups(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["id"], r["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def balance_train(rows, rng):
    """Drop surplus public/legacy rows so that no genre is dominated by one class (anti domain-shortcut)."""
    protected = lambda r: r["generator_id"].startswith(("claude-", "gpt-")) or r["origin"] in ("hybrid",) or r["access_path"].startswith(("claude-code", "api:", "user_paste", "local_model", "programmatic")) and not r["access_path"].startswith("public_dataset")  # noqa: E731
    by = defaultdict(list)
    for r in rows:
        by[schema.genre_of(r["domain"])].append(r)
    keep, log = [], {}
    for g, rs in by.items():
        ai = [r for r in rs if r["label"] == "ai"]
        hu = [r for r in rs if r["label"] == "human"]
        lo, hi = (0.25, 0.75) if g == "student_essay" else (1 / 3, 2 / 3)
        # AI too many -> drop droppable AI; AI too few -> drop droppable human
        def trim(majority, minority, ratio):
            cap = int(len(minority) * ratio)
            drop = [r for r in majority if not protected(r)]
            rng.shuffle(drop)
            excess = max(0, len(majority) - cap)
            gone = {id(r) for r in drop[:excess]}
            return [r for r in majority if id(r) not in gone]
        if len(ai) > 0 and len(hu) > 0:
            share = len(ai) / (len(ai) + len(hu))
            if share > hi:
                ai = trim(ai, hu, hi / (1 - hi))
            elif share < lo:
                hu = trim(hu, ai, (1 - lo) / lo)
        log[g] = {"ai": len(ai), "human": len(hu)}
        keep += ai + hu
    return keep, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true", help="write the locked test (once, before modelling)")
    args = ap.parse_args()
    rng = random.Random(SEED)
    reg = registry.load()
    rows = drop_dups(load_all(reg))
    for r in rows:
        r["domain"] = r.get("domain") or "other"
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    train, tlog = balance_train(by_split["train"], rng)
    dev, locked = by_split["dev"], by_split["locked"]
    # cross-split near-duplicates (5-gram J >= 0.5, e.g. the same human text in two corpora): never drop a locked row;
    # drop the train row if its twin is locked, otherwise drop the dev row
    from scripts.common import dedup
    docs = [(s, r["id"], r["text"]) for s, rs in (("train", train), ("dev", dev), ("locked", locked)) for r in rs]
    drop = set()
    for a, ia, b, ib, _ in dedup.cross_group_near_duplicates(docs, k=5, threshold=0.5):
        if a == "locked":
            drop.add((b, ib))
        elif b == "locked":
            drop.add((a, ia))
        else:
            drop.add(("dev", ia if a == "dev" else ib))
    train = [r for r in train if ("train", r["id"]) not in drop]
    dev = [r for r in dev if ("dev", r["id"]) not in drop]
    print(f"dropped {len(drop)} train/dev rows that near-duplicate another split: {sorted(drop)[:5]}")

    def clean(rs):
        return [{k: r[k] for k in KEEP if k in r} for r in rs]
    os.makedirs(os.path.join(D, "splits"), exist_ok=True)
    io_utils.write_jsonl(os.path.join(D, "splits", "train.jsonl.gz"), clean(train))
    io_utils.write_jsonl(os.path.join(D, "splits", "dev.jsonl.gz"), clean(dev))
    man = {"seed": SEED, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "files": {n: {"sha256": io_utils.sha256_file(os.path.join(D, "splits", n)), "n": c}
                     for n, c in (("train.jsonl.gz", len(train)), ("dev.jsonl.gz", len(dev)))},
           "train_genre_balance": tlog}
    if args.lock:
        os.makedirs(os.path.join(D, "locked"), exist_ok=True)
        lai = [r for r in locked if r["label"] == "ai" or r["origin"] == "hybrid" or r["attack_id"] != "none" and r["generator_id"].startswith(("claude-", "gpt-"))]
        lhu = [r for r in locked if r not in lai]
        io_utils.write_jsonl(os.path.join(D, "locked", "locked_ai.jsonl.gz"), clean(lai))
        io_utils.write_jsonl(os.path.join(D, "locked", "locked_human.jsonl.gz"), clean(lhu))
        lm = {"created": man["created"], "seed": SEED, "untested": [g for g in ("gpt-6-astra",)
              if not glob.glob(os.path.join(D, "corpus", "frontier", g, "*.jsonl"))],
              "files": {n: {"sha256": io_utils.sha256_file(os.path.join(D, "locked", n)), "n": c}
                        for n, c in (("locked_ai.jsonl.gz", len(lai)), ("locked_human.jsonl.gz", len(lhu)))},
              "note": "locked_human.jsonl.gz contains third-party text (W&I+LOCNESS is non-commercial) and is not tracked by git"}
        with open(os.path.join(D, "locked", "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(lm, f, indent=1)
        open(os.path.join(D, "locked", "ACCESS_LOG.md"), "a", encoding="utf-8").close()
        print("LOCKED written:", {n: i["n"] for n, i in lm["files"].items()}, "untested:", lm["untested"])
    else:
        io_utils.write_jsonl(os.path.join(D, "splits", "locked_PREVIEW.jsonl.gz"), clean(locked))
        print("locked PREVIEW written (not locked); use --lock to freeze")
    with open(os.path.join(D, "splits", "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)

    def table(rs, name):
        c = Counter((r["label"], schema.attack_family(r["attack_id"]), r["origin"]) for r in rs)
        print(f"\n[{name}] n={len(rs)}  AI={sum(1 for r in rs if r['label'] == 'ai')} human={sum(1 for r in rs if r['label'] == 'human')}  "
              f"ESL={sum(1 for r in rs if r.get('esl'))}")
        gen = Counter(r["generator_id"].split(":")[0] if r["generator_id"].startswith(("human", "raid", "mage", "hc3")) else r["generator_id"] for r in rs if r["label"] == "ai")
        print("   AI sources:", dict(gen.most_common(8)))
        print("   by attack family:", dict(Counter(schema.attack_family(r["attack_id"]) for r in rs)))
        print("   by genre x label:", {g: dict(Counter(r["label"] for r in rs if schema.genre_of(r["domain"]) == g)) for g in sorted({schema.genre_of(r["domain"]) for r in rs})})
    table(train, "train")
    table(dev, "dev")
    table(locked, "locked")


if __name__ == "__main__":
    main()
