#!/usr/bin/env python
"""
Train a candidate detector on the REAL train split (CPU). Keeps the shipped runtime interface: a 4-logit sequence classifier
(MiniLM-L6-v2 by default) whose logits the engine mean-pools over ~100-word chunks.

  python scripts/train_detector.py --name h1h2h9 --epochs 2 [--backbone sentence-transformers/all-MiniLM-L6-v2]
                                   [--attack-weight 2.0] [--no-attacks] [--hold-out-family A3] [--hold-out-gen claude-opus-5-5]

Classes (runtime order): 0 human, 1 human & AI-refined (hybrids with ai_share < 0.5), 2 AI & AI-refined (attacked AI, hybrids >= 0.5),
3 AI-generated (raw AI). Input hygiene (H9) is applied to every text: scripts.common.textnorm.normalize_text + strip_markdown + newline
flattening. Training examples are random <=100-word chunks (matching the engine). --hold-out-* implement leave-one-family-out
(drop that attack family / generator family from training only). Output: models/candidates/<name>/ (HF weights + tokenizer + train_log.json).
Nothing here reads the locked test.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import io_utils, schema, textnorm  # noqa: E402

D = io_utils.DATA


def prep(text):
    t = textnorm.strip_markdown(textnorm.normalize_text(text))
    return " ".join(t.split())


def label_of(r):
    if r["origin"] == "hybrid":
        return 2 if r.get("ai_share", 0.5) >= 0.5 else 1
    if r["label"] == "human":
        return 0
    return 3 if r["attack_id"] == "none" else 2


def chunks(text, max_words=100):
    from backend.document_parser import split_sentences
    out, cur, n = [], [], 0
    for s in split_sentences(text) or [text]:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out or [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--chunks-per-doc", type=int, default=3)
    ap.add_argument("--attack-weight", type=float, default=2.0, help="sampling weight of attacked-AI/hybrid rows (H2)")
    ap.add_argument("--no-attacks", action="store_true", help="ablation: drop all attacked/hybrid rows from training")
    ap.add_argument("--hold-out-family", default=None, help="leave-one-attack-family-out, e.g. A3")
    ap.add_argument("--hold-out-gen", default=None, help="leave-one-generator-family-out, e.g. claude (drops generator ids starting with it)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = io_utils.read_jsonl(os.path.join(D, "splits", "train.jsonl.gz"))
    kept = []
    for r in rows:
        fam = schema.attack_family(r["attack_id"])
        if args.no_attacks and (fam != "none" or r["origin"] == "hybrid") and r["label"] == "ai":
            continue
        if args.hold_out_family and fam == args.hold_out_family:
            continue
        if args.hold_out_gen and r["generator_id"].startswith(args.hold_out_gen):
            continue
        kept.append(r)
    if args.max_docs:
        random.shuffle(kept)
        kept = kept[:args.max_docs]
    labels = np.array([label_of(r) for r in kept])
    print(f"train docs: {len(kept)} (dropped {len(rows) - len(kept)}); class counts: {np.bincount(labels, minlength=4).tolist()}")
    docs = [chunks(prep(r["text"])) for r in kept]
    w = np.array([args.attack_weight if (r["origin"] in ("ai_attacked", "hybrid")) else 1.0 for r in kept], float)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_per_epoch = len(kept) * args.chunks_per_doc
    steps = args.epochs * (n_per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / max(1, steps)))
    counts = np.bincount(labels, minlength=4)
    raw_w = len(labels) / (4 * np.maximum(counts, 1))
    raw_w[counts < 50] = 1.0                      # (near-)empty classes must not get huge weights
    cls_w = torch.tensor(np.clip(raw_w, 0.5, 3.0), dtype=torch.float32)
    print("class weights:", [round(float(x), 2) for x in cls_w])
    p = w / w.sum()
    log, t0, step = [], time.time(), 0
    model.train()
    for ep in range(args.epochs):
        idx_docs = np.random.choice(len(kept), size=n_per_epoch, replace=True, p=p)
        for b in range(0, len(idx_docs) - args.batch + 1, args.batch):
            bi = idx_docs[b:b + args.batch]
            texts = [random.choice(docs[i]) for i in bi]
            y = torch.tensor(labels[bi], dtype=torch.long)
            enc = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            loss = F.cross_entropy(model(**enc).logits, y, weight=cls_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                log.append({"step": step, "loss": float(loss), "elapsed_s": round(time.time() - t0)})
                print(f"ep{ep} step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(REPO, "models", "candidates", args.name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "n_docs": len(kept), "class_counts": np.bincount(labels, minlength=4).tolist(), "log": log,
               "train_seconds": round(time.time() - t0)}, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
