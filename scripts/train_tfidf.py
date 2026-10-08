#!/usr/bin/env python
"""
Train the TF-IDF n-gram detector (runtime mode `tfidf`) and write models/tfidf_v2/.

  python scripts/build_modern_human.py   # once: extra modern human negatives
  python scripts/train_tfidf.py

Recipe (scratch experiment 2026-10-07, "TF-IDF v2"): char_wb 2-5-grams (100k) + word 1-2-grams (50k), sublinear TF-IDF, logistic
regression (C=10, balanced). Positives: Claude 5.5 rows of the TRAIN split. Negatives: human rows of the TRAIN split plus
data/corpus/human/modern_human_train.jsonl. Never reads data/locked.

Outputs:
  tfidf_model.json.gz   vocabularies, idf, coefficients (read by backend/tfidf_detector.py, numpy only)
  runtime_config.json   decision_threshold = dev clean-human 1%-FPR decision value of the EXPORTED model (all dev clean humans);
                        the runtime reports ai_score = sigmoid(decision - decision_threshold), so its threshold is 0.5
Before writing the config the exported model is checked against sklearn on every dev row (max |diff| must be < 1e-6).
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.tfidf_detector import WORD_TOKEN_PATTERN, TfidfDetector  # noqa: E402
from backend.runtime_engine import RUNTIME_CONFIGS, TFIDF_DIR  # noqa: E402
from scripts.common import io_utils, metrics  # noqa: E402
from scripts.train_detector import prep  # noqa: E402

CHAR_RANGE, WORD_RANGE = (2, 5), (1, 2)


def fit_model(texts, labels, char_min_df=3, word_min_df=2, C=10.0):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_RANGE, min_df=char_min_df, sublinear_tf=True, max_features=100000)
    vw = TfidfVectorizer(analyzer="word", ngram_range=WORD_RANGE, min_df=word_min_df, sublinear_tf=True, lowercase=True,
                         token_pattern=WORD_TOKEN_PATTERN, max_features=50000)
    X = hstack([vc.fit_transform(texts), vw.fit_transform(texts)]).tocsr()
    lr = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=0).fit(X, labels)
    return vc, vw, lr


def sklearn_decision(model, texts):
    from scipy.sparse import hstack
    vc, vw, lr = model
    return lr.decision_function(hstack([vc.transform(texts), vw.transform(texts)]).tocsr())


def export_model(model, path):
    vc, vw, lr = model
    out = {"char_ngram_range": list(CHAR_RANGE), "word_ngram_range": list(WORD_RANGE), "word_token_pattern": WORD_TOKEN_PATTERN,
           "char_vocab": {k: int(v) for k, v in sorted(vc.vocabulary_.items())}, "char_idf": vc.idf_.tolist(),
           "word_vocab": {k: int(v) for k, v in sorted(vw.vocabulary_.items())}, "word_idf": vw.idf_.tolist(),
           "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(json.dumps(out, separators=(",", ":")).encode("utf-8"))


def is_claude(r):
    return r["label"] == "ai" and r["origin"] in ("ai_raw", "ai_attacked") and r["generator_id"].startswith("claude-")


def is_clean_neg(r):
    return r["label"] == "human" and r["origin"] == "human" and r["attack_id"] == "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=10.0)
    args = ap.parse_args()
    extra_path = os.path.join(io_utils.DATA, "corpus", "human", "modern_human_train.jsonl")
    if not os.path.exists(extra_path):
        sys.exit(f"missing {extra_path}: run python scripts/build_modern_human.py first")
    train = io_utils.read_jsonl(os.path.join(io_utils.DATA, "splits", "train.jsonl.gz"))
    rows = [r for r in train if is_claude(r) or (r["label"] == "human" and r["origin"] == "human")]
    rows += [json.loads(ln) for ln in open(extra_path, encoding="utf-8")]
    labels = [int(r["label"] == "ai") for r in rows]
    print(f"train rows: {len(rows)} ({sum(labels)} Claude, {len(rows) - sum(labels)} human)", flush=True)

    t0 = time.time()
    model = fit_model([prep(r["text"]) for r in rows], labels, C=args.C)
    print(f"fit in {time.time() - t0:.0f}s", flush=True)
    model_path = os.path.join(TFIDF_DIR, "tfidf_model.json.gz")
    export_model(model, model_path)
    print(f"wrote {model_path} ({os.path.getsize(model_path) / 1e6:.2f} MB)", flush=True)

    dev = io_utils.read_jsonl(os.path.join(io_utils.DATA, "splits", "dev.jsonl.gz"))
    dev_texts = [prep(r["text"]) for r in dev]
    runtime = TfidfDetector(model_path).decision(dev_texts)
    diff = float(np.max(np.abs(runtime - sklearn_decision(model, dev_texts))))
    print(f"runtime vs sklearn on {len(dev)} dev rows: max |diff| = {diff:.2e}", flush=True)
    if diff >= 1e-6:
        sys.exit("REFUSED: exported model does not reproduce sklearn")

    neg = runtime[[is_clean_neg(r) for r in dev]]
    t_dec = metrics.threshold_at_fpr(neg, 0.01)
    commit = os.popen(f'git -C "{REPO}" rev-parse --short HEAD').read().strip()
    cfg = {"candidate": "tfidf_v2", "threshold": 0.5, "decision_threshold": t_dec,
           "threshold_rule": f"tfidf: dev clean-human 1% FPR of the exported model's decision value (n_dev_neg={len(neg)}, "
                             f"commit {commit}); ai_score = sigmoid(decision - decision_threshold)",
           "score": "sigmoid(decision - decision_threshold)", "C": args.C, "n_train_claude": sum(labels),
           "n_train_human": len(labels) - sum(labels)}
    with open(RUNTIME_CONFIGS["tfidf"], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print("wrote", RUNTIME_CONFIGS["tfidf"], cfg)


if __name__ == "__main__":
    main()
