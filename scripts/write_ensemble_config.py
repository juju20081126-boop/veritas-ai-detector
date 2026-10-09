#!/usr/bin/env python
"""
Write models/ensemble/runtime_config.json for the opt-in `ensemble` runtime mode (backend/runtime_engine.py), which flags a text
when ANY component mode reaches its own cut.

  python scripts/write_ensemble_config.py                                  # tfidf OR frontier
  python scripts/write_ensemble_config.py --components tfidf multi_teacher

Cuts: every component gets the same false-positive budget a; a is the largest value on a 0.005-point grid for which the union
flags at most --fpr of the calibration negatives = ALL dev clean humans + the calibration half (sha1(id) even) of
data/corpus/human/modern_human_heldout.jsonl. The other half is the test half: its union FPR is recorded in the config and is never
used to pick a cut. Dev alone is not enough: at its own dev 1%-FPR cut frontier flags 7.3% of the unseen modern human texts.
Never reads data/locked.

Scores come from the runtime engine end to end (scripts.detectors.baselines), cached like scripts/eval_frontier.py in
data/eval/scores/<detector>__{dev,modern_heldout}.jsonl. Before a dev cache is trusted, 8 of its rows are re-scored and must match
within 1e-3 (a cache from an older model is refused).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.runtime_engine import RUNTIME_CONFIGS  # noqa: E402
from scripts.common import io_utils, metrics  # noqa: E402
from scripts.detectors.baselines import get_detector  # noqa: E402
from scripts.eval_frontier import cached_scores, is_clean_neg, load_split  # noqa: E402

DETECTOR_OF = {"tfidf": "tfidf_runtime", "frontier": "frontier_onnx", "multi_teacher": "multi_teacher_onnx"}
MODERN = os.path.join(io_utils.DATA, "corpus", "human", "modern_human_heldout.jsonl")


def is_calibration_half(r):
    return int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2 == 0


def check_cache(det, rows, cache):
    fresh = get_detector(det).score([r["text"] for r in rows])
    diff = max(abs(cache[r["id"]] - s) for r, s in zip(rows, fresh))
    if diff > 1e-3:
        sys.exit(f"REFUSED: data/eval/scores/{det}__dev.jsonl does not match the current model (max diff {diff:.4f}); delete it and re-run")


def union_cuts(N, fpr):
    """Largest equal per-column budget a (grid of 200 steps up to fpr) whose per-column cuts flag <= fpr of the rows of N in union."""
    best = None
    for a in np.arange(1, 201) * fpr / 200:
        cuts = np.array([metrics.threshold_at_fpr(N[:, j], a) for j in range(N.shape[1])])
        if (N > cuts).any(axis=1).mean() <= fpr:
            best = (float(a), cuts)
    return best


def rate(flags):
    return f"{int(np.sum(flags))}/{len(flags)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="+", choices=list(DETECTOR_OF), default=["tfidf", "frontier"])
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()
    comps = args.components
    if len(set(comps)) != len(comps) or len(comps) < 2:
        sys.exit("need at least two distinct components")

    dev_neg = [r for r in load_split("dev") if is_clean_neg(r)]
    modern = io_utils.read_jsonl(MODERN)
    cal = [r for r in modern if is_calibration_half(r)]
    test = [r for r in modern if not is_calibration_half(r)]
    scores = {}
    for c in comps:
        det = DETECTOR_OF[c]
        fac = lambda det=det: get_detector(det)  # noqa: E731
        dev_sc = cached_scores(det, "dev", dev_neg, fac)
        check_cache(det, dev_neg[:8], dev_sc)
        scores[c] = {**dev_sc, **cached_scores(det, "modern_heldout", modern, fac)}

    mat = lambda rows: np.array([[scores[c][r["id"]] for c in comps] for r in rows])  # noqa: E731
    a, cuts = union_cuts(mat(dev_neg + cal), args.fpr)
    flags = lambda rows: (mat(rows) > cuts).any(axis=1)  # noqa: E731
    commit = os.popen(f'git -C "{REPO}" rev-parse --short HEAD').read().strip()
    names = [json.load(open(RUNTIME_CONFIGS[c], encoding="utf-8")).get("candidate", c) for c in comps]
    cfg = {"candidate": "ensemble(" + " OR ".join(names) + ")",
           "components": {c: float(t) for c, t in zip(comps, cuts)},
           "threshold": 0.5,
           "threshold_rule": (f"OR of {', '.join(comps)}: equal per-component budget {100 * a:.3f}% so the union flags <= "
                              f"{100 * args.fpr:g}% of dev clean humans (n={len(dev_neg)}) + modern-human calibration half "
                              f"(n={len(cal)}), commit {commit}"),
           "score": "sigmoid(max_c [logit(ai_score_c) - logit(cut_c)])",
           "calibration": {"budget_each": a, "union_fpr_dev_clean_human": rate(flags(dev_neg)),
                           "union_fpr_dev_esl": rate(flags([r for r in dev_neg if r.get("esl")])),
                           "union_fpr_modern_calibration_half": rate(flags(cal)),
                           "union_fpr_modern_test_half": rate(flags(test))}}
    out = RUNTIME_CONFIGS["ensemble"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(cfg, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
