"""Metrics used by the evaluation scripts: Wilson CIs, AUROC, threshold at a target FPR, paired bootstrap."""

import math

import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n -> (point, low, high). n == 0 -> (nan, nan, nan)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def auroc(pos, neg):
    """AUROC via the rank-sum statistic (ties handled by average ranks). Higher score = more AI."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(neg_scores, target_fpr=0.01):
    """Smallest threshold t such that the fraction of negatives with score > t is <= target_fpr (flag if score > t)."""
    neg = np.sort(np.asarray(neg_scores, float))
    if len(neg) == 0:
        return float("nan")
    k = int(math.floor(target_fpr * len(neg)))        # number of negatives allowed above the threshold
    return float(neg[len(neg) - k - 1]) if k < len(neg) else float(neg[0] - 1.0)


def rate_above(scores, t):
    """(k, n) = number of scores strictly above t, and total."""
    s = np.asarray(scores, float)
    return int((s > t).sum()), int(len(s))


def paired_bootstrap_diff(scores_a, scores_b, t_a, t_b, n_boot=10000, seed=0):
    """Paired bootstrap of TPR(a) - TPR(b) over the SAME positive texts (arrays aligned by text). Returns (diff, lo, hi)."""
    a = (np.asarray(scores_a, float) > t_a).astype(float)
    b = (np.asarray(scores_b, float) > t_b).astype(float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
