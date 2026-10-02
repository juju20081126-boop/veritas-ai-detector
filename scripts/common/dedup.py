"""Exact cross-split near-duplicate detection via a sorted shingle index (vectorized with numpy)."""

import re

import numpy as np

_TOK = re.compile(r"\w+", re.UNICODE)


def shingle_hashes(text, k=5):
    """Unique 64-bit hashes of the word k-grams of `text` (process-local hash; fine for in-run dedup)."""
    w = _TOK.findall(text.lower())
    if len(w) < k:
        grams = [tuple(w)] if w else []
    else:
        grams = [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]
    if not grams:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.fromiter((hash(g) for g in grams), dtype=np.int64, count=len(grams)))


def cross_group_near_duplicates(docs, k=5, threshold=0.5, df_cap=64):
    """
    docs: list of (group, doc_id, text). Returns [(group_a, id_a, group_b, id_b, jaccard)] for pairs in
    DIFFERENT groups whose k-gram Jaccard >= threshold. Shingles shared by more than df_cap docs
    (boilerplate) are ignored, which can only lower recall for very generic text, never create false pairs.
    """
    n = len(docs)
    if n == 0:
        return []
    group_names = sorted({g for g, _, _ in docs})
    gidx = {g: i for i, g in enumerate(group_names)}
    grp = np.array([gidx[g] for g, _, _ in docs], dtype=np.int32)
    sets = [shingle_hashes(t, k) for _, _, t in docs]
    sizes = np.array([len(s) for s in sets], dtype=np.int64)
    if sizes.sum() == 0:
        return []
    H = np.concatenate(sets)
    O = np.concatenate([np.full(len(s), i, dtype=np.int32) for i, s in enumerate(sets)])
    order = np.argsort(H, kind="stable")
    H, O = H[order], O[order]
    bounds = np.flatnonzero(np.diff(H)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(H)]))
    lengths = ends - starts
    keys = []
    for L in range(2, df_cap + 1):
        sel = np.flatnonzero(lengths == L)
        if sel.size == 0:
            continue
        members = O[starts[sel][:, None] + np.arange(L)[None, :]]  # (m, L)
        for a in range(L):
            for b in range(a + 1, L):
                x, y = members[:, a].astype(np.int64), members[:, b].astype(np.int64)
                m = grp[x] != grp[y]
                if m.any():
                    lo = np.minimum(x[m], y[m])
                    hi = np.maximum(x[m], y[m])
                    keys.append(lo * n + hi)
    if not keys:
        return []
    allk = np.concatenate(keys)
    uniq, counts = np.unique(allk, return_counts=True)
    lo, hi = uniq // n, uniq % n
    jac = counts / (sizes[lo] + sizes[hi] - counts)
    keep = jac >= threshold
    out = []
    for a, b, j in zip(lo[keep], hi[keep], jac[keep]):
        out.append((docs[a][0], docs[a][1], docs[b][0], docs[b][1], float(j)))
    return out
