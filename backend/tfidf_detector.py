"""
Veritas AI — TF-IDF n-gram detector runtime (numpy only, no scikit-learn).

Scores text with a model trained by scripts/train_tfidf.py: char_wb 2-5-gram and word 1-2-gram TF-IDF (sublinear tf, smoothed idf,
per-block L2 norm) followed by a logistic-regression decision function. The analyzers below reproduce scikit-learn's
TfidfVectorizer(lowercase=True) exactly; scripts/tests/test_runtime_api.py::test_tfidf_runtime_matches_sklearn checks parity.
Callers apply the same input hygiene as training (backend.runtime_engine._frontier_prep) before calling decision().
"""

import gzip
import json
import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

_WHITE_SPACES = re.compile(r"\s\s+")
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def char_wb_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> List[str]:
    """sklearn's _char_wb_ngrams on lowercased text: n-grams inside space-padded words only."""
    out = []
    for w in _WHITE_SPACES.sub(" ", text.lower()).split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:  # a short word (w_len < n) is counted once
                break
    return out


def word_ngrams(text: str, token_re: "re.Pattern", min_n: int = 1, max_n: int = 2) -> List[str]:
    """sklearn's word analyzer (lowercase, token_pattern findall, no stop words) followed by _word_ngrams."""
    tokens = token_re.findall(text.lower())
    out = list(tokens) if min_n == 1 else []
    for n in range(max(min_n, 2), min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


class TfidfDetector:
    def __init__(self, path: str):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            m = json.load(f)
        self.char_vocab: Dict[str, int] = m["char_vocab"]
        self.word_vocab: Dict[str, int] = m["word_vocab"]
        self.char_idf = np.asarray(m["char_idf"], dtype=np.float64)
        self.word_idf = np.asarray(m["word_idf"], dtype=np.float64)
        coef = np.asarray(m["coef"], dtype=np.float64)
        self.char_coef, self.word_coef = coef[:len(self.char_idf)], coef[len(self.char_idf):]
        self.intercept = float(m["intercept"])
        self.char_range = tuple(m.get("char_ngram_range", (2, 5)))
        self.word_range = tuple(m.get("word_ngram_range", (1, 2)))
        self.token_re = re.compile(m.get("word_token_pattern", WORD_TOKEN_PATTERN))

    @staticmethod
    def _block(grams: List[str], vocab: Dict[str, int], idf: np.ndarray, coef: np.ndarray) -> float:
        """coef . l2norm((1 + log tf) * idf) over the n-grams present in the vocabulary."""
        counts = Counter(g for g in grams if g in vocab)
        if not counts:
            return 0.0
        idx = np.fromiter((vocab[g] for g in counts), dtype=np.int64, count=len(counts))
        tf = np.fromiter((1.0 + math.log(c) for c in counts.values()), dtype=np.float64, count=len(counts))
        x = tf * idf[idx]
        return float(x @ coef[idx]) / float(np.sqrt(x @ x))

    def decision(self, texts: List[str]) -> np.ndarray:
        """Logistic-regression decision value per text (higher = more AI)."""
        return np.array([self.intercept
                         + self._block(char_wb_ngrams(t, *self.char_range), self.char_vocab, self.char_idf, self.char_coef)
                         + self._block(word_ngrams(t, self.token_re, *self.word_range), self.word_vocab, self.word_idf,
                                       self.word_coef)
                         for t in texts], dtype=np.float64)
