"""Evaluation wrapper for a trained candidate (models/candidates/<name>): normalise -> ~100-word chunks -> mean-pooled logits -> P(class 2)+P(class 3)."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


class StudentDetector:
    def __init__(self, name, threads=6, max_len=160, temperature=1.0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from scripts.train_detector import chunks, prep
        torch.set_num_threads(threads)
        path = os.path.join(REPO, "models", "candidates", name)
        self.name, self.torch, self.max_len, self.T = name, torch, max_len, temperature
        self.chunks, self.prep = chunks, prep
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def score(self, texts, batch=32):
        out = []
        for t in texts:
            cs = self.chunks(self.prep(t))
            logits = []
            for i in range(0, len(cs), batch):
                enc = self.tok(cs[i:i + batch], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                with self.torch.no_grad():
                    logits.append(self.model(**enc).logits)
            lg = self.torch.cat(logits).mean(0) / self.T
            p = self.torch.softmax(lg, dim=-1)
            out.append(float(p[2] + p[3]))
        return np.array(out, float)
