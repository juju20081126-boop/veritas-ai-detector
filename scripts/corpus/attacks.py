"""
Programmatic and local-model attacks used ONLY to build evaluation/training data (see data/research/attack_catalogue.md).

  A4  T5 paraphraser on CPU (humarin/chatgpt_paraphraser_on_T5_base), sentence by sentence
  A5  back-translation en->zh->en and en->de->en (MarianMT, CPU)
  A6  hybrid / mixed authorship: interleave AI and human sentences
  A7  character-level: homoglyphs, zero-width characters, typos

Plus the quality gate (sentence-embedding cosine >= 0.80, length ratio, not a near copy). Never import this from backend/.
"""

import random
import re

from scripts.common import io_utils

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text):
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    out = []
    for p in paras:
        out += [s.strip() for s in _SENT.split(p.replace("\n", " ")) if s.strip()]
    return out


# ----------------------------------------------------------------------------- A7 character-level
_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і",
         "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "T": "Т", "H": "Н", "B": "В",
         "M": "М", "K": "К"}
_KEYS = {"a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl",
         "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb",
         "w": "qe", "x": "zc", "y": "tu", "z": "x"}


def attack_homoglyph(text, rng, rate=0.03):
    return "".join(_HOMO[ch] if ch in _HOMO and rng.random() < rate else ch for ch in text)


def attack_zwsp(text, rng, every=10):
    words = text.split(" ")
    for i in range(len(words)):
        if len(words[i]) >= 4 and rng.random() < 1.0 / every:
            k = rng.randint(1, len(words[i]) - 1)
            words[i] = words[i][:k] + "​" + words[i][k:]
    return " ".join(words)


def attack_typo(text, rng, per100=4):
    words = text.split(" ")
    n = max(1, round(len(words) / 100 * per100))
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    rng.shuffle(idx)
    for i in idx[:n]:
        w = words[i]
        k = rng.randint(1, len(w) - 2)
        kind = rng.choice(["swap", "drop", "dup", "key"])
        if kind == "swap":
            w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
        elif kind == "drop":
            w = w[:k] + w[k + 1:]
        elif kind == "dup":
            w = w[:k] + w[k] + w[k:]
        else:
            ch = w[k].lower()
            w = w[:k] + (rng.choice(_KEYS[ch]) if ch in _KEYS else w[k]) + w[k + 1:]
        words[i] = w
    return " ".join(words)


# ----------------------------------------------------------------------------- A6 hybrid
def interleave(ai_text, human_text, share_ai, rng):
    """Build a text with ~share_ai of its sentences from the AI text and the rest from the human text.
    Returns (text, ai_word_share)."""
    a, h = split_sentences(ai_text), split_sentences(human_text)
    if len(a) < 4 or len(h) < 2:
        return None, 0.0
    total = len(a)
    ai_slots = set(rng.sample(range(total), max(1, round(total * share_ai))))
    ai_i = h_i = 0
    out, ai_words, all_words = [], 0, 0
    for slot in range(total):
        if slot in ai_slots or h_i >= len(h):
            s = a[ai_i % len(a)]
            ai_i += 1
            ai_words += io_utils.word_count(s)
        else:
            s = h[h_i]
            h_i += 1
        all_words += io_utils.word_count(s)
        out.append(s)
    return " ".join(out), ai_words / max(1, all_words)


# ----------------------------------------------------------------------------- A4 / A5 local models
class T5Paraphraser:
    NAME = "humarin/chatgpt_paraphraser_on_T5_base"

    def __init__(self, threads=6):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(self.NAME).eval()

    def paraphrase(self, text, batch=16):
        sents = split_sentences(text)
        outs = []
        for i in range(0, len(sents), batch):
            chunk = sents[i:i + batch]
            enc = self.tok(["paraphrase: " + s for s in chunk], return_tensors="pt", padding=True, truncation=True, max_length=160)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_length=160, num_beams=4, repetition_penalty=2.5, no_repeat_ngram_size=3)
            outs += self.tok.batch_decode(gen, skip_special_tokens=True)
        return " ".join(o.strip() for o in outs)


class BackTranslator:
    PAIRS = {"zh": ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"),
             "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en")}

    def __init__(self, lang, threads=6):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        fwd, back = self.PAIRS[lang]
        self.tok_f, self.m_f = MarianTokenizer.from_pretrained(fwd), MarianMTModel.from_pretrained(fwd).eval()
        self.tok_b, self.m_b = MarianTokenizer.from_pretrained(back), MarianMTModel.from_pretrained(back).eval()

    def _tr(self, tok, model, sents, batch=16):
        out = []
        for i in range(0, len(sents), batch):
            enc = tok(sents[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=200)
            with self.torch.no_grad():
                g = model.generate(**enc, max_length=220, num_beams=3)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    def run(self, text):
        sents = split_sentences(text)
        mid = self._tr(self.tok_f, self.m_f, sents)
        back = self._tr(self.tok_b, self.m_b, mid)
        return " ".join(b.strip() for b in back)


# ----------------------------------------------------------------------------- quality gate
class QualityGate:
    def __init__(self, min_cos=0.80):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.min_cos = min_cos

    def check(self, parent, child):
        """Return (ok, reason, cosine)."""
        import numpy as np
        from scripts.common import dedup
        if not child.strip():
            return False, "empty", 0.0
        wp, wc = io_utils.word_count(parent), io_utils.word_count(child)
        if not (0.6 <= wc / max(1, wp) <= 1.6):
            return False, "length_ratio", 0.0
        e = self.model.encode([parent, child], normalize_embeddings=True)
        cos = float(np.dot(e[0], e[1]))
        if cos < self.min_cos:
            return False, "low_similarity", cos
        a, b = dedup.shingle_hashes(parent), dedup.shingle_hashes(child)
        if len(a) and len(b) and len(set(a.tolist()) & set(b.tolist())) / len(set(a.tolist()) | set(b.tolist())) >= 0.9:
            return False, "near_copy", cos
        return True, "ok", cos
