"""
Text cleaning.

fix_tokenization(): repairs PTB/Moses tokenization artifacts that some source corpora ship with ("was n't", "can ’ t",
``quotes'', " ." before sentence ends). Corpus-building only: it undoes the SOURCE's preprocessing so that the label is
not leaked through formatting. It is applied to human text only when a source needs it.

normalize_text(): detector-input hygiene (hypothesis H9), applied identically to training, evaluation and inference
text: NFKC, strip zero-width/soft-hyphen characters, map curly quotes to ASCII, fold Cyrillic/Greek homoglyphs inside
mixed-script words back to Latin, collapse whitespace. It deliberately keeps dashes, markdown and punctuation habits.
"""

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("‚"): "'", ord("‛"): "'",
           ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("«"): '"', ord("»"): '"'}
# common Cyrillic / Greek look-alikes -> Latin
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "н": "h", "к": "k",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "ο": "o", "ν": "v", "ι": "i", "ρ": "p", "Α": "A", "Β": "B", "Ε": "E",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
}
_MIXED = re.compile(r"\w*[A-Za-z]\w*[Ͱ-ϿЀ-ӿ]\w*|\w*[Ͱ-ϿЀ-ӿ]\w*[A-Za-z]\w*")


def _fold_word(m):
    return "".join(_CONFUSABLES.get(ch, ch) for ch in m.group(0))


def normalize_text(text):
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_ZERO_WIDTH).translate(_QUOTES)
    t = _MIXED.sub(_fold_word, t)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def fix_tokenization(text):
    t = text
    t = re.sub(r"``|''", '"', t)
    t = re.sub(r"(\w) n't\b", r"\1n't", t)
    t = re.sub(r"(\w) ?([’'])\s?(s|ve|re|ll|m|d|t)\b", r"\1\2\3", t)
    t = re.sub(r"“ +", "“", t)
    t = re.sub(r" +”", "”", t)
    t = re.sub(r" +([.,;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[]) +", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_EMPH = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.M)
_MD_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.M)


def strip_markdown(text):
    """Remove markdown syntax (headings, emphasis markers, bullet glyphs, rules) so chat-UI formatting is not a cue."""
    t = _MD_RULE.sub("", text)
    t = _MD_HEADING.sub("", t)
    t = _MD_EMPH.sub(r"\2", t)
    t = _MD_BULLET.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def has_markdown(text):
    return bool(_MD_HEADING.search(text) or re.search(r"\*\*\S.+?\S\*\*", text) or _MD_BULLET.search(text))
