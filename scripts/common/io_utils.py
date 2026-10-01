"""I/O helpers: JSONL (optionally gzip), hashing, word counts, length buckets."""

import gzip
import hashlib
import json
import os
import re
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")

_WORD_RX = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path):
    rows = []
    with open_text(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Atomic write (temp file + replace). Creates parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(tmp, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_id(text):
    """Stable 16-hex id of the exact text (used for dedup and parent links)."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def word_count(text):
    return len(_WORD_RX.findall(text))


LENGTH_BUCKETS = ("<50", "50-100", "100-250", "250-600", "600+")


def length_bucket(words):
    if words < 50:
        return "<50"
    if words < 100:
        return "50-100"
    if words < 250:
        return "100-250"
    if words < 600:
        return "250-600"
    return "600+"
