#!/usr/bin/env python
"""
Human student essays from W&I+LOCNESS (BEA-2019 shared task): real non-native learner essays (CEFR A/B/C) and native
university-student essays (LOCNESS). Free for NON-COMMERCIAL research use only, so the texts stay out of git
(data/corpus/human/ is ignored); only the registry entry and hashes are tracked.

  python scripts/corpus/build_esl.py
"""

import io
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402
from scripts.corpus import registry  # noqa: E402

URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
CACHE = os.path.join(io_utils.DATA, "cache", "wi+locness_v2.1.bea19.tar.gz")
OUT = os.path.join(io_utils.DATA, "corpus", "human", "wi_locness.jsonl")
MIN_WORDS = 60


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not os.path.exists(CACHE):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, CACHE)
    sha = io_utils.sha256_file(CACHE)
    docs = []
    with tarfile.open(CACHE, "r:gz") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile() and "/json/" in m.name and m.name.endswith(".json")]
        print("json members:", sorted(os.path.basename(n) for n in names))
        for n in sorted(names):
            base = os.path.basename(n)            # e.g. A.train.json, N.dev.json
            level = base.split(".")[0]
            f = tf.extractfile(n)
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                text = r.get("text", "")
                w = io_utils.word_count(text)
                if w < MIN_WORDS:
                    continue
                esl = level in ("A", "B", "C")
                docs.append({
                    "doc_id": f"wi:{r.get('id')}", "text": text.strip(), "domain": "student_essay", "esl": esl,
                    "cefr": r.get("cefr", level), "user": f"wiu:{r.get('userid') or r.get('id')}", "words": w,
                    "source_file": base, "corpus": "wi_locness", "revision": sha[:10]})
    io_utils.write_jsonl(OUT, docs)
    registry.register("human_corpora", "wi_locness", {
        "hf_repo": None, "url": URL, "revision": sha[:10], "sha256_archive": sha,
        "license": "non-commercial research use only (BEA 2019 shared task page: 'All corpora are subject to similar licences "
                   "and may only be used for non-commercial purposes')",
        "published_before": "2019-08-01", "verified_human": True, "domain": "student_essay",
        "note": "ESL = W&I levels A/B/C (non-native learners); LOCNESS (N) = native university students"})
    esl = [d for d in docs if d["esl"]]
    nat = [d for d in docs if not d["esl"]]
    users = len({d["user"] for d in docs})
    print(f"docs={len(docs)} ESL={len(esl)} native={len(nat)} distinct authors={users}")
    print("by file:", dict(Counter(d["source_file"] for d in docs)))
    print("by CEFR:", dict(Counter(d["cefr"] for d in docs)))
    for name, xs in (("ESL", esl), ("native", nat)):
        w = sorted(d["words"] for d in xs)
        print(f"{name} words: min={w[0]} p25={w[len(w)//4]} median={w[len(w)//2]} p75={w[3*len(w)//4]} max={w[-1]}")
    print("sample ESL:", repr(esl[0]["text"][:220]))


if __name__ == "__main__":
    main()
