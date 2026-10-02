#!/usr/bin/env python
"""
Validate and ingest raw generation files written by Claude Code subagents.

  python scripts/corpus/ingest_generations.py            # ingest every batch whose raw file exists
  python scripts/corpus/ingest_generations.py --batch opus__train__018

Raw format (written by the worker):   === prompt_id: <id> ===\n<reply text>\n\n=== prompt_id: ...
Output: data/corpus/frontier/<generator>/<batch_id>.jsonl  (tracked: generations are not reproducible)

Each row follows scripts/common/schema.py with provenance generator_id / access_path / date / prompt_id and extra
fields task_type, target_words, style, batch_id, quality_flags. Hard flags (too_short, too_long, refusal, meta_words,
duplicate_or_missing) exclude a text from the splits; a conversational preamble/outro line is stripped (a user pastes only
the body) and recorded as `stripped_wrapper`.
"""

import argparse
import datetime
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils, textnorm  # noqa: E402

FRONTIER = os.path.join(io_utils.DATA, "corpus", "frontier")
ALIAS = {"opus": "claude-code-subagent:opus", "sonnet": "claude-code-subagent:sonnet", "haiku": "claude-code-subagent:haiku"}
HARD = {"too_short", "too_long", "refusal", "meta_words", "duplicate_or_missing"}

_PRE = re.compile(r"^(sure|certainly|of course|absolutely|okay|great)\b[^\n]{0,160}$|^here(?:'s| is| are)\b[^\n]{0,200}:\s*$", re.I)
_POST = re.compile(r"^(let me know|feel free|i hope (this|that)|would you like|hope this helps|if you(?:'d| would) like)\b", re.I)
_REFUSE = re.compile(r"\b(i'?m sorry, but|i can(?:'|no)t (help|assist|write)|as an ai\b|i (cannot|can't) fulfill)", re.I)
_META = re.compile(r"\(\s*(?:approx(?:imately|\.)?|about|roughly)?\s*\d{2,4}\s*words?\s*\)|\bword count\s*:", re.I)


def parse_raw(path):
    txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^=== prompt_id: (\S+) ===[ \t]*$", txt)
    items = []
    for i in range(1, len(parts), 2):
        items.append((parts[i], parts[i + 1].strip()))
    return items, parts[0].strip()


def strip_wrapper(text):
    lines = text.split("\n")
    stripped = False
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _PRE.match(lines[0].strip()):
        lines.pop(0)
        stripped = True
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _POST.match(lines[-1].strip()):
        lines.pop()
        stripped = True
    while lines and (not lines[-1].strip() or re.fullmatch(r"-{3,}", lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).strip(), stripped


def ingest_batch(b, prompts_by_id, date):
    raw = os.path.join(REPO, b["out_path"])
    tasks = json.load(open(os.path.join(REPO, b["tasks_path"]), encoding="utf-8"))
    want = [t["prompt_id"] for t in tasks]
    items, junk = parse_raw(raw)
    seen, rows, report = set(), [], {"batch_id": b["batch_id"], "expected": len(want), "parsed": len(items), "leading_junk": bool(junk), "flags": {}}
    alias = b["batch_id"].split("__")[0]
    for pid, body in items:
        flags = []
        if pid not in want or pid in seen:
            flags.append("duplicate_or_missing")
        seen.add(pid)
        p = prompts_by_id.get(pid)
        if p is None:
            continue
        text, stripped = strip_wrapper(body)
        if stripped:
            flags.append("stripped_wrapper")
        w = io_utils.word_count(text)
        r = w / max(1, p["target_words"])
        asked_length = bool(re.search(r"\b\d{2,4}[- ]words?\b", p["instruction"], re.I))
        if (asked_length and r < 0.45) or w < 25:
            flags.append("too_short")
        if asked_length and r > 2.2:
            flags.append("too_long")
        if _REFUSE.search(text[:400]):
            flags.append("refusal")
        if _META.search(text):
            flags.append("meta_words")
        for f in flags:
            report["flags"][f] = report["flags"].get(f, 0) + 1
        rows.append({
            "id": io_utils.text_id(text), "text": text, "label": "ai", "origin": "ai_raw",
            "generator_id": b["generator_id"], "access_path": ALIAS[alias], "date": date,
            "prompt_id": pid, "attack_id": "none", "parent_id": None, "domain": p["domain"], "esl": False,
            "words": w, "markdown": textnorm.has_markdown(text), "split": p["split"], "task_type": p["task_type"],
            "target_words": p["target_words"], "style": p["style"], "batch_id": b["batch_id"], "quality_flags": flags,
        })
    missing = [pid for pid in want if pid not in seen]
    report["missing"] = missing
    out_dir = os.path.join(FRONTIER, b["generator_id"])
    io_utils.write_jsonl(os.path.join(out_dir, b["batch_id"] + ".jsonl"), rows)
    report["rows"] = len(rows)
    report["usable"] = sum(1 for r in rows if not (set(r["quality_flags"]) & HARD))
    report["words_median"] = sorted(r["words"] for r in rows)[len(rows) // 2] if rows else 0
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()
    mpath = os.path.join(FRONTIER, "_manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8"))
    prompts = {p["prompt_id"]: p for p in io_utils.read_jsonl(os.path.join(io_utils.DATA, "corpus", "prompts.jsonl"))}
    date = datetime.date.today().isoformat()
    ingest_generations_prompts = prompts  # noqa: F841 (kept for readability)
    n_ok = 0
    for b in manifest["batches"]:
        if args.batch and b["batch_id"] != args.batch:
            continue
        if not os.path.exists(os.path.join(REPO, b["out_path"])):
            continue
        rep = ingest_batch(b, prompts, date)
        b["status"] = "ingested" if not rep["missing"] and rep["usable"] > 0 else "needs_review"
        b["report"] = {k: rep[k] for k in ("rows", "usable", "flags", "missing", "words_median")}
        n_ok += 1
        print(f"{b['batch_id']}: rows={rep['rows']}/{rep['expected']} usable={rep['usable']} median_words={rep['words_median']} "
              f"flags={rep['flags']} missing={rep['missing']} status={b['status']}")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("batches processed:", n_ok)


if __name__ == "__main__":
    main()
