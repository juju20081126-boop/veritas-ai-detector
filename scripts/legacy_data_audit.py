"""
Audit of the legacy (pre-2026-10-01) synthetic data / evaluation pipeline.

Stdlib only. Reproduces the evidence in scratch/GOAL_BRIEF.md section 1 and writes it to
data/eval/legacy_audit.json so the claims "legacy data is synthetic / leaky / hard-coded" are
re-checkable by anyone with one command:

    python scripts/legacy_data_audit.py

It finds the legacy data in either data/{raw,processed} (before quarantine) or
data/_quarantine_synthetic/{raw,processed} (after), and legacy scripts in either scripts/ or
scripts/legacy_synthetic/.
"""

import glob
import json
import os
import re
import statistics
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
QUAR = os.path.join(DATA, "_quarantine_synthetic")
SCRIPT_DIRS = [os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts", "legacy_synthetic")]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return set(tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1)))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_dir(sub):
    for base in (DATA, QUAR):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            return d
    return None


def find_script(name):
    for d in SCRIPT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def grep_file(path, pattern, flags=0):
    hits = []
    if not path:
        return hits
    rx = re.compile(pattern, flags)
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if rx.search(line):
                hits.append((i, line.strip()[:140]))
    return hits


def main():
    out = {}
    print("=" * 78)
    print("LEGACY DATA AUDIT (re-run of GOAL_BRIEF section 1)")
    print("=" * 78)

    # 1. raw 'public dataset' placeholders
    raw_dir = find_dir("raw")
    raw_sets = {}
    print("\n[1] data/raw/*_samples.jsonl  (claimed: RAID, M4, HC3, DetectRL, MAGE, ESL samples)")
    out["raw"] = {}
    if raw_dir:
        for p in sorted(glob.glob(os.path.join(raw_dir, "*_samples.jsonl"))):
            rows = load_jsonl(p)
            uniq = set(r["text"] for r in rows)
            raw_sets[os.path.basename(p)] = frozenset(uniq)
            out["raw"][os.path.basename(p)] = {"rows": len(rows), "unique_texts": len(uniq)}
            print(f"    {os.path.basename(p):28s} rows={len(rows):3d} unique_texts={len(uniq)}")
        identical = len(set(raw_sets.values())) == 1 if raw_sets else False
        out["raw"]["identical_text_sets_across_files"] = identical
        print(f"    -> same set of unique texts in all {len(raw_sets)} files: {identical}")

    # 2. leakage between train and the test sets
    proc = find_dir("processed")
    print("\n[2] train/test leakage (exact copy or 5-gram Jaccard >= 0.5 vs any train text)")
    out["leakage"] = {}
    if proc:
        train = load_jsonl(os.path.join(proc, "train.jsonl"))
        tr_texts = {r["text"] for r in train}
        tr_sh = [shingles(r["text"]) for r in train]
        out["train"] = {
            "rows": len(train),
            "unique_texts": len(tr_texts),
            "has_teacher_soft_probs": any("teacher_soft_probs" in r for r in train),
            "median_words": statistics.median(len(r["text"].split()) for r in train),
            "domain_counts_top": Counter(r.get("domain") for r in train).most_common(3),
        }
        print(f"    train: rows={len(train)} unique={len(tr_texts)} median_words={out['train']['median_words']} "
              f"has_teacher_soft_probs={out['train']['has_teacher_soft_probs']} top_domains={out['train']['domain_counts_top']}")
        for name in ("test_indist", "test_paraphrased", "test_unseen_qwen", "test_unseen_deepseek", "test_esl"):
            p = os.path.join(proc, name + ".jsonl")
            if not os.path.exists(p):
                continue
            rows = load_jsonl(p)
            exact = sum(r["text"] in tr_texts for r in rows)
            near = 0
            for r in rows:
                s = shingles(r["text"])
                if any(jaccard(s, t) >= 0.5 for t in tr_sh):
                    near += 1
            out["leakage"][name] = {"n": len(rows), "exact_copies_in_train": exact, "near_dup_in_train": near}
            print(f"    {name:22s} n={len(rows):3d} exact_in_train={exact:3d} near_dup(J>=0.5)={near:3d}")

        # 3. how different are 'paraphrased' texts from their un-paraphrased sources?
        print("\n[3] 'paraphrased' test texts vs nearest un-paraphrased AI text in the repo")
        ai_pool = []
        for p in glob.glob(os.path.join(proc, "*.jsonl")):
            for r in load_jsonl(p):
                if r.get("label") == "ai_generated":
                    ai_pool.append(shingles(r["text"]))
        para = load_jsonl(os.path.join(proc, "test_paraphrased.jsonl"))
        best = []
        for r in para:
            s = shingles(r["text"])
            best.append(max((jaccard(s, a) for a in ai_pool), default=0.0))
        out["paraphrase_similarity"] = {
            "n": len(best),
            "median_jaccard_to_source": round(statistics.median(best), 3),
            "share_ge_0.9": round(sum(b >= 0.9 for b in best) / len(best), 3),
            "share_ge_0.5": round(sum(b >= 0.5 for b in best) / len(best), 3),
        }
        print(f"    n={len(best)} median J={out['paraphrase_similarity']['median_jaccard_to_source']} "
              f"share>=0.9={out['paraphrase_similarity']['share_ge_0.9']} share>=0.5={out['paraphrase_similarity']['share_ge_0.5']}")

    # 4. source-level evidence (templates, hard-coded metrics, silent fallbacks, unforwarded flags)
    print("\n[4] source-level evidence")
    checks = [
        ("generate_frontier_data.py", r"def synthesize_model_fingerprint", "canned-template generator exists"),
        ("download_datasets.py", r"offline research seeds|synthetic research mirrors", "silent fallback to synthetic 'dataset' seeds"),
        ("evaluate_models.py", r"teacher_tpr_at_1fpr\s*=\s*0?\.\d+", "hard-coded teacher TPR literal"),
        ("generate_notebooks.py", r"Accuracy on val: 96\.8%", "hard-coded teacher notebook output"),
        ("train_student.py", r"smoothed one-hot", "student trained on label-smoothed one-hot (no teacher soft probs)"),
        ("refresh_pipeline.py", r"generate_frontier_data\.py", "generator call (check whether --new_models is forwarded)"),
    ]
    out["source_evidence"] = []
    for fname, pat, what in checks:
        path = find_script(fname)
        hits = grep_file(path, pat, re.I)
        rel = os.path.relpath(path, REPO) if path else None
        out["source_evidence"].append({"file": rel, "what": what, "hits": hits[:3]})
        status = "FOUND" if hits else "not found"
        loc = f"{rel}:{hits[0][0]}" if hits else (rel or fname + " (missing)")
        print(f"    [{status:9s}] {what:62s} {loc}")
    rp = find_script("refresh_pipeline.py")
    if rp:
        fwd = [h for h in grep_file(rp, r"new_models")]
        forwarded = any("generate_frontier_data" in h[1] and "new_models" in h[1] for h in fwd)
        out["new_models_forwarded_to_generator"] = forwarded
        print(f"    refresh_pipeline.py forwards --new_models to the generator: {forwarded}")

    # 5. QuillBot comparison sheet: any real verdicts?
    sheet = os.path.join(DATA, "quillbot_comparison_sheet.md")
    if os.path.exists(sheet):
        txt = open(sheet, encoding="utf-8").read()
        pending = txt.count("*(Pending)*")
        out["quillbot_sheet"] = {"pending_cells": pending, "expected_cells": 60}
        print(f"\n[5] data/quillbot_comparison_sheet.md: {pending} '(Pending)' cells (30 QuillBot verdicts + 30 Veritas verdicts expected)")

    os.makedirs(os.path.join(DATA, "eval"), exist_ok=True)
    out_path = os.path.join(DATA, "eval", "legacy_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    sys.exit(main())
