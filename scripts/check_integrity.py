#!/usr/bin/env python
"""
Integrity gates for the Veritas real-data pipeline. FAIL-CLOSED: a gate with nothing to verify fails.

    python scripts/check_integrity.py                  # check data/splits + data/locked
    python scripts/check_integrity.py --legacy-demo    # run the gates on the quarantined synthetic data (expected FAIL)
    python scripts/check_integrity.py --json out.json  # also save the report

Gates
  G1 PROVENANCE    every row has the schema fields and real provenance (model, access path, date, prompt id,
                   attack id); human rows come from registered pre-ChatGPT corpora; AI rows from registered generators
  G2 NO-SYNTHETIC  no synthetic/template markers, no text from data/_quarantine_synthetic, no active template generator
  G3 METRIC-LINT   no hard-coded metric literals (tpr/fpr/f1/auc/ece/acc/recall/precision = 0.xyz) in scripts/ or backend/
  G4 SPLITS        prompt/source groups disjoint across train/dev/locked; 0 exact and 0 near-duplicates (5-gram J>=0.5)
  G5 LOCKED        locked file SHA-256 matches data/locked/MANIFEST.json; <= 3 locked evaluations in ACCESS_LOG.md
  G6 COUNTS        locked-test minimums: human >= 1000 (>= 300 ESL); per frontier model >= 150 raw and >= 40 per
                   attack family (A1,A2,A3,A4,A5,A7) -- or the model is declared UNTESTED in the manifest

Exit code 0 only if every gate passes.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import dedup, io_utils, schema  # noqa: E402

DATA = os.path.join(REPO, "data")
SPLITS_DIR = os.path.join(DATA, "splits")
LOCKED_DIR = os.path.join(DATA, "locked")
REGISTRY_PATH = os.path.join(DATA, "corpus", "registry.json")
QUAR_DIR = os.path.join(DATA, "_quarantine_synthetic")

MIN_HUMAN = 1000
MIN_ESL = 300
MIN_RAW_PER_MODEL = 150
MIN_PER_FAMILY = 40
MAX_LOCKED_ACCESSES = 3

_ASSIGN = re.compile(
    r"""^\s*(?P<lhs>["']?[A-Za-z_][\w.]*["']?)\s*(?:=|:)\s*(?P<val>-?\d+\.\d+(?:[eE]-?\d+)?)\s*[,)]*\s*(?:#.*)?$"""
)
_METRIC_TOKENS = {"tpr", "fpr", "f1", "auc", "auroc", "ece", "acc", "accuracy", "recall", "precision",
                  "sensitivity", "specificity"}
_EXEMPT_TOKENS = {"target", "limit", "cap", "budget", "threshold", "thr", "min", "max", "alpha", "default",
                  "required", "floor", "ceil", "tol", "tolerance", "eps", "lr", "weight", "margin",
                  "temperature", "bound"}
_NEUTRAL_VALUES = {0.0, 1.0, 100.0}


# ----------------------------------------------------------------------------- loading
def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _shim_legacy(r, fname, i):
    """Map a legacy synthetic row onto the new schema so the gates can show why it fails."""
    text = r["text"]
    lab = "human" if r.get("label") in ("human", "human_ai_refined") else "ai"
    return {
        "id": io_utils.text_id(text), "text": text, "label": lab,
        "origin": "human" if lab == "human" else "ai_raw",
        "generator_id": str(r.get("generator", "")), "access_path": "legacy:" + fname,
        "date": "", "prompt_id": str(r.get("tag", "")), "attack_id": "none", "parent_id": None,
        "domain": str(r.get("domain", "")), "esl": r.get("generator") == "human_esl",
        "words": r.get("word_count", io_utils.word_count(text)),
    }


def load_rows(legacy_demo=False):
    rows = {"train": [], "dev": [], "locked": []}
    if legacy_demo:
        for p in sorted(glob.glob(os.path.join(QUAR_DIR, "processed", "*.jsonl"))):
            name = os.path.basename(p)
            split = {"train.jsonl": "train", "val.jsonl": "dev"}.get(name, "locked")
            for i, r in enumerate(io_utils.read_jsonl(p)):
                rows[split].append(_shim_legacy(r, name, i))
        return rows
    for split in ("train", "dev"):
        p = _first_existing(os.path.join(SPLITS_DIR, split + ".jsonl.gz"), os.path.join(SPLITS_DIR, split + ".jsonl"))
        if p:
            rows[split] = io_utils.read_jsonl(p)
    p = os.path.join(LOCKED_DIR, "locked_test.jsonl.gz")
    if os.path.exists(p):
        rows["locked"] = io_utils.read_jsonl(p)
    return rows


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_manifest():
    p = os.path.join(LOCKED_DIR, "MANIFEST.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


# ----------------------------------------------------------------------------- gates
def _kinds(msgs):
    return Counter(re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "#", m.split(": ", 1)[-1]) for m in msgs).most_common(5)


def gate_provenance(rows_by_split, registry):
    n = sum(len(v) for v in rows_by_split.values())
    if n == 0:
        return False, "no rows loaded - nothing to verify (fail closed)", []
    problems = []
    for split, rows in rows_by_split.items():
        for r in rows:
            errs = schema.validate_record(r, registry)  # registry=None skips only the registry look-ups
            if errs:
                problems.append(f"[{split}] {r.get('id', '?')}: " + "; ".join(errs))
    if registry is None:
        return False, ("data/corpus/registry.json is missing; "
                       f"{len(problems)}/{n} rows violate the schema/provenance rules"), \
            [f"{c} x {k}" for k, c in _kinds(problems)] + problems[:3]
    if problems:
        det = [f"{c} x {k}" for k, c in _kinds(problems)] + problems[:3]
        return False, f"{len(problems)}/{n} rows violate the schema/provenance rules", det
    return True, f"{n} rows, all with registered provenance", []


def _active_script_files():
    files = []
    for base in ("scripts", "backend"):
        for p in glob.glob(os.path.join(REPO, base, "**", "*.py"), recursive=True):
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            if "/.venv/" in rel or "__pycache__" in rel:
                continue
            files.append((p, rel))
    return files


def gate_no_synthetic(rows_by_split):
    problems = []
    n = sum(len(v) for v in rows_by_split.values())
    if n == 0:
        problems.append("no rows loaded - nothing to verify (fail closed)")
    # (a) texts that come from the quarantined synthetic pool
    quarantined = set()
    for p in glob.glob(os.path.join(QUAR_DIR, "**", "*.jsonl"), recursive=True):
        for r in io_utils.read_jsonl(p):
            quarantined.add(io_utils.text_id(r["text"]))
    hits = sum(1 for rows in rows_by_split.values() for r in rows if r.get("id") in quarantined
               or io_utils.text_id(r.get("text", "")) in quarantined)
    if hits:
        problems.append(f"{hits} rows are texts from data/_quarantine_synthetic")
    # (b) explicit synthetic markers in any metadata field
    marked = sum(1 for rows in rows_by_split.values() for r in rows
                 if any(schema.BANNED_WORDS.search(str(r.get(k, ""))) for k in
                        ("generator_id", "access_path", "source", "tag", "origin")))
    if marked:
        problems.append(f"{marked} rows carry synthetic/template/legacy markers in provenance fields")
    # (c) no active code path may synthesize 'model' text
    bad_code = []
    rx = re.compile(r"def synthesize_model_fingerprint|offline research seeds|synthetic research mirrors")
    for p, rel in _active_script_files():
        if rel.startswith("scripts/legacy_synthetic/") or rel in ("scripts/check_integrity.py", "scripts/legacy_data_audit.py"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if rx.search(line):
                    bad_code.append(f"{rel}:{i}")
    if bad_code:
        problems.append("active code can synthesize text: " + ", ".join(bad_code[:5]))
    if problems:
        return False, "; ".join(problems), []
    return True, f"{n} rows free of synthetic markers; no active template generator", []


def lint_line(line):
    """True if `line` assigns a literal float to a metric-named variable/dict key (e.g. tpr = 0.962)."""
    if "integrity: allow" in line:
        return False
    m = _ASSIGN.match(line)
    if not m:
        return False
    lhs = m.group("lhs").strip("'\"").lower()
    toks = set(t for t in re.split(r"[^a-z0-9]+", lhs) if t)
    if not (toks & _METRIC_TOKENS) or (toks & _EXEMPT_TOKENS):
        return False
    return float(m.group("val")) not in _NEUTRAL_VALUES


def gate_metric_lint():
    flagged = []
    for p, rel in _active_script_files():
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if lint_line(line):
                    flagged.append(f"{rel}:{i}: {line.strip()[:90]}")
    if flagged:
        return False, f"{len(flagged)} hard-coded metric literal(s)", flagged[:8]
    return True, "no hard-coded metric literals in scripts/ or backend/", []


def gate_splits(rows_by_split):
    nonempty = {s: r for s, r in rows_by_split.items() if r}
    if len(nonempty) < 3:
        return False, f"need train, dev and locked splits; found {sorted(nonempty)}", []
    problems, det = [], []
    names = sorted(nonempty)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga = {r["prompt_id"] for r in nonempty[a]}
            gb = {r["prompt_id"] for r in nonempty[b]}
            both = ga & gb
            if both:
                problems.append(f"{len(both)} prompt/source groups shared by {a} and {b}")
                det.append(f"e.g. {sorted(both)[:3]}")
            ia = {r["id"] for r in nonempty[a]}
            ib = {r["id"] for r in nonempty[b]}
            ex = ia & ib
            if ex:
                problems.append(f"{len(ex)} exact duplicate texts across {a} and {b}")
    docs = [(s, r["id"], r["text"]) for s, rows in nonempty.items() for r in rows]
    pairs = dedup.cross_group_near_duplicates(docs, k=5, threshold=0.5)
    if pairs:
        problems.append(f"{len(pairs)} near-duplicate pairs (5-gram Jaccard >= 0.5) across splits")
        det += [f"{a}:{ia} ~ {b}:{ib} J={j:.2f}" for a, ia, b, ib, j in pairs[:3]]
    lab = {}
    for rows in nonempty.values():
        for r in rows:
            lab.setdefault(r["id"], set()).add(r["label"])
    clash = sum(1 for v in lab.values() if len(v) > 1)
    if clash:
        problems.append(f"{clash} identical texts labelled both human and ai")
    if problems:
        return False, "; ".join(problems), det
    return True, f"{len(docs)} rows: groups disjoint, 0 exact dups, 0 near-dups across {names}", []


def gate_locked(rows_by_split):
    m = load_manifest()
    if m is None:
        return False, "data/locked/MANIFEST.json is missing", []
    problems = []
    for name, info in m.get("files", {}).items():
        p = os.path.join(LOCKED_DIR, name)
        if not os.path.exists(p):
            problems.append(f"{name} missing")
            continue
        if io_utils.sha256_file(p) != info.get("sha256"):
            problems.append(f"{name}: SHA-256 does not match the manifest")
        if info.get("n") is not None and len(io_utils.read_jsonl(p)) != info["n"]:
            problems.append(f"{name}: row count differs from the manifest")
    if not m.get("files"):
        problems.append("manifest lists no files")
    log = os.path.join(LOCKED_DIR, "ACCESS_LOG.md")
    entries = []
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            entries = [ln.strip() for ln in f if re.match(r"^- \d{4}-\d{2}-\d{2}T", ln)]
    bad = [e for e in entries if "model=" not in e or "cmd=" not in e]
    if len(entries) > MAX_LOCKED_ACCESSES:
        problems.append(f"locked test evaluated {len(entries)} times (> {MAX_LOCKED_ACCESSES})")
    if bad:
        problems.append(f"{len(bad)} access-log entries lack model=/cmd=")
    if problems:
        return False, "; ".join(problems), []
    return True, f"SHA-256 matches manifest; {len(entries)}/{MAX_LOCKED_ACCESSES} locked evaluations used", []


def locked_counts(locked_rows, registry, manifest):
    humans = [r for r in locked_rows if r["label"] == "human"]
    esl = [r for r in humans if r.get("esl")]
    declared_untested = set((manifest or {}).get("untested", []))
    frontier = [g for g, v in (registry or {}).get("generators", {}).items() if v.get("kind") == "frontier"]
    table = {"human": len(humans), "esl": len(esl), "models": {}}
    for g in frontier:
        src = [r for r in locked_rows if r["label"] == "ai" and r["generator_id"] == g]
        raw = sum(1 for r in src if r["attack_id"] == "none")
        fam = Counter(schema.attack_family(r["attack_id"]) for r in src if r["attack_id"] != "none")
        table["models"][g] = {"raw": raw, "families": dict(fam), "untested": g in declared_untested}
    return table


def gate_counts(rows_by_split, registry):
    locked = rows_by_split.get("locked", [])
    if not locked:
        return False, "no locked rows loaded", []
    table = locked_counts(locked, registry, load_manifest())
    problems = []
    if table["human"] < MIN_HUMAN:
        problems.append(f"human {table['human']} < {MIN_HUMAN}")
    if table["esl"] < MIN_ESL:
        problems.append(f"ESL {table['esl']} < {MIN_ESL}")
    if not table["models"]:
        problems.append("registry lists no frontier generators")
    for g, t in table["models"].items():
        if t["untested"]:
            if t["raw"] or t["families"]:
                problems.append(f"{g} is declared UNTESTED but has rows")
            continue
        if t["raw"] < MIN_RAW_PER_MODEL:
            problems.append(f"{g}: raw {t['raw']} < {MIN_RAW_PER_MODEL} (add samples or declare UNTESTED)")
        for fam in schema.T3_FAMILIES:
            if t["families"].get(fam, 0) < MIN_PER_FAMILY:
                problems.append(f"{g}: {fam} {t['families'].get(fam, 0)} < {MIN_PER_FAMILY}")
    det = [f"locked human={table['human']} (ESL={table['esl']})"]
    for g, t in table["models"].items():
        if t["untested"]:
            det.append(f"{g:22s} UNTESTED (declared)")
        else:
            fam = " ".join(f"{f}={t['families'].get(f, 0)}" for f in schema.T3_FAMILIES)
            det.append(f"{g:22s} raw={t['raw']:4d} {fam} | A6={t['families'].get('A6', 0)} A8={t['families'].get('A8', 0)}")
    if problems:
        return False, "; ".join(problems[:6]) + (f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""), det
    return True, "locked-test minimums met", det


# ----------------------------------------------------------------------------- runner
def run_all(legacy_demo=False):
    rows = load_rows(legacy_demo)
    registry = load_registry()
    results = []

    def add(gid, name, res):
        ok, summary, details = res
        results.append({"gate": gid, "name": name, "pass": bool(ok), "summary": summary, "details": details})

    add("G1", "PROVENANCE", gate_provenance(rows, registry))
    add("G2", "NO-SYNTHETIC", gate_no_synthetic(rows))
    add("G3", "METRIC-LINT", gate_metric_lint())
    add("G4", "SPLITS", gate_splits(rows))
    add("G5", "LOCKED", gate_locked(rows))
    add("G6", "COUNTS", gate_counts(rows, registry))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-demo", action="store_true", help="run the gates on data/_quarantine_synthetic (expected FAIL)")
    ap.add_argument("--json", default=None, help="write the report to this path")
    args = ap.parse_args()

    results = run_all(args.legacy_demo)
    print("=" * 78)
    print("INTEGRITY GATES" + ("  [legacy-demo: quarantined synthetic data]" if args.legacy_demo else ""))
    print("=" * 78)
    for r in results:
        print(f"{r['gate']} {r['name']:<13s} {'PASS' if r['pass'] else 'FAIL'}  {r['summary']}")
        for d in r["details"]:
            print(f"      {d}")
    ok = all(r["pass"] for r in results)
    npass = sum(r["pass"] for r in results)
    print("-" * 78)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}  ({npass}/{len(results)} gates passed)")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"overall_pass": ok, "gates": results}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
