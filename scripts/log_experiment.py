#!/usr/bin/env python
"""
Append one row per detector of an eval_frontier.py result file to data/eval/experiments.csv (the experiment log).

  python scripts/log_experiment.py data/eval/results/dev_h1h2h9.json --hypothesis "H1+H2+H9" --note "MiniLM 2 epochs, attack weight 2"

Every number is copied from the result JSON (nothing is typed in by hand). Columns: when, commit, split, detector, hypothesis,
note, threshold, AUROC, realized FPR (all / ESL / native), TPR (pooled / raw / attacked) and TPR per generator|family.
"""

import argparse
import csv
import datetime
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "data", "eval", "experiments.csv")
BASE = ["when", "commit", "split", "detector", "hypothesis", "note", "threshold_1pct_fpr_dev", "auroc", "fpr_clean_human",
        "fpr_esl", "fpr_native", "tpr_pooled", "tpr_raw", "tpr_attacked"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    data = json.load(open(args.result, encoding="utf-8"))
    rows = []
    for r in data["results"]:
        row = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "commit": r.get("commit", ""), "split": r["split"],
               "detector": r["detector"], "hypothesis": args.hypothesis, "note": args.note,
               "threshold_1pct_fpr_dev": round(r["threshold_1pct_fpr_dev"], 5), "auroc": round(r["auroc_pooled"], 4)}
        for k in BASE[8:]:
            row[k] = round(r[k]["rate"], 4) if r[k]["n"] else ""
        for gf, c in r["tpr_by_generator_family"].items():
            row[f"tpr:{gf}"] = round(c["rate"], 4) if c["n"] else ""
        rows.append(row)
    old = list(csv.DictReader(open(CSV, encoding="utf-8"))) if os.path.exists(CSV) else []
    cols = BASE + sorted({k for r in old + rows for k in r} - set(BASE))
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old + rows)
    print(f"logged {len(rows)} row(s) -> {CSV}")


if __name__ == "__main__":
    main()
