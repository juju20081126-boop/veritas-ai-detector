#!/usr/bin/env python
"""
Write a student mode's runtime config (read by backend/runtime_engine.py; path from runtime_engine.STUDENT_MODES, e.g.
models/frontier/frontier_config.json) from a DEV result file.

  python scripts/write_frontier_config.py --candidate h1h2h9 --result data/eval/results/dev_frontier_onnx.json --detector frontier_onnx
  python scripts/write_frontier_config.py --mode multi_teacher --candidate multi_teacher_distilled \
      --result data/eval/results/dev_multi_teacher_onnx.json --detector multi_teacher_onnx

The threshold is the detector's own dev clean-human 1%-FPR threshold as computed by scripts/eval_frontier.py. Refuses locked
results: the threshold must never come from the locked test. Bootstrap order (the INT8 runtime needs a config before it can be
scored): first write the config from the PyTorch candidate's dev result (--detector cand:<name>), score frontier_onnx on dev,
then re-run this with --detector frontier_onnx so the shipped threshold matches the shipped INT8 model (for --mode multi_teacher:
<mode>_onnx = multi_teacher_onnx).
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.runtime_engine import STUDENT_MODES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(STUDENT_MODES), default="frontier")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()
    data = json.load(open(args.result, encoding="utf-8"))
    if data["args"]["split"] != "dev":
        sys.exit("REFUSED: thresholds come from the dev split only")
    res = next((r for r in data["results"] if r["detector"] == args.detector), None)
    if res is None:
        sys.exit(f"detector {args.detector} not in {args.result}")
    cfg = {"candidate": args.candidate, "threshold": res["threshold_1pct_fpr_dev"], "temperature": 1.0, "max_len": args.max_len,
           "threshold_rule": f"dev clean-human 1% FPR of {args.detector} (n_dev_neg={res['n_dev_neg']}, commit {res.get('commit', '?')})",
           "score": "P(AI-generated) + P(AI-generated & AI-refined)"}
    out = os.path.join(*STUDENT_MODES[args.mode])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(cfg, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, cfg)


if __name__ == "__main__":
    main()
