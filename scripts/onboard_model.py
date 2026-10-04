#!/usr/bin/env python
"""
New-model onboarding: does a freshly released model slip past the detector?

  python scripts/onboard_model.py --model openai/gpt-6-astra --n 100 --backend openrouter      # needs OPENROUTER_API_KEY
  python scripts/onboard_model.py --model claude-sonnet-5-5 --n 100 --backend anthropic         # needs ANTHROPIC_API_KEY
  python scripts/onboard_model.py --model gpt-6-astra --file pasted_outputs.txt                 # user-pasted ChatGPT outputs
  python scripts/onboard_model.py --model claude-opus-5-5 --corpus dev                          # already-ingested real texts (dev only)

Prompts come from the dev split of data/corpus/prompts.jsonl (never the locked test). Texts are scored by a detector
(--detector shipped | cand:<name>) at the threshold fixed on dev clean-human text for 1% FPR (the same rule as
scripts/eval_frontier.py). Prints TPR with a Wilson 95% CI and whether the model "evades" (TPR below --min-tpr).
FAIL-CLOSED: with no API key, no --file and no --corpus it exits with an error; it never simulates model output.
Generated texts are saved to data/corpus/onboard/<model>.jsonl with provenance.
"""

import argparse
import datetime
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import io_utils, metrics  # noqa: E402

D = io_utils.DATA
KEYS = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def dev_prompts(n):
    ps = [p for p in io_utils.read_jsonl(os.path.join(D, "corpus", "prompts.jsonl")) if p["split"] == "dev"]
    return ps[:n]


def generate(backend, model, prompts):
    key = os.environ.get(KEYS[backend])
    if not key:
        sys.exit(f"FAIL-CLOSED: {KEYS[backend]} is not set; refusing to simulate {model}. Use --file or --corpus instead.")
    out = []
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        for p in prompts:
            r = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": p["instruction"]}])
            out.append((p, "".join(b.text for b in r.content if getattr(b, "type", "") == "text")))
    else:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1" if backend == "openrouter" else None)
        for p in prompts:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": p["instruction"]}])
            out.append((p, r.choices[0].message.content or ""))
    return out


def read_file(path):
    txt = open(path, encoding="utf-8").read()
    if path.endswith(".jsonl"):
        return [r["text"] for r in (json.loads(l) for l in txt.splitlines() if l.strip())]
    blocks = [b.strip() for b in re.split(r"\n\s*-{3,}\s*\n|\n={3,}[^\n]*\n", txt) if b.strip()]
    return blocks


def dev_threshold(det_name, detector_factory):
    from scripts.eval_frontier import cached_scores, is_clean_neg, load_split, thin
    dev = thin(load_split("dev"), 1500, 300)
    neg = [r for r in dev if is_clean_neg(r)]
    sc = cached_scores(det_name, "dev", neg, detector_factory)
    return metrics.threshold_at_fpr([sc[r["id"]] for r in neg if r["id"] in sc], 0.01), len(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", choices=list(KEYS), default=None)
    ap.add_argument("--file", default=None, help="user-pasted outputs (.jsonl with 'text', or blocks separated by '---')")
    ap.add_argument("--corpus", choices=["dev"], default=None, help="use already-ingested real texts of this model (dev split only)")
    ap.add_argument("--detector", default="shipped", help="shipped | cand:<name>")
    ap.add_argument("--min-tpr", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    if args.file:
        texts, source = read_file(args.file)[: args.n], f"user_paste:{os.path.basename(args.file)}"
    elif args.corpus:
        import glob
        texts = [r["text"] for f in glob.glob(os.path.join(D, "corpus", "frontier", args.model, "*.jsonl"))
                 for r in io_utils.read_jsonl(f) if r["split"] == "dev" and not r.get("quality_flags")][: args.n]
        source = f"corpus:frontier/{args.model}/dev"
    elif args.backend:
        pairs = generate(args.backend, args.model, dev_prompts(args.n))
        texts, source = [t for _, t in pairs if t.strip()], f"api:{args.backend}"
        rows = [{"id": io_utils.text_id(t), "text": t, "label": "ai", "origin": "ai_raw", "generator_id": args.model,
                 "access_path": source, "date": datetime.date.today().isoformat(), "prompt_id": p["prompt_id"], "attack_id": "none",
                 "parent_id": None, "domain": p["domain"], "esl": False, "words": io_utils.word_count(t)} for p, t in pairs if t.strip()]
        io_utils.write_jsonl(os.path.join(D, "corpus", "onboard", f"{args.model.replace('/', '_')}.jsonl"), rows)
    else:
        sys.exit("FAIL-CLOSED: give --backend (with an API key), --file or --corpus; this tool never simulates a model.")
    if not texts:
        sys.exit("no texts to score")

    from scripts.detectors.baselines import get_detector
    fac = lambda: get_detector(args.detector, threads=args.threads)  # noqa: E731
    t, n_neg = dev_threshold(args.detector, fac)
    scores = fac().score(texts)
    k, n = metrics.rate_above(scores, t)
    p, lo, hi = metrics.wilson(k, n)
    print(f"model={args.model} source={source} detector={args.detector} texts={n}")
    print(f"threshold (dev clean-human 1% FPR, n_neg={n_neg}) = {t:.4f}")
    print(f"TPR = {100 * p:.1f}% [{100 * lo:.1f}-{100 * hi:.1f}] ({k}/{n})")
    print("VERDICT:", "EVADES (below min TPR)" if p < args.min_tpr else "detected at or above min TPR")


if __name__ == "__main__":
    main()
