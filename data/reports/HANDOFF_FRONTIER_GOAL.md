# Handoff: frontier-model detection goal (Claude → Antigravity)

Written 2026-10-04 ~21:20 (Taiwan time) by Claude Code at branch `claude/work`, commit after `be1a86e`.
Goal spec: `scratch/GOAL_CONDITION.txt` (the /goal text) and `scratch/GOAL_BRIEF.md` (full playbook). Read both first.

## 0. Where to work (important)

- Work **only** in `C:\Users\justi\AI detector-claude` on branch **`claude/work`**. All data (splits, locked test, score
  caches, ESL corpus) lives untracked in this worktree; the `main` checkout does not have it.
- Python: `C:\Users\justi\AI detector-claude\.venv\Scripts\python.exe` (3.11). Never use or install into the Hermes venv.
  Run every command from the worktree root (`cd "C:\Users\justi\AI detector-claude"`).
- PowerShell: double quotes in `git commit -m` get mangled → write the message to a file and `git commit -F file`
  (write it without BOM: `[IO.File]::WriteAllText($f, $msg, (New-Object Text.UTF8Encoding $false))`).
- Commit attribution line used so far: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (use your own if you prefer).
- **Never push, never merge into `main`, never delete data.** The user decides merges.

## 1. Hard rules (from the goal; breaking any of these invalidates the result)

1. **Locked test** (`data/locked/`): never look at it, never tune/choose thresholds/features on it. It may be evaluated
   at most 3 times in total; `scripts/eval_frontier.py --split locked` refuses after 3 and logs each call in
   `data/locked/ACCESS_LOG.md`. **Access #1 is used** (baselines, running now). Plan: **access #2 = ONE call with every
   final model** (final candidate + `frontier_onnx` + all LOFO candidates). Keep #3 as a spare.
2. All selection (which candidate, epochs, weights) uses the **dev** split only.
3. No hard-coded metrics, no typed-in numbers: every number in the report must come from a result JSON / printed output.
   `scripts/check_integrity.py` gate G3 lints for metric literals.
4. No simulated model output. GPT-6 Astra has **no real samples** → it is reported as **UNTESTED**, never faked.
5. No automation of detector/paraphraser web UIs.

## 2. State right now

| Item | State |
|---|---|
| Integrity gates | `check_integrity.py` → **7/7 PASS** (re-verified after merging main, commit 456cbb9) |
| pytest | `python -m pytest scripts/tests -q` → **18 passed, 2 skipped** (the 2 frontier tests unskip once `models/frontier/` exists) |
| Locked test | frozen. Human 1,380 (ESL 404). Opus: raw 174, A1 99, A2 71, A3 49, A4 40, A5 40, A7 40, A6 22. Sonnet: raw 174, A1 99, A2 44, A3 45, A4 40, A5 40, A7 40, A6 19. GPT-6 Astra UNTESTED |
| Shipped-model locked baseline | done (printed in `scratch/locked_baselines.log`): AUROC 0.59, TPR@1%FPR 0.2% pooled, raw AI 0.0%, realized FPR 0.3% → the shipped detector catches essentially nothing |
| **Background job 1** | locked baselines for `shipped hc3_roberta openai_roberta modernbert desklib` → writes `data/eval/results/locked_baselines.json` **only at the end** (several hours; desklib is slow). Log: `scratch/locked_baselines.log` |
| **Background job 2** | training candidate `h1h2h9` (MiniLM-L6, 4-class, 2 epochs, attack weight 2, input hygiene H9) → `models/candidates/h1h2h9/`. Log: `scratch/train_h1h2h9.log`. ~5.3 s/step, 2,140 steps → finishes ≈ 00:10 |
| Runtime integration | done, behind a flag: `VERITAS_DETECTOR=frontier` or `QuillBotDetectorEngine(mode="frontier")`. Default stays `shipped`. Needs `models/frontier/{student_model_int8.onnx, tokenizer/, frontier_config.json}` |
| onboard tool | `scripts/onboard_model.py` done, fails closed without keys (tested) |

Check jobs: `Get-Content scratch\train_h1h2h9.log -Tail 2; Get-Content scratch\locked_baselines.log -Tail 2`
and `Get-Process python`. **If job 1 dies, do not simply re-run it** — a re-run is logged as locked access #2. Tell the
user first. (Scores are cached in `data/eval/scores/`, so a re-run is fast, but it still costs an access.)

## 3. Remaining steps, in order

Replace `PY` with `.venv\Scripts\python.exe`. Score caches are keyed by detector name
(`data/eval/scores/<name>__<split>.jsonl`): **use a new `--name` for every training run**; if you re-export the ONNX model,
delete `data/eval/scores/frontier_onnx__*.jsonl` first.

### Step A: dev baselines file (fast, cached)
```
PY scripts/eval_frontier.py --split dev --detectors shipped hc3_roberta openai_roberta modernbert desklib --neg-cap 1500 --public-cap 300 --threads 4 --out data/eval/baselines_dev.json
PY scripts/log_experiment.py data/eval/baselines_dev.json --hypothesis baseline --note "pre-change detectors, raw text"
```

### Step B: evaluate h1h2h9 on dev (after job 2 prints `saved ...`)
```
PY scripts/eval_frontier.py --split dev --detectors cand:h1h2h9 --neg-cap 1500 --public-cap 300 --threads 6 --out data/eval/results/dev_h1h2h9.json
PY scripts/log_experiment.py data/eval/results/dev_h1h2h9.json --hypothesis "H1+H2+H9" --note "MiniLM-L6 4-class, 2 ep, attack weight 2, input hygiene"
```

### Step C: ablation + leave-one-family-out (LOFO) training runs
Run **one at a time** (CPU only; parallel runs slow everything). `--epochs 1` is acceptable for these to save time; say so in the report.
```
PY scripts/train_detector.py --name abl_noatk  --epochs 1 --no-attacks               # H2 ablation: no attacked data
PY scripts/train_detector.py --name lofo_noclaude --epochs 1 --hold-out-gen claude-    # T4: train without Claude 5.x, test on Opus/Sonnet
PY scripts/train_detector.py --name lofo_noA3  --epochs 1 --hold-out-family A3        # leave-one-attack-out: humanizer
PY scripts/train_detector.py --name lofo_noA4  --epochs 1 --hold-out-family A4        # leave-one-attack-out: T5 paraphraser
```
After each: eval on dev (same command as Step B with `cand:<name>`) and `log_experiment.py` with a clear hypothesis.
For LOFO, the number that matters is TPR on the **held-out** family (e.g. `claude-opus-5-5|none` for lofo_noclaude, the `A3`
rows for lofo_noA3).

### Step D: pick the final candidate (dev only) and improve if time allows
Pick the candidate with the best dev TPR@1%FPR that keeps dev realized FPR ≤1.5% and ESL FPR ≤2× native. Optional
improvements if a target misses on dev: more epochs, `--attack-weight 3`, larger `--chunks-per-doc`. Log every try,
including failures.

### Step E: export + integrate + verify
```
PY scripts/export_onnx.py --hf-dir models/candidates/<FINAL> --out-dir models/frontier
PY scripts/write_frontier_config.py --candidate <FINAL> --result data/eval/results/dev_<FINAL>.json --detector cand:<FINAL>
PY scripts/eval_frontier.py --split dev --detectors frontier_onnx --neg-cap 1500 --public-cap 300 --out data/eval/results/dev_frontier_onnx.json
PY scripts/write_frontier_config.py --candidate <FINAL> --result data/eval/results/dev_frontier_onnx.json --detector frontier_onnx
del models\frontier\student_model.onnx          # FP32 intermediate, not used at runtime (only this file)
PY -m pytest scripts/tests -q                   # all must pass, 0 skipped
PY scripts/benchmark_target.py --mode frontier --runs 3
PY scripts/onboard_model.py --model claude-opus-5-5 --corpus dev --detector frontier_onnx
PY scripts/onboard_model.py --model gpt-6-astra   # must print FAIL-CLOSED (no key) -> evidence that Astra is not simulated
```
Check `frontier_onnx` dev numbers are close to `cand:<FINAL>` (INT8 quantisation should cost ≤1-2 points).

### Step F: final locked evaluation (access #2, ONE call)
Only after Step E and after job 1 has finished.
```
PY scripts/eval_frontier.py --split locked --detectors cand:<FINAL> frontier_onnx cand:lofo_noclaude cand:lofo_noA3 cand:lofo_noA4 cand:abl_noatk --threads 6 --model-hash final:<FINAL>@<commit> --out data/eval/results/locked_final.json
PY scripts/log_experiment.py data/eval/results/locked_final.json --hypothesis final --note "locked access #2"
PY scripts/check_integrity.py
```
Do not change anything after seeing these numbers. They are the result, good or bad.

### Step G: report + handoff + commit
Write `data/reports/FRONTIER_DETECTION_REPORT.md` (structure in GOAL_BRIEF Phase 7): 1-page summary; locked counts table;
baseline-vs-final tables with 95% CIs (pooled, per generator, per attack, per length, per genre, ESL vs native); LOFO table;
T1-T4 each marked MET / NOT MET against the locked numbers; what worked / what did not (from `data/eval/experiments.csv`);
limitations (GPT-6 Astra UNTESTED, subagent-generated text instead of API, small cells, Haiku as the only humanizer,
train has only ~88 ESL essays); exact reproduction commands; next steps. **Copy numbers from the JSON files, never round
by hand.** Last line: `STATUS: TARGETS MET` only if T1-T4 are all met on the locked test, otherwise `STATUS: TARGETS NOT MET`
with a diagnosis. Note T3 is defined "pooled over the 3 models"; with Astra untested it can at most be met for 2 models.

Targets (locked): T1 realized human FPR ≤1.5% and ESL ≤2× native · T2 raw TPR ≥90% per model · T3 TPR ≥70% per
paraphrase/humanizer family and ≥15 points above baseline · T4 LOFO TPR ≥60%.

Then append a `HANDOFF.md` entry (files changed, commands, and the README/EVAL_REPORT claims that must be corrected:
any legacy "96%"/teacher accuracy numbers come from synthetic data and must be replaced by the locked numbers), and commit:
`git add data/reports scripts backend models/frontier data/eval/experiments.csv HANDOFF.md` (check `git status` first;
`data/locked/locked_human.jsonl.gz` and `data/corpus/human/wi_locness*` must stay untracked: W&I+LOCNESS is non-commercial).

## 4. Key files

| File | Purpose |
|---|---|
| `scripts/eval_frontier.py` | the only evaluator (dev threshold at 1% FPR, Wilson CIs, slices, locked access guard) |
| `scripts/train_detector.py` | CPU trainer; flags `--no-attacks --hold-out-family --hold-out-gen --attack-weight --epochs` |
| `scripts/detectors/baselines.py` | detector registry: `shipped`, `frontier_onnx`, `cand:<name>`, open-source baselines |
| `scripts/log_experiment.py` | result JSON → `data/eval/experiments.csv` |
| `scripts/write_frontier_config.py` | dev result → `models/frontier/frontier_config.json` (refuses locked) |
| `scripts/export_onnx.py` | `--hf-dir/--out-dir` INT8 export |
| `scripts/benchmark_target.py` | runtime envelope check, `--mode frontier` |
| `scripts/onboard_model.py` | new-model check, fails closed |
| `scripts/check_integrity.py` | gates G1-G7 |
| `backend/runtime_engine.py` | engine; frontier mode behind `VERITAS_DETECTOR` |
| `data/research/` | sources (17), detector teardown, attack catalogue, ranked hypotheses: done |
| `scratch/PROGRESS.md`, `scratch/GOAL_BRIEF.md` | Claude's resume notes and the full goal playbook |

## 5. Known issues / open items

- Train has only ~30 AI student essays, so genre balancing keeps only ~88 ESL human essays in train. ESL FPR is the
  biggest risk for T1. Fixing it needs more AI essay generation (Claude subagents); not approved by the user yet.
- Dev/train attack data is partial (train A1 60, A3 14, …); six interrupted attack-generation agents were not relaunched.
- `frontend/tests/smoke.spec` is an exact duplicate of `smoke.spec.js` (Antigravity's area; harmless).
- Phase 4 black-box probing of commercial detectors: no API keys → mark as pending in the report (hand-collection CSV possible).
