# Veritas AI — Evaluation Report (2026 Frontier Benchmark)

> [!IMPORTANT]
> **Bottom line.** On a locked, held-out test of real text, Veritas rarely flags human writing (0.4% false-positive rate), but it also **misses almost all text written by Claude Opus 5.5 and Claude Sonnet 5.5**. It catches 3–5% of raw output and about 1% of paraphrased or "humanized" output. A "Human-written" verdict is therefore **not evidence that a text was written by a person**. Do not use Veritas as the sole basis for any academic-integrity or hiring decision.

**Source of every number below:** `data/eval/results/locked_final.json` (locked access #3) and the locked baseline table in
`data/reports/FRONTIER_DETECTION_REPORT.md` §3.2 (locked access #1). Full methodology, dev-split results and ablations are in
that report. The earlier synthetic-data metrics (85% TPR, 0.00% ESL FPR, 0.6392 macro-F1, 80% paraphrase detection, ECE 0.0306)
were measured on leaky, template-generated data and are **retracted**; see `data/eval/legacy_audit.json`.

---

## 1. Protocol

| Item | Setting |
|---|---|
| Test set | `data/locked/`: 2,416 rows. 1,281 clean human (877 native, 404 ESL learner), 60 attacked human, 39 human/AI hybrids, 348 raw AI, 647 attacked AI |
| AI generators | Claude Opus 5.5 and Claude Sonnet 5.5 (174 raw texts each). GPT-6 Astra is **untested** (no API access; not simulated) |
| Attacks | A1 LLM paraphrase, A2 two-pass paraphrase, A3 humanizer prompt, A4 T5 paraphraser, A5 back-translation (de/zh), A7 typos / homoglyphs / zero-width characters |
| Threshold | Fixed on the **dev** split at the 99th percentile of clean-human scores (1% dev FPR), then frozen. Never tuned on the locked test |
| Splits | Group-disjoint train / dev / locked (0 exact and 0 near duplicates across splits); 7/7 integrity gates pass (`scripts/check_integrity.py`) |
| Confidence intervals | Wilson 95%, shown as `[low–high]`; counts shown as `(hits/n)` |
| Locked budget | **Exhausted: 3 of 3 evaluations used.** Any new model must be judged on dev, or on a newly collected held-out set |

---

## 2. Target verdicts (deployed INT8 model unless noted)

| Target | Criterion | Locked result | Verdict |
|---|---|---|:---:|
| **T1 Fairness** | Clean-human FPR ≤ 1.5% **and** ESL FPR ≤ 2× native FPR | FPR **0.4%** [0.2–0.9] ✔. ESL **0.7%** (3/404) vs native **0.2%** (2/877) = **3.3×** ✘ (champion PyTorch model: 6.5×). The gap rests on 3 vs 1–2 false positives and is not statistically significant (Fisher p = 0.18 / 0.095) | **PARTIALLY MET** |
| **T2 Raw frontier detection** | Raw TPR ≥ 90% per model | Opus 5.5 **3.4%** [1.6–7.3] (6/174); Sonnet 5.5 **5.2%** [2.7–9.5] (9/174) | **NOT MET** |
| **T3 Adversarial robustness** | Attacked TPR ≥ 70% | Pooled attacked **1.1%** [0.5–2.2] (7/647); A3 humanizer **0.0%** (0/94) | **NOT MET** |
| **T4 Generalization (leave-one-family-out)** | Held-out family TPR ≥ 60% | Claude held out: Opus **0.6%**, Sonnet **1.7%**; A3 held out: **0.0%** (0/94); A4 held out: **1.2%** (1/80) | **NOT MET** |

**Overall: targets not met.** The frontier model is a modest improvement on the legacy detector (pooled TPR 2.2% vs 0.2%) and
is fair to native writers, but it does not detect 2026 frontier-model text at a 1% false-positive rate.

---

## 3. Detector comparison on the locked test

TPR = share of AI texts flagged; FPR = share of human texts wrongly flagged. Threshold = 1% FPR on dev.

| Detector | AUROC | Clean-human FPR | ESL FPR | Native FPR | Raw AI TPR | Attacked AI TPR | Pooled TPR |
|---|---|---|---|---|---|---|---|
| **`frontier_onnx`** (deployed INT8, default since v2.0.0) | 0.637 | 0.4% [0.2–0.9] | 0.7% [0.3–2.2] | 0.2% [0.1–0.8] | 4.3% [2.6–7.0] | 1.1% [0.5–2.2] | **2.2%** [1.5–3.3] |
| `cand:h1h2h9` (champion, PyTorch) | 0.645 | 0.3% [0.1–0.8] | 0.7% [0.3–2.2] | 0.1% [0.0–0.6] | 4.6% [2.8–7.3] | 1.5% [0.8–2.8] | **2.6%** [1.8–3.8] |
| `shipped` (legacy, `VERITAS_DETECTOR=shipped`) | 0.593 | 0.3% [0.1–0.8] | 0.2% [0.0–1.4] | 0.3% [0.1–1.0] | 0.0% [0.0–1.1] | 0.3% [0.1–1.1] | **0.2%** [0.1–0.7] |
| `hc3_roberta` (public baseline) | 0.629 | 1.0% [0.6–1.7] | 0.7% [0.3–2.2] | 1.1% [0.6–2.1] | 3.4% [2.0–5.9] | 1.7% [1.0–3.0] | **2.3%** [1.5–3.4] |
| `openai_roberta` (public baseline) | 0.578 | 0.5% [0.2–1.0] | 0.7% [0.3–2.2] | 0.3% [0.1–1.0] | 1.4% [0.6–3.3] | 2.2% [1.3–3.6] | **1.9%** [1.2–3.0] |
| `modernbert` (public baseline) | 0.843 | 0.0% [0.0–0.3] | 0.0% [0.0–0.9] | 0.0% [0.0–0.4] | 0.0% [0.0–1.1] | 0.0% [0.0–0.6] | **0.0%** [0.0–0.4] |

Notes:
- Since v2.0.0 the app runs the **frontier INT8 detector by default**. Set `VERITAS_DETECTOR=shipped` to use the legacy model.
- `modernbert` ranks texts best (AUROC 0.843), but its scores saturate on dev human text (1% FPR threshold = 1.0), so it flags nothing at this operating point.
- At a looser 5% dev FPR threshold, `frontier_onnx` catches 5.0% [3.8–6.6] of AI text while wrongly flagging 2.7% [2.0–3.8] of human text.

---

## 4. Per generator × attack (`frontier_onnx`)

| Attack | Claude Opus 5.5 | Claude Sonnet 5.5 |
|---|---|---|
| None (raw) | 3.4% (6/174) | 5.2% (9/174) |
| A1 LLM paraphrase | 0.0% (0/99) | 1.0% (1/99) |
| A2 two-pass paraphrase | 0.0% (0/71) | 2.3% (1/44) |
| A3 humanizer prompt | 0.0% (0/49) | 0.0% (0/45) |
| A4 T5 paraphraser | 0.0% (0/40) | 5.0% (2/40) |
| A5 back-translation | 0.0% (0/40) | 2.5% (1/40) |
| A7 typos / homoglyphs / zero-width | 2.5% (1/40) | 2.5% (1/40) |

Attacked human controls: 0/60 wrongly flagged (0.0% [0.0–6.0]).

---

## 5. Short-text sensitivity (`frontier_onnx`)

| Length (words) | AI texts flagged (TPR) | Human texts wrongly flagged (FPR) |
|---|---|---|
| 50–100 | 33.3% [18.6–52.2] (9/27) | **0.9%** [0.2–4.8] (1/114) |
| 100–250 | 1.7% [0.8–3.4] (7/414) | 0.5% [0.2–1.3] (4/774) |
| 250–600 | 1.2% [0.5–2.6] (6/507) | 0.0% [0.0–1.0] (0/385) |
| 600+ | 0.0% [0.0–7.6] (0/47) | 0.0% [0.0–32.4] (0/8) |

Short texts carry the highest false-positive rate, and the samples are too small for confident per-bucket claims (only 27 AI
texts and 114 human texts under 100 words). The higher TPR in the 50–100 bucket rests on 9 hits and should not be read as
better short-text detection. The UI now shows a short-text notice for any analysis under 150 words, and a warning under 80 words.

---

## 6. Fairness detail (`frontier_onnx`)

| Human writing group | False-positive rate |
|---|---|
| Native English | 0.2% [0.1–0.8] (2/877) |
| ESL, all levels | 0.7% [0.3–2.2] (3/404) |
| ESL CEFR A (beginner) | 0.0% [0.0–2.5] (0/151) |
| ESL CEFR B (intermediate) | 0.0% [0.0–2.4] (0/155) |
| ESL CEFR C (advanced) | 3.1% [1.0–8.6] (3/98) |
| Student essays (all) | 1.0% [0.4–2.5] (4/413) |

All three ESL false positives come from advanced (CEFR C) learners. Student essays have the highest false-positive rate of any
genre, and AI-written student essays were caught 0/139 times, so **classroom use is the weakest case for this detector**.

---

## 7. Generalization (leave-one-family-out)

Each row is a model retrained with one family removed from training, then scored on that family in the locked test.

| Held-out family | Model | TPR on held-out family | Pooled AUROC |
|---|---|---|---|
| All Claude 5.x generations | `cand:lofo_noclaude` | Opus 0.6% (1/174), Sonnet 1.7% (3/174) | 0.375 |
| A3 humanizer | `cand:lofo_noA3` | 0.0% (0/94) | 0.457 |
| A4 T5 paraphraser | `cand:lofo_noA4` | 1.2% (1/80) | 0.522 |
| Ablation: no attacked training data | `cand:abl_noatk` | Attacked 0.2% (1/647); ESL FPR rises to 2.5% | 0.445 |

---

## 8. Hardware envelope (`scripts/benchmark_target.py --mode frontier`, 2 CPU threads)

| Metric | Limit | Measured | Status |
|---|---|---|---|
| Model + assets on disk | ≤ 500 MB | 22.64 MB (`models/frontier/student_model_int8.onnx` 21.96 MB) | PASS |
| Peak process RAM | ≤ 1,500 MB | 177.3 MB | PASS |
| Latency, 500 words | ≤ 15 s | 0.229 s | PASS |
| Runtime dependencies | No PyTorch | `onnxruntime` CPU + `tokenizers` | PASS |

Source: `data/eval/results/benchmark_target_results_frontier.json`.
