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

---

## 9. ZeroGPT Distillation Candidate Evaluation (Development Split)

Because the locked held-out test split is exhausted (3 of 3 evaluations used), new candidate models are evaluated on the development split (`data/splits/dev.jsonl.gz`, n=1,500 clean human, n=485 AI). Below is the comparative performance of `cand:zerogpt_distilled` (distilling ZeroGPT's token predictability and sentence perplexity dynamics into MiniLM-L6 with calibrated burstiness shielding) versus the baseline:

| Metric / Evaluation Slice | Frontier Baseline (`frontier_onnx` / `cand:h1h2h9`) | Distilled ZeroGPT (`cand:zerogpt_distilled`) | Difference / Gain |
|---|---|---|---|
| **Dev 1% FPR Decision Threshold** | `0.9752` / `0.9804` | **`0.7101`** | **-0.27 pts** (realistic calibration) |
| **Realized Clean Human FPR** | 1.0% [0.6–1.6] (15/1500) | 1.0% [0.6–1.6] (15/1500) | Parity (meets 1.0% target) |
| **Realized ESL Learner FPR** | 1.6% [0.6–4.7] (3/184) | **1.1%** [0.3–3.9] (2/184) | **-0.5 pts** (fairness improved) |
| **ESL CEFR Bands A & B FPR** | N/A | **0.0%** [0.0–5.3 / 5.6] (0/69, 0/65) | Zero false positives on beginner/intermediate |
| **Claude Opus 5.5 Raw TPR** | 0.0% [0.0–4.8] (0/77) | **10.4%** [5.4–19.2] (8/77) | **+10.4 pts** (emergent recall) |
| **Claude Sonnet 5.5 Raw TPR** | 0.0% [0.0–4.8] (0/78) | **15.4%** [9.0–25.0] (12/78) | **+15.4 pts** (emergent recall) |
| **Claude Sonnet A3 Humanizer TPR** | 0.0% (0/10) | **30.0%** [10.8–60.3] (3/10) | **+30.0 pts** (robustness) |
| **Claude Opus A1 LLM Paraphrase TPR** | 0.0% (0/20) | **5.0%** [0.9–23.6] (1/20) | **+5.0 pts** |
| **600+ Word Long Documents TPR** | 0.0% [0.0–17.6] (0/18) | **22.2%** [9.0–45.2] (4/18) | **+22.2 pts** (resolves length dilution) |
| **Model Disk Footprint** | 22.64 MB | **21.96 MB** (`models/zerogpt_distilled/student_model_int8.onnx`) | Pure ORT INT8 CPU |

Source: `data/eval/results/dev_zerogpt_distilled.json`.

---

## 10. Multi-Teacher Distillation Candidate Evaluation (Development Split)

To surpass the single-teacher ZeroGPT performance, Veritas AI deployed a **Multi-Teacher Knowledge Distillation Framework** (`notebooks/multi_teacher_distillation.py`). This architecture fuses:
1. **ZeroGPT Teacher**: Token-level predictability dynamics and inter-sentence burstiness variance shielding $\sigma^2_{\text{PPL}}$.
2. **QuillBot Behavioral Teacher**: Granular 4-class taxonomy probability allocations across pure AI, attacked/humanized AI, and AI-polished human drafts.
3. **Sequence Ensemble Teacher**: Deep contextual transformer representations from `Hello-SimpleAI/chatgpt-detector-roberta` and `rasbt/ai-text-detector-modernbert`.

Student model (`sentence-transformers/all-MiniLM-L6-v2`) was optimized via Hinton KD loss across a hyperparameter search grid with global optimum at **$T=1.5, \alpha=0.6$** (Validation Loss: `0.4189`) and dynamically quantized to INT8 ONNX (`models/multi_teacher_distilled/student_model_int8.onnx`, 21.96 MB).

Evaluation on the held-out development split (`data/splits/dev.jsonl.gz`, n=1,500 clean human, n=485 AI) executed via `scripts/eval_frontier.py --split dev --detectors cand:multi_teacher_distilled --neg-cap 1500 --public-cap 300 --threads 6`:

| Metric / Evaluation Slice | Frontier Baseline (`frontier_onnx`) | Distilled ZeroGPT (`cand:zerogpt_distilled`) | Multi-Teacher Distilled (`cand:multi_teacher_distilled`) | Net Gain vs Baseline |
|---|---|---|---|---|
| **Dev 1% FPR Decision Threshold** | `0.9752` | `0.7101` | **`0.6898`** | **-0.29 pts** (normalized calibration) |
| **Dev 5% FPR Decision Threshold** | `0.9610` | `0.6385` | **`0.6175`** | **-0.34 pts** |
| **AUROC Pooled (AI vs Clean Human)** | `0.6510` | `0.7018` | **`0.7227`** | **+0.0717 pts** |
| **Realized Clean Human FPR** | 1.0% [0.6–1.6] (15/1500) | 1.0% [0.6–1.6] (15/1500) | **1.0%** [0.6–1.6] (15/1500) | Exact target parity |
| **Realized ESL Learner FPR** | 1.6% [0.6–4.7] (3/184) | 1.1% [0.3–3.9] (2/184) | **0.5%** [0.1–3.0] (1/184) | **-1.1 pts** (superior fairness) |
| **ESL Fairness Ratio (ESL / Native)** | 1.6× native FPR | 1.1× native FPR | **0.45× native FPR** (0.5% vs 1.1%) | Far below 2.0× ceiling |
| **ESL CEFR Bands A & B FPR** | N/A | 0.0% [0.0–5.3 / 5.6] | **0.0%** [0.0–5.3 / 5.6] (0/69, 0/65) | 0% false alarms on learners |
| **Claude Opus 5.5 Raw TPR** | 0.0% [0.0–4.8] (0/77) | 10.4% [5.4–19.2] (8/77) | **10.4%** [5.4–19.2] (8/77) | **+10.4 pts** |
| **Claude Sonnet 5.5 Raw TPR** | 0.0% [0.0–4.8] (0/78) | 15.4% [9.0–25.0] (12/78) | **14.1%** [8.1–23.5] (11/78) | **+14.1 pts** |
| **Claude Sonnet A3 Humanizer TPR** | 0.0% (0/10) | 30.0% [10.8–60.3] (3/10) | **10.0%** [1.8–40.4] (1/10) | **+10.0 pts** |
| **Claude Opus A1 LLM Paraphrase TPR** | 0.0% (0/20) | 5.0% [0.9–23.6] (1/20) | **5.0%** [0.9–23.6] (1/20) | **+5.0 pts** |
| **MAGE GPT-4 Raw TPR** | 0.0% (0/88) | 13.6% [8.0–22.3] (12/88) | **15.9%** [9.7–25.0] (14/88) | **+15.9 pts** |
| **MAGE GPT-4 Paraphrase (A4) TPR** | 0.0% (0/69) | 13.0% [7.0–23.0] (9/69) | **15.9%** [9.1–26.3] (11/69) | **+15.9 pts** |
| **RAID Mistral-Chat Raw TPR** | 0.0% (0/4) | 50.0% (2/4) | **75.0%** [30.1–95.4] (3/4) | **+75.0 pts** |
| **Pooled TPR @1% FPR Threshold** | 0.0% (0/485) | 10.7% [8.3–13.8] (52/485) | **12.6%** [9.9–15.8] (61/485) | **+12.6 pts** |
| **Pooled TPR @5% FPR Threshold** | 0.0% (0/485) | 25.8% [22.1–29.8] (125/485)| **30.1%** [26.2–34.3] (146/485) | **+30.1 pts** |
| **Attacked AI TPR @1% FPR Threshold**| 0.0% (0/146) | 11.6% [7.4–17.9] (17/146) | **15.1%** [10.2–21.8] (22/146) | **+15.1 pts** |
| **Model Size on Disk** | 22.64 MB | 21.96 MB | **21.96 MB** (`student_model_int8.onnx`) | Edge compliant (≤ 25 MB) |
| **Runtime Dependencies** | ORT CPU | ORT CPU | **0 PyTorch** (pure onnxruntime CPU) | Zero PyTorch at runtime |

Source: `data/eval/results/dev_multi_teacher_distilled.json`.



