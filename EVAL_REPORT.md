# Veritas AI — Comprehensive Evaluation & Forensic Verification Report

> [!WARNING]
> **Legacy metrics (measured on synthetic, leaky data; teacher figures never measured) — superseded; see `data/reports/FRONTIER_DETECTION_REPORT.md` when it exists.**

**Project**: QuillBot-Style Offline AI Writing Detector  
**Model Architecture**: Student Knowledge Distillation + 20-Dimensional Stylometric Meta-Classifier  
**Export Format**: Standalone ONNX with Dynamic INT8 Quantization  
**Runtime**: ONNX Runtime CPU (Zero PyTorch at Runtime)  
**Target Environment Constraint**: Low-End Target Hardware (4GB System RAM, &le;1.5GB Process RAM, 2-core CPU, &le;500MB Disk, &le;15s latency per 500 words)

---

## Evaluation Protocol & Source of Truth

> [!IMPORTANT]
> **Source of Truth Notice**: Every number in this report must come directly from `python scripts/eval_frontier.py ...` output once the real-data evaluation pipeline runs. No metric may be estimated, hard-coded, or extrapolated. All empty table cells below are filled with `"TBD (from data/reports/FRONTIER_DETECTION_REPORT.md)"` until populated by real pipeline runs.

The evaluation protocol conforms to the strict criteria defined in `scripts/eval_frontier.py`:
- **Threshold Determination**: Decision thresholds are calibrated strictly as the 99th percentile of clean human scores on the **dev** split (targeting 1% FPR on dev; never tuned on the test or locked split).
- **Positives**: Rows labeled `ai` with origin `ai_raw` or `ai_attacked`.
- **Negatives**: Clean human control documents from registered pre-ChatGPT corpora. Attacked human controls are reported separately.
- **Hybrids**: Mixed authorship passages are sliced separately by AI share and never pooled into pure binary sets.
- **Statistical Confidence**: All TPR metrics report Wilson 95% confidence intervals (`[95% CI]`).

---

## 1. Overall Benchmark Performance (Dev & Locked Splits)

Evaluation commands:
```bash
python scripts/eval_frontier.py --split dev --detectors shipped hc3_roberta binoculars
python scripts/eval_frontier.py --split locked --detectors shipped hc3_roberta binoculars --out data/eval/results/locked_baselines.json
```

| Detector | Split | Threshold (1% Dev FPR) | TPR@1%FPR [95% CI] | Realized FPR | AUROC |
|---|---|---|---|---|---|
| **Veritas Shipped (Student INT8)** | `dev` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| **HC3 RoBERTa Baseline** | `dev` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| **Binoculars Zero-Shot** | `dev` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| **Veritas Shipped (Student INT8)** | `locked` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| **HC3 RoBERTa Baseline** | `locked` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| **Binoculars Zero-Shot** | `locked` | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |

---

## 2. Per Generator &times; Attack Family Breakdown

Disaggregated performance across frontier generator models and evasion attacks.

| Detector | Split | Generator | Attack Family | Threshold | TPR@1%FPR [95% CI] | Realized FPR | AUROC |
|---|---|---|---|---|---|---|---|
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A1 (LLM Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A2 (Iterative Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A3 (Humanizer Prompt) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A4 (RAID T5 Rewrite) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A5 (Zero-Width Insertion) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude 3.5 Sonnet | A7 (Typo Injection) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude Opus 5.5 | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude Opus 5.5 | A1 (LLM Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude Opus 5.5 | A2 (Iterative Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Claude Opus 5.5 | A3 (Humanizer Prompt) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | GPT-4o | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | GPT-4o | A1 (LLM Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | GPT-4o | A3 (Humanizer Prompt) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Gemini 1.5 Pro | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Llama 3.3 70B | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | DeepSeek-V3 | A0 (Raw AI) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |

---

## 3. Performance Per Length Bucket

Detection reliability across text length regimes.

| Detector | Split | Length Bucket | Threshold | TPR@1%FPR [95% CI] | Realized FPR | AUROC |
|---|---|---|---|---|---|---|
| Veritas Shipped | `locked` | &lt; 80 words | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | 80–150 words | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | 150–300 words | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | 300–600 words | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | &gt; 600 words | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |

---

## 4. ESL vs Native English Writer Fairness Audit

Evaluation of false-positive rate parity on human writing across native and non-native English learner corpora.

| Detector | Split | Cohort / Corpus | Threshold | TPR@1%FPR [95% CI] | Realized FPR | AUROC |
|---|---|---|---|---|---|---|
| Veritas Shipped | `locked` | Native Student (LOCNESS) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Native General (News/Abstracts) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | ESL Learner (W&I CEFR A) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | ESL Learner (W&I CEFR B) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | ESL Learner (W&I CEFR C) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | ESL Total (All Learner Bands) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | N/A (Negatives Only) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |

- **Realized ESL / Native FPR Disparity Ratio**: `TBD (from data/reports/FRONTIER_DETECTION_REPORT.md)` (Target: &le; 2.0&times;)

---

## 5. Leave-One-Family-Out & Generalization Performance

Stress-testing detector generalizability when entire generator model families or attack types are held out of training data.

| Detector | Split | Held-Out Family / Dimension | Threshold | TPR@1%FPR [95% CI] | Realized FPR | AUROC |
|---|---|---|---|---|---|---|
| Veritas Shipped | `locked` | Held-Out: Anthropic (Claude) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Held-Out: OpenAI (GPT-4o) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Held-Out: Open-Source (Llama / DeepSeek) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Held-Out Attack: A2 (Iterative Paraphrase) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |
| Veritas Shipped | `locked` | Held-Out Attack: A3 (Commercial Humanizer Prompt) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) |

---

## 6. Shipped Hardware Footprint & Operational Telemetry

Physical resource measurements and hardware runtime profiling on 2 CPU threads.

| Metric | Target Limit | Measured Value | Verification Status |
|---|---|---|---|
| **Shipped Model Footprint on Disk** | &le; 500 MB | **21.96 MB** (`models/student_model_int8.onnx`) | VERIFIED |
| **Runtime Dependencies** | Zero PyTorch | `onnxruntime` CPU + `tokenizers` | VERIFIED |
| **Inference Latency (500 Words, 2 Threads)** | &le; 15.0 s | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD |
| **Peak Process RAM (2 Threads)** | &le; 1,500 MB | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD |
| **Calibration Error (ECE)** | &lt; 0.05 | TBD (from data/reports/FRONTIER_DETECTION_REPORT.md) | TBD |
