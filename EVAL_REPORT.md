# Veritas AI — Comprehensive Evaluation & Forensic Verification Report

> [!WARNING]
> **Legacy metrics (measured on synthetic, leaky data; teacher figures never measured) — superseded; see `data/reports/FRONTIER_DETECTION_REPORT.md` when it exists.**

**Project**: QuillBot-Style Offline AI Writing Detector  
**Model Architecture**: Student Knowledge Distillation (MiniLM-L6-v2, 22.7M parameters) + 20-Dimensional Stylometric Meta-Classifier  
**Export Format**: Standalone ONNX with Dynamic INT8 Quantization  
**Runtime**: ONNX Runtime CPU (Zero PyTorch at Runtime)  
**Target Environment Constraint**: Low-End Target Hardware (4GB System RAM, ≤1.5GB Process RAM, 2-core CPU, ≤500MB Disk, ≤15s latency per 500 words)

---

## 1. Executive Summary & Verification Scorecard

> [!WARNING]
> **Retraction Notice**: An audit (`scripts/legacy_data_audit.py` -> `data/eval/legacy_audit.json`) revealed that legacy metrics below were measured on synthetic, leaky data (exact training text duplicates in test splits: 20/80 indist, 10/50 paraphrased, 16/30 ESL). Furthermore, teacher figures (`96.2% TPR`) were hard-coded in evaluation scripts and never executed. All metrics in this table are retracted and superseded. Honest frontier evaluation benchmarks will be published in `data/reports/FRONTIER_DETECTION_REPORT.md`.

| Evaluation Metric | Target Threshold | Teacher Ensemble (7B + DeBERTa) | Shipped Student (Local INT8 ONNX) | Status | Empirical Note |
|---|---|---|---|---|---|
| **In-Distribution TPR (@ ≤1% FPR)** | Teacher ≥95% | ~~96.20%~~ | ~~85.00%~~ | **RETRACTED** | Teacher never trained; 20/80 test leakage |
| **Teacher-vs-Student TPR Gap** | ≤ 5.0% Gap | — | ~~11.20%~~ | **RETRACTED** | Unmeasured teacher baseline |
| **Unseen Model: Qwen 2.5-72B** | Honest TPR (@ 1% FPR) | ~~94.5%~~ | ~~80.0%~~ (12/15 detected) | **RETRACTED** | Synthetic template data |
| **Unseen Model: DeepSeek-V3** | Honest TPR (@ 1% FPR) | ~~93.8%~~ | ~~73.3%~~ (11/15 detected) | **RETRACTED** | Synthetic template data |
| **Paraphrased / Humanized AI Text** | Honest Detection Rate | ~~88.2%~~ | ~~80.0%~~ (40/50 detected) | **RETRACTED** | Regex word-swaps, not true paraphrasing; 10/50 leaked |
| **ESL Human False-Positive Rate** | ≤ 2.0× Native FPR | ~~0.8%~~ | ~~0.00%~~ | **RETRACTED** | 16/30 leaked copies; synthetic data |
| **4-Class Macro-F1 Score** | Balanced 4-Class F1 | ~~0.958~~ | ~~0.6392~~ | **RETRACTED** | Measured on leaked synthetic test split |
| **Expected Calibration Error (ECE)** | ECE < 0.05 | ~~0.021~~ | ~~0.0306~~ | **RETRACTED** | Contradicted across reports (0.0306/0.0351/0.038) |
| **500-Word Latency (Simulated 2-Thread)** | ≤ 15.0 seconds | N/A | ~~0.167 seconds~~ | **UNVERIFIED** | Contradicts README (~0.34s) |
| **Peak Process RAM (RSS)** | ≤ 1,500 MB (1.5GB) | > 16 GB | ~~178.5 MB~~ | **UNVERIFIED** | Synthetic benchmark run |
| **Disk Footprint (Models + App)** | ≤ 500 MB | > 30 GB | **196.10 MB** (21.96 MB INT8 model) | **VALIDATED** | File size on disk |

---

## 2. 4-Class Discrimination Performance & Confusion Matrix

> [!WARNING]
> **Retracted**: The 80-sample test split suffered from 25% exact train-set leakage (20/80 duplicates). Metrics derived from this split are invalid and superseded.

The detector classifies text into QuillBot's 4 distinct forensic categories:
1. `Human-written`
2. `Human-written & AI-refined`
3. `AI-generated & AI-refined`
4. `AI-generated`
*(Plus an `Uncertain` designation when top probability falls below the confidence threshold).*

### 4×4 Confusion Matrix on Held-Out Test Split (80 Samples — Retracted)

| True Class \ Predicted Class | Human-written | Human-written & AI-refined | AI-generated & AI-refined | AI-generated | Per-Class F1 (Legacy) |
|---|---|---|---|---|---|
| **Human-written** | ~~13~~ | 5 | 1 | 1 | ~~0.703~~ |
| **Human-written & AI-refined** | 4 | ~~15~~ | 1 | 0 | ~~0.750~~ |
| **AI-generated & AI-refined** | 0 | 0 | ~~7~~ | 13 | ~~0.438~~ |
| **AI-generated** | 0 | 0 | 3 | ~~17~~ | ~~0.667~~ |
| **Overall Macro-F1** | — | — | — | — | ~~0.6392~~ *(Retracted)* |

---

## 3. Generalization on Unseen Generator Models (Leave-One-Model-Out)

> [!WARNING]
> **Retracted**: Evaluation was performed on synthetic 4-text template data rather than genuine API generations from Qwen-2.5-72B and DeepSeek-V3.

- **Qwen-2.5-72B**: ~~80.0% TPR (12/15 detected)~~ (*Retracted / synthetic template data*).
- **DeepSeek-V3**: ~~73.3% TPR (11/15 detected)~~ (*Retracted / synthetic template data*).

---

## 4. Adversarial & Paraphrased Robustness

> [!WARNING]
> **Retracted**: The "paraphrasing" attack evaluated here consisted of ~20 regex word swaps (median 5-gram Jaccard to source = 1.0) with 10/50 samples duplicated from training data. Genuine adversarial paraphrasing (e.g. T5, QuillBot, DIPPER) was not measured.

- **Paraphrased AI Detection Rate**: ~~86.0% (43/50)~~ / ~~80.0% (40/50)~~ (*Retracted / regex substitution on leaked data*).

---

## 5. Non-Native English (ESL) Fairness Audit

> [!WARNING]
> **Retracted**: 16 of the 30 ESL test samples were exact duplicates of the training set (`test_esl` leakage: 16/30). The reported 0.00% false-positive rate is an artifact of data leakage.

- **Native-Writer False-Positive Rate**: ~~0.00% (0 / 20)~~ (*Retracted*)
- **ESL / Non-Native False-Positive Rate**: ~~0.00% (0 / 30)~~ (*Retracted*)
- **ESL / Native Disparity Ratio**: ~~1.00×~~ (*Retracted*)

---

## 6. Calibration: Expected Calibration Error (ECE)

> [!WARNING]
> **Retracted**: ECE figures (reported variously as 0.0306, 0.0351, and 0.038) were computed on the leaky synthetic test set and are not statistically reliable.

- Calibration method: Post-quantization temperature scaling ($T_{\text{cal}} = 1.55$) applied to meta-classifier logits.
- Measured ECE: ~~0.0306~~ (*Retracted*).

---

## 7. Target Hardware Simulation Telemetry

> [!NOTE]
> Hardware constraints (2 CPU threads, INT8 ONNX, memory limit) remain target design requirements. The latency and RAM measurements below were obtained on synthetic sample runs and should be treated as preliminary indications rather than verified production benchmarks.

| Benchmark Text Length | Sentence Count | Mean Latency (s) | Latency Std Dev (s) | Peak Process RAM (MB) | Low-End Compliance |
|---|---|---|---|---|---|
| **80 words** (Short) | 6 sentences | 0.030s | 0.005s | 96.5 MB | Preliminary |
| **500 words** (Standard) | 36 sentences | ~0.17s - 0.34s | 0.004s | 124.1 MB | Preliminary |
| **1,000 words** (Long) | 72 sentences | 0.326s | 0.002s | 155.9 MB | Preliminary |
| **2,000 words** (Max) | 143 sentences | 0.613s | 0.006s | 178.5 MB | Preliminary |

---

## 8. What Quantization Cost

> [!NOTE]
> Model file size on disk is verified (21.96 MB INT8 model). However, the accuracy and TPR comparison below relied on the synthetic, unmeasured teacher baseline.

| Metric | FP32 Base Model | INT8 Dynamic Quantization | Delta / Cost |
|---|---|---|---|
| **Model Size on Disk** | 86.77 MB | 21.96 MB | **-74.7% (-64.81 MB)** (Verified) |
| **In-Distribution TPR (@ 1% FPR)** | ~~85.0%~~ | ~~85.0%~~ | *Retracted (Synthetic split)* |
| **4-Class Macro-F1** | ~~0.641~~ | ~~0.639~~ | *Retracted (Synthetic split)* |
| **500-Word Latency (CPU)** | 0.395s | 0.167s | Preliminary |
| **Peak Process RAM** | 225 MB | 178.5 MB | Preliminary |

---

## 9. Architectural Breakthrough: Hierarchical Chunk Pooling

### Diagnosis of Multi-Paragraph AI Essay Attenuation
When analyzing long-form essays (>500 words) such as standard academic papers or ChatGPT expository essays, sequence classification heads trained on paragraph-scale text (~70-100 words) previously suffered from sequence distribution shift and token truncation at 512 tokens.

### Solution: Hierarchical Paragraph-Window Pooling
1. **Paragraph & Sentence Window Decomposition**: The runtime engine segments multi-paragraph texts into coherent ~70–100 word chunks.
2. **Micro-batched ONNX Inference**: Chunks are processed via batch ONNX inference using dynamic sequence padding.
3. **Logit Mean Pooling**: Logits across all chunks are mean-pooled (`np.mean(chunk_logits, axis=0, keepdims=True)`), aiming to capture autoregressive sequence predictability across the entire essay without truncation.
4. *(Legacy performance claims of >91.5% confidence, ECE 0.0306, and ESL 0.00% FPR were measured on leaked synthetic data and are retracted pending evaluation on the real frontier corpus).*
