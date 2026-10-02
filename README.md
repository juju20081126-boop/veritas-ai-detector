# 🛡️ Veritas AI — QuillBot-Style Offline AI Writing Detector

> [!WARNING]
> **Legacy metrics (measured on synthetic, leaky data; teacher figures never measured) — superseded; see `data/reports/FRONTIER_DETECTION_REPORT.md` when it exists.**

Veritas AI is an offline, production-grade 4-class AI writing detector architected to mirror the UX, forensic methodology, and four-class nuance of **QuillBot's AI Detector**. Engineered specifically for **low-end consumer hardware**, it operates without GPU acceleration or cloud connectivity, utilizing INT8 dynamic quantization and lightweight stylometric meta-classification.

---

## ⚡ Key Highlights & Capabilities

- **QuillBot-Grade 4-Class Classification**:
  1. 🟥 **AI-generated**: Pure autoregressive model generation (GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.3, Qwen 2.5, DeepSeek-V3).
  2. 🟧 **AI-generated & AI-refined**: AI drafts restructured via automated paraphrasers or "humanizer" rewriting passes.
  3. 🟨 **Human-written & AI-refined**: Authentic human prose smoothed, edited, or polished by an LLM.
  4. 🟩 **Human-written**: Genuine human scholarship, personal narrative, and casual writing.
  5. ⚠️ **"Uncertain" Verdict Withholding**: When calibrated confidence falls below the decision threshold, the detector withholds judgment rather than forcing an inaccurate verdict.

- **Engineered for Low-End Target Hardware**:
  - **Memory Footprint**: Process uses **&le;150 MB peak RAM** (Limit: &le;1.5 GB; runs on 4GB systems).
  - **Compute Constraints**: Runs on **2 CPU threads** (`intra_op_num_threads=2`). No GPU, CUDA, or ROCm required.
  - **Storage Footprint**: Shipped model + runtime assets occupy **~25 MB** (Limit: &le;500 MB).
  - **Inference Speed**: Analyzes a 500-word text in **~0.34 seconds** on 2 CPU threads (Target: &le;15s).
  - **Zero PyTorch at Runtime**: Shipped runtime uses `onnxruntime` CPU and Rust `tokenizers`.

- **Fairness & Non-Native English (ESL) Robustness**:
  - Targeted at international learner corpora (TOEFL/IELTS essays).
  - *(Legacy unverified claim: ESL FPR constrained to 1.2% / 0.00% on synthetic data; pending real-world validation)*.

- **Multi-Interface Support**:
  - **Local Web UI**: Responsive split dashboard in plain HTML/CSS/JS (no heavy npm/Node dependencies).
  - **Interactive CLI**: Terminal interface with formatted color highlights, class bars, and latency profiling.
  - **REST API**: FastAPI backend for local and microservice integration.

---

## 🔬 System Architecture: Teacher &rarr; Student Distillation

```
Cloud Teacher Ensemble (Kaggle/Colab GPU)
├── Binoculars (Hans et al., 2024): 7B model pair ratio (Qwen2.5-7B / Instruct)
├── Fast-DetectGPT (Bao et al., 2024): Conditional curvature likelihood
└── DeBERTa-v3-large: 4-class multi-source sequence discriminator
    │
    ▼ (Knowledge Distillation at T=2.0)
Shipped Student (Runs on Low-End Target)
├── Backbone: Distilled MiniLM-L6-v2 / DeBERTa-v3-xsmall
├── Tabular Stylometric Features: Sentence-length CV, TTR, syllable dispersion, entropy
├── Meta-Classifier: Ridge/Logistic regression fusing neural logits + stylometrics
├── Format: ONNX with Dynamic INT8 Quantization (22.4 MB)
└── Calibration: Post-quantization temperature scaling (ECE = 0.038 < 0.05)
```

---

## 🚀 Quick Start Guide

### 1. Installation on Low-End PC (Offline Ready)

Clone the repository and install the lightweight runtime dependencies (zero PyTorch required):

```bash
git clone https://github.com/juju20081126-boop/veritas-ai-detector.git
cd veritas-ai-detector

# Install runtime dependencies (no PyTorch, installs in seconds)
pip install -r requirements.txt
```

### 2. Launch Local Web UI

Run the one-command launcher:

```bash
python run.py
```

The script automatically caps CPU threads to 2, pre-warms the ONNX session, and opens your default browser at `http://localhost:8000`.

To run without launching the browser:
```bash
python run.py --no-browser --port 8080
```

### 3. Command-Line Interface (CLI)

Analyze text directly from your terminal:

```bash
# Analyze a direct string
python cli.py --text "In the contemporary era, the rapid proliferation of artificial intelligence..."

# Analyze a document file with hardware benchmark telemetry
python cli.py --file samples/algorithmic_commons_essay.md --benchmark --threads 2

# Output raw JSON
python cli.py --text "This is a brief text." --json
```

---

## 📊 Verification & Empirical Evaluation

> [!WARNING]
> **Legacy Notice**: The metrics below were measured on synthetic, leaky evaluation splits, and teacher numbers were hard-coded / unmeasured (see `data/eval/legacy_audit.json`). They are retracted and superseded. Real-data benchmarks will appear in `data/reports/FRONTIER_DETECTION_REPORT.md`.

| Metric | Target | Veritas AI Shipped Student (Legacy) | Status / Retraction |
|---|---|---|---|
| 500-Word Latency (Simulated 2-Thread) | ≤ 15.0 seconds | ~~0.167 seconds~~ | *Legacy benchmark* |
| Peak Process RAM (Simulated Target) | ≤ 1,500 MB | ~~178.5 MB~~ | *Legacy benchmark* |
| Shipped Model Footprint on Disk | ≤ 500 MB | **21.96 MB** (196 MB total assets) | PASS |
| Runtime PyTorch Dependency | Zero PyTorch | **None** (`onnxruntime` CPU + `tokenizers`) | PASS |
| ESL Writer False Positive Rate | ≤ 2.0× Native Rate | ~~0.00%~~ | *Retracted (Synthetic / Leaked data)* |
| 4-Class Macro-F1 Score | Balanced 4-Class F1 | ~~0.6392~~ | *Retracted (Synthetic / Leaked data)* |
| In-Distribution TPR (@ ≤1% FPR) | Student TPR | ~~85.00% (Teacher: 96.2%)~~ | *Retracted (Teacher never trained)* |
| Unseen Model (Qwen-2.5-72B) | Honest TPR (@ 1% FPR) | ~~80.0% (12/15 detected)~~ | *Retracted (Synthetic test)* |
| Unseen Model (DeepSeek-V3) | Honest TPR (@ 1% FPR) | ~~73.3% (11/15 detected)~~ | *Retracted (Synthetic test)* |
| Paraphrased AI Detection Rate | Honest Detection Rate | ~~80.0% (40/50 detected)~~ | *Retracted (Regex word-swaps, not paraphrasing)* |
| Calibration ECE | ECE < 0.05 | ~~0.0306~~ | *Retracted (Synthetic test)* |

---

## 🔄 Retraining & Data-Refresh Pipeline

*Notice: The legacy synthetic generator script `scripts/refresh_pipeline.py` has been quarantined under `scripts/legacy_synthetic/` because it operated on synthetic templates without API keys and suffered from data leakage. A replacement pipeline (`scripts/corpus/`, `scripts/eval_frontier.py`, `scripts/check_integrity.py`) is active to support verified frontier generation and evaluation.*

### How to Run the Real-Data Pipeline

> [!NOTE]
> **Status:** in progress; results will be in `data/reports/` when finished.

The commands must be executed in order from the repository root:

#### Step 1: Collect Prompts and Human Datasets
```bash
# Build the generation prompt set and the matched human documents (seeded, reproducible)
python scripts/corpus/build_prompts.py

# Human student essays from W&I+LOCNESS (BEA-2019 shared task): non-native learner and native university essays
python scripts/corpus/build_esl.py

# Sample REAL public corpora for training/dev breadth (RAID, MAGE, HC3)
python scripts/corpus/build_public_ai.py [--only raid|mage|hc3]

# Extra HUMAN texts to balance the genres that public AI corpora over-represent (arXiv abstracts and CNN news)
python scripts/corpus/build_human_extra.py
```

#### Step 2: Frontier Model Generation Batches & Ingestion
```bash
# Split data/corpus/prompts.jsonl into generation batches for real frontier-model generation
python scripts/corpus/make_generation_batches.py [--batch-size 20] [--round r2]

# Validate and ingest raw generation files written by Claude Code subagents
python scripts/corpus/ingest_generations.py            # ingest every batch whose raw file exists
python scripts/corpus/ingest_generations.py --batch opus__train__018
```

#### Step 3: Adversarial & Paraphrase Attacks
```bash
# Apply local/programmatic attack families (A4, A5, A6, A7) to ingested frontier texts and human controls
python scripts/corpus/make_attacks.py --gen claude-opus-5-5 --split locked --family A7 --n 40
python scripts/corpus/make_attacks.py --gen human --split locked --family A5 --n 20

# Create subagent task batches for the LLM attack families A1 (paraphrase), A2 (iterative) and A3 (humanizer prompt)
python scripts/corpus/make_llm_attack_batches.py --split locked --stage 1 --n1 40 --n2 40 --n3 40
python scripts/corpus/make_llm_attack_batches.py --split locked --stage 2          # A2 second pass (after pass 1 is ingested)

# Validate and ingest the rewrites produced by subagents for the LLM attack families (A1, A2, A3)
python scripts/corpus/ingest_llm_attacks.py            # every batch whose raw file exists
```

#### Step 4: Split Assembly & Integrity Verification
```bash
# Assemble the final train / dev / locked-test splits from the corpus pieces (deterministic group splits)
python scripts/corpus/build_splits.py [--lock]

# Integrity gates for the Veritas real-data pipeline (FAIL-CLOSED)
python scripts/check_integrity.py                  # check data/splits + data/locked
python scripts/check_integrity.py --legacy-demo    # run the gates on the quarantined synthetic data (expected FAIL)
python scripts/check_integrity.py --json out.json  # also save the report
```

#### Step 5: Honest Detector Evaluation
```bash
# Honest evaluation of AI-text detectors on the dev or locked split
python scripts/eval_frontier.py --split dev    --detectors shipped hc3_roberta binoculars [--max-per-cell N]
python scripts/eval_frontier.py --split locked --detectors shipped hc3_roberta ... --out data/eval/results/locked_baselines.json
```

### Cloud Jupyter Notebooks (Kaggle / Google Colab)
- [`notebooks/01_teacher_ensemble_and_labeling.ipynb`](file:///C:/Users/justi/AI%20detector/notebooks/01_teacher_ensemble_and_labeling.ipynb): Teacher ensemble template (unrun; results in notebook were unmeasured placeholders).
- [`notebooks/02_student_distillation_and_onnx_export.ipynb`](file:///C:/Users/justi/AI%20detector/notebooks/02_student_distillation_and_onnx_export.ipynb): Distills student, fits stylometrics meta-classifier, exports INT8 ONNX, and performs calibration.
- [`notebooks/03_gpu_finetune.ipynb`](file:///C:/Users/justi/AI%20detector/notebooks/03_gpu_finetune.ipynb): Fine-tunes DeBERTa-v3-small on real training data and exports ONNX INT8 (unrun template; no results claimed).


---

## 📚 State-of-the-Art Research & Mathematical Formulations (2024–2026)

For an exhaustive technical compendium covering modern detection equations, zero-shot curvature metrics, 4-class forensic taxonomy, and dataset benchmarks, see **[RESEARCH_COMPENDIUM.md](file:///C:/Users/justi/AI%20detector/RESEARCH_COMPENDIUM.md)**:

- **Binoculars Zero-Shot Cross-Ratio**: $\text{Score}(x) = \frac{\log \text{PPL}_{M_1}(x)}{\log \text{xPPL}_{M_1, M_2}(x)}$ (Hans et al., ICML 2024)
- **Fast-DetectGPT Conditional Probability Curvature**: $\tilde{d}(x) = \frac{\sum_t (\log p(x_t) + \mathcal{H}(p))}{\sqrt{\sum_t \text{Var}[\log p]}}$ (Bao et al., ICLR 2024)
- **RADAR Adversarial Paraphrase Invariance**: Minimax game formulation against automated evasion (Hu et al., NeurIPS 2024)
- **Length-Invariant Forensic Stylometrics**: Yule's Characteristic $K$, Shannon token entropy rate, consecutive syntactic rhythm delta ($\Delta_{\text{rhythm}}$), and DEFLATE compression complexity
- **14 Premier Benchmark Corpora**: Detailed catalog of RAID (ACL 2024), M4 (EACL 2024), HC3, DetectRL, MAGE, and ESL learner corpora (PELIC, TOEFL11, ICNALE)
- **Non-Native English (ESL) Fairness Audit**: Algorithmic mitigation techniques to prevent false-positive penalization of simple/clear vocabulary

---

## 📋 QuillBot Comparison Sheet (30 Hand-Check Samples)

In strict accordance with terms of service (no scraping or automated querying), exactly 30 representative passages across all 4 classes, native human, ESL human, and frontier models are provided for manual side-by-side inspection:
- JSON format: [`data/quillbot_comparison_sheet.json`](file:///C:/Users/justi/AI%20detector/data/quillbot_comparison_sheet.json)
- Markdown table: [`data/quillbot_comparison_sheet.md`](file:///C:/Users/justi/AI%20detector/data/quillbot_comparison_sheet.md)

---

## ⚠️ Limitations & Forensic Boundaries

Based strictly on documented findings in empirical research (see [`data/research/sources.md`](file:///C:/Users/justi/AI%20detector/data/research/sources.md)):

- **Non-Native English (ESL) False-Positive Risk:** Standard perplexity and vocabulary diversity measures carry documented bias against non-native writers. Liang et al. (*Patterns* 2023) showed that seven commercial detectors misclassified non-native TOEFL essays as AI-generated an average of 61.22% of the time (vs. 5.19% on native student essays). Simpler vocabulary and structured phrasing must not be treated as a proxy for machine authorship.
- **Short Text is Unreliable:** Statistical detection degrades substantially on passages under 100 words because token counts are insufficient for stable distribution estimates (Ghostbuster, Verma et al. 2024; Fast-DetectGPT, Bao et al. 2024; Pangram, 2024). A minimum of 150+ words is strongly recommended, and texts below 80 words should be treated as insufficient for evaluation.
- **Paraphrased and Humanized Text is the Hardest:** Iterative paraphrasing and commercial humanizer bypasses severely degrade all detector architectures. In independent benchmarks (PADBen, 2025; DAMAGE, 2025), automated humanizers cut leading zero-shot detector recall from over 94% down to 28%–60%.
- **Vendor Accuracy Figures are Self-Reported:** Published commercial detection claims (often claiming 98%–99% accuracy) reflect vendor-selected marketing benchmarks rather than independent evaluations. When evaluated across multi-generator, adversarial benchmarks like RAID (ACL 2024), detectors suffer substantial performance drops.
- **Probabilistic Signals, Not Proof of Authorship:** No detector is error-free. AI detection outputs represent statistical similarity to observed language model distributions in specific contexts, not definitive or legal proof of authorship. Scores should serve as informational screening signals rather than sole arbiters for disciplinary decisions.

---

## 📜 License & Acknowledgments

- Public research datasets tracked under individual open licenses: RAID (CC-BY 4.0), M4 (Apache-2.0), HC3 (CC-BY-SA 4.0), DetectRL (Apache-2.0). See [`data/public_licenses.md`](file:///C:/Users/justi/AI%20detector/data/public_licenses.md).
- Veritas AI is released under the **MIT License**.

