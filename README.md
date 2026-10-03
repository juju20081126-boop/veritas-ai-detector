# 🛡️ Veritas AI — QuillBot-Style Offline AI Writing Detector

> [!WARNING]
> **Legacy metrics (measured on synthetic, leaky data; teacher figures never measured) — superseded; see `data/reports/FRONTIER_DETECTION_REPORT.md` when it exists.**

Veritas AI is an offline, production-grade 4-class AI writing detector architected to mirror the UX, forensic methodology, and four-class nuance of **QuillBot's AI Detector**. Engineered specifically for **low-end consumer hardware**, it operates without GPU acceleration or cloud connectivity, utilizing INT8 dynamic quantization and lightweight stylometric meta-classification.

---

## Table of Contents

- [Key Highlights & Capabilities](#key-highlights--capabilities)
- [How It Works: Current Engine Architecture](#how-it-works-current-engine-architecture)
  - [1. Quantized ONNX INT8 Neural Student](#1-quantized-onnx-int8-neural-student)
  - [2. 20 Tabular Stylometric Features](#2-20-tabular-stylometric-features)
  - [3. Meta-Classifier Fusion](#3-meta-classifier-fusion)
  - [4. Temperature Calibration & Confidence Gating](#4-temperature-calibration--confidence-gating)
  - [5. Hierarchical Chunking & Sentence-Level Smoothing](#5-hierarchical-chunking--sentence-level-smoothing)
  - [6. FastAPI Server Endpoints & Response Schema](#6-fastapi-server-endpoints--response-schema)
- [Quick Start Guide](#quick-start-guide)
  - [1. Installation on Low-End PC (Offline Ready)](#1-installation-on-low-end-pc-offline-ready)
  - [2. Launch Local Web UI](#2-launch-local-web-ui)
  - [3. Command-Line Interface (CLI)](#3-command-line-interface-cli)
- [Verification & Empirical Evaluation](#verification--empirical-evaluation)
- [Retraining & Data-Refresh Pipeline](#retraining--data-refresh-pipeline)
  - [How to Run the Real-Data Pipeline](#how-to-run-the-real-data-pipeline)
  - [Cloud Jupyter Notebooks (Kaggle / Google Colab)](#cloud-jupyter-notebooks-kaggle--google-colab)
- [State-of-the-Art Research & Mathematical Formulations (2024–2026)](#state-of-the-art-research--mathematical-formulations-20242026)
- [QuillBot Comparison Sheet (30 Hand-Check Samples)](#quillbot-comparison-sheet-30-hand-check-samples)
- [Hand-collecting detector verdicts (no automation, ToS-safe, <=200 texts)](#hand-collecting-detector-verdicts-no-automation-tos-safe-200-texts)
- [Limitations](#limitations)
- [License & Acknowledgments](#license--acknowledgments)

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

## 🔬 How It Works: Current Engine Architecture

The runtime engine in `backend/runtime_engine.py` and the server in `backend/server.py` implement an offline, low-resource detection architecture designed to run on modest consumer hardware (2 CPU threads, &le;150 MB RAM, zero PyTorch).

```
Raw Input Document (Pasted Text / Uploaded File)
  │
  ├─► Regex Sentence Splitter & ~100-Word Paragraph Chunking
  │
  ├─► 20-Dimensional Stylometric Feature Extraction (backend/stylometrics.py)
  │     (Burstiness CV, TTR, ARI, Syllable Variance, Entropy, Rhythm Δ, DEFLATE, Discourse Markers)
  │
  ├─► ONNX INT8 Student Model (models/student_model_int8.onnx via onnxruntime)
  │     (Rust tokenizers, max sequence length 512, 2 CPU threads)
  │
  ├─► Meta-Classifier Fusion (models/meta_classifier.json)
  │     (Z-score normalized stylometrics + neural logits fused via learned weights)
  │
  ├─► Temperature Calibration & Confidence Gating
  │     (Post-quantization scaling, uncertainty thresholding)
  │
  ├─► Hierarchical Sentence Smoothing (40% sentence + 60% chunk context)
  │
  ▼
QuillBot-Style 4-Class Breakdown, Word-Weighted AI %, and Sentence Highlights
```

### 1. Quantized ONNX INT8 Neural Student
- **Model Asset**: `models/student_model_int8.onnx` (~22 MB).
- **Runtime**: Evaluated via `onnxruntime.InferenceSession` with `intra_op_num_threads=2`. Zero PyTorch or CUDA dependencies are loaded at runtime.
- **Tokenizer**: Hugging Face Rust `tokenizers` loaded from `models/tokenizer/tokenizer.json`. Operates with a maximum sequence length of 512 tokens with truncation and padding.
- **Output**: 4-class raw neural logits corresponding to:
  1. `human` (0)
  2. `human_ai_refined` (1)
  3. `ai_ai_refined` (2)
  4. `ai_generated` (3)

### 2. 20 Tabular Stylometric Features
The engine extracts a 20-dimensional stylometric feature vector (`backend/stylometrics.py`) invariant to passage topic:
1. **Sentence length mean**: Mean words per sentence.
2. **Sentence length variance**: Spread of sentence lengths across the document.
3. **Sentence length CV ($\lambda_{\text{auth}}$)**: Coefficient of variation ($\sigma / \mu$), capturing human syntactic burstiness versus uniform LLM cadence.
4. **Type-Token Ratio (TTR)**: Unique tokens divided by total tokens (lexical diversity).
5. **Root TTR (Guiraud's Index)**: $V / \sqrt{N}$, mitigating length bias.
6. **Automated Readability Index (ARI)**: Character- and sentence-based structural readability level.
7. **Flesch Reading Ease**: Syllable- and sentence-based readability score.
8. **Mean syllables per word**: Average lexical complexity.
9. **Syllable count variance**: Polysyllabic distribution across words.
10. **Punctuation density**: Frequency of punctuation marks per 100 words.
11. **Comma frequency**: Average commas per sentence.
12. **Semicolon and colon frequency**: Density of complex clause delimiters.
13. **Question mark frequency**: Rhetorical question density.
14. **Shannon token entropy rate**: Information density per token position.
15. **Consecutive rhythm delta ($\Delta_{\text{rhythm}}$)**: Mean absolute difference between adjacent sentence lengths.
16. **DEFLATE compression ratio**: Algorithmic information complexity under Lempel-Ziv compression.
17. **AI discourse transition density**: Frequency of formulaic LLM discourse markers (*furthermore*, *delve*, *moreover*, *testament*, *in summary*).
18. **Personal voice marker frequency**: Frequency of first-person and experiential pronouns (*I*, *my*, *we*, *personally*).
19. **Long clause ratio**: Proportion of sentences with $\ge 35$ words.
20. **Short clause ratio**: Proportion of sentences with $\le 6$ words.

*Forensic Guardrail*: An authentic voice check protects literary human prose. When $\lambda_{\text{auth}} \ge 0.70$, personal voice markers are present, and AI transition markers are absent, the detector prevents false-positive escalation.

### 3. Meta-Classifier Fusion
- **Parameters**: `models/meta_classifier.json` stores `scaler_mean`, `scaler_std`, `meta_weights`, and `meta_intercept`.
- **Z-Score Normalization**: Each stylometric feature is standardized:
  $$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$
- **Linear Logit Fusion**: The 4-class neural logits from the ONNX student model are combined with the 20 normalized stylometric features:
  $$\mathbf{z}_{\text{fused}} = \mathbf{W}_{\text{neural}} \mathbf{z}_{\text{onnx}} + \mathbf{W}_{\text{style}} \tilde{\mathbf{x}}_{\text{style}} + \mathbf{b}$$

### 4. Temperature Calibration & Confidence Gating
- **Temperature Scaling**: Platt-style post-quantization calibration adjusts the fused logits:
  $$p_c = \frac{\exp(z_c / T)}{\sum_{k=1}^4 \exp(z_k / T)}$$
  where $T$ is the empirically fit `calibration_temperature` from `models/meta_classifier.json`.
- **Confidence Gating & Uncertain Verdict**:
  If the top calibrated probability is below the decision threshold (default $0.40$), or if the top probability is $< 0.45$ and the margin between the top two classes is $< 0.04$, the engine withholds judgment and assigns the verdict **Uncertain**.

### 5. Hierarchical Chunking & Sentence-Level Smoothing
- **Paragraph Chunking**: Texts exceeding typical sentence lengths are partitioned into hierarchical context chunks (~100 words) using sentence boundary preservation.
- **Context Blending**: To eliminate erratic classification flickering across short clauses, each sentence's prediction blends its local sentence-level logits ($40\%$) with the surrounding chunk context ($60\%$):
  $$\mathbf{p}_{\text{sentence}}^{\text{blended}} = 0.40 \cdot \mathbf{p}_{\text{local}} + 0.60 \cdot \mathbf{p}_{\text{chunk}}$$
- **QuillBot Headline & Percentage**:
  The document-level AI percentage is computed as the word-weighted coverage of sentences classified as `ai_generated` or `ai_ai_refined`:
  $$\text{AI Coverage \%} = \frac{\sum_{s \in \text{AI Sentences}} \text{Words}(s)}{\sum_{s \in \text{All Sentences}} \text{Words}(s)} \times 100$$
  This generates QuillBot-style headlines such as `"82% of text is likely AI"` or `"100% of text is likely Human"`.

### 6. FastAPI Server Endpoints & Response Schema
The backend server (`backend/server.py`) provides the following endpoints:

#### `POST /api/detect`
- **Request Body**:
  ```json
  {
    "text": "String of text to analyze (minimum 5 words)",
    "confidence_threshold": 0.40,
    "filename": "Pasted Text"
  }
  ```
- **Response JSON Keys**:
  - `summary`:
    - `verdict`: `"Human-written"`, `"Human-written & AI-refined"`, `"AI-generated & AI-refined"`, `"AI-generated"`, or `"Uncertain"`.
    - `verdict_description`: Textual explanation of the forensic classification.
    - `badge`: HTML badge label.
    - `is_uncertain`: Boolean indicating if judgment was withheld.
    - `confidence` / `confidence_pct`: Overall confidence score.
    - `quillbot_headline`: QuillBot headline string (e.g., `"78% of text is likely AI"`).
    - `quillbot_headline_class`: CSS badge class (`badge-ai`, `badge-ai-refined`, `badge-human`).
    - `quillbot_ai_pct` / `quillbot_human_pct`: Aggregate word-weighted percentage coverage.
    - `word_count`, `character_count`, `sentence_count`: Text statistics.
    - `length_warning`: Warning message if text has fewer than 80 words.
    - `elapsed_seconds`: Wall-clock analysis duration.
  - `calibrated_probabilities`: Dictionary mapping each of the 4 classes to its calibrated probability float.
  - `percentages`: Dictionary with percentage coverage breakdown for `ai_generated`, `ai_ai_refined`, `human_ai_refined`, and `human`.
  - `quillbot_breakdown`: Structured breakdown containing `headline`, `headline_class`, `ai_percentage`, `human_percentage`, and `segments`.
  - `sentences`: List of sentence analysis dictionaries:
    - `index`: 0-based sentence position.
    - `text`: Sentence string.
    - `class_label`: Assigned class.
    - `class_key`: Key identifier (`ai_generated`, `ai_ai_refined`, `human_ai_refined`, `human`).
    - `color_class` / `highlight_class`: CSS styling classes.
    - `confidence`: Confidence score.
    - `ai_likelihood_pct`: Combined AI probability percentage ($p_{\text{ai\_gen}} + p_{\text{ai\_ref}}$).
    - `probabilities`: 4-class calibrated probabilities for this sentence.
    - `reasons`: Forensic explanation strings (e.g., detected AI transitions, personal voice markers, clause lengths).
  - `stylometrics`: Extracted 20 tabular features and guardrail metrics.
  - `mathematical_equations`: Mathematical formulation calculations for display.

#### `POST /api/upload`
- Accepts multipart file upload (`.txt`, `.pdf`, `.docx`).
- Parses document using `backend/document_parser.py` and returns the detection JSON schema plus `metadata` (page count, author metadata, filename).

#### `GET /api/health`
- Returns system telemetry: `status`, `architecture`, `engine_runtime`, `device`, `cpu_threads`, `process_ram_mb`, `target_ram_cap_mb`, and `timestamp`.

#### `GET /api/samples`
- Returns pre-loaded 4-class benchmark archetype sample texts.

#### `GET /api/comparison-sheet`
- Returns the 30-sample side-by-side QuillBot comparison sheet from `data/quillbot_comparison_sheet.json`.

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
- [`notebooks/01_teacher_ensemble_and_labeling.ipynb`](notebooks/01_teacher_ensemble_and_labeling.ipynb): Teacher ensemble template (unrun; results in notebook were unmeasured placeholders).
- [`notebooks/02_student_distillation_and_onnx_export.ipynb`](notebooks/02_student_distillation_and_onnx_export.ipynb): Student distillation, stylometrics meta-classifier, INT8 ONNX export and calibration (unrun template; no results claimed).
- [`notebooks/03_gpu_finetune.ipynb`](notebooks/03_gpu_finetune.ipynb): Fine-tunes DeBERTa-v3-small on real training data and exports ONNX INT8 (unrun template; no results claimed).


---

## 📚 State-of-the-Art Research & Mathematical Formulations (2024–2026)

For an exhaustive technical compendium covering modern detection equations, zero-shot curvature metrics, 4-class forensic taxonomy, and dataset benchmarks, see **[RESEARCH_COMPENDIUM.md](RESEARCH_COMPENDIUM.md)**:

- **Binoculars Zero-Shot Cross-Ratio**: $\text{Score}(x) = \frac{\log \text{PPL}_{M_1}(x)}{\log \text{xPPL}_{M_1, M_2}(x)}$ (Hans et al., ICML 2024)
- **Fast-DetectGPT Conditional Probability Curvature**: $\tilde{d}(x) = \frac{\sum_t (\log p(x_t) + \mathcal{H}(p))}{\sqrt{\sum_t \text{Var}[\log p]}}$ (Bao et al., ICLR 2024)
- **RADAR Adversarial Paraphrase Invariance**: Minimax game formulation against automated evasion (Hu et al., NeurIPS 2024)
- **Length-Invariant Forensic Stylometrics**: Yule's Characteristic $K$, Shannon token entropy rate, consecutive syntactic rhythm delta ($\Delta_{\text{rhythm}}$), and DEFLATE compression complexity
- **14 Premier Benchmark Corpora**: Detailed catalog of RAID (ACL 2024), M4 (EACL 2024), HC3, DetectRL, MAGE, and ESL learner corpora (PELIC, TOEFL11, ICNALE)
- **Non-Native English (ESL) Fairness Audit**: Algorithmic mitigation techniques to prevent false-positive penalization of simple/clear vocabulary

---

## 📋 QuillBot Comparison Sheet (30 Hand-Check Samples)

In strict accordance with terms of service (no scraping or automated querying), exactly 30 representative passages across all 4 classes, native human, ESL human, and frontier models are provided for manual side-by-side inspection:
- JSON format: [`data/quillbot_comparison_sheet.json`](data/quillbot_comparison_sheet.json)
- Markdown table: [`data/quillbot_comparison_sheet.md`](data/quillbot_comparison_sheet.md)

---

## Hand-collecting detector verdicts (no automation, ToS-safe, <=200 texts)

To benchmark third-party commercial detectors safely and in compliance with Terms of Service:
1. Claude produces a numbered CSV containing test passages (`id`, `text`).
2. The user pastes each text into a detector's web page **BY HAND**.
3. The user records the verdict in a result column.

**Rules:**
- Maximum ~200 texts per detector to avoid abuse and maintain manual feasibility.
- Never automate a detector or paraphraser website, never scrape, and never send requests programmatically.

---

## Limitations

Every point below is a claim tagged [documented] in data/research/sources.md. The S-numbers refer to entries in that file.

- Non-native English writers can be wrongly flagged as AI. Liang et al. (Patterns 2023, S11) ran seven detectors on 91 TOEFL essays by non-native writers and 88 essays by US 8th-graders. On average, 61.22% of the TOEFL essays were labelled AI-written, against about 5.19% of the native essays. The authors link this to lower perplexity and less varied vocabulary. Those were 2023 detectors, but the result is the standing worst case.
- Short text is unreliable. Ghostbuster (S10) degrades substantially on text of 100 tokens or fewer. Fast-DetectGPT (S7) reports that accuracy rises steadily with passage length. Pangram (S1) states that shorter text is harder: in its humanizer test, texts under 50 words reached 73.32% TPR at 1% FPR, compared with 100% at full length.
- Paraphrased and humanized text is the hardest case. PADBen (S3) finds that iterative paraphrasing is the hardest attack, and that detectors break on the intermediate paraphrase steps. In DAMAGE (S16), detection of humanized text fell to 60.04% for GPTZero and 28.23% for Binoculars (TPR at 5% FPR). DIPPER paraphrasing (S9) cut DetectGPT from 70.3% to 4.6% TPR at 1% FPR.
- Vendor accuracy figures are self-reported. For example, Pangram's figures (S1, S13) come from Pangram's own technical reports. The independent RAID benchmark (S14) found that detectors are biased toward the domains and models they were trained on, and are "not yet robust enough for high-stakes use".
- A score is a signal, not proof of authorship. Pangram (S1) says its detector is statistical and that the same text can score differently in different contexts. The Pangram authors also advise against using a detector as the only basis for a decision (S13). Binoculars (S6) flagged famous memorised texts, such as the US Constitution, as machine-generated.

---

## 📜 License & Acknowledgments

- Public research datasets tracked under individual open licenses: RAID (CC-BY 4.0), M4 (Apache-2.0), HC3 (CC-BY-SA 4.0), DetectRL (Apache-2.0). See [`data/public_licenses.md`](data/public_licenses.md).
- Veritas AI is released under the **MIT License**.

