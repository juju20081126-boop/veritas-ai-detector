# 🛡️ VERITAS AI — Institutional-Grade AI Writing & Originality Detector

**Veritas AI** is a state-of-the-art AI content and authenticity detection platform architected to match and exceed the forensic capabilities of enterprise academic integrity suites such as **Turnitin**, **GPTZero**, and **CopyLeaks**.

---

## ⚡ Key Highlights & Capabilities

- **Multi-Signal Ensemble Forensic Engine**:
  - **Neural Discriminator**: Fine-tuned RoBERTa transformer sequence classifier specifically calibrated for generative AI artifacts.
  - **Causal Perplexity Loss (GPT-2)**: Measures token-level cross-entropy loss and log-probability distributions.
  - **Token Predictability Spectrum (GLTR / Binoculars methodology)**: Measures token rank distribution across Top-10, Top-100, and Top-1000 probability spaces.
  - **Burstiness & Uniformity Analysis**: Quantifies sentence-to-sentence perplexity variance ($CV$) and sentence length standard deviation. Generative AI generates flat, monotonic text; authentic human writing exhibits high burstiness.
  - **Forensic Stylometrics**: Computes Type-Token Ratio (TTR), Root TTR (Guiraud's index), Hapax Legomena ratio, and academic readability metrics (Flesch-Kincaid Grade Level, Gunning Fog, Flesch Reading Ease).
  - **LLM Cliché & Hallmark Lexical Database**: Flags overused generative transitional markers and filler phrases (*e.g., "delve", "multifaceted", "rich tapestry", "stands as a testament to", "pivotal role", "in conclusion"*).

- **Turnitin-Grade Sentence-by-Sentence Heatmap**:
  - Color-coded sentence tagging:
    - 🟥 **High AI Likelihood** ($\ge 75\%$)
    - 🟧 **Likely AI** ($55\% - 74\%$)
    - 🟨 **Mixed / Paraphrased** ($35\% - 54\%$)
    - 🟩 **Human Written** ($< 35\%$)
  - Interactive **Sentence Inspector**: Click on any highlighted sentence to examine its individual perplexity, top-10 token predictability, neural score, and forensic rationale.

- **Multi-Format Ingestion**:
  - Native drag-and-drop parsing for **PDF** (`.pdf`), **Microsoft Word** (`.docx`), **Plain Text** (`.txt`), and **Markdown** (`.md`).

- **Official Turnitin-Style PDF Report Generation**:
  - Generates downloadable, publication-grade **Originality & Authenticity Inspection Reports** via ReportLab with digital verification IDs, master AI score gauges, forensic parameter tables, and annotated sentence matrices.

- **Preloaded Real-World Sample Library**:
  - Test pure ChatGPT 4o essays, Claude 3.5 Sonnet technical papers, authentic human archival history, personal narrative prose, and mixed student submissions with a single click.

---

## 🚀 Quick Start Guide

### 1. Launch the Application
Run the launcher from PowerShell or terminal in the project directory:

```powershell
python run.py
```

The script will automatically allocate an available port (default `http://localhost:8000`), pre-warm the neural models, and launch your default browser.

To start without automatically popping open the browser:
```powershell
python run.py --no-browser
```

---

## 🔬 System Architecture

```
AI detector/
├── backend/
│   ├── engine.py              # Ensemble AI detection pipeline (Neural + PPL + Burstiness + Ranks)
│   ├── cliches.py             # Curated database of LLM hallmarks, buzzwords, and transitions
│   ├── stylometrics.py        # Lexical diversity, readability indexes, syntactic variance
│   ├── document_parser.py     # PDF, DOCX, TXT parser with abbreviation-aware segmentation
│   ├── pdf_report.py          # ReportLab generator for Turnitin-style inspection PDF reports
│   └── server.py              # FastAPI application & REST API endpoints
├── frontend/
│   ├── index.html             # Turnitin-style split workspace dashboard
│   ├── style.css              # Glassmorphism, dark/light themes, animated circular gauges
│   └── app.js                 # Real-time state management, heatmap renderer, inspector logic
├── samples/
│   └── sample_data.py         # Realistic benchmark essays for instant demonstration
├── run.py                     # Single-command launcher
├── requirements.txt           # Dependency specifications
└── README.md                  # Documentation
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/detect` | `POST` | Analyzes text: `{"text": "...", "filename": "..."}` |
| `/api/upload` | `POST` | Uploads and analyzes PDF, DOCX, or TXT file (`multipart/form-data`) |
| `/api/report` | `POST` | Generates official Turnitin-style PDF audit report |
| `/api/samples` | `GET` | Returns preloaded sample benchmark texts |
| `/api/health` | `GET` | Engine status and compute device telemetry |
