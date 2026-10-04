# Demonstration & Reference Samples

> [!WARNING]
> **Status: Demo & Reference Samples Only**
>
> Passages in this directory are intended solely for local manual testing, CLI formatting checks, and UI walkthroughs:
> - **`samples/sample_data.py` (Synthetic Demo Texts)**: Hand-crafted demo passages (`SYNTHETIC_DEMO = True`) for quick UI smoke tests. Not genuine model outputs and not evaluation evidence.
> - **`samples/real_samples.py` (Frontier Generation Extracts)**: Exactly 20 real model generation passages (10 Claude Opus 5.5, 10 Claude Sonnet 5.5) extracted READ-ONLY from `data/corpus/frontier/` with full provenance metadata. Provided for offline sanity checks. Not benchmark evaluation data.
>
> Genuine empirical evaluation data, full benchmark corpora, and locked test splits are located strictly in `data/corpus/`, `data/splits/`, `data/eval/`, and `data/reports/`.
