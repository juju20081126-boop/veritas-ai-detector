# Quarantined legacy scripts — synthetic-data pipeline

Moved here on 2026-10-01 by Claude Code. Each file starts with a guard (`raise SystemExit`) so it cannot be run by accident.

| Script | Problem |
|---|---|
| `generate_frontier_data.py` | Falls back to canned template paragraphs (`synthesize_model_fingerprint`) when no API key is set. |
| `build_dataset.py`, `expand_data_richness.py`, `update_corpus.py` | Hard-coded seed texts labelled as specific models/humans; write `data/processed/*`. |
| `refine_data.py` | "Paraphrasing" is ~20 regex word swaps; median Jaccard to the source is 1.0. |
| `download_datasets.py` | Silently falls back to "offline research seeds" instead of failing. |
| `refresh_pipeline.py` | Chains the scripts above; accepts `--new_models` but never forwards it to the generator. |
| `evaluate_models.py` | Teacher metrics were hard-coded (`teacher_tpr_at_1fpr = 0.962`); now `nan`/not measured. Evaluates on leaky tiny sets. |
| `generate_notebooks.py` | Writes teacher notebooks that print hard-coded results (`Accuracy on val: 96.8% | Macro-F1: 0.962`). |

Replacements (real data, fail-closed): `scripts/check_integrity.py`, `scripts/eval_frontier.py` and the corpus/attack builders in `scripts/`.
Evidence: `python scripts/legacy_data_audit.py` -> `data/eval/legacy_audit.json`.
