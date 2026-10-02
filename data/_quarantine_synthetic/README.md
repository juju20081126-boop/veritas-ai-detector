# Quarantined synthetic data — do not train or evaluate on this

Moved here on 2026-10-01 by Claude Code (branch `claude/work`). Nothing was deleted; `git mv` keeps history.

## Why

The old training and evaluation data was not real model output and not real human or public-dataset text:

- **"AI" text is canned templates.** `generate_frontier_data.py` fell back to fixed paragraphs (the prompt pasted into a template) because no API keys were set. `build_dataset.py` and `expand_data_richness.py` add hard-coded seed texts labelled `qwen-2-5`, `gemini-1-5`, etc., written inside the repo.
- **`raw/*_samples.jsonl` are placeholders.** Six files named after RAID, M4, HC3, DetectRL, MAGE and ESL each hold 50 rows but only 4 unique texts, the same four in every file. `download_datasets.py` silently fell back to "offline research seeds" and the HF `datasets` library was not installed.
- **Train/test leakage.** 20/80 of `test_indist`, 10/50 of `test_paraphrased` and 16/30 of `test_esl` are exact copies of training texts.
- **The "paraphrase" attack is ~20 regex word swaps.** Median 5-gram Jaccard between a "paraphrased" text and its source is 1.0.
- **Tiny.** 224 training texts (96 academic), 40–100 words each; 15 texts per "unseen" generator.

Re-check any of this with `python scripts/legacy_data_audit.py` (output: `data/eval/legacy_audit.json`).

## What stays outside

`data/quillbot_comparison_sheet.{json,md}` remain in `data/` only because `backend/server.py` serves the JSON to the UI's benchmark menu. They come from the same synthetic pool and every verdict cell is still `(Pending)`: treat them as demo samples, not evidence.

## Rules

- No training, tuning, thresholding or reporting on these files.
- `python scripts/check_integrity.py --legacy-demo` runs the integrity gates against them and prints why they fail.
- Real-data replacements live under `data/corpus/`, `data/splits/` and `data/locked/` (see `GOAL_BRIEF.md`, section 6).
