# Ranked hypotheses (updated after Phase 1 reading; measured results go to data/eval/experiments.csv)

Ranking = expected gain on paraphrased frontier text x probability it works / cost on CPU. "Evidence" cites data/research/sources.md.

| Rank | Id | Hypothesis | Why (evidence) | Expected gain | Cost | Test |
|---|---|---|---|---|---|---|
| 1 | H1 | Real, diverse, domain-balanced data (>= 20k texts, many generators incl. 2026 frontier) fixes most of the gap | The old data was synthetic (legacy audit); Ghostbuster S10 shows supervised encoders over-fit domains; Base-model paper S5 shows distribution shift dominates | very high | low (data already built) | retrain MiniLM student on the new train split; compare with the shipped model on dev |
| 2 | H2 | Train on attacked + intermediate paraphrase states (RADAR-style, offline), oversample them, hold out one attack family at a time | RADAR S8 (+31.6% AUROC under unseen paraphraser), DAMAGE S16 (0.68% humanizer data oversampled 18x -> 98.26% TPR), PADBen S3 (intermediate states) | high | low-medium | leave-one-attack-out on A1/A2/A3/A4/A5 |
| 3 | H9 | Input hygiene: NFKC, zero-width/homoglyph folding, curly->ASCII quotes, flatten newlines, strip markdown | RAID S14 and S17: character attacks are the cheapest wins for attackers; our own data audit found tokenization/paragraph artifacts that leak labels | high on A7, removes shortcuts | very low | A7 TPR before/after; confound probe (G7) |
| 4 | H8 | Hard-negative mining + synthetic mirrors | Pangram S13 (FPR -100x..-1000x) and S1 (active learning) | high for FPR (T1) | medium | mine FPs on a large held-out human pool, add AI mirrors from matched prompts, retrain |
| 5 | H5 | Model-agnostic discourse/structure features, data-driven lexicon replacing `backend/cliches.py` | S4: text-feature model lost only 0.0526 F1 under paraphrase vs 0.196 for Binoculars | medium | low | log-odds features per generator; ablate; check survival under A1-A5 |
| 6 | H3 | Stronger backbone within the envelope (DeBERTa-v3-small / MiniLM-L12 INT8) + longer windows | Pangram uses a much larger backbone; our latency has ~90x slack | medium-high | medium (CPU training time) | compare 3 backbones on dev |
| 7 | H6 | Sentence/window-level learning with learned pooling, length-bucket thresholds, abstention below 80 words | Pangram S1 (window aggregation, short text weak), Ghostbuster S10 (<=100 tokens) | medium | low | pooled vs learned aggregation per length bucket |
| 8 | H7 | Per-length-bucket thresholds at 1% FPR on dev human; conformal-style abstention | RAID S14 protocol (fixed FPR); Pangram reports by length | medium for T1 | low | realized FPR on locked (single use) |
| 9 | H10 | Calibrated ensemble of encoder logits + discourse features (+ LM features if they help) | S4: ensembles with Binoculars win pre-attack but lose most under attack | medium | low | meta-classifier on dev; check complementary errors |
| 10 | H4 | Small-LM probability features (Binoculars/Fast-DetectGPT style with a ~0.5B pair) | S6/S7 strong on raw text; S4/S16 collapse under paraphrase/humanizer | low-medium (raw only) | medium (CPU LM scoring) | adopt only if it adds on A1-A5 |
| 11 | H12 | Binary-first vs 4-class multi-task | QuillBot-style taxonomy needs real "AI-refined" data (ARB S2) | unknown | low | multi-task vs binary head |
| 12 | H11 | Retrieval/semantic fingerprint vs a local bank of our own generations | S9 (needs provider-side bank) | low | low | skip unless time remains |
| 13 | H13 (new) | "Base-model completion" slice (A10) in training and dev | S5 | medium for robustness to new evasion routes | medium (CPU generation) | dev slice only first |
| 14 | H14 (new) | Mixed-authorship handling: sentence-level labels from interleaved A6 texts | Pangram S1 (token-wise heads, interleaving results), ARB S2 | medium for the "AI-refined" classes | medium | A6 slice by ai_share |
