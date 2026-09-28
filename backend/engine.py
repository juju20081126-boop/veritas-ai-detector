"""
Ensemble AI Writing Detection Engine
Combines:
1. Deep Transformer Neural Sequence Classification (RoBERTa Academic AI Detector)
2. Causal Language Model Perplexity (GPT-2 Cross-Entropy Loss)
3. Token Predictability & Rank Spectrum Analysis (GLTR / Binoculars methodology)
4. Perplexity Burstiness & Syntactic Variance (Coefficient of Variation)
5. Stylometric Forensics (Lexical Diversity & Readability)
6. Lexical Hallmarks & AI Cliché Density
"""

import time
import re
import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

from backend.cliches import detect_cliches_in_sentence, analyze_document_cliches
from backend.stylometrics import analyze_stylometrics
from backend.document_parser import split_sentences


class AIDetectorEngine:
    _instance: Optional["AIDetectorEngine"] = None

    def __init__(self):
        print("[Engine] Initializing Veritas AI Detection Engine...")
        torch.set_num_threads(4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Engine] Utilizing compute device: {self.device}")

        # 1. Load Causal Language Model (GPT-2) for Perplexity & Token Log-Probs
        print("[Engine] Loading GPT-2 causal language model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.lm_model.eval()

        # 2. Load Neural Discriminator (RoBERTa Academic AI Detector)
        print("[Engine] Loading RoBERTa academic AI detector...")
        self.classifier = pipeline(
            "text-classification",
            model="andreas122001/roberta-academic-detector",
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512
        )
        print("[Engine] Veritas AI Detection Engine is fully armed and ready.")

    @classmethod
    def get_instance(cls) -> "AIDetectorEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_sentence_perplexity_and_ranks(self, sentence: str) -> Dict[str, Any]:
        """
        Computes token-level perplexity and token rank spectrum under GPT-2.
        """
        text = sentence.strip()
        if not text:
            return {
                "perplexity": 60.0,
                "token_count": 0,
                "top10_ratio": 0.5,
                "top100_ratio": 0.8,
                "ranks": []
            }

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        num_tokens = input_ids.shape[1]

        if num_tokens < 2:
            return {
                "perplexity": 50.0,
                "token_count": num_tokens,
                "top10_ratio": 0.5,
                "top100_ratio": 0.8,
                "ranks": []
            }

        with torch.no_grad():
            outputs = self.lm_model(input_ids)
            logits = outputs.logits

        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits, shift_labels).cpu().numpy()
        mean_loss = float(np.mean(token_losses))
        sentence_ppl = float(np.exp(mean_loss))

        # Token rank analysis
        token_ranks = []
        top10_count = 0
        top100_count = 0
        top1000_count = 0

        for t_idx in range(shift_logits.shape[0]):
            target_token_id = shift_labels[t_idx].item()
            token_str = self.tokenizer.decode([target_token_id])
            pos_logits = shift_logits[t_idx]
            target_logit = pos_logits[target_token_id].item()
            rank = int((pos_logits > target_logit).sum().item()) + 1

            if rank <= 10:
                top10_count += 1
                category = "top10"
            elif rank <= 100:
                top100_count += 1
                category = "top100"
            elif rank <= 1000:
                top1000_count += 1
                category = "top1000"
            else:
                category = "beyond1000"

            token_ranks.append({
                "token": token_str,
                "rank": rank,
                "loss": float(token_losses[t_idx]),
                "category": category
            })

        n_eval = max(1, len(token_ranks))
        top10_ratio = top10_count / n_eval
        top100_ratio = (top10_count + top100_count) / n_eval

        return {
            "perplexity": round(sentence_ppl, 2),
            "token_count": num_tokens,
            "top10_ratio": round(top10_ratio, 3),
            "top100_ratio": round(top100_ratio, 3),
            "ranks": token_ranks
        }

    def compute_neural_classifier_batch(self, sentences: List[str]) -> List[float]:
        """
        Passes sentences through the RoBERTa academic AI detector.
        Returns neural AI probability in [0.0, 1.0].
        """
        if not sentences:
            return []

        cleaned = [s if len(s.strip()) > 3 else "This is valid text." for s in sentences]

        try:
            results = self.classifier(cleaned, batch_size=16)
        except Exception as e:
            print(f"[Engine] Warning in classifier batch: {e}")
            return [0.5] * len(sentences)

        scores = []
        for r in results:
            label = r["label"]
            score = float(r["score"])
            # in andreas122001/roberta-academic-detector:
            # "machine-generated" = AI generated
            # "human-produced" = Human written
            if label == "machine-generated":
                ai_prob = score
            else:
                ai_prob = 1.0 - score
            scores.append(ai_prob)

        return scores

    def analyze_document(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive Turnitin-grade forensic audit of the input text.
        """
        start_time = time.time()
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty. Please provide content to analyze.")

        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)
        word_count = len(words)
        char_count = len(text)

        # 1. Batch Neural Classification
        neural_scores = self.compute_neural_classifier_batch(sentences)

        # 2. Per-sentence Language Modeling & Cliché Detection
        sentence_analyses = []
        ppl_values = []
        top10_ratios = []

        for idx, sent in enumerate(sentences):
            lm_info = self.compute_sentence_perplexity_and_ranks(sent)
            ppl = lm_info["perplexity"]
            ppl_values.append(ppl)
            top10_ratios.append(lm_info["top10_ratio"])

            cliche_info = detect_cliches_in_sentence(sent)
            neural_p = neural_scores[idx] if idx < len(neural_scores) else 0.5

            # Calibrated Perplexity score: Sigmoidal curve centered at PPL=45
            ppl_p = 1.0 / (1.0 + math.exp((ppl - 45.0) / 14.0))
            ppl_p = max(0.01, min(0.99, ppl_p))

            # Token predictability: High top-10 ratio indicates AI
            rank_p = max(0.0, min(1.0, (lm_info["top10_ratio"] - 0.40) / 0.35))

            # Cliché marker component
            cliche_p = cliche_info["score"]

            # Ensemble sentence probability
            combined_p = (
                0.60 * neural_p +
                0.20 * ppl_p +
                0.10 * rank_p +
                0.10 * cliche_p
            )

            # If neural classifier is extremely confident (>0.95), reinforce it
            if neural_p > 0.95:
                combined_p = max(0.85, combined_p)
            elif neural_p < 0.05 and ppl > 60.0:
                combined_p = min(0.15, combined_p)

            combined_p = round(float(max(0.0, min(1.0, combined_p))), 3)

            # Categorization label
            if combined_p >= 0.75:
                category = "highly_likely_ai"
                category_label = "Highly Likely AI"
                color_class = "ai-high"
            elif combined_p >= 0.55:
                category = "likely_ai"
                category_label = "Likely AI"
                color_class = "ai-moderate"
            elif combined_p >= 0.35:
                category = "mixed"
                category_label = "Mixed / Paraphrased"
                color_class = "ai-mixed"
            else:
                category = "human"
                category_label = "Likely Human"
                color_class = "human-clear"

            reasons = []
            if neural_p >= 0.75:
                reasons.append("Neural syntactic patterns closely match generative AI transformer weights.")
            if ppl <= 28.0:
                reasons.append(f"Low perplexity ({ppl:.1f}), showing high algorithmic predictability.")
            elif ppl >= 70.0:
                reasons.append(f"High perplexity ({ppl:.1f}), characteristic of authentic human phrasing.")
            if lm_info["top10_ratio"] >= 0.65:
                reasons.append(f"{int(lm_info['top10_ratio']*100)}% of tokens reside in top-10 next-token probabilities.")
            if cliche_info["matches"]:
                cliche_words = [f"'{m['term']}'" for m in cliche_info["matches"][:3]]
                reasons.append(f"Contains characteristic LLM signposts/clichés: {', '.join(cliche_words)}.")

            if not reasons:
                reasons.append("Linguistic metrics fall within balanced baseline parameters.")

            sentence_analyses.append({
                "index": idx,
                "sentence": sent,
                "ai_probability": combined_p,
                "ai_percentage": round(combined_p * 100, 1),
                "category": category,
                "category_label": category_label,
                "color_class": color_class,
                "perplexity": ppl,
                "neural_score": round(neural_p, 3),
                "top10_ratio": lm_info["top10_ratio"],
                "cliches": cliche_info["matches"],
                "reasons": reasons
            })

        # 3. Burstiness & Distribution Statistics
        mean_ppl = float(np.mean(ppl_values)) if ppl_values else 40.0
        std_ppl = float(np.std(ppl_values)) if ppl_values else 0.0
        cv_ppl = (std_ppl / mean_ppl) if mean_ppl > 0 else 0.0
        burstiness_index = round(float(cv_ppl), 3)

        # 4. Stylometrics & Document Clichés
        doc_cliches = analyze_document_cliches(sentences)
        stylometrics = analyze_stylometrics(text, sentences)

        # 5. Document-Level Aggregate Calculation
        sentence_weights = [max(1, len(s["sentence"].split())) for s in sentence_analyses]
        total_weight = sum(sentence_weights)
        weighted_ai_base = sum(s["ai_probability"] * w for s, w in zip(sentence_analyses, sentence_weights)) / total_weight

        # Global modifiers
        burstiness_mod = 0.0
        if burstiness_index > 0.55:
            burstiness_mod = -0.08  # High burstiness human discount
        elif burstiness_index < 0.28:
            burstiness_mod = +0.06  # Machine uniformity penalty

        cliche_mod = min(0.08, doc_cliches["average_density_score"] * 0.15)

        final_ai_score = weighted_ai_base + burstiness_mod + cliche_mod
        final_ai_score = max(0.0, min(1.0, final_ai_score))
        final_ai_percentage = round(final_ai_score * 100, 1)

        # Verdict
        if final_ai_percentage >= 75.0:
            verdict = "Highly Likely AI-Generated"
            verdict_desc = "The document exhibits strong transformer structural patterns, unnaturally uniform sentence progression, and low overall perplexity consistent with generative AI language models (ChatGPT, Claude, Gemini)."
            verdict_badge = "badge-danger"
        elif final_ai_percentage >= 50.0:
            verdict = "Mixed AI and Human Composition"
            verdict_desc = "The text contains substantial segments characteristic of AI synthesis or automated paraphrasing, interspersed with human-authored passages."
            verdict_badge = "badge-warning"
        elif final_ai_percentage >= 25.0:
            verdict = "Mostly Human with Minor AI Assistance"
            verdict_desc = "The writing appears predominantly human-authored, with occasional formulaic phrasing or AI-assisted grammatical refinement."
            verdict_badge = "badge-info"
        else:
            verdict = "Authentic Human Writing"
            verdict_desc = "The text demonstrates high natural burstiness, authentic idiosyncratic vocabulary, diverse sentence rhythm, and high perplexity consistent with genuine human scholarship."
            verdict_badge = "badge-success"

        confidence_factor = min(0.99, 0.88 + (min(500, word_count) / 500) * 0.10)
        confidence_percentage = round(confidence_factor * 100, 1)

        counts = {
            "highly_likely_ai": sum(1 for s in sentence_analyses if s["category"] == "highly_likely_ai"),
            "likely_ai": sum(1 for s in sentence_analyses if s["category"] == "likely_ai"),
            "mixed": sum(1 for s in sentence_analyses if s["category"] == "mixed"),
            "human": sum(1 for s in sentence_analyses if s["category"] == "human"),
            "total_sentences": len(sentence_analyses)
        }

        elapsed_seconds = round(time.time() - start_time, 2)

        return {
            "summary": {
                "overall_ai_score": round(final_ai_score, 3),
                "overall_ai_percentage": final_ai_percentage,
                "human_percentage": round(100.0 - final_ai_percentage, 1),
                "verdict": verdict,
                "verdict_description": verdict_desc,
                "verdict_badge": verdict_badge,
                "confidence_percentage": confidence_percentage,
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": len(sentence_analyses),
                "reading_time_minutes": round(word_count / 200, 1),
                "filename": filename or "Pasted Document",
                "elapsed_seconds": elapsed_seconds
            },
            "metrics": {
                "average_perplexity": round(mean_ppl, 1),
                "perplexity_std": round(std_ppl, 1),
                "burstiness_index": burstiness_index,
                "burstiness_label": "High (Human)" if burstiness_index > 0.40 else "Low (AI Uniform)",
                "lexical_diversity": stylometrics["lexical_diversity"],
                "readability": stylometrics["readability"],
                "syntax_variance": stylometrics["syntax_variance"],
                "total_ai_markers": doc_cliches["total_markers"]
            },
            "counts": counts,
            "sentences": sentence_analyses,
            "doc_cliches": doc_cliches
        }
