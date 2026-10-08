"""
Veritas AI — Shipped Offline Student Runtime Engine (QuillBot Behavior)
Zero-PyTorch Runtime: Pure ONNX Runtime CPU + Rust Tokenizers + NumPy.

Target Hardware Envelope:
- Peak RAM: <= 1.5 GB
- CPU Threads: 2 intra-op threads
- Latency: <= 15s per 500 words
- Classes:
  1. AI-generated
  2. AI-generated & AI-refined
  3. Human-written & AI-refined
  4. Human-written
  + "Uncertain" when confidence falls below threshold.
"""

import os
import re
import math
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from backend.document_parser import split_sentences
from backend.textnorm import normalize_text, strip_markdown
from backend.stylometrics import (
    analyze_stylometrics,
    extract_stylometrics_feature_vector,
    AI_SINGLE_WORDS,
    AI_PHRASES,
    HUMAN_SINGLE_WORDS,
    HUMAN_PHRASES
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "student_model_int8.onnx")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer", "tokenizer.json")
META_CONFIG_PATH = os.path.join(MODELS_DIR, "meta_classifier.json")

# Frontier detector (trained on real Claude Opus/Sonnet 5.5 text and paraphrase attacks; see
# data/reports/FRONTIER_DETECTION_REPORT.md) is the default since v2.0.0. The legacy model is still available with
# VERITAS_DETECTOR=shipped or QuillBotDetectorEngine(mode="shipped"). The response schema is identical in both modes, plus a
# "detector" block. Frontier mode decides AI vs human with the dev-calibrated threshold stored in frontier_config.json.
# multi_teacher (opt-in, VERITAS_DETECTOR=multi_teacher) is Antigravity's multi-teacher distilled MiniLM student
# (notebooks/multi_teacher_distillation.py); it shares the frontier decision rule and reads runtime_config.json, whose threshold
# is the INT8 model's own dev clean-human 1%-FPR threshold (scripts/write_frontier_config.py --mode multi_teacher).
FRONTIER_DIR = os.path.join(MODELS_DIR, "frontier")
MULTI_TEACHER_DIR = os.path.join(MODELS_DIR, "multi_teacher_distilled")
STUDENT_MODES = {  # mode -> (model dir, runtime config file name)
    "frontier": (FRONTIER_DIR, "frontier_config.json"),
    "multi_teacher": (MULTI_TEACHER_DIR, "runtime_config.json"),
}
DETECTOR_MODES = ("shipped",) + tuple(STUDENT_MODES)

CLASS_NAMES = [
    "Human-written",
    "Human-written & AI-refined",
    "AI-generated & AI-refined",
    "AI-generated"
]

CLASS_KEYS = [
    "human",
    "human_ai_refined",
    "ai_ai_refined",
    "ai_generated"
]

# Color and style mapping mirroring QuillBot UI
CLASS_METADATA = {
    "Human-written": {
        "key": "human",
        "color_class": "badge-human",
        "highlight_class": "highlight-human",
        "description": "Text displays natural syntactic cadence, authentic idiosyncratic phrasing, and human burstiness.",
        "badge": "badge-success"
    },
    "Human-written & AI-refined": {
        "key": "human_ai_refined",
        "color_class": "badge-human-refined",
        "highlight_class": "highlight-human-refined",
        "description": "Original human text that has been smoothed, edited, or rephrased with AI assistance.",
        "badge": "badge-warning"
    },
    "AI-generated & AI-refined": {
        "key": "ai_ai_refined",
        "color_class": "badge-ai-refined",
        "highlight_class": "highlight-ai-refined",
        "description": "AI-generated text that has undergone subsequent paraphrasing or humanizer restructuring.",
        "badge": "badge-orange"
    },
    "AI-generated": {
        "key": "ai_generated",
        "color_class": "badge-ai",
        "highlight_class": "highlight-ai",
        "description": "Text exhibits direct machine-generation signatures, uniform token predictability, and canonical structures.",
        "badge": "badge-danger"
    },
    "Uncertain": {
        "key": "uncertain",
        "color_class": "badge-uncertain",
        "highlight_class": "highlight-neutral",
        "description": "Text displays mixed or borderline statistical signals below the decision threshold. Verdict withheld.",
        "badge": "badge-secondary"
    }
}


class QuillBotDetectorEngine:
    _instance: Optional["QuillBotDetectorEngine"] = None

    def __init__(self, threads: int = 2, mode: Optional[str] = None):
        self.mode = (mode or os.environ.get("VERITAS_DETECTOR", "frontier")).strip().lower()
        if self.mode not in DETECTOR_MODES:
            raise ValueError(f"Unknown detector mode {self.mode!r}; expected one of {DETECTOR_MODES}.")
        print(f"[RuntimeEngine] Initializing Offline QuillBot-behavior Student Engine (mode={self.mode}, threads={threads})...")

        # 1. Initialize ONNX Runtime Session with target hardware thread caps
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path, tokenizer_path = ONNX_MODEL_PATH, TOKENIZER_PATH
        self.student_config: Dict[str, Any] = {}
        if self.mode in STUDENT_MODES:
            model_dir, cfg_name = STUDENT_MODES[self.mode]
            onnx_path = os.path.join(model_dir, "student_model_int8.onnx")
            tokenizer_path = os.path.join(model_dir, "tokenizer", "tokenizer.json")
            cfg_path = os.path.join(model_dir, cfg_name)
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"{self.mode} detector config not found at {cfg_path}. Export the {self.mode} model first.")
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.student_config = json.load(f)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"INT8 ONNX model not found at {onnx_path}. "
                "Run 'python scripts/export_onnx.py' first."
            )

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        # 2. Initialize Fast Rust Tokenizer (Zero PyTorch)
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()

        # 3. Load Meta-classifier and Calibration Parameters
        self.meta_config = {}
        if os.path.exists(META_CONFIG_PATH):
            with open(META_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.meta_config = json.load(f)
        
        self.calibration_temperature = self.meta_config.get("calibration_temperature", 1.0)
        self.meta_weights = np.array(self.meta_config.get("meta_weights", [])) if "meta_weights" in self.meta_config else None
        self.meta_intercept = np.array(self.meta_config.get("meta_intercept", [])) if "meta_intercept" in self.meta_config else None
        self.feature_mean = np.array(self.meta_config.get("feature_mean", [])) if "feature_mean" in self.meta_config else None
        self.feature_std = np.array(self.meta_config.get("feature_std", [])) if "feature_std" in self.meta_config else None

        print("[RuntimeEngine] Shipped Student Engine successfully armed (Zero PyTorch, ONNX INT8).")

    @classmethod
    def get_instance(cls, threads: int = 2, mode: Optional[str] = None) -> "QuillBotDetectorEngine":
        if cls._instance is None:
            cls._instance = cls(threads=threads, mode=mode)
        return cls._instance

    @staticmethod
    def _frontier_prep(text: str) -> str:
        """Input hygiene identical to training (scripts/train_detector.py prep): NFKC, zero-width/homoglyph folding, no markdown, one line."""
        return " ".join(strip_markdown(normalize_text(text)).split())

    @staticmethod
    def _sentence_chunks(text: str, max_words: int = 100) -> List[str]:
        """Sentence-packed ~100-word chunks, identical to the training chunker."""
        out, cur, n = [], [], 0
        for s in split_sentences(text) or [text]:
            w = len(s.split())
            if cur and n + w > max_words:
                out.append(" ".join(cur))
                cur, n = [], 0
            cur.append(s)
            n += w
        if cur:
            out.append(" ".join(cur))
        return out or [text]

    def _extract_stylometrics_vec(self, text: str, sentences: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extracts 20-dimensional forensic and information-theoretic feature vector."""
        return extract_stylometrics_feature_vector(text, sentences)

    def _run_onnx_inference(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> np.ndarray:
        """Batch inference using ONNX Runtime CPU with dynamic sequence padding and micro-batching."""
        if not texts:
            return np.empty((0, 4), dtype=np.float32)

        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()

        all_logits = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            encodings = self.tokenizer.encode_batch(chunk)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            ort_inputs = {}
            if "input_ids" in self.input_names:
                ort_inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                ort_inputs["attention_mask"] = attention_mask

            outputs = self.session.run(["logits"], ort_inputs)
            all_logits.append(outputs[0])

        return np.concatenate(all_logits, axis=0) if len(all_logits) > 1 else all_logits[0]

    def _apply_meta_classifier_and_calibration(
        self, logits: np.ndarray, feat_vec: np.ndarray, stylometrics: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Fuses neural logits with stylometric features and applies temperature calibration."""
        if self.meta_weights is not None and self.feature_mean is not None and len(self.feature_mean) > 0:
            norm_feats = (feat_vec - self.feature_mean) / self.feature_std
            combined = np.concatenate([logits[0], norm_feats], axis=0).reshape(1, -1)
            raw_scores = np.dot(combined, self.meta_weights.T) + self.meta_intercept
        else:
            raw_scores = logits

        # Forensic Likelihood Guardrail using Formal Mathematical Formulations
        # Prevents authentic human authorial voice (e.g. David Sedaris, memoirs, creative essays)
        # from being falsely penalized as "AI-refined" when genuine human personal markers and burstiness are present.
        if stylometrics is not None:
            math_eqs = stylometrics.get("mathematical_equations", {})
            lambda_auth = math_eqs.get("authorial_affinity_lambda", 0.0)
            phi_disc = math_eqs.get("discourse_polarity_phi", 0.0)
            human_marker_rate = stylometrics.get("discourse_punctuation", {}).get("human_marker_rate", 0.0)
            ai_marker_rate = stylometrics.get("discourse_punctuation", {}).get("ai_marker_rate", 0.0)
            rhythm_delta = stylometrics.get("syntax_variance", {}).get("rhythm_delta", 0.0)

            is_human_side = (logits[0][0] > logits[0][2] and logits[0][0] > logits[0][3])
            is_literary_author = (
                (human_marker_rate >= 3.0 and rhythm_delta >= 12.0) or
                (human_marker_rate >= 5.0 and rhythm_delta >= 9.0) or
                (human_marker_rate >= 6.0) or
                (lambda_auth >= 0.70 and phi_disc <= -0.50)
            ) and (ai_marker_rate == 0.0) and (logits[0][0] >= logits[0][1] - 0.2)

            if is_human_side and is_literary_author:
                boost = 0.8 + 0.3 * (human_marker_rate / 2.0)
                if raw_scores[0][1] > raw_scores[0][0]:
                    raw_scores[0][0] = raw_scores[0][1] + boost
                else:
                    raw_scores[0][0] += boost

        # Temperature calibration
        temp = max(0.1, self.calibration_temperature)
        scaled = raw_scores / temp
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return probs[0]

    def _chunk_text(self, text: str, max_chunk_words: int = 100) -> List[str]:
        """
        Hierarchical chunking: splits long documents into coherent paragraph or sentence chunks
        (~70-100 words), matching the sequence length distribution the student transformer was distilled on.
        """
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not raw_paras:
            raw_paras = [text]

        chunks = []
        for para in raw_paras:
            para_words = para.split()
            if len(para_words) <= max_chunk_words:
                chunks.append(para)
            else:
                para_sents = split_sentences(para)
                curr_chunk = []
                curr_word_count = 0
                for sent in para_sents:
                    sent_words = len(sent.split())
                    if curr_chunk and (curr_word_count + sent_words > max_chunk_words):
                        chunks.append(" ".join(curr_chunk))
                        curr_chunk = [sent]
                        curr_word_count = sent_words
                    else:
                        curr_chunk.append(sent)
                        curr_word_count += sent_words
                if curr_chunk:
                    chunks.append(" ".join(curr_chunk))

        return chunks if chunks else [text]

    def analyze_text(self, text: str, confidence_threshold: float = 0.40) -> Dict[str, Any]:
        """
        Main analysis method matching QuillBot's exact UX & Output Spec:
        - 4 classes + 'Uncertain'
        - Calibrated probabilities
        - Per-sentence score and color-coded highlighting
        - 80–2,000 word guidance check
        """
        start_time = time.time()
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        words = re.findall(r"\b[a-zA-Z0-9'-]+\b", text)
        word_count = len(words)
        char_count = len(text)

        # Word count check (80 - 2,000 words guidance)
        is_length_warning = False
        length_warning_msg = None
        if word_count < 80:
            is_length_warning = True
            length_warning_msg = f"Input contains {word_count} words. QuillBot recommends 80–2,000 words for optimal forensic accuracy."
        elif word_count > 2000:
            is_length_warning = True
            length_warning_msg = f"Input contains {word_count} words. Analyzing the first 2,000 words for optimal performance."

        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        # 1. Stylometric feature extraction
        feat_vec, stylometrics = self._extract_stylometrics_vec(text, sentences)

        # 2. Document-level neural inference (hierarchical chunk pooling)
        student = self.mode in STUDENT_MODES
        if student:
            # Same preprocessing, chunking and pooling as training/evaluation; no meta-classifier or guardrail (both were fit on the
            # retired synthetic data), plain temperature softmax.
            max_len = int(self.student_config.get("max_len", 160))
            chunks = self._sentence_chunks(self._frontier_prep(text))
            chunk_logits = self._run_onnx_inference(chunks, max_length=max_len, batch_size=16)
            doc_logits = np.mean(chunk_logits, axis=0) / max(0.1, float(self.student_config.get("temperature", 1.0)))
            exp_d = np.exp(doc_logits - np.max(doc_logits))
            calibrated_probs = exp_d / np.sum(exp_d)
        else:
            chunks = self._chunk_text(text, max_chunk_words=100)
            chunk_logits = self._run_onnx_inference(chunks, max_length=256, batch_size=16)
            doc_logits = np.mean(chunk_logits, axis=0, keepdims=True)
            calibrated_probs = self._apply_meta_classifier_and_calibration(doc_logits, feat_vec, stylometrics=stylometrics)

        # Form probability dictionary
        prob_dict = {
            "Human-written": round(float(calibrated_probs[0]), 4),
            "Human-written & AI-refined": round(float(calibrated_probs[1]), 4),
            "AI-generated & AI-refined": round(float(calibrated_probs[2]), 4),
            "AI-generated": round(float(calibrated_probs[3]), 4)
        }

        # 3. Verdict Determination & Uncertainty Gating
        # 3. Sentence-by-sentence Inference & Highlights (QuillBot Alignment)
        sent_inputs = [self._frontier_prep(s) or s for s in sentences] if student else sentences
        sent_logits = self._run_onnx_inference(sent_inputs, max_length=128, batch_size=16)
        S = len(sentences)

        # Raw sentence softmax probabilities
        raw_s_probs = np.zeros((S, 4), dtype=np.float32)
        for i, s_logit in enumerate(sent_logits):
            exp_l = np.exp(s_logit - np.max(s_logit))
            raw_s_probs[i] = exp_l / np.sum(exp_l)

        # Phase 2 Formulation: Context-Aware Markovian Local Pacing Smoothing
        # \tilde{P}(y_i = k) = \lambda_{self} P(y_i = k) + \frac{1 - \lambda_{self}}{2} [ P(y_{i-1} = k) + P(y_{i+1} = k) ]
        lambda_self = 0.75
        smoothed_probs = np.zeros_like(raw_s_probs)
        for i in range(S):
            prev_p = raw_s_probs[max(0, i - 1)]
            next_p = raw_s_probs[min(S - 1, i + 1)]
            smoothed_probs[i] = lambda_self * raw_s_probs[i] + ((1.0 - lambda_self) / 2.0) * (prev_p + next_p)
            smoothed_probs[i] /= np.sum(smoothed_probs[i])

        # Bayesian Document Prior Blend
        # \bar{P}(y_i = k) \propto P_{sent}(k)^{1 - \alpha} \times P_{doc}(k)^\alpha
        alpha_blend = 0.50
        doc_prior = np.array(calibrated_probs, dtype=np.float32)
        blended_probs = np.zeros_like(smoothed_probs)
        for i in range(S):
            p_sent = smoothed_probs[i] + 1e-6
            p_doc = doc_prior + 1e-6
            log_p = (1.0 - alpha_blend) * np.log(p_sent) + alpha_blend * np.log(p_doc)
            exp_p = np.exp(log_p - np.max(log_p))
            blended_probs[i] = exp_p / np.sum(exp_p)

        # Word counts per sentence for weighted aggregation
        sent_words_counts = [max(1, len(re.findall(r"\b[a-zA-Z0-9'-]+\b", s))) for s in sentences]
        total_words_sum = sum(sent_words_counts)

        class_word_counts = {k: 0 for k in range(4)}
        for i in range(S):
            top_sent_class_idx = int(np.argmax(blended_probs[i]))
            class_word_counts[top_sent_class_idx] += sent_words_counts[i]

        # QuillBot Segment Breakdown: p_k = (\sum_{i: \hat{y}_i = k} w_i) / W * 100%
        qb_coverage = {
            "human": round(class_word_counts[0] / total_words_sum * 100, 1),
            "human_ai_refined": round(class_word_counts[1] / total_words_sum * 100, 1),
            "ai_ai_refined": round(class_word_counts[2] / total_words_sum * 100, 1),
            "ai_generated": round(class_word_counts[3] / total_words_sum * 100, 1),
        }
        total_ai_pct = round(qb_coverage["ai_generated"] + qb_coverage["ai_ai_refined"], 1)
        total_human_pct = round(100.0 - total_ai_pct, 1)

        # QuillBot Headline String & Pill Class
        if total_ai_pct >= 50.0:
            qb_headline = f"{int(round(total_ai_pct))}% of text is likely AI"
            qb_headline_class = "badge-ai"
        elif total_ai_pct == 0.0:
            qb_headline = "100% of text is likely Human"
            qb_headline_class = "badge-human"
        else:
            qb_headline = f"{int(round(total_ai_pct))}% of text is likely AI"
            qb_headline_class = "badge-ai-refined"

        # 4. Final Verdict Determination
        top_idx = int(np.argmax(calibrated_probs))
        sorted_probs = np.sort(calibrated_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]
        top_prob = float(calibrated_probs[top_idx])

        ai_score = float(calibrated_probs[2] + calibrated_probs[3])
        threshold = float(self.student_config.get("threshold", 0.5))
        if student:
            # The evaluated decision rule: AI iff P(AI-generated) + P(AI-generated & AI-refined) >= the dev 1%-FPR threshold.
            is_uncertain = False
            if ai_score >= threshold:
                verdict = CLASS_NAMES[3] if calibrated_probs[3] >= calibrated_probs[2] else CLASS_NAMES[2]
                final_conf = ai_score
            else:
                verdict = CLASS_NAMES[0] if calibrated_probs[0] >= calibrated_probs[1] else CLASS_NAMES[1]
                final_conf = 1.0 - ai_score
        elif top_prob < confidence_threshold or (top_prob < 0.45 and margin < 0.04):
            verdict = "Uncertain"
            is_uncertain = True
            final_conf = top_prob
        else:
            is_uncertain = False
            if total_ai_pct >= 50.0:
                verdict = "AI-generated" if qb_coverage["ai_generated"] >= qb_coverage["ai_ai_refined"] else "AI-generated & AI-refined"
                final_conf = max(top_prob, total_ai_pct / 100.0)
            else:
                verdict = "Human-written" if qb_coverage["human"] >= qb_coverage["human_ai_refined"] else "Human-written & AI-refined"
                final_conf = max(top_prob, total_human_pct / 100.0)

        verdict_meta = CLASS_METADATA[verdict]

        sentence_analyses = []
        for idx, sent in enumerate(sentences):
            s_probs = blended_probs[idx]
            s_top_idx = int(np.argmax(s_probs))
            s_class = CLASS_NAMES[s_top_idx]
            s_prob = float(s_probs[s_top_idx])
            s_meta = CLASS_METADATA[s_class]

            ai_presence = float(s_probs[2] + s_probs[3])

            reasons = []
            s_lower = sent.lower()
            s_words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", sent)
            s_words_lower = [w.lower() for w in s_words]

            # Detect markers present in this specific sentence
            sent_ai_markers = [w for w in s_words_lower if w in AI_SINGLE_WORDS]
            for p in AI_PHRASES:
                if p in s_lower:
                    sent_ai_markers.append(p)
            sent_ai_markers = sorted(list(set(sent_ai_markers)))

            sent_human_markers = [w for w in s_words_lower if w in HUMAN_SINGLE_WORDS]
            for p in HUMAN_PHRASES:
                if p in s_lower:
                    sent_human_markers.append(p)
            sent_human_markers = sorted(list(set(sent_human_markers)))

            if s_class == "AI-generated":
                reasons.append("High sequence predictability characteristic of autoregressive LLMs.")
            elif s_class == "AI-generated & AI-refined":
                reasons.append("Restructured syntax exhibiting AI generation signatures beneath paraphrased syntax.")
            elif s_class == "Human-written & AI-refined":
                reasons.append("Human sentence cadence with AI-assisted lexical smoothing or punctuation standardization.")
            else:
                reasons.append("Natural stylistic variation and human syntactic burstiness.")

            if sent_ai_markers:
                reasons.append(f"AI transition / marker detected: {', '.join(repr(m) for m in sent_ai_markers[:3])}")
            if sent_human_markers:
                reasons.append(f"Personal voice marker: {', '.join(repr(m) for m in sent_human_markers[:3])}")
            if len(s_words) >= 35:
                reasons.append(f"Complex clause construction ({len(s_words)} words)")
            elif 0 < len(s_words) <= 6:
                reasons.append(f"Punchy rhetorical clause ({len(s_words)} words)")

            sentence_analyses.append({
                "index": idx,
                "text": sent,
                "class_label": s_class,
                "class_key": s_meta["key"],
                "color_class": s_meta["color_class"],
                "highlight_class": s_meta["highlight_class"],
                "confidence": round(s_prob, 3),
                "ai_likelihood_pct": round(ai_presence * 100, 1),
                "probabilities": {
                    "Human-written": round(float(s_probs[0]), 3),
                    "Human-written & AI-refined": round(float(s_probs[1]), 3),
                    "AI-generated & AI-refined": round(float(s_probs[2]), 3),
                    "AI-generated": round(float(s_probs[3]), 3)
                },
                "reasons": reasons
            })

        elapsed_seconds = round(time.time() - start_time, 3)

        return {
            "summary": {
                "verdict": verdict,
                "verdict_description": verdict_meta["description"],
                "badge": verdict_meta["badge"],
                "is_uncertain": is_uncertain,
                "confidence": round(final_conf, 4),
                "confidence_pct": round(final_conf * 100, 1),
                "quillbot_headline": qb_headline,
                "quillbot_headline_class": qb_headline_class,
                "quillbot_ai_pct": total_ai_pct,
                "quillbot_human_pct": total_human_pct,
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": len(sentences),
                "length_warning": length_warning_msg,
                "elapsed_seconds": elapsed_seconds
            },
            "calibrated_probabilities": prob_dict,
            "percentages": {
                "ai_generated": qb_coverage["ai_generated"],
                "ai_ai_refined": qb_coverage["ai_ai_refined"],
                "human_ai_refined": qb_coverage["human_ai_refined"],
                "human": qb_coverage["human"]
            },
            "quillbot_breakdown": {
                "headline": qb_headline,
                "headline_class": qb_headline_class,
                "ai_percentage": total_ai_pct,
                "human_percentage": total_human_pct,
                "segments": qb_coverage
            },
            "sentences": sentence_analyses,
            "detector": {
                "mode": self.mode,
                "model": self.student_config.get("candidate", "shipped") if student else "shipped",
                "ai_score": round(ai_score, 4),
                "threshold": round(threshold, 4) if student else None,
                "threshold_rule": self.student_config.get("threshold_rule") if student else None,
            },
            "stylometrics": stylometrics,
            "mathematical_equations": stylometrics.get("mathematical_equations", {}) if stylometrics else {}
        }
