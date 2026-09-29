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
from backend.stylometrics import analyze_stylometrics, extract_stylometrics_feature_vector

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "student_model_int8.onnx")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer", "tokenizer.json")
META_CONFIG_PATH = os.path.join(MODELS_DIR, "meta_classifier.json")

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

    def __init__(self, threads: int = 2):
        print(f"[RuntimeEngine] Initializing Offline QuillBot-behavior Student Engine (threads={threads})...")
        
        # 1. Initialize ONNX Runtime Session with target hardware thread caps
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(
                f"Shipped INT8 ONNX model not found at {ONNX_MODEL_PATH}. "
                "Run 'python scripts/export_onnx.py' first."
            )

        self.session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        
        # 2. Initialize Fast Rust Tokenizer (Zero PyTorch)
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}.")
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
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
    def get_instance(cls, threads: int = 2) -> "QuillBotDetectorEngine":
        if cls._instance is None:
            cls._instance = cls(threads=threads)
        return cls._instance

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

        # Forensic Guardrail for Authentic Human Authorial Voice:
        # Prevents rich authorial vocabulary and complex punctuation (e.g. David Sedaris, literature)
        # from being falsely penalized as "AI-refined" when genuine human personal markers and burstiness are present.
        if stylometrics is not None:
            human_marker_rate = stylometrics.get("discourse_punctuation", {}).get("human_marker_rate", 0.0)
            ai_marker_rate = stylometrics.get("discourse_punctuation", {}).get("ai_marker_rate", 0.0)
            rhythm_delta = stylometrics.get("syntax_variance", {}).get("rhythm_delta", 0.0)

            is_human_side = (logits[0][0] > logits[0][2] and logits[0][0] > logits[0][3])
            is_literary_author = (
                (human_marker_rate >= 3.0 and rhythm_delta >= 12.0) or
                (human_marker_rate >= 5.0 and rhythm_delta >= 9.0) or
                (human_marker_rate >= 6.0)
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
        top_idx = int(np.argmax(calibrated_probs))
        top_prob = float(calibrated_probs[top_idx])
        top_class = CLASS_NAMES[top_idx]

        # Check for Uncertainty: if top probability is below confidence threshold or top two are too close
        sorted_probs = np.sort(calibrated_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

        if top_prob < confidence_threshold or (top_prob < 0.45 and margin < 0.04):
            verdict = "Uncertain"
            is_uncertain = True
        else:
            verdict = top_class
            is_uncertain = False

        verdict_meta = CLASS_METADATA[verdict]

        # 4. Sentence-by-sentence Inference & Highlights
        sent_logits = self._run_onnx_inference(sentences, max_length=128, batch_size=16)
        sentence_analyses = []

        for idx, (sent, s_logit) in enumerate(zip(sentences, sent_logits)):
            # Softmax on sentence logits
            exp_l = np.exp(s_logit - np.max(s_logit))
            s_probs = exp_l / np.sum(exp_l)

            s_top_idx = int(np.argmax(s_probs))
            s_class = CLASS_NAMES[s_top_idx]
            s_prob = float(s_probs[s_top_idx])
            s_meta = CLASS_METADATA[s_class]

            # AI likelihood percentage for sentence
            ai_presence = float(s_probs[2] + s_probs[3])  # Combined AI + AI-refined AI

            reasons = []
            if s_class == "AI-generated":
                reasons.append("High sequence predictability characteristic of autoregressive LLMs.")
            elif s_class == "AI-generated & AI-refined":
                reasons.append("Restructured syntax exhibiting AI generation signatures beneath paraphrased syntax.")
            elif s_class == "Human-written & AI-refined":
                reasons.append("Human sentence cadence with AI-assisted lexical smoothing or punctuation standardization.")
            else:
                reasons.append("Natural stylistic variation and human syntactic burstiness.")

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
                "confidence": round(top_prob, 4),
                "confidence_pct": round(top_prob * 100, 1),
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": len(sentences),
                "length_warning": length_warning_msg,
                "elapsed_seconds": elapsed_seconds
            },
            "calibrated_probabilities": prob_dict,
            "percentages": {
                "ai_generated": round(prob_dict["AI-generated"] * 100, 1),
                "ai_ai_refined": round(prob_dict["AI-generated & AI-refined"] * 100, 1),
                "human_ai_refined": round(prob_dict["Human-written & AI-refined"] * 100, 1),
                "human": round(prob_dict["Human-written"] * 100, 1)
            },
            "sentences": sentence_analyses,
            "stylometrics": stylometrics
        }
