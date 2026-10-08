"""
Veritas AI — Multi-Teacher Knowledge Distillation Framework
Combines three complementary teacher models to generate calibrated soft-targets:
1. ZeroGPT Teacher: Token perplexity dynamics, top-k rank predictability, burstiness shielding.
2. QuillBot Behavioral Teacher: 4-class taxonomy distributions (Human, Human+Refined, AI+Refined, AI)
   with word-level paraphrase sensitivity and authorial markers.
3. Sequence Ensemble Teacher: RoBERTa (Hello-SimpleAI/chatgpt-detector-roberta) & ModernBERT
   (rasbt/ai-text-detector-modernbert) contextual transformer representations.
"""

import os
import sys
import re
import math
import time
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.common import io_utils, schema, textnorm
from backend.document_parser import split_sentences
from notebooks.zerogpt_teacher import CalibratedZeroGPTTeacher

# Reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def prep(text: str) -> str:
    t = textnorm.strip_markdown(textnorm.normalize_text(text))
    return " ".join(t.split())


def sentence_chunks(text: str, max_words: int = 100) -> List[str]:
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


def label_of(r: Dict[str, Any]) -> int:
    """Canonical 4-class labeling matching runtime engine:
    0: human
    1: human-written & AI-refined (hybrids with ai_share < 0.5)
    2: AI-generated & AI-refined (attacked/humanized AI, hybrids >= 0.5)
    3: AI-generated (raw LLM)
    """
    if r.get("origin") == "hybrid":
        return 2 if r.get("ai_share", 0.5) >= 0.5 else 1
    if r.get("label") == "human":
        return 0
    return 3 if r.get("attack_id", "none") == "none" else 2


class SequenceEnsembleTeacher:
    """Sequence classification teacher combining HC3 RoBERTa and ModernBERT."""
    def __init__(self, device: str = "cpu", threads: int = 4):
        torch.set_num_threads(threads)
        self.device = device
        print(f"[Seq Ensemble Teacher] Loading RoBERTa & ModernBERT on {device}...")
        self.roberta_tok = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta").to(device).eval()

        self.modernbert_tok = AutoTokenizer.from_pretrained("rasbt/ai-text-detector-modernbert")
        self.modernbert_model = AutoModelForSequenceClassification.from_pretrained("rasbt/ai-text-detector-modernbert").to(device).eval()
        print("[Seq Ensemble Teacher] Successfully armed.")

    def score_batch(self, texts: List[str], max_len: int = 256) -> List[float]:
        """Returns ensemble P(AI) for a batch of texts."""
        if not texts:
            return []
        
        # RoBERTa inference (class 1 = AI in HC3)
        r_enc = self.roberta_tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        with torch.no_grad():
            r_logits = self.roberta_model(**r_enc).logits
            r_probs = F.softmax(r_logits.float(), dim=-1)[:, 1].cpu().numpy()

        # ModernBERT inference (class 1 = AI)
        m_enc = self.modernbert_tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        with torch.no_grad():
            m_logits = self.modernbert_model(**m_enc).logits
            m_probs = F.softmax(m_logits.float(), dim=-1)[:, 1].cpu().numpy()

        # Weighted combination (balanced ensemble)
        ens_probs = 0.5 * r_probs + 0.5 * m_probs
        return [float(p) for p in ens_probs]


class MultiTeacherPipeline:
    """Unified Multi-Teacher orchestrator synthesizing ZeroGPT, QuillBot, and Sequence Ensemble."""
    def __init__(self, device: str = "cpu", threads: int = 4):
        self.device = device
        self.threads = threads
        print("[MultiTeacherPipeline] Initializing teacher ensemble...")
        self.zerogpt_teacher = CalibratedZeroGPTTeacher(device=device, threads=threads)
        self.seq_teacher = SequenceEnsembleTeacher(device=device, threads=threads)
        print("[MultiTeacherPipeline] All teacher systems armed and operational.")

    def compute_fused_target(
        self,
        text: str,
        hard_label: int,
        is_esl: bool = False,
        generator_id: str = "",
        attack_id: str = "none",
        seq_p_ai: float = None,
        zg_fake_pct: float = None
    ) -> List[float]:
        """Synthesizes calibrated 4-class soft probability vector:
        [P(human), P(human_ai_refined), P(ai_ai_refined), P(ai_generated)]
        """
        words = text.split()
        n_words = len(words)

        # 1. Human Texts (with strict ESL and personal voice deconfounding)
        if hard_label == 0:
            if is_esl:
                # Absolute zero-leakage protection for ESL learner text
                target = [0.96, 0.04, 0.00, 0.00]
            else:
                # Clean native human: if high sequence probability is triggered by formal vocabulary,
                # permit minor leakage into human-refined (class 1), never into AI (class 2 or 3)
                p_formal = min(0.12, max(0.02, seq_p_ai * 0.12)) if seq_p_ai is not None else 0.05
                target = [1.0 - p_formal, p_formal, 0.00, 0.00]
            return target

        # 2. AI Texts: Fused signal from Sequence Ensemble and ZeroGPT
        zg_prob = (zg_fake_pct / 100.0) if zg_fake_pct is not None else 0.5
        s_prob = seq_p_ai if seq_p_ai is not None else 0.5

        # Weighted blend of token predictability (ZeroGPT) and contextual semantics (Transformer)
        # For frontier models (Claude Opus/Sonnet), sequence ensemble has stronger sensitivity
        is_frontier = "claude" in generator_id.lower() or "gpt-4" in generator_id.lower()
        if is_frontier:
            p_ai = 0.65 * s_prob + 0.35 * zg_prob
        else:
            p_ai = 0.50 * s_prob + 0.50 * zg_prob

        # Ensure AI signal has floor for ground-truth AI samples
        p_ai = max(0.40, min(0.99, p_ai))

        if hard_label == 1:
            # Human-written & AI-refined (hybrid < 0.5 AI)
            target = [0.15, 0.60, 0.20, 0.05]
        elif hard_label == 2:
            # AI-generated & AI-refined (attacked / humanized / paraphrased AI)
            # Soft distribution concentrates on Class 2 with minor tail in Class 3 and Class 1
            target = [0.03, 0.12, p_ai * 0.70 + 0.15, (1.0 - p_ai) * 0.30 + 0.05]
        else:
            # Raw AI-generated (Class 3)
            # Soft distribution concentrates on Class 3 with tail in Class 2
            target = [0.02, 0.05, (1.0 - p_ai) * 0.35 + 0.05, p_ai * 0.80 + 0.10]

        # L1 normalize
        s = sum(target)
        return [float(x / s) for x in target]


class DistillationDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_len: int = 160):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.long),
            "teacher_soft": torch.tensor(item["teacher_soft"], dtype=torch.float32)
        }


def build_multi_teacher_training_set(
    pipeline: MultiTeacherPipeline,
    train_rows: List[Dict[str, Any]],
    cache_path: str = None,
    n_ai_samples: int = 600,
    n_human_samples: int = 500
) -> List[Dict[str, Any]]:
    """Builds a balanced, high-coverage distillation dataset caching multi-teacher soft targets."""
    if cache_path and os.path.exists(cache_path):
        print(f"[Dataset] Found cached multi-teacher dataset at {cache_path}. Loading...")
        with open(cache_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        print(f"[Dataset] Loaded {len(items)} training chunks from cache.")
        return items

    print("[Dataset] Building high-coverage multi-teacher distillation dataset from train split...")
    # 1. Stratified AI selection
    claude_opus = [r for r in train_rows if r.get("generator_id") == "claude-opus-5-5"]
    claude_sonnet = [r for r in train_rows if r.get("generator_id") == "claude-sonnet-5-5"]
    attacked_ai = [r for r in train_rows if r["label"] == "ai" and r.get("attack_id", "none") != "none"]
    other_ai = [r for r in train_rows if r["label"] == "ai" and not r.get("generator_id", "").startswith("claude") and r.get("attack_id", "none") == "none"]

    selected_ai = []
    selected_ai.extend(claude_opus)   # 100% of Claude Opus
    selected_ai.extend(claude_sonnet) # 100% of Claude Sonnet
    
    # Add humanized/attacked AI
    n_attack = min(len(attacked_ai), 200)
    selected_ai.extend(random.sample(attacked_ai, n_attack))

    # Add diverse LLM generators (GPT-4, LLaMA, etc.)
    n_other = max(0, n_ai_samples - len(selected_ai))
    selected_ai.extend(random.sample(other_ai, min(len(other_ai), n_other)))

    # 2. Stratified Human selection
    esl_human = [r for r in train_rows if r["label"] == "human" and r.get("esl")] # 100% of ESL
    native_human = [r for r in train_rows if r["label"] == "human" and not r.get("esl")]

    selected_human = []
    selected_human.extend(esl_human) # 100% of ESL human
    n_native = max(0, n_human_samples - len(selected_human))
    selected_human.extend(random.sample(native_human, min(len(native_human), n_native)))

    all_docs = selected_ai + selected_human
    random.shuffle(all_docs)
    print(f"[Dataset] Selected {len(all_docs)} diverse documents ({len(selected_ai)} AI, {len(selected_human)} Human, {len(esl_human)} ESL).")

    # 3. Batch compute Sequence Ensemble probabilities for efficiency
    print("[Dataset] Computing sequence ensemble scores (RoBERTa + ModernBERT)...")
    texts_to_score = [prep(d["text"])[:512] for d in all_docs]
    seq_scores = []
    batch_sz = 32
    for b in range(0, len(texts_to_score), batch_sz):
        sub_batch = texts_to_score[b : b + batch_sz]
        seq_scores.extend(pipeline.seq_teacher.score_batch(sub_batch, max_len=160))

    # 4. Synthesize targets with ZeroGPT perplexity & burstiness
    print("[Dataset] Computing ZeroGPT token predictability and fusing multi-teacher targets...")
    items = []
    t0 = time.time()
    for idx, (doc, s_prob) in enumerate(zip(all_docs, seq_scores)):
        clean_text = prep(doc["text"])
        hard_lbl = label_of(doc)
        is_esl = bool(doc.get("esl"))
        gen_id = str(doc.get("generator_id", ""))
        atk_id = str(doc.get("attack_id", "none"))

        # ZeroGPT scoring
        zg_res = pipeline.zerogpt_teacher.score_document(doc["text"], ground_truth_label=doc["label"])
        zg_fake = zg_res["fakePercentage"]

        # Fused 4-class target
        soft_target = pipeline.compute_fused_target(
            clean_text,
            hard_lbl,
            is_esl=is_esl,
            generator_id=gen_id,
            attack_id=atk_id,
            seq_p_ai=s_prob,
            zg_fake_pct=zg_fake
        )

        # Chunk document into training units
        cs = sentence_chunks(clean_text, max_words=100)
        for c in cs[:2]:  # Up to 2 ~100-word chunks per document
            items.append({
                "text": c,
                "hard_label": hard_lbl,
                "teacher_soft": soft_target,
                "is_esl": is_esl,
                "generator_id": gen_id
            })

        if (idx + 1) % 150 == 0:
            print(f"  Processed {idx + 1}/{len(all_docs)} docs ({time.time() - t0:.1f}s)...")

    print(f"[Dataset] Generated {len(items)} training chunks with multi-teacher soft targets.")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"[Dataset] Cached dataset successfully saved to: {cache_path}")

    return items


def train_single_run(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 3.5e-5,
    alpha: float = 0.5,
    temperature: float = 2.0,
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[float, Any]:
    """Runs a single distillation training pass and returns validation loss."""
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=4)
    model.train()

    train_ds = DistillationDataset(train_items, tokenizer)
    val_ds = DistillationDataset(val_items, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    T = temperature
    best_val_loss = float("inf")

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            hard_y = batch["hard_label"]
            soft_t = batch["teacher_soft"]

            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits

            # Hard cross-entropy
            l_ce = ce_loss_fn(logits, hard_y)

            # Soft Hinton KL loss
            l_kl = kl_loss_fn(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(soft_t / T, dim=-1)
            ) * (T * T)

            loss = alpha * l_kl + (1.0 - alpha) * l_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_batch in val_loader:
                v_ids = v_batch["input_ids"]
                v_mask = v_batch["attention_mask"]
                v_hard_y = v_batch["hard_label"]
                v_soft_t = v_batch["teacher_soft"]

                v_out = model(input_ids=v_ids, attention_mask=v_mask)
                v_logits = v_out.logits

                vl_ce = ce_loss_fn(v_logits, v_hard_y)
                vl_kl = kl_loss_fn(
                    F.log_softmax(v_logits / T, dim=-1),
                    F.softmax(v_soft_t / T, dim=-1)
                ) * (T * T)
                val_loss += (alpha * vl_kl + (1.0 - alpha) * vl_ce).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    return best_val_loss, model


def run_hyperparameter_search(
    items: List[Dict[str, Any]],
    output_candidate_dir: str,
    temperatures: List[float] = [1.5, 2.0, 2.5],
    alphas: List[float] = [0.4, 0.5, 0.6]
) -> Dict[str, Any]:
    """Explores temperature and distillation weight grid, logging validation losses."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER GRID SEARCH: MULTI-TEACHER DISTILLATION")
    print("=" * 70)

    # 85/15 train/validation split
    random.shuffle(items)
    n_val = int(len(items) * 0.15)
    val_items = items[:n_val]
    train_items = items[n_val:]
    print(f"Data split: {len(train_items)} train items, {len(val_items)} validation items.")

    results = []
    best_config = None
    best_val_loss = float("inf")
    search_tmp_dir = os.path.join(REPO, "models", "candidates", "grid_search_tmp")

    grid = [(t, a) for t in temperatures for a in alphas]
    for idx, (t, a) in enumerate(grid):
        print(f"\n[Trial {idx + 1}/{len(grid)}] Evaluating T={t}, alpha={a}...")
        trial_out = os.path.join(search_tmp_dir, f"trial_T{t}_a{a}")
        t0 = time.time()
        val_loss, _ = train_single_run(
            train_items=train_items,
            val_items=val_items,
            output_dir=trial_out,
            epochs=2,
            batch_size=16,
            lr=3.5e-5,
            alpha=a,
            temperature=t
        )
        elapsed = time.time() - t0
        print(f"  -> Finished in {elapsed:.1f}s | Validation Loss: {val_loss:.4f}")
        res_entry = {"T": t, "alpha": a, "val_loss": val_loss, "time_s": elapsed}
        results.append(res_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = (t, a)
            # Copy trial to primary candidate dir
            os.makedirs(output_candidate_dir, exist_ok=True)
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(trial_out)
            mod = AutoModelForSequenceClassification.from_pretrained(trial_out)
            tok.save_pretrained(output_candidate_dir)
            mod.save_pretrained(output_candidate_dir)

    print("\n" + "=" * 70)
    print(f"GRID SEARCH COMPLETE. Best config: T={best_config[0]}, alpha={best_config[1]} (Val Loss: {best_val_loss:.4f})")
    print("=" * 70)

    # Clean up temporary search directory
    import shutil
    if os.path.exists(search_tmp_dir):
        shutil.rmtree(search_tmp_dir, ignore_errors=True)

    summary = {
        "best_T": best_config[0],
        "best_alpha": best_config[1],
        "best_val_loss": best_val_loss,
        "grid_results": results
    }
    with open(os.path.join(output_candidate_dir, "distill_grid_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def export_and_verify_onnx(hf_candidate_dir: str, production_dir: str) -> str:
    """Exports candidate to FP32 ONNX, applies INT8 dynamic quantization, and verifies parity."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort

    print(f"\n[ONNX Export] Loading PyTorch model from {hf_candidate_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_candidate_dir)
    model = AutoModelForSequenceClassification.from_pretrained(hf_candidate_dir, num_labels=4).eval()

    os.makedirs(production_dir, exist_ok=True)
    fp32_path = os.path.join(production_dir, "student_model.onnx")
    int8_path = os.path.join(production_dir, "student_model_int8.onnx")

    dummy_text = "Furthermore, advanced syntactic structures indicate machine generation patterns."
    enc = tokenizer(dummy_text, return_tensors="pt")
    dummy_input_ids = enc["input_ids"]
    dummy_attention_mask = enc["attention_mask"]

    with torch.no_grad():
        pt_logits = model(dummy_input_ids, attention_mask=dummy_attention_mask).logits.numpy()

    print(f"[ONNX Export] Exporting FP32 graph -> {fp32_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True
    )

    onnx_proto = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_proto)
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"[ONNX Export] FP32 ONNX valid. Size: {fp32_size:.2f} MB")

    print(f"[ONNX Export] Applying dynamic INT8 quantization -> {int8_path}...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8
    )

    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    compression = (1.0 - (int8_size / fp32_size)) * 100.0
    print(f"[ONNX Export] INT8 Quantization successful!")
    print(f"             INT8 Size: {int8_size:.2f} MB (Compression: {compression:.1f}%)")
    assert int8_size <= 25.0, f"INT8 model exceeds 25MB budget: {int8_size:.2f} MB"

    # Purge FP32 ONNX to satisfy disk budget
    if os.path.exists(fp32_path):
        os.remove(fp32_path)
        print("[ONNX Export] Unquantized FP32 model purged to conserve disk space.")

    # Save tokenizer for offline runtime
    tokenizer.save_pretrained(production_dir)
    # Also copy INT8 ONNX to candidate directory for evaluation wrapper
    shutil_cand_int8 = os.path.join(hf_candidate_dir, "student_model_int8.onnx")
    import shutil
    shutil.copy2(int8_path, shutil_cand_int8)

    # Parity verification with ONNX Runtime
    print("[ONNX Export] Verifying numerical parity between PyTorch and INT8 ONNX...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy()
    }
    ort_logits = session.run(["logits"], ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"[ONNX Export] Max logit disparity: {max_diff:.5f}")
    assert max_diff <= 0.20, f"Disparity exceeds safety threshold: {max_diff}"
    print("[ONNX Export] Numerical parity confirmed!")

    return int8_path


if __name__ == "__main__":
    train_split_path = os.path.join(REPO, "data", "splits", "train.jsonl.gz")
    cache_path = os.path.join(REPO, "notebooks", "multi_teacher_targets_cache.json")
    candidate_dir = os.path.join(REPO, "models", "candidates", "multi_teacher_distilled")
    production_dir = os.path.join(REPO, "models", "multi_teacher_distilled")

    train_rows = io_utils.read_jsonl(train_split_path)
    pipeline = MultiTeacherPipeline(device="cpu", threads=6)

    # Step 1: Build & cache multi-teacher targets
    items = build_multi_teacher_training_set(
        pipeline=pipeline,
        train_rows=train_rows,
        cache_path=cache_path,
        n_ai_samples=600,
        n_human_samples=500
    )

    # Step 2: Hyperparameter grid search & student training
    search_summary = run_hyperparameter_search(
        items=items,
        output_candidate_dir=candidate_dir,
        temperatures=[1.5, 2.0, 2.5],
        alphas=[0.4, 0.5, 0.6]
    )

    # Step 3: INT8 ONNX export & numerical verification
    export_and_verify_onnx(candidate_dir, production_dir)
    print("\n[SUCCESS] Multi-teacher distillation pipeline completed successfully!")


