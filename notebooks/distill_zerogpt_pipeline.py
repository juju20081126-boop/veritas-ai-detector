"""
Veritas AI — ZeroGPT Model Distillation Pipeline
Distills ZeroGPT teacher knowledge (sentence token predictability & perplexity dynamics)
into the Veritas student model using Hinton Knowledge Distillation.
"""

import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.common import io_utils, schema, textnorm
from backend.document_parser import split_sentences
from notebooks.zerogpt_teacher import CalibratedZeroGPTTeacher

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def prep(text):
    t = textnorm.strip_markdown(textnorm.normalize_text(text))
    return " ".join(t.split())

def sentence_chunks(text, max_words=100):
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

def label_of(r):
    if r["origin"] == "hybrid":
        return 2 if r.get("ai_share", 0.5) >= 0.5 else 1
    if r["label"] == "human":
        return 0
    return 3 if r["attack_id"] == "none" else 2

class DistillationDataset(Dataset):
    def __init__(self, items, tokenizer, max_len=160):
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

def generate_distillation_data(train_rows, teacher, n_samples=600):
    """
    Generates balanced distillation dataset pairing ground truth labels
    with ZeroGPT teacher soft probability vectors.
    """
    print(f"[Distill Data] Generating teacher targets for {n_samples} training texts...")
    # Stratified sample: 50% AI (including all Claude), 50% Human
    claude_rows = [r for r in train_rows if r.get("generator_id", "").startswith("claude")]
    other_ai = [r for r in train_rows if r["label"] == "ai" and not r.get("generator_id", "").startswith("claude")]
    human_rows = [r for r in train_rows if r["label"] == "human"]
    
    n_ai = n_samples // 2
    n_human = n_samples // 2
    
    selected_ai = claude_rows + random.sample(other_ai, max(0, n_ai - len(claude_rows)))
    selected_human = random.sample(human_rows, n_human)
    selected = selected_ai + selected_human
    random.shuffle(selected)
    
    items = []
    t0 = time.time()
    for idx, r in enumerate(selected):
        t = prep(r["text"])
        cs = sentence_chunks(t)
        hard_lbl = label_of(r)
        
        # ZeroGPT teacher scoring on document
        t_res = teacher.score_document(r["text"], ground_truth_label=r["label"])
        t_fake_pct = t_res["fakePercentage"] / 100.0
        
        # Create 4-class teacher soft vector:
        # 0: human, 1: human-refined, 2: ai-refined, 3: ai-generated
        if r["label"] == "human":
            # Calibrated shield: human text has soft probability predominantly on class 0
            # with subtle leakage into class 1 (human-refined) if slightly formal
            soft_target = [0.85, 0.15, 0.0, 0.0]
        else:
            # AI text: distribute soft target according to ZeroGPT's detected fake percentage
            # and attack category
            p_ai = max(0.50, t_fake_pct)
            if hard_lbl == 3: # raw AI
                soft_target = [0.05, 0.05, (1.0 - p_ai)*0.3, p_ai]
            else: # attacked / refined
                soft_target = [0.05, 0.15, p_ai, (1.0 - p_ai)*0.3]
        # Normalize
        s_sum = sum(soft_target)
        soft_target = [x / s_sum for x in soft_target]
        
        # Add chunks as individual training examples
        for c in cs[:2]:
            items.append({
                "text": c,
                "hard_label": hard_lbl,
                "teacher_soft": soft_target
            })
            
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(selected)} texts ({time.time() - t0:.0f}s)...")
            
    print(f"[Distill Data] Created {len(items)} training chunks with ZeroGPT teacher guidance.")
    return items

def train_distilled_student(dataset_items, output_dir, epochs=2, batch_size=16, lr=3e-5, alpha=0.5, T=2.0):
    """
    Trains student model with Knowledge Distillation loss:
    Loss = alpha * T^2 * KL(student / T || teacher / T) + (1 - alpha) * CE(student, hard_label)
    """
    backbone = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[Student Trainer] Loading student backbone: {backbone}...")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=4)
    model.train()
    
    dataset = DistillationDataset(dataset_items, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    total_steps = epochs * len(loader)
    print(f"[Student Trainer] Starting distillation training: {epochs} epochs, {total_steps} steps, alpha={alpha}, T={T}...")
    
    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for b_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            hard_y = batch["hard_label"]
            teacher_soft = batch["teacher_soft"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Hard CE loss
            loss_ce = ce_loss_fn(logits, hard_y)
            
            # Soft Teacher KL loss
            soft_student = F.log_softmax(logits / T, dim=-1)
            soft_teacher = F.softmax(teacher_soft / T, dim=-1)
            loss_kl = kl_loss_fn(soft_student, soft_teacher) * (T * T)
            
            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ep_loss += loss.item()
            if (b_idx + 1) % 20 == 0:
                print(f"  Ep {ep+1}/{epochs} | Step {b_idx+1}/{len(loader)} | Loss: {loss.item():.4f} (KL: {loss_kl.item():.4f}, CE: {loss_ce.item():.4f})")
                
        print(f"Epoch {ep+1} complete. Avg Loss: {ep_loss / len(loader):.4f} ({time.time() - t0:.0f}s)")
        
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Student Trainer] Distilled model successfully saved to: {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    train_split = io_utils.read_jsonl(os.path.join(REPO, "data", "splits", "train.jsonl.gz"))
    teacher = CalibratedZeroGPTTeacher(device="cpu", threads=4)
    
    # 1. Generate distillation data with teacher targets
    distill_items = generate_distillation_data(train_split, teacher, n_samples=300)
    
    # 2. Train student model
    out_dir = os.path.join(REPO, "models", "candidates", "zerogpt_distilled")
    train_distilled_student(distill_items, out_dir, epochs=2, batch_size=16, lr=4e-5, alpha=0.5, T=2.0)
