"""
Student Knowledge Distillation & Stylometric Meta-Classifier Training
Distills a compact student model (MiniLM-L6-v2, ~22M params) from teacher ensemble
soft probability distributions and fits the stylometric meta-classifier.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.stylometrics import FEATURE_NAMES, extract_stylometrics_feature_vector
from backend.document_parser import split_sentences

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def extract_features_vector(text: str) -> np.ndarray:
    """Extracts 20-dimensional forensic & information-theoretic stylometric feature vector."""
    sents = split_sentences(text)
    if not sents:
        sents = [text]
    vec, _ = extract_stylometrics_feature_vector(text, sents)
    return vec


class DistillationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Precompute stylometric feature vectors once
        self.features = []
        for item in self.samples:
            self.features.append(extract_features_vector(item["text"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        label = item["class_id"]

        # If teacher soft probs exist, use them; otherwise use smoothed one-hot target
        if "teacher_soft_probs" in item:
            soft_target = np.array(item["teacher_soft_probs"], dtype=np.float32)
        else:
            # Label smoothing approximation of teacher target
            soft_target = np.full(4, 0.05 / 3.0, dtype=np.float32)
            soft_target[label] = 0.95

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        feat_vec = self.features[idx]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "soft_target": torch.tensor(soft_target, dtype=torch.float32),
            "features": torch.tensor(feat_vec, dtype=torch.float32),
            "text": text
        }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(bin_mask):
            bin_acc = float(np.mean(accuracies[bin_mask]))
            bin_conf = float(np.mean(confidences[bin_mask]))
            bin_size = int(np.sum(bin_mask))
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


def train_student_model(epochs: int = 4, batch_size: int = 8, lr: float = 3e-5, alpha: float = 0.5, temperature: float = 2.0):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Training Student on device: {device}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")

    train_ds = DistillationDataset(train_path, tokenizer)
    val_ds = DistillationDataset(val_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Differential learning rates: 3e-5 for pretrained transformer trunk, 1e-3 for freshly initialized classifier head
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": lr},
        {"params": model.classifier.parameters(), "lr": 1e-3}
    ], weight_decay=0.01)
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    print(f"[Train] Commencing knowledge distillation ({len(train_ds)} train, {len(val_ds)} val, {epochs} epochs)...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            soft_targets = batch["soft_target"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Hard cross-entropy loss
            loss_ce = ce_loss_fn(logits, labels)

            # Soft teacher distillation loss (temperature-scaled KL)
            log_soft_student = F.log_softmax(logits / temperature, dim=-1)
            loss_kl = kl_loss_fn(log_soft_student, soft_targets) * (temperature ** 2)

            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} | Distillation Loss: {avg_loss:.4f}", flush=True)

    # Evaluate on validation set and collect neural logits + stylometric features
    model.eval()
    val_logits_list = []
    val_labels_list = []
    val_feats_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            feats = batch["features"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

            val_logits_list.append(logits)
            val_labels_list.append(labels)
            val_feats_list.append(feats)

    val_logits = np.concatenate(val_logits_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    val_feats = np.concatenate(val_feats_list, axis=0)

    # Collect training set logits and stylometric features
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_logits_list = []
    train_labels_list = []
    train_feats_list = []

    with torch.no_grad():
        for batch in train_eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            feats = batch["features"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

            train_logits_list.append(logits)
            train_labels_list.append(labels)
            train_feats_list.append(feats)

    train_logits = np.concatenate(train_logits_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    train_feats = np.concatenate(train_feats_list, axis=0)

    # 1. Feature normalization parameters based on training distribution
    feat_mean = np.mean(train_feats, axis=0).tolist()
    feat_std = (np.std(train_feats, axis=0) + 1e-6).tolist()

    # 2. Fit stylometric fusion meta-classifier on train set
    from sklearn.linear_model import LogisticRegression
    norm_train_feats = (train_feats - np.array(feat_mean)) / np.array(feat_std)
    train_combined = np.concatenate([train_logits, norm_train_feats], axis=1)

    meta_clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    meta_clf.fit(train_combined, train_labels)

    # 3. Post-quantization probability calibration via Temperature Scaling on held-out validation set
    norm_val_feats = (val_feats - np.array(feat_mean)) / np.array(feat_std)
    val_combined = np.concatenate([val_logits, norm_val_feats], axis=1)
    val_fused_raw = np.dot(val_combined, meta_clf.coef_.T) + meta_clf.intercept_

    # Optimize temperature directly to minimize Expected Calibration Error (ECE)
    best_T = 1.0
    best_ece = 1.0
    for T_cand in np.linspace(0.8, 5.0, 211):
        scaled = val_fused_raw / T_cand
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        e = compute_ece(p, val_labels)
        if e < best_ece:
            best_ece = e
            best_T = float(T_cand)

    cal_temp = best_T
    scaled = val_fused_raw / cal_temp
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    cal_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    val_preds = np.argmax(cal_probs, axis=1)
    val_acc = float(np.mean(val_preds == val_labels))
    ece = compute_ece(cal_probs, val_labels)

    print(f"[Train] Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"[Train] Calibration Temperature: {cal_temp:.4f}")
    print(f"[Train] Post-Distillation ECE: {ece:.4f} (Target: < 0.05 -> {'PASS' if ece < 0.05 else 'FAIL'})")

    # Save PyTorch student model
    torch_save_path = os.path.join(MODELS_DIR, "student_pytorch.pt")
    torch.save(model.state_dict(), torch_save_path)
    print(f"[Train] Saved PyTorch student checkpoint to {torch_save_path}")

    # Save Tokenizer files for standalone runtime
    tokenizer_dir = os.path.join(MODELS_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"[Train] Saved tokenizer files to {tokenizer_dir}")

    # Save Meta-Classifier and Calibration Parameters
    meta_config = {
        "feature_names": FEATURE_NAMES,
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "meta_weights": meta_clf.coef_.tolist(),
        "meta_intercept": meta_clf.intercept_.tolist(),
        "calibration_temperature": cal_temp,
        "val_accuracy": val_acc,
        "val_ece": ece,
        "classes": ["human", "human_ai_refined", "ai_ai_refined", "ai_generated"]
    }
    config_path = os.path.join(MODELS_DIR, "meta_classifier.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=2)
    print(f"[Train] Saved meta-classifier & calibrator parameters to {config_path}")

    return model, tokenizer, meta_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train distilled student model and stylometrics meta-classifier.")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    args = parser.parse_args()
    train_student_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
