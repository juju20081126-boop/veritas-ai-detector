"""
Baseline detectors, scored on RAW text exactly as a user would paste it (no input normalisation, no tuning on our data).

  shipped        the pre-change Veritas student (ONNX INT8 + stylometric meta-classifier), score = P(AI-generated) + P(AI-generated & AI-refined)
  hc3_roberta    Hello-SimpleAI/chatgpt-detector-roberta       (RoBERTa-base trained on HC3, ChatGPT-era)
  openai_roberta openai-community/roberta-base-openai-detector (GPT-2 output detector, 2019)
  modernbert     rasbt/ai-text-detector-modernbert             (ModernBERT-base fine-tuned, open)
  desklib        desklib/ai-text-detector-v1.01                (DeBERTa-v3-large based, open, MIT)
  binoculars     Binoculars-style score with a small base/instruct pair (Qwen2.5-0.5B), lower ratio = more AI

Long texts are scored on up to 3 windows and averaged (classifiers) or on the first 512 tokens (binoculars).
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


class ShippedDetector:
    """The runtime engine end to end. mode="shipped" is the pre-change baseline; mode="frontier" (detector name `frontier_onnx`)
    is the exported INT8 frontier model as deployed, used to check that the shipped artefact matches the PyTorch candidate;
    mode="multi_teacher" (`multi_teacher_onnx`) is the INT8 multi-teacher distilled student as deployed; mode="tfidf"
    (`tfidf_runtime`) is the numpy TF-IDF n-gram model as deployed; mode="ensemble" (`ensemble_runtime`) is the OR of the
    modes in models/ensemble/runtime_config.json as deployed."""

    def __init__(self, threads=2, mode="shipped"):
        from backend.runtime_engine import QuillBotDetectorEngine
        self.name = {"shipped": "shipped", "tfidf": "tfidf_runtime", "ensemble": "ensemble_runtime"}.get(mode, f"{mode}_onnx")
        self.engine = QuillBotDetectorEngine(threads=threads, mode=mode)

    def score(self, texts):
        out = []
        for t in texts:
            try:
                r = self.engine.analyze_text(t)
                p = r["calibrated_probabilities"]
                out.append(p["AI-generated"] + p["AI-generated & AI-refined"])
            except Exception:
                out.append(0.0)
        return np.array(out, float)


class HFSeqClassifier:
    def __init__(self, name, model_id, ai_index=None, ai_names=("fake", "ai", "chatgpt", "machine", "generated", "label_1"),
                 max_length=512, threads=6, trust_remote_code=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch, self.name, self.max_length = torch, name, max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()
        if ai_index is None:
            id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
            hits = [i for i, v in id2label.items() if any(a in v for a in ai_names)]
            ai_index = hits[0] if len(hits) == 1 else 1
        self.ai_index = ai_index

    def _windows(self, text):
        ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        step = self.max_length - 2
        wins = [ids[i:i + step] for i in range(0, max(1, len(ids)), step)][:3]
        return [self.tok.decode(w) for w in wins if w]

    def score(self, texts, batch=16):
        out = []
        for t in texts:
            ws = self._windows(t) or [t]
            enc = self.tok(ws, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            p = self.torch.softmax(logits, dim=-1)[:, self.ai_index].mean().item()
            out.append(p)
        return np.array(out, float)


class DesklibDetector:
    """desklib/ai-text-detector-v1.01: transformer encoder + mean pooling + 1-logit head (code from its model card)."""
    name = "desklib"

    def __init__(self, threads=6, max_length=768):
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch, self.max_length = torch, max_length
        mid = "desklib/ai-text-detector-v1.01"
        self.tok = AutoTokenizer.from_pretrained(mid)
        cfg = AutoConfig.from_pretrained(mid)

        class Model(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                h = self.model(input_ids, attention_mask=attention_mask)[0]
                m = attention_mask.unsqueeze(-1).expand(h.size()).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)
        self.net = Model(cfg)
        try:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(mid, "model.safetensors"))
        except Exception:
            sd = torch.load(hf_hub_download(mid, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()

    def score(self, texts):
        out = []
        for t in texts:
            enc = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_length)
            with self.torch.no_grad():
                logit = self.net(enc["input_ids"], enc["attention_mask"]).squeeze()
            out.append(float(self.torch.sigmoid(logit)))
        return np.array(out, float)


class BinocularsSmall:
    """Binoculars-style: log-PPL under the observer divided by the observer/performer cross-perplexity. Returns -ratio (higher = more AI)."""
    name = "binoculars_small"

    def __init__(self, observer="Qwen/Qwen2.5-0.5B", performer="Qwen/Qwen2.5-0.5B-Instruct", threads=6, max_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch, self.max_tokens = torch, max_tokens
        self.tok = AutoTokenizer.from_pretrained(observer)
        self.obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=torch.float32).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=torch.float32).eval()

    def score(self, texts):
        torch = self.torch
        out = []
        for t in texts:
            enc = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_tokens)
            ids = enc["input_ids"]
            if ids.shape[1] < 8:
                out.append(0.0)
                continue
            with torch.no_grad():
                lo = self.obs(**enc).logits[:, :-1]
                lp = self.perf(**enc).logits[:, :-1]
            labels = ids[:, 1:]
            logp_o = torch.log_softmax(lo, dim=-1)
            ppl = -logp_o.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
            xppl = -(torch.softmax(lp, dim=-1) * logp_o).sum(-1).mean()
            out.append(-float(ppl / xppl))
        return np.array(out, float)


def get_detector(name, threads=6):
    if name.startswith("cand:"):
        from scripts.detectors.student import StudentDetector
        return StudentDetector(name[5:], threads=threads)
    if name == "shipped":
        return ShippedDetector(threads=2)
    if name == "frontier_onnx":
        return ShippedDetector(threads=2, mode="frontier")
    if name == "multi_teacher_onnx":
        return ShippedDetector(threads=2, mode="multi_teacher")
    if name == "tfidf_runtime":
        return ShippedDetector(threads=2, mode="tfidf")
    if name == "ensemble_runtime":
        return ShippedDetector(threads=2, mode="ensemble")
    if name == "hc3_roberta":
        return HFSeqClassifier("hc3_roberta", "Hello-SimpleAI/chatgpt-detector-roberta", ai_index=1, threads=threads)
    if name == "openai_roberta":
        return HFSeqClassifier("openai_roberta", "openai-community/roberta-base-openai-detector", threads=threads)
    if name == "modernbert":
        return HFSeqClassifier("modernbert", "rasbt/ai-text-detector-modernbert", threads=threads)
    if name == "desklib":
        return DesklibDetector(threads=threads)
    if name == "binoculars":
        return BinocularsSmall(threads=threads)
    raise ValueError(f"unknown detector {name}")
