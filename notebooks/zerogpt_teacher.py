"""
Veritas AI — Calibrated ZeroGPT Teacher Model
Implements ZeroGPT's DeepAnalyse token-perplexity & rank formulation
with calibrated deconfounding against simple/formal human false positives.
"""

import os
import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import re
import math
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from backend.document_parser import split_sentences

class CalibratedZeroGPTTeacher:
    """
    Teacher model implementing ZeroGPT DeepAnalyse mechanics with
    inter-sentence burstiness calibration to eliminate human false positives.
    """
    def __init__(self, device="cpu", threads=4):
        torch.set_num_threads(threads)
        self.device = device
        print(f"[ZeroGPT Teacher] Loading GPT-2 causal language model on {device}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.lm_model.eval()
        print("[ZeroGPT Teacher] Successfully armed.")

    def compute_sentence_stats(self, sentence: str):
        text = sentence.strip()
        words = text.split()
        if len(words) < 3:
            return None
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 3:
            return None
        with torch.no_grad():
            outputs = self.lm_model(input_ids)
            logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().numpy()
        
        nll = float(-np.mean(token_logp))
        ppl = float(np.exp(min(15.0, nll)))
        top10_ratio = float(np.mean(token_logp > -2.30)) # log(0.1) ~ -2.30
        return {
            "nll": nll,
            "ppl": ppl,
            "top10_ratio": top10_ratio,
            "word_count": len(words)
        }

    def score_document(self, text: str, ground_truth_label: str = None):
        """
        Computes ZeroGPT fakePercentage, sentence AI flags, and soft teacher targets.
        """
        sents = split_sentences(text)
        if not sents:
            return {"fakePercentage": 0.0, "sentence_soft_probs": [], "ai_words": 0, "total_words": 0}
        
        sent_stats = [self.compute_sentence_stats(s) for s in sents]
        valid_stats = [st for st in sent_stats if st is not None]
        if not valid_stats:
            return {"fakePercentage": 0.0, "sentence_soft_probs": [0.0]*len(sents), "ai_words": 0, "total_words": len(text.split())}
        
        # Document-level burstiness (variance of sentence perplexities)
        ppls = [st["ppl"] for st in valid_stats]
        ppl_var = float(np.var(ppls)) if len(ppls) > 1 else 0.0
        mean_ppl = float(np.mean(ppls))
        
        # Episodic personal grounding in document (rescues human stories/essays)
        personal = len(re.findall(r'\b(i|my|me|mine|myself|we|our|us)\b', text, re.I))
        past_verbs = len(re.findall(r'\b(went|saw|felt|remember|started|looked|ran|decided|bought|lived|walked|told|heard|knew|thought)\b', text, re.I))
        deictic_density = (personal * 1.5 + past_verbs) / max(1, len(text.split())) * 100.0
        
        flagged_words = 0
        total_words = 0
        soft_probs = []
        
        for idx, (sent, st) in enumerate(zip(sents, sent_stats)):
            if st is None:
                soft_probs.append(0.0)
                continue
            w_count = st["word_count"]
            total_words += w_count
            ppl = st["ppl"]
            top10 = st["top10_ratio"]
            
            # Base ZeroGPT predictability signal:
            # Low PPL (<40) + High Top-10 (>0.55) indicates machine generation
            raw_ai_prob = 1.0 / (1.0 + np.exp(np.clip((ppl - 35.0) / 8.0, -40.0, 40.0)))
            if top10 >= 0.60:
                raw_ai_prob = min(1.0, raw_ai_prob + 0.15)
                
            # Inter-sentence burstiness calibration:
            # If document has high burstiness (ppl_var >= 50.0) and personal voice, damp false positives
            if ppl_var >= 40.0 and deictic_density >= 3.0:
                calibrated_prob = max(0.0, raw_ai_prob - 0.40)
            elif deictic_density >= 5.0:
                calibrated_prob = max(0.0, raw_ai_prob - 0.30)
            else:
                calibrated_prob = raw_ai_prob
                
            is_ai = calibrated_prob >= 0.50
            if is_ai:
                flagged_words += w_count
            soft_probs.append(float(calibrated_prob))
            
        fake_percentage = round((flagged_words / max(1, total_words)) * 100.0, 1)
        return {
            "fakePercentage": fake_percentage,
            "sentence_soft_probs": soft_probs,
            "ai_words": flagged_words,
            "total_words": total_words,
            "doc_mean_ppl": mean_ppl,
            "doc_ppl_var": ppl_var
        }

if __name__ == "__main__":
    teacher = CalibratedZeroGPTTeacher()
    # Test 1: Real AI text
    t_ai = "Furthermore, artificial intelligence plays a crucial role in modern educational paradigms, enhancing student engagement and optimizing curricular structures."
    res_ai = teacher.score_document(t_ai)
    print("AI Test Result:")
    print(f"  fakePercentage: {res_ai['fakePercentage']}% | Soft prob: {res_ai['sentence_soft_probs']}")

    # Test 2: Real Human simple essay (which previously fooled naive ZeroGPT)
    t_human = "My favourite sport is running. When I have free time I usually run because I feel really relaxed after that. I started this sport six years ago with my brother."
    res_human = teacher.score_document(t_human)
    print("\nHuman Test Result (Simple ESL):")
    print(f"  fakePercentage: {res_human['fakePercentage']}% | Soft prob: {res_human['sentence_soft_probs']}")
