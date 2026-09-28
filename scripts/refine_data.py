"""
AI-Refinement Pipeline for Veritas AI
Implements two intermediate 4-class target categories:
1. human_ai_refined: Authentic human draft polished/edited by an LLM.
2. ai_ai_refined: AI-generated text passed through paraphrasing / stylistic restructuring.

Note: Strictly for dataset creation and adversarial robustness testing per safety guidelines.
"""

import os
import re
import json
import random
from typing import List, Dict, Any


def polish_human_text_to_refined(text: str) -> str:
    """
    Simulates LLM polishing / editing of human writing:
    - Tightens rambling clauses
    - Corrects colloquial inconsistencies
    - Introduces formal transitional words (e.g. 'furthermore', 'consequently')
    - Standardizes punctuation while preserving core human premise
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    refined_sentences = []

    transitions = ["Furthermore,", "Consequently,", "In particular,", "Moreover,", "Notably,"]
    
    for i, s in enumerate(sentences):
        s_clean = s.strip()
        if not s_clean:
            continue
        
        # Polish heuristics
        s_clean = re.sub(r'\b(a lot of|lots of)\b', 'substantial', s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(r'\b(kind of|sort of)\b', 'relatively', s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(r'\b(get|getting)\b', 'obtain', s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(r'\b(big)\b', 'significant', s_clean, flags=re.IGNORECASE)
        
        # Occasionally insert an LLM transitional marker
        if i == 1 and random.random() < 0.6 and not any(s_clean.startswith(t) for t in transitions):
            s_clean = f"{random.choice(transitions)} {s_clean[0].lower() + s_clean[1:]}"
            
        refined_sentences.append(s_clean)

    return " ".join(refined_sentences)


def paraphrase_ai_text_to_refined(text: str) -> str:
    """
    Simulates paraphrasing / restructuring of AI text:
    - Inverts dependent/independent clauses
    - Replaces hallmark LLM buzzwords (delve -> explore, multifaceted -> diverse)
    - Adds natural contractions
    - Injects minor colloquial sentence breaks to disrupt uniform perplexity
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paraphrased_sentences = []

    replacements = {
        r'\bdelve into\b': 'look closely at',
        r'\bmultifaceted\b': 'complex and varied',
        r'\bpivotal role\b': 'key function',
        r'\brich tapestry\b': 'interconnected dynamic',
        r'\bstands as a testament to\b': 'clearly demonstrates',
        r'\bfoster(?:ing)?\b': 'encouraging',
        r'\bleverage(?:ing)?\b': 'utilizing',
        r'\bconsequently\b': 'as a direct result',
        r'\bfundamentally\b': 'essentially',
        r'\bin conclusion\b': 'overall'
    }

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        
        for pat, rep in replacements.items():
            s_clean = re.sub(pat, rep, s_clean, flags=re.IGNORECASE)

        # Clause inversion on sentences starting with 'While' or 'Although'
        if s_clean.startswith("While ") and "," in s_clean:
            parts = s_clean[6:].split(",", 1)
            if len(parts) == 2:
                s_clean = f"{parts[1].strip().capitalize()}, although {parts[0].strip().lower()}"

        paraphrased_sentences.append(s_clean)

    return " ".join(paraphrased_sentences)


def create_refined_dataset(input_jsonl: str, output_jsonl: str):
    """
    Reads existing human and AI samples, produces balanced refined datasets:
    - human -> human_ai_refined
    - ai_generated -> ai_ai_refined
    """
    refined_records = []
    
    if not os.path.exists(input_jsonl):
        print(f"[Refine] Input file {input_jsonl} not found.")
        return

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            orig_text = item["text"]
            orig_label = item.get("label", "")

            if orig_label == "human":
                polished_text = polish_human_text_to_refined(orig_text)
                refined_records.append({
                    "text": polished_text,
                    "label": "human_ai_refined",
                    "original_label": "human",
                    "refinement_type": "llm_polished_human",
                    "source": item.get("source", "curated"),
                    "word_count": len(polished_text.split())
                })
            elif orig_label == "ai_generated":
                paraphrased_text = paraphrase_ai_text_to_refined(orig_text)
                refined_records.append({
                    "text": paraphrased_text,
                    "label": "ai_ai_refined",
                    "original_label": "ai_generated",
                    "refinement_type": "paraphrased_ai",
                    "source": item.get("source", "curated"),
                    "word_count": len(paraphrased_text.split())
                })

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in refined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Refine] Generated {len(refined_records)} refined samples -> {output_jsonl}")


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/frontier_generated.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/refined_samples.jsonl"
    create_refined_dataset(inp, out)
