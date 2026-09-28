"""
Stylometric and Forensic Linguistic Analysis Module
Computes lexical diversity, syntactic uniformity, readability indexes,
and statistical distribution metrics indicative of AI vs. human generation.
"""

import math
import re
from typing import List, Dict, Any
import numpy as np


def count_syllables(word: str) -> int:
    """Estimates syllable count using heuristic vowel cluster counting."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Remove common non-pronounced endings
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    vowels = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(vowels))


def compute_readability(text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
    """
    Computes standard academic readability indices:
    - Flesch Reading Ease
    - Flesch-Kincaid Grade Level
    - Gunning Fog Index
    - Automated Readability Index (ARI)
    """
    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))
    
    syllables = sum(count_syllables(w) for w in words)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    characters = sum(len(w) for w in words)
    
    words_per_sentence = num_words / num_sentences
    syllables_per_word = syllables / num_words
    complex_word_ratio = (complex_words / num_words) * 100
    
    # 1. Flesch Reading Ease: 206.835 - 1.015*(words/sent) - 84.6*(syllables/word)
    flesch_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    flesch_ease = max(0.0, min(100.0, flesch_ease))
    
    # 2. Flesch-Kincaid Grade Level: 0.39*(words/sent) + 11.8*(syllables/word) - 15.59
    fk_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    fk_grade = max(1.0, min(20.0, fk_grade))
    
    # 3. Gunning Fog Index: 0.4 * ( (words/sent) + 100*(complex_words/words) )
    gunning_fog = 0.4 * (words_per_sentence + complex_word_ratio)
    gunning_fog = max(1.0, min(25.0, gunning_fog))
    
    # 4. Automated Readability Index (ARI)
    ari = (4.71 * (characters / num_words)) + (0.5 * words_per_sentence) - 21.43
    ari = max(1.0, min(22.0, ari))
    
    return {
        "flesch_reading_ease": round(float(flesch_ease), 1),
        "flesch_kincaid_grade": round(float(fk_grade), 1),
        "gunning_fog": round(float(gunning_fog), 1),
        "ari": round(float(ari), 1)
    }


def compute_lexical_diversity(words: List[str]) -> Dict[str, float]:
    """
    Computes vocabulary richness and diversity:
    - Type-Token Ratio (TTR)
    - Root TTR (Guiraud's R = V / sqrt(N))
    - Hapax Legomena Ratio (% of words occurring only once)
    - Yule's K Characteristic (text length independent richness)
    """
    num_tokens = len(words)
    if num_tokens == 0:
        return {"ttr": 0.0, "root_ttr": 0.0, "hapax_ratio": 0.0, "yule_k": 0.0}
        
    lower_words = [w.lower() for w in words]
    vocab = set(lower_words)
    num_types = len(vocab)
    
    # Type-Token Ratio
    ttr = num_types / num_tokens
    
    # Root TTR (Guiraud's Index)
    root_ttr = num_types / math.sqrt(num_tokens) if num_tokens > 0 else 0.0
    
    # Frequency spectrum
    freq_counts: Dict[str, int] = {}
    for w in lower_words:
        freq_counts[w] = freq_counts.get(w, 0) + 1
        
    hapax_count = sum(1 for count in freq_counts.values() if count == 1)
    hapax_ratio = (hapax_count / num_types) if num_types > 0 else 0.0
    
    # Yule's Characteristic K
    m1 = num_tokens
    m2 = sum(c ** 2 for c in freq_counts.values())
    yule_k = 10000.0 * (m2 - m1) / (m1 ** 2) if m1 > 1 else 0.0
    
    return {
        "ttr": round(float(ttr), 3),
        "root_ttr": round(float(root_ttr), 2),
        "hapax_ratio": round(float(hapax_ratio), 3),
        "yule_k": round(float(yule_k), 2)
    }


def compute_syntactic_variance(sentences: List[str]) -> Dict[str, float]:
    """
    Analyzes sentence length distributions and syntactic uniformity.
    LLMs exhibit an unnaturally uniform sentence length (low variance & low CV).
    Humans naturally mix punchy clauses with lengthy compound sentences.
    """
    if not sentences:
        return {
            "mean_length": 0.0,
            "std_length": 0.0,
            "cv_length": 0.0,
            "uniformity_score": 0.0
        }
        
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()]
    if not lengths:
        lengths = [1]
        
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    cv_len = (std_len / mean_len) if mean_len > 0 else 0.0
    
    # Uniformity score: Higher means more robotically uniform (typical CV for AI is 0.20-0.35)
    # Human CV is typically > 0.45 - 0.70
    if cv_len < 0.25:
        uniformity_score = 0.90  # Extremely uniform (Strong AI indicator)
    elif cv_len < 0.38:
        uniformity_score = 0.70  # Moderately uniform
    elif cv_len < 0.50:
        uniformity_score = 0.40  # Balanced
    else:
        uniformity_score = 0.15  # Natural human bursty variance
        
    return {
        "mean_length": round(mean_len, 1),
        "std_length": round(std_len, 1),
        "cv_length": round(float(cv_len), 3),
        "uniformity_score": round(float(uniformity_score), 2)
    }


def analyze_stylometrics(text: str, sentences: List[str]) -> Dict[str, Any]:
    """Runs the full stylometric forensic pipeline."""
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)
    
    readability = compute_readability(text, words, sentences)
    lexical = compute_lexical_diversity(words)
    syntax = compute_syntactic_variance(sentences)
    
    # Calculate a composite stylometric AI probability component (0.0 to 1.0)
    # AI correlates with: high uniformity, moderate-to-low Hapax ratio, specific FK grades (11-14)
    stylometric_ai_score = (
        syntax["uniformity_score"] * 0.55 +
        (1.0 - min(1.0, lexical["hapax_ratio"] * 1.4)) * 0.45
    )
    
    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len(sentences),
        "readability": readability,
        "lexical_diversity": lexical,
        "syntax_variance": syntax,
        "stylometric_ai_score": round(float(stylometric_ai_score), 3)
    }
