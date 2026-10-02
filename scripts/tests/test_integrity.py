"""Known-answer tests for the integrity gates, schema validation and near-duplicate detection."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts import check_integrity as ci  # noqa: E402
from scripts.common import dedup, io_utils, schema  # noqa: E402

REGISTRY = {
    "generators": {"claude-opus-5-5": {"kind": "frontier"}},
    "human_corpora": {"wikitext-103": {"published_before": "2017-01-01"},
                      "new-forum": {"published_before": "2024-01-01"}},
    "public_ai_corpora": {"raid": {}},
}


def _ai(text, **kw):
    r = {"id": io_utils.text_id(text), "text": text, "label": "ai", "origin": "ai_raw",
         "generator_id": "claude-opus-5-5", "access_path": "claude-code-subagent:opus", "date": "2026-10-01",
         "prompt_id": "p1", "attack_id": "none", "parent_id": None, "domain": "essay", "esl": False,
         "words": io_utils.word_count(text)}
    r.update(kw)
    return r


def _human(text, **kw):
    r = _ai(text, label="human", origin="human", generator_id="human:wikitext-103",
            access_path="corpus:wikitext-103@v1", prompt_id="doc1")
    r.update(kw)
    return r


# ---------------------------------------------------------------- metric lint
def test_lint_flags_hard_coded_metrics():
    assert ci.lint_line("    teacher_tpr_at_1fpr = 0.962")
    assert ci.lint_line('        "tpr_at_1fpr": 0.9,')
    assert ci.lint_line("macro_f1 = 0.6392")
    assert ci.lint_line("ece = 0.0306  # measured")


def test_lint_ignores_parameters_and_neutral_values():
    assert not ci.lint_line("    target_fpr = 0.01")
    assert not ci.lint_line("max_fpr = 0.015")
    assert not ci.lint_line("acc = 0.0")
    assert not ci.lint_line("confidence_threshold = 0.40")
    assert not ci.lint_line("tpr = 0.9  # integrity: allow (test fixture)")
    assert not ci.lint_line("x = 0.962")


def test_active_code_passes_metric_lint():
    ok, summary, details = ci.gate_metric_lint()
    assert ok, (summary, details)


# ---------------------------------------------------------------- schema
def test_valid_records_pass():
    assert schema.validate_record(_ai("A real model wrote this text."), REGISTRY) == []
    assert schema.validate_record(_human("A person wrote this."), REGISTRY) == []


def test_synthetic_provenance_rejected():
    errs = schema.validate_record(_ai("x y z", access_path="synthetic:template"), REGISTRY)
    assert any("allow-list" in e for e in errs) and any("banned word" in e for e in errs)


def test_unregistered_generator_and_post_chatgpt_human_rejected():
    assert any("not in registry" in e for e in schema.validate_record(_ai("t", generator_id="gpt-x"), REGISTRY))
    late = _human("t", access_path="corpus:new-forum@v1", generator_id="human:new-forum")
    assert any("published before" in e for e in schema.validate_record(late, REGISTRY))


def test_attacked_row_needs_parent():
    errs = schema.validate_record(_ai("t", origin="ai_attacked", attack_id="A1_light"), REGISTRY)
    assert any("parent_id" in e for e in errs)


# ---------------------------------------------------------------- dedup
def test_near_duplicate_detected_across_groups_only():
    base = " ".join(f"word{i}" for i in range(60))
    near = base + " extra tail words here"
    other = " ".join(f"zzz{i}" for i in range(60))
    docs = [("train", "a", base), ("dev", "b", near), ("dev", "c", other), ("train", "d", base)]
    pairs = dedup.cross_group_near_duplicates(docs, threshold=0.5)
    keys = {(p[1], p[3]) for p in pairs}
    assert ("a", "b") in keys and ("b", "d") in keys
    assert not any("c" in k for k in keys)
    assert all(p[0] != p[2] for p in pairs)  # never pairs inside one group


# ---------------------------------------------------------------- metrics
def test_wilson_and_auroc_and_threshold():
    from scripts.common import metrics
    p, lo, hi = metrics.wilson(12, 15)
    assert abs(p - 0.8) < 1e-9 and 0.5 < lo < 0.6 and 0.9 < hi < 0.96          # 12/15 -> roughly 55-93%
    assert metrics.wilson(0, 30)[2] > 0.10                                       # 0/30 still allows ~11%
    assert metrics.auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert metrics.auroc([1, 1], [1, 1]) == 0.5
    neg = list(range(100))                                                        # scores 0..99
    t = metrics.threshold_at_fpr(neg, 0.01)
    assert metrics.rate_above(neg, t) == (1, 100)                                 # exactly 1% of negatives above t
    assert metrics.rate_above(list(range(50)), 100) == (0, 50)


def test_paired_bootstrap_detects_a_clear_gain():
    from scripts.common import metrics
    a = [0.9] * 80 + [0.1] * 20
    b = [0.9] * 40 + [0.1] * 60
    d, lo, hi = metrics.paired_bootstrap_diff(a, b, 0.5, 0.5, n_boot=2000)
    assert abs(d - 0.4) < 1e-9 and lo > 0.2 and hi <= 0.6


# ---------------------------------------------------------------- text normalisation
def test_normalize_text_removes_attack_characters_and_keeps_style():
    from scripts.common import textnorm
    raw = "Thе quіck​ fox “jumps” — over\r\nthe  lazy dоg."
    out = textnorm.normalize_text(raw)
    assert "​" not in out and "е" not in out and "і" not in out and "о" not in out
    assert '"jumps"' in out and "—" in out and "  " not in out
    assert textnorm.strip_markdown("# Title\n\n**bold** and _it_\n- item") == "Title\n\nbold and it\nitem"
    assert textnorm.fix_tokenization("It was n't me , she said .") == "It wasn't me, she said."


# ---------------------------------------------------------------- balance gate
def test_balance_gate_flags_dominated_genre():
    def r(label, dom, words=200, i=[0]):
        i[0] += 1
        return _ai(f"text number {i[0]} " * 20, label=label, domain=dom, words=words)
    bad = {"train": [r("ai", "news") for _ in range(250)] + [r("human", "news") for _ in range(10)]}
    assert not ci.gate_balance(bad)[0]
    good = {"train": [r("ai", "news") for _ in range(130)] + [r("human", "news") for _ in range(120)]}
    assert ci.gate_balance(good)[0]
    assert not ci.gate_balance({"train": []})[0]


# ---------------------------------------------------------------- split gate
def test_split_gate_fails_on_shared_group_and_passes_when_disjoint():
    t1, t2, t3 = ("alpha " * 30).strip(), ("beta " * 30).strip(), ("gamma " * 30).strip()
    bad = {"train": [_ai(t1 + " one", prompt_id="g1")], "dev": [_ai(t2 + " two", prompt_id="g1")],
           "locked": [_ai(t3 + " three", prompt_id="g3")]}
    assert not ci.gate_splits(bad)[0]
    good = {"train": [_ai(t1 + " one", prompt_id="g1")], "dev": [_ai(t2 + " two", prompt_id="g2")],
            "locked": [_ai(t3 + " three", prompt_id="g3")]}
    assert ci.gate_splits(good)[0]


def test_gates_fail_closed_on_empty():
    empty = {"train": [], "dev": [], "locked": []}
    assert not ci.gate_provenance(empty, REGISTRY)[0]
    assert not ci.gate_splits(empty)[0]
    assert not ci.gate_counts(empty, REGISTRY)[0]
