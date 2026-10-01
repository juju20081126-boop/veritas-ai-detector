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
