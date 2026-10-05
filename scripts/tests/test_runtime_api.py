"""Runtime engine: backward-compatible response schema in both detector modes, the frontier decision rule, input hygiene,
and fail-closed onboarding. Frontier tests are skipped until models/frontier/ has been exported."""

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from backend import runtime_engine  # noqa: E402
from backend.textnorm import normalize_text, strip_markdown  # noqa: E402

TEXT = ("The committee met on Tuesday to discuss the budget. Several members raised concerns about rising costs, "
        "and the chair promised a revised proposal by next month. ") * 4
TOP_KEYS = {"summary", "calibrated_probabilities", "percentages", "quillbot_breakdown", "sentences", "stylometrics",
            "mathematical_equations"}
SUMMARY_KEYS = {"verdict", "verdict_description", "badge", "is_uncertain", "confidence", "confidence_pct", "quillbot_headline",
                "quillbot_headline_class", "quillbot_ai_pct", "quillbot_human_pct", "word_count", "character_count",
                "sentence_count", "length_warning", "elapsed_seconds"}
HAS_FRONTIER = os.path.exists(os.path.join(runtime_engine.FRONTIER_DIR, "frontier_config.json"))


def _check_schema(r):
    assert TOP_KEYS <= set(r)
    assert SUMMARY_KEYS <= set(r["summary"])
    assert set(r["calibrated_probabilities"]) == set(runtime_engine.CLASS_NAMES)
    assert abs(sum(r["calibrated_probabilities"].values()) - 1.0) < 1e-2
    assert r["sentences"] and {"text", "class_label", "probabilities", "reasons"} <= set(r["sentences"][0])


def test_default_mode_is_frontier(monkeypatch):
    monkeypatch.delenv("VERITAS_DETECTOR", raising=False)
    e = runtime_engine.QuillBotDetectorEngine(threads=1)
    assert e.mode == "frontier"
    r = e.analyze_text(TEXT)
    _check_schema(r)
    assert r["detector"]["mode"] == "frontier"


def test_legacy_mode_still_available():
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="shipped")
    r = e.analyze_text(TEXT)
    _check_schema(r)
    assert r["detector"]["mode"] == "shipped"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        runtime_engine.QuillBotDetectorEngine(threads=1, mode="magic")


@pytest.mark.skipif(not HAS_FRONTIER, reason="models/frontier not exported yet")
def test_frontier_schema_and_threshold_rule():
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="frontier")
    r = e.analyze_text(TEXT)
    _check_schema(r)
    d = r["detector"]
    assert d["mode"] == "frontier" and d["threshold"] is not None
    p = r["calibrated_probabilities"]
    assert abs(p["AI-generated"] + p["AI-generated & AI-refined"] - d["ai_score"]) < 1e-3
    is_ai_verdict = r["summary"]["verdict"] in ("AI-generated", "AI-generated & AI-refined")
    assert is_ai_verdict == (d["ai_score"] >= d["threshold"])


@pytest.mark.skipif(not HAS_FRONTIER, reason="models/frontier not exported yet")
def test_frontier_ignores_obfuscation():
    """Zero-width characters and homoglyphs (attack A7) must not move the frontier score."""
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="frontier")
    attacked = TEXT.replace("e", "е", 7).replace(" ", " ​", 9)
    assert abs(e.analyze_text(TEXT)["detector"]["ai_score"] - e.analyze_text(attacked)["detector"]["ai_score"]) < 1e-3


def test_textnorm_hygiene():
    assert normalize_text("pаper​ “quoted”") == 'paper "quoted"'
    assert strip_markdown("## Title\n\n**bold** and - item") == "Title\n\nbold and - item"


def test_onboard_fails_closed_without_keys():
    env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    script = os.path.join(REPO, "scripts", "onboard_model.py")
    for extra in ([], ["--backend", "openrouter"], ["--backend", "anthropic"]):
        p = subprocess.run([sys.executable, script, "--model", "gpt-6-astra", *extra], env=env, capture_output=True, text=True,
                           timeout=120)
        assert p.returncode != 0
        assert "FAIL-CLOSED" in p.stderr
