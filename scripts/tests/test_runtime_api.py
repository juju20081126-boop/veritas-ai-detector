"""Runtime engine: backward-compatible response schema in both detector modes, the frontier decision rule, input hygiene,
and fail-closed onboarding. Frontier tests are skipped until models/frontier/ has been exported."""

import math
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
# Modes that decide with a dev-calibrated threshold, limited to those whose runtime config has been written.
THRESHOLD_MODES = [m for m, cfg in runtime_engine.RUNTIME_CONFIGS.items() if os.path.exists(cfg)]


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


def test_threshold_modes_registered():
    assert {"frontier", "multi_teacher", "tfidf", "ensemble"} <= set(runtime_engine.DETECTOR_MODES)
    assert set(runtime_engine.RUNTIME_CONFIGS) == set(runtime_engine.DETECTOR_MODES) - {"shipped"}


@pytest.mark.parametrize("mode", THRESHOLD_MODES)
def test_threshold_mode_schema_and_rule(mode):
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode=mode)
    r = e.analyze_text(TEXT)
    _check_schema(r)
    d = r["detector"]
    assert d["mode"] == mode and d["threshold"] is not None
    assert d["threshold"] == round(float(e.runtime_config["threshold"]), 4)
    p = r["calibrated_probabilities"]
    assert abs(p["AI-generated"] + p["AI-generated & AI-refined"] - d["ai_score"]) < 1e-3
    is_ai_verdict = r["summary"]["verdict"] in ("AI-generated", "AI-generated & AI-refined")
    assert is_ai_verdict == (d["ai_score"] >= d["threshold"])


@pytest.mark.skipif("multi_teacher" not in THRESHOLD_MODES, reason="models/multi_teacher_distilled runtime config not written yet")
def test_multi_teacher_reports_its_own_model():
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="multi_teacher")
    d = e.analyze_text(TEXT)["detector"]
    assert d["model"] == "multi_teacher_distilled"
    assert "multi_teacher" in d["threshold_rule"]


@pytest.mark.skipif("tfidf" not in THRESHOLD_MODES, reason="models/tfidf_v2 not trained yet (scripts/train_tfidf.py)")
def test_tfidf_score_is_relative_to_its_decision_threshold():
    """tfidf ai_score = sigmoid(decision - decision_threshold), so 0.5 is exactly the dev clean-human 1%-FPR cut."""
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="tfidf")
    d = e.analyze_text(TEXT)["detector"]
    assert d["model"] == "tfidf_v2" and d["threshold"] == 0.5
    dec = float(e.tfidf.decision([e._frontier_prep(TEXT)])[0])
    expected = 1.0 / (1.0 + math.exp(-(dec - float(e.runtime_config["decision_threshold"]))))
    assert abs(d["ai_score"] - expected) < 1e-3


def test_tfidf_runtime_matches_sklearn(tmp_path):
    """The numpy-only TF-IDF scorer reproduces sklearn's TfidfVectorizer + LogisticRegression decision_function."""
    pytest.importorskip("sklearn")
    from backend.tfidf_detector import TfidfDetector
    from scripts.train_tfidf import export_model, fit_model
    texts = ["The committee met on Tuesday.  It  discussed the budget!", "Honestly, I can't believe it's already October...",
             "Furthermore, it is important to note that robust frameworks matter.", "lol idk, maybe? we'll see tmrw",
             "In conclusion, these findings underscore a pivotal shift.", "My grandmother's recipe uses café-style milk.",
             "Overall, the results highlight key considerations.", "We drove to the lake and the dog jumped in again"] * 3
    labels = [0, 0, 1, 0, 1, 0, 1, 0] * 3
    model = fit_model(texts, labels, char_min_df=1, word_min_df=1)
    path = str(tmp_path / "m.json.gz")
    export_model(model, path)
    probe = ["Robust frameworks underscore the budget.", "naïve  café\nnewline\ttab", "", "?!", "It's important; it's pivotal."]
    vc, vw, lr = model
    from scipy.sparse import hstack
    want = lr.decision_function(hstack([vc.transform(probe), vw.transform(probe)]).tocsr())
    got = TfidfDetector(path).decision(probe)
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-6


def test_ensemble_or_rule():
    """Ensemble AI score >= 0.5 exactly when some component reaches its cut; outputs are distributions, monotone in each input."""
    import numpy as np
    e = runtime_engine.QuillBotDetectorEngine.__new__(runtime_engine.QuillBotDetectorEngine)
    e.runtime_config = {"components": {"a": 0.6, "b": 0.98}}
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), size=(2, 500))
    # rows: a exactly at its cut | no AI mass anywhere | a below, b exactly at its cut | a all AI
    P[0, :4] = [[0.4, 0, 0, 0.6], [1, 0, 0, 0], [0.5, 0, 0, 0.5], [0, 0, 0, 1]]
    P[1, :4] = [[0.98, 0.02, 0, 0], [1, 0, 0, 0], [0.02, 0, 0, 0.98], [1, 0, 0, 0]]
    out = e._ensemble_probs(P)
    assert np.all(out >= 0) and np.allclose(out.sum(1), 1.0)
    ai = out[:, 2] + out[:, 3]
    reached = ((P[..., 2] + P[..., 3]) >= np.array([0.6, 0.98])[:, None]).any(0)
    assert np.array_equal(ai >= 0.5 - 1e-9, reached)
    assert abs(ai[0] - 0.5) < 1e-9 and ai[1] < 0.5 and abs(ai[2] - 0.5) < 1e-6 and ai[3] > 0.99
    bumped = P.copy()
    bumped[1, :, 3] += 0.05
    bumped[1] /= bumped[1].sum(-1, keepdims=True)
    assert np.all(e._ensemble_probs(bumped)[:, 2:].sum(1) >= ai - 1e-12)


@pytest.mark.skipif("ensemble" not in THRESHOLD_MODES, reason="models/ensemble not written yet (scripts/write_ensemble_config.py)")
def test_ensemble_reports_components_and_follows_them():
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode="ensemble")
    solo = {m: runtime_engine.QuillBotDetectorEngine(threads=1, mode=m) for m in e.runtime_config["components"]}
    human = "We drove up to the lake on Saturday, and of course the dog jumped straight in before we'd even parked. " * 3
    for text in (TEXT, human):
        d = e.analyze_text(text)["detector"]
        assert d["threshold"] == 0.5 and list(d["components"]) == list(e.runtime_config["components"])
        for m, c in d["components"].items():
            assert abs(c["ai_score"] - solo[m].analyze_text(text)["detector"]["ai_score"]) < 1e-4
            assert c["threshold"] == round(float(e.runtime_config["components"][m]), 4)
        assert (d["ai_score"] >= 0.5) == any(c["flagged"] for c in d["components"].values())


@pytest.mark.parametrize("mode", THRESHOLD_MODES)
def test_threshold_mode_ignores_obfuscation(mode):
    """Zero-width characters and homoglyphs (attack A7) must not move the score."""
    e = runtime_engine.QuillBotDetectorEngine(threads=1, mode=mode)
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
