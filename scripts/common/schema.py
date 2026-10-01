"""
Record schema and provenance rules for the real-data pipeline.

Every row in data/corpus/*, data/splits/* and data/locked/* is a dict with these fields:

  id           16-hex id of the exact text (io_utils.text_id)
  text         the text
  label        "human" | "ai"         (AI-origin vs human-origin; hybrids are labelled by ai_share >= 0.5)
  origin       "human" | "ai_raw" | "ai_attacked" | "hybrid"
  generator_id for AI rows: the model that wrote the ORIGINAL text (also for attacked rows);
               for human rows: "human:<corpus>"
  access_path  how this exact text was obtained (allow-list below)
  date         YYYY-MM-DD the text was produced/downloaded
  prompt_id    group key: the prompt (AI) or source document (human); attacked rows inherit the parent's
  attack_id    "none" or "<family>_<variant>", e.g. "A1_medium"; family = text before "_"
  parent_id    id of the text this one was derived from (attacked/hybrid rows) else null
  domain       free-form domain tag
  esl          bool, human rows only (real learner corpora)
  words        word count

Optional: attack_tool, ai_share (0..1), markdown (bool), split ("train"|"dev"|"locked"), license.
"""

import datetime
import re

REQUIRED_FIELDS = (
    "id", "text", "label", "origin", "generator_id", "access_path", "date",
    "prompt_id", "attack_id", "parent_id", "domain", "esl", "words",
)
LABELS = ("human", "ai")
ORIGINS = ("human", "ai_raw", "ai_attacked", "hybrid")
SPLITS = ("train", "dev", "locked")

ATTACK_FAMILIES = {
    "A1": "LLM paraphrase (light/medium/heavy)",
    "A2": "iterative paraphrase (2-3 passes)",
    "A3": "humanizer prompt",
    "A4": "open-source paraphraser (CPU)",
    "A5": "back-translation",
    "A6": "hybrid / mixed authorship",
    "A7": "character-level (homoglyph/zero-width/typo)",
    "A8": "commercial humanizer (hand-run)",
    "A9": "lexical/format perturbation (synonym, article deletion, case, spelling, paragraphs)",
}
# Families that count toward target T3 (A6 is a mixed-authorship slice, A8 is optional/hand-run).
T3_FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A7")

# Allow-list of provenance access paths. Anything else (e.g. "synthetic:*") is rejected.
ACCESS_PATH_PATTERNS = [
    re.compile(r"^claude-code-subagent:(opus|sonnet|haiku|fable)$"),
    re.compile(r"^api:(anthropic|openai|openrouter|google|together|groq)$"),
    re.compile(r"^user_paste:[\w.\-]+$"),
    re.compile(r"^public_dataset:[\w./\-]+@[\w.\-]+$"),
    re.compile(r"^corpus:[\w./\-]+@[\w.\-]+$"),
    re.compile(r"^local_model:[\w./\-]+$"),
    re.compile(r"^programmatic:[\w.\-]+$"),
]
BANNED_WORDS = re.compile(r"synthetic|template|fingerprint|simulat|legacy|quarantine|mock|fake|placeholder", re.I)

# ChatGPT public release: human corpora must predate it, or be flagged verified_human in the registry.
HUMAN_CUTOFF = "2022-11-30"


# Coarse genre of each raw `domain` tag. Used to keep AI:human balanced within a genre (a domain that only one
# class has would become a shortcut) and for slicing reports.
GENRE = {
    "news": "news", "xsum": "news", "cnn": "news",
    "abstracts": "academic", "academic": "academic", "pubmed": "academic", "sci": "academic", "wiki_csai": "academic",
    "wp": "creative", "creative": "creative", "books": "creative", "poetry": "creative", "story": "creative",
    "imdb": "reviews", "yelp": "reviews",
    "eli5": "forum_qa", "reddit_eli5": "forum_qa", "tldr": "forum_qa", "cmv": "forum_qa", "open_qa": "forum_qa",
    "explain": "forum_qa", "finance": "forum_qa", "medicine": "forum_qa", "general": "forum_qa", "dialogsum": "forum_qa",
    "email": "email", "student_essay": "student_essay",
}


def genre_of(domain):
    return GENRE.get(str(domain), "other")


def attack_family(attack_id):
    if not attack_id or attack_id == "none":
        return "none"
    return str(attack_id).split("_")[0]


def access_path_ok(path):
    return isinstance(path, str) and any(p.match(path) for p in ACCESS_PATH_PATTERNS)


def validate_record(rec, registry=None):
    """Return a list of human-readable problems with one record (empty list = OK)."""
    errs = []
    for k in REQUIRED_FIELDS:
        if k not in rec:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs
    if rec["label"] not in LABELS:
        errs.append(f"bad label {rec['label']!r}")
    if rec["origin"] not in ORIGINS:
        errs.append(f"bad origin {rec['origin']!r}")
    if not isinstance(rec["text"], str) or not rec["text"].strip():
        errs.append("empty text")
    if not access_path_ok(rec["access_path"]):
        errs.append(f"access_path {rec['access_path']!r} not in allow-list")
    for k in ("generator_id", "access_path", "prompt_id", "attack_id"):
        if BANNED_WORDS.search(str(rec.get(k, ""))):
            errs.append(f"banned word in {k}: {rec.get(k)!r}")
    try:
        d = datetime.date.fromisoformat(rec["date"])
        if d > datetime.date.today():
            errs.append(f"date in the future: {rec['date']}")
    except Exception:
        errs.append(f"bad date {rec.get('date')!r}")
    if not rec["prompt_id"]:
        errs.append("empty prompt_id")
    fam = attack_family(rec["attack_id"])
    if fam != "none" and fam not in ATTACK_FAMILIES:
        errs.append(f"unknown attack family {rec['attack_id']!r}")
    if rec["label"] == "ai" and rec["origin"] == "ai_attacked" and not rec["parent_id"]:
        errs.append("attacked AI row without parent_id")
    if rec["origin"] == "human" and rec["label"] != "human":
        errs.append("origin=human but label!=human")
    if rec["label"] == "human" and rec["origin"] == "human":
        if fam == "none" and not str(rec["access_path"]).startswith("corpus:"):
            errs.append("human row must come from a registered corpus (access_path 'corpus:<name>@<rev>')")
        if fam != "none" and not rec["parent_id"]:
            errs.append("attacked human row without parent_id")
    if registry is not None:
        gens = registry.get("generators", {})
        hum = registry.get("human_corpora", {})
        pub = registry.get("public_ai_corpora", {})
        ap = str(rec["access_path"])
        if rec["label"] == "human" and rec["origin"] == "human" and fam == "none":
            name = ap[len("corpus:"):].split("@")[0] if ap.startswith("corpus:") else None
            c = hum.get(name)
            if c is None:
                errs.append(f"human corpus {name!r} not in registry")
            elif not (c.get("verified_human") or str(c.get("published_before", "9999")) <= HUMAN_CUTOFF):
                errs.append(f"human corpus {name!r} neither verified_human nor published before {HUMAN_CUTOFF}")
        if rec["label"] == "ai":
            if ap.startswith("public_dataset:"):
                name = ap[len("public_dataset:"):].split("@")[0]
                if name not in pub:
                    errs.append(f"public AI corpus {name!r} not in registry")
            elif rec["generator_id"] not in gens:
                errs.append(f"generator {rec['generator_id']!r} not in registry")
    return errs
