"""data/corpus/registry.json: the single list of registered generators, human corpora and public AI corpora."""

import datetime
import json
import os

from scripts.common.io_utils import DATA

REGISTRY_PATH = os.path.join(DATA, "corpus", "registry.json")

DEFAULT = {
    "_note": "Written by scripts/corpus/*. check_integrity.py G1 rejects rows whose source is not registered here.",
    "generators": {
        "claude-opus-5-5": {
            "kind": "frontier", "vendor": "anthropic", "family": "claude-5", "api_id": "claude-opus-5-5",
            "access_paths": ["claude-code-subagent:opus"], "verified_model_id": None, "verification": "pending",
        },
        "claude-sonnet-5-5": {
            "kind": "frontier", "vendor": "anthropic", "family": "claude-5", "api_id": "claude-sonnet-5-5",
            "access_paths": ["claude-code-subagent:sonnet"], "verified_model_id": None, "verification": "pending",
        },
        "gpt-6-astra": {
            "kind": "frontier", "vendor": "openai", "family": "openai-2026", "api_id": "openai/gpt-6-astra (OpenRouter slug; verify)",
            "released": "2026-09-03", "access_paths": ["api:openrouter", "api:openai", "user_paste:chatgpt"],
            "verified_model_id": None, "verification": "no access yet (no API key set in this environment)",
        },
        "claude-haiku-4-5": {
            "kind": "tool", "vendor": "anthropic", "family": "claude-4", "api_id": "claude-haiku-4-5-20251001",
            "access_paths": ["claude-code-subagent:haiku"], "verified_model_id": None, "verification": "pending",
            "note": "used as paraphraser/humanizer (attack tool) and optionally as extra generator",
        },
    },
    "human_corpora": {},
    "public_ai_corpora": {},
}


def load():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT))


def save(reg):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False, sort_keys=False)


def register(section, name, entry):
    """Add or update one entry (idempotent). Adds today's date as `registered`."""
    reg = load()
    entry = dict(entry)
    entry.setdefault("registered", datetime.date.today().isoformat())
    reg.setdefault(section, {})[name] = {**reg.get(section, {}).get(name, {}), **entry}
    save(reg)
    return reg
