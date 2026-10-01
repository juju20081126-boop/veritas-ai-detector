#!/usr/bin/env python
"""
Bookkeeping for subagent batches (frontier generation and LLM attacks).

  python scripts/corpus/manifest_tool.py mark frontier opus__locked__r3_000=a8ddac3b9c8526de4 ...   # mark running + record agent id
  python scripts/corpus/manifest_tool.py status frontier|attacked
  python scripts/corpus/manifest_tool.py verify          # resolved model ids from the subagent transcripts

`verify` reads each recorded agent transcript (.claude/projects/.../subagents/agent-<id>.jsonl) and extracts the
`"model":"..."` values of its assistant messages, then writes the resolved id into data/corpus/registry.json
(generators.<id>.verified_model_id / verification) if every transcript of that alias agrees.
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scripts.common import io_utils  # noqa: E402
from scripts.corpus import registry  # noqa: E402

PATHS = {"frontier": os.path.join(io_utils.DATA, "corpus", "frontier", "_manifest.json"),
         "attacked": os.path.join(io_utils.DATA, "corpus", "attacked", "_manifest.json")}
AGENTS = {"frontier": os.path.join(io_utils.DATA, "corpus", "frontier", "_agents.json"),
          "attacked": os.path.join(io_utils.DATA, "corpus", "attacked", "_agents.json")}
TRANSCRIPT_ROOT = os.path.join(os.path.expanduser("~"), ".claude", "projects")
ALIAS_TO_GEN = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-4-5"}


def _load(p, default):
    return json.load(open(p, encoding="utf-8-sig")) if os.path.exists(p) else default


def mark(kind, pairs):
    m, ag = _load(PATHS[kind], {"batches": []}), _load(AGENTS[kind], {})
    ids = {b["batch_id"]: b for b in m["batches"]}
    for pair in pairs:
        bid, aid = pair.split("=")
        if bid not in ids:
            print("unknown batch", bid)
            continue
        ids[bid]["status"], ids[bid]["agent_id"] = "running", aid
        ag[bid] = {"agent_id": aid, "alias": ids[bid].get("worker") or bid.split("__")[0]}
    json.dump(m, open(PATHS[kind], "w", encoding="utf-8"), indent=1)
    json.dump(ag, open(AGENTS[kind], "w", encoding="utf-8"), indent=1)
    status(kind)


def status(kind):
    m = _load(PATHS[kind], {"batches": []})
    c = Counter(b["status"] for b in m["batches"])
    print(kind, dict(c))


def verify():
    seen = defaultdict(Counter)
    n = 0
    for kind in AGENTS:
        for bid, a in _load(AGENTS[kind], {}).items():
            hits = glob.glob(os.path.join(TRANSCRIPT_ROOT, "*", "*", "subagents", f"agent-{a['agent_id']}.jsonl"))
            if not hits:
                continue
            txt = open(hits[0], encoding="utf-8").read()
            for mid in re.findall(r'"model":"([^"]+)"', txt):
                seen[a["alias"]][mid] += 1
            n += 1
    reg = registry.load()
    for alias, ctr in seen.items():
        gen = ALIAS_TO_GEN.get(alias)
        print(alias, dict(ctr))
        if gen and len(ctr) == 1:
            mid = next(iter(ctr))
            reg["generators"][gen]["verified_model_id"] = mid
            reg["generators"][gen]["verification"] = f"message.model in {sum(ctr.values())} assistant messages of {n} subagent transcripts (all equal)"
    registry.save(reg)
    print("transcripts checked:", n)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "mark":
        mark(sys.argv[2], sys.argv[3:])
    elif cmd == "status":
        status(sys.argv[2])
    elif cmd == "verify":
        verify()
