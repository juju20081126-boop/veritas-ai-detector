# AGENTS.md — Parallel Agent Workflow

Two agents work on this repo at the same time: **Claude Code** and **Antigravity CLI**.
Each works in its own git worktree on its own branch, so they never share a working directory.

| Agent | Worktree | Branch |
|---|---|---|
| Antigravity CLI | `AI detector/` (main checkout) | `main` |
| Claude Code | `../AI detector-claude/` | `claude/work` |

## Ownership

| Path | Owner |
|---|---|
| `backend/` | Claude |
| `models/` | Claude |
| `scripts/` | Claude |
| `data/` | Claude |
| `audit_clean.py`, `scratch_audit.py`, `scratch/` | Claude |
| `frontend/` | Antigravity |
| `notebooks/` | Antigravity |
| `samples/` | Antigravity |
| `.github/` | Antigravity |
| `README.md`, `EVAL_REPORT.md`, `RESEARCH_COMPENDIUM.md`, `MATHEMATICAL_EQUATIONS.md`, `QUILLBOT_REVERSE_ENGINEERING_PLAN.md` | Antigravity |
| `cli.py`, `run.py`, `requirements*.txt`, `.gitignore`, `*.json` results | Shared — do not edit without a note in `HANDOFF.md` first |
| `AGENTS.md`, `HANDOFF.md` | Shared — append to `HANDOFF.md` only; edit `AGENTS.md` only with the user's approval |

## Rules

1. **Never edit files outside your owned paths.** If you need a change in another agent's area, describe it in `HANDOFF.md` instead of making it.
2. **Commit small and often.** One logical change per commit, with a clear message. Don't leave large uncommitted work.
3. **Log changes in `HANDOFF.md`.** After each commit or meaningful step, add an entry: date, agent, what changed, and anything the other agent needs to know or do.
4. Stay on your own branch. Don't commit to the other agent's branch or force-push.
5. Merge into `main` only when the user says so.
