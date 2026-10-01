# HANDOFF

Append-only log. Newest entries at the bottom.

## 2026-09-29 — Claude Code — frontend/ ownership swap and redesign

**Ownership change (approved by the user in chat):** Claude now owns `frontend/` for the redesign.
Antigravity: please don't edit `frontend/` until this branch (`claude/work`) is merged or the user says otherwise.
`AGENTS.md` has NOT been changed yet; the user still needs to approve updating the ownership table.

**What changed (branch `claude/work`, files in `frontend/`):**
- `index.html`: restructured into a floating "window" with header, hero, sample chips, editor card and results card. All element IDs that `app.js` uses are unchanged. Copy rewritten in plain sentence case.
- `style.css`: full rewrite. Light theme is a white panel with an ink halftone. Dark theme is a black window with a blue glow. Geist and Geist Mono fonts. Class colors follow an "ink density" ramp (darker means more AI).
- `halftone.js` (new): draws the hero dot field on a canvas and re-settles it after each analysis. `app.js` dispatches a `veritas:result` event with `{ aiShare }`.
- `app.js`: small edits only (button markup, copy strings, the `veritas:result` event, inspector heading class). No API or logic changes.

**Needs from Antigravity:** nothing right now. If you have pending `frontend/` edits, commit them on `main` and tell the user so the merge can be planned.

**Preview:** run from this worktree with `python run.py --port 8001 --no-browser` (use the Hermes venv Python, which has uvicorn).

## 2026-09-29 — Claude Code — frontend redesign, round 2

Rebuilt `frontend/` (branch `claude/work`) as a monochrome split card in a Times-style serif (Newsreader).
- `index.html`, `style.css`, `app.js`: rewritten. Same API calls; no tabs, theme toggle or sample chips; errors show inline instead of alerts.
- `halftone.js`: ordered-dither halftone; `backdrop.js` (new): painted forest backdrop.
- Antigravity: still please stay out of `frontend/` until this is merged.

## 2026-10-01 — Claude Code — frontend redesign, round 3 ("ink and highlighter")

Rebuilt `frontend/index.html`, `style.css`, `app.js` on `claude/work`, starting from Antigravity's original feature set (user asked for a more professional, premium version of the original).
- Look: document-on-a-desk. Human sentences stay plain ink; AI sentences get a highlighter mark (pink = AI, amber = paraphrased AI, teal underline = AI-refined human). Literata for document text, Schibsted Grotesk for UI. Light + dark themes (defaults to system).
- All original features kept: samples + 30-sample benchmark, upload/drag-drop/paste, uncertainty cutoff, 4-class breakdown, sentence inspector, writing signals, engine telemetry, copy summary.
- Added: per-sentence probability bars, prev/next + arrow-key stepping, flagged AI vocabulary chips (from `stylometrics.discourse_punctuation.detected_ai_samples`), live engine status from `/api/health`, inline errors instead of `alert()`.
- Fixed: uploads now fill the editor (`/api/upload` returns no `document_text`, so text is rebuilt from `sentences`); verdict icon now maps the real badge names (`badge-danger/orange/warning/success`); sentence text rendered with `textContent` (was innerHTML).
- `halftone.js` and `backdrop.js` from round 2 are no longer referenced; safe to delete.
- No API or backend changes.
