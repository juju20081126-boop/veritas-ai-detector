/**
 * Veritas — frontend logic.
 * Sends text to /api/detect (or a file to /api/upload), then fills the report:
 * score, 4-class breakdown, verdict, marked-up sentences, inspector and signals.
 */

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);

  const textInput = $("textInput");
  const heatmapViewer = $("heatmapViewer");
  const btnModeEdit = $("btnModeEdit");
  const btnModeHeatmap = $("btnModeHeatmap");
  const heatmapSentCountBadge = $("heatmapSentCountBadge");
  const btnAnalyze = $("btnAnalyze");
  const btnClear = $("btnClear");
  const btnPaste = $("btnPaste");
  const btnUpload = $("btnUpload");
  const fileUploadInput = $("fileUploadInput");
  const editorDropZone = $("editorDropZone");
  const dragDropOverlay = $("dragDropOverlay");
  const sampleSelect = $("sampleSelect");
  const qbComparisonGroup = $("qbComparisonGroup");
  const thresholdInput = $("thresholdInput");
  const themeToggleBtn = $("themeToggleBtn");
  const btnCopyReport = $("btnCopyReport");
  const formError = $("formError");

  const wordCountLabel = $("wordCountLabel");
  const sentCountLabel = $("sentCountLabel");
  const wordGuideBadge = $("wordGuideBadge");

  const resultsPlaceholder = $("resultsPlaceholder");
  const activeResultsContent = $("activeResultsContent");

  const inspectorTitle = $("inspectorTitle");
  const inspectorContent = $("inspectorContent");
  const btnPrevSentence = $("btnPrevSentence");
  const btnNextSentence = $("btnNextSentence");

  const engineStatus = $("engineStatus");
  const engineStatusText = $("engineStatusText");

  let currentAnalysisData = null;
  let currentSentences = [];
  let activeSentence = -1;
  let archetypeSamples = {};
  let comparisonSheetData = [];

  // The backend's 4 classes, in spectrum order (most AI first).
  const CLASSES = [
    { key: "ai_generated", label: "AI-generated", short: "AI", tone: "ai" },
    { key: "ai_ai_refined", label: "AI-generated, then paraphrased", short: "paraphrased AI", tone: "ai-refined" },
    { key: "human_ai_refined", label: "Human-written, AI-refined", short: "AI-refined", tone: "human-refined" },
    { key: "human", label: "Human-written", short: "human", tone: "human" },
  ];
  const PROB_KEYS = {
    "AI-generated": "ai",
    "AI-generated & AI-refined": "ai-refined",
    "Human-written & AI-refined": "human-refined",
    "Human-written": "human",
  };
  const TONE_FROM_CLASS_KEY = {
    ai_generated: "ai", ai_ai_refined: "ai-refined",
    human_ai_refined: "human-refined", human: "human", uncertain: "uncertain",
  };
  const TONE_FROM_BADGE = {
    "badge-danger": "ai", "badge-ai": "ai",
    "badge-orange": "ai-refined", "badge-ai-refined": "ai-refined",
    "badge-warning": "human-refined", "badge-human-refined": "human-refined",
    "badge-success": "human", "badge-human": "human",
  };
  const toneVar = (tone) => `var(--c-${tone})`;
  // Show the backend's class names in the same words the rest of the page uses.
  const DISPLAY_LABEL = {
    "AI-generated & AI-refined": "AI-generated, then paraphrased",
    "Human-written & AI-refined": "Human-written, AI-refined",
  };
  const displayLabel = (label) => DISPLAY_LABEL[label] || label;
  const SHORT_LABEL = {
    "AI-generated": "AI",
    "AI-generated & AI-refined": "Paraphrased AI",
    "Human-written & AI-refined": "AI-refined human",
    "Human-written": "Human",
  };

  // ---------- theme ----------

  function syncThemeButton() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    themeToggleBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  syncThemeButton();
  themeToggleBtn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("veritas_theme", next); } catch (e) { /* storage unavailable */ }
    syncThemeButton();
  });

  // ---------- engine status ----------

  async function checkEngine() {
    try {
      const resp = await fetch("/api/health");
      if (!resp.ok) throw new Error();
      const h = await resp.json();
      engineStatus.classList.add("is-online");
      engineStatusText.textContent = `Offline engine ready, ${h.cpu_threads || 2} CPU threads`;
      if (h.cpu_threads) $("telemetryThreads").textContent = h.cpu_threads;
      if (h.process_ram_mb) $("telemetryMemory").textContent = `${Math.round(h.process_ram_mb)} MB of ${Number(h.target_ram_cap_mb || 1500).toLocaleString()} MB`;
    } catch (e) {
      engineStatus.classList.add("is-offline");
      engineStatusText.textContent = "Engine not responding";
    }
  }

  // ---------- errors ----------

  function showError(msg) {
    formError.textContent = msg;
    formError.hidden = false;
  }
  function clearError() {
    formError.hidden = true;
    formError.textContent = "";
  }

  // ---------- counters ----------

  function updateTextCounters() {
    const text = textInput.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const sents = text.trim() ? text.split(/[.!?]+/).filter((s) => s.trim().length > 0).length : 0;

    wordCountLabel.textContent = `${words.toLocaleString()} ${words === 1 ? "word" : "words"}`;
    sentCountLabel.textContent = `${sents.toLocaleString()} ${sents === 1 ? "sentence" : "sentences"}`;

    if (words === 0) {
      wordGuideBadge.className = "guide";
      wordGuideBadge.textContent = "80–2,000 words recommended";
    } else if (words < 80) {
      wordGuideBadge.className = "guide warn";
      wordGuideBadge.textContent = `Add ${80 - words} more for a reliable result`;
    } else if (words > 2000) {
      wordGuideBadge.className = "guide warn";
      wordGuideBadge.textContent = "Only the first 2,000 words are checked";
    } else {
      wordGuideBadge.className = "guide ok";
      wordGuideBadge.textContent = "Good length";
    }
  }
  textInput.addEventListener("input", () => {
    updateTextCounters();
    clearError();
    markLoadedChip(null);
  });

  // ---------- view modes ----------

  function switchToEditMode() {
    btnModeEdit.classList.add("active");
    btnModeHeatmap.classList.remove("active");
    btnModeEdit.setAttribute("aria-selected", "true");
    btnModeHeatmap.setAttribute("aria-selected", "false");
    textInput.hidden = false;
    heatmapViewer.hidden = true;
  }

  function switchToHeatmapMode() {
    if (!currentAnalysisData) return;
    btnModeHeatmap.classList.add("active");
    btnModeEdit.classList.remove("active");
    btnModeHeatmap.setAttribute("aria-selected", "true");
    btnModeEdit.setAttribute("aria-selected", "false");
    textInput.hidden = true;
    heatmapViewer.hidden = false;
  }

  btnModeEdit.addEventListener("click", () => { switchToEditMode(); textInput.focus(); });
  btnModeHeatmap.addEventListener("click", switchToHeatmapMode);

  function resetResults() {
    currentAnalysisData = null;
    currentSentences = [];
    activeSentence = -1;
    resultsPlaceholder.hidden = false;
    activeResultsContent.hidden = true;
    btnModeHeatmap.disabled = true;
    heatmapSentCountBadge.hidden = true;
    heatmapViewer.innerHTML = "";
    switchToEditMode();
  }

  // ---------- input actions ----------

  btnClear.addEventListener("click", () => {
    textInput.value = "";
    updateTextCounters();
    clearError();
    markLoadedChip(null);
    resetResults();
    textInput.focus();
  });

  btnPaste.addEventListener("click", async () => {
    try {
      const clipText = await navigator.clipboard.readText();
      if (!clipText) return;
      textInput.value = clipText;
      updateTextCounters();
      clearError();
      switchToEditMode();
    } catch (e) {
      showError("Your browser blocked clipboard access. Press Ctrl+V in the text box instead.");
      textInput.focus();
    }
  });

  btnUpload.addEventListener("click", () => fileUploadInput.click());
  fileUploadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) await handleFileUpload(file);
    fileUploadInput.value = "";
  });

  // Drag a file anywhere onto the sheet.
  let dragDepth = 0;
  editorDropZone.addEventListener("dragenter", (e) => {
    if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
    e.preventDefault();
    dragDepth++;
    dragDropOverlay.hidden = false;
  });
  editorDropZone.addEventListener("dragover", (e) => {
    if (!dragDropOverlay.hidden) e.preventDefault();
  });
  editorDropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dragDropOverlay.hidden = true;
  });
  editorDropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDepth = 0;
    dragDropOverlay.hidden = true;
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) await handleFileUpload(file);
  });

  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runAnalysis();
    }
  });
  btnAnalyze.addEventListener("click", runAnalysis);

  // ---------- busy state ----------

  const analyzeLabel = btnAnalyze.querySelector(".btn-label");
  function setBusy(busy, label) {
    btnAnalyze.disabled = busy;
    btnUpload.disabled = busy;
    btnAnalyze.setAttribute("aria-busy", busy ? "true" : "false");
    const spinner = btnAnalyze.querySelector(".spinner");
    if (busy && !spinner) {
      const s = document.createElement("span");
      s.className = "spinner";
      s.setAttribute("aria-hidden", "true");
      btnAnalyze.prepend(s);
    } else if (!busy && spinner) {
      spinner.remove();
    }
    analyzeLabel.textContent = busy ? label : "Check text";
  }

  async function readError(resp, fallback) {
    try {
      const err = await resp.json();
      return err.detail || fallback;
    } catch (e) {
      return fallback;
    }
  }

  function getThreshold() {
    const v = parseFloat(thresholdInput.value);
    if (Number.isNaN(v)) return 0.40;
    return Math.min(0.70, Math.max(0.30, v));
  }

  async function runAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
      showError("Paste or type some text first. 80 words or more gives a reliable result.");
      switchToEditMode();
      textInput.focus();
      return;
    }
    clearError();
    setBusy(true, "Checking");
    try {
      const resp = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, confidence_threshold: getThreshold() }),
      });
      if (!resp.ok) throw new Error(await readError(resp, "The engine couldn't check this text."));
      const data = await resp.json();
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message === "Failed to fetch"
        ? "Can't reach the Veritas engine. Make sure run.py is still running, then try again."
        : err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleFileUpload(file) {
    const okType = /\.(pdf|docx|txt)$/i.test(file.name);
    if (!okType) {
      showError(`"${file.name}" isn't supported. Upload a PDF, Word (.docx) or .txt file.`);
      return;
    }
    clearError();
    const formData = new FormData();
    formData.append("file", file);
    setBusy(true, "Reading file");
    try {
      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      if (!resp.ok) throw new Error(await readError(resp, "That file couldn't be read."));
      const data = await resp.json();
      // The upload endpoint returns sentences rather than the raw text, so rebuild it.
      const text = data.document_text || data.text || (data.sentences || []).map((s) => s.text).join(" ");
      textInput.value = text;
      updateTextCounters();
      markLoadedChip(null);
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message === "Failed to fetch"
        ? "Can't reach the Veritas engine. Make sure run.py is still running, then try again."
        : err.message);
    } finally {
      setBusy(false);
    }
  }

  // ---------- rendering ----------

  function fmtPct(v) {
    const n = Number(v) || 0;
    return `${n % 1 === 0 ? n.toFixed(0) : n.toFixed(1)}%`;
  }

  function verdictTone(summary) {
    if (summary.is_uncertain) return "uncertain";
    return TONE_FROM_BADGE[summary.badge] || TONE_FROM_BADGE[summary.quillbot_headline_class] || "uncertain";
  }

  const ICONS = {
    human: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "human-refined": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
    "ai-refined": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-15.7-6L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 15.7 6L21 16"/><path d="M16 21h5v-5"/></svg>',
    ai: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 8V4"/><circle cx="12" cy="3.5" r=".5"/><path d="M9 13v1.5M15 13v1.5"/></svg>',
    uncertain: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9.2 9a3 3 0 0 1 5.6 1c0 2-3 2.5-3 4.5"/><path d="M12 18.5h.01"/></svg>',
  };

  function renderAnalysisResults(data) {
    const summary = data.summary || {};
    const pcts = data.percentages || {};
    const sentences = data.sentences || [];
    currentSentences = sentences;

    resultsPlaceholder.hidden = true;
    activeResultsContent.hidden = false;
    btnModeHeatmap.disabled = sentences.length === 0;

    // Count the sentences that carry a mark.
    const marked = sentences.filter((s) => s.highlight_class === "highlight-ai" || s.highlight_class === "highlight-ai-refined").length;
    heatmapSentCountBadge.textContent = marked;
    heatmapSentCountBadge.hidden = marked === 0;

    // Score
    const aiPct = Number(summary.quillbot_ai_pct != null ? summary.quillbot_ai_pct : ((pcts.ai_generated || 0) + (pcts.ai_ai_refined || 0)));
    const scoreEl = $("qbHeadlineBanner");
    // Color the score by whichever AI class dominates; human blue when nothing is flagged.
    scoreEl.dataset.tone = aiPct <= 0 ? "human"
      : (Number(pcts.ai_ai_refined) || 0) > (Number(pcts.ai_generated) || 0) ? "ai-refined" : "ai";
    $("scoreNumber").textContent = aiPct % 1 === 0 ? aiPct.toFixed(0) : aiPct.toFixed(1);
    $("qbHeadlineText").textContent = "of the text is likely AI";

    // Spectrum + rows
    const barIds = { ai_generated: "stackBarAI", ai_ai_refined: "stackBarAIRefined", human_ai_refined: "stackBarHumanRefined", human: "stackBarHuman" };
    const pctIds = { ai_generated: "pctAIGen", ai_ai_refined: "pctAIRefined", human_ai_refined: "pctHumanRefined", human: "pctHuman" };
    const ariaParts = [];
    CLASSES.forEach((c) => {
      const v = Number(pcts[c.key]) || 0;
      $(barIds[c.key]).style.width = `${v}%`;
      const pctEl = $(pctIds[c.key]);
      pctEl.textContent = fmtPct(v);
      pctEl.parentElement.classList.toggle("is-zero", v === 0);
      ariaParts.push(`${c.label} ${fmtPct(v)}`);
    });
    $("spectrumBar").setAttribute("aria-label", ariaParts.join(", "));

    // Verdict
    const tone = verdictTone(summary);
    const iconBox = $("verdictIconBox");
    iconBox.dataset.tone = tone;
    iconBox.innerHTML = ICONS[tone] || ICONS.uncertain;
    $("verdictTitle").textContent = summary.is_uncertain ? "Not enough signal to decide" : displayLabel(summary.verdict || "—");
    $("confidenceTag").textContent = summary.confidence_pct != null ? `${summary.confidence_pct}% confidence` : "";
    $("verdictDescription").textContent = summary.verdict_description || "";
    $("uncertainAlertBanner").hidden = !summary.is_uncertain;
    const lw = $("lengthWarning");
    lw.hidden = !summary.length_warning;
    lw.textContent = summary.length_warning || "";

    const counts = { ai: 0, "ai-refined": 0, "human-refined": 0, human: 0 };
    sentences.forEach((s) => {
      const t = TONE_FROM_CLASS_KEY[s.class_key];
      if (t in counts) counts[t]++;
    });
    const total = sentences.length;
    const flaggedCount = counts.ai + counts["ai-refined"];
    $("sentenceCountsSummary").textContent = total
      ? `${flaggedCount} of ${total} sentences marked as AI`
      : "";

    // Signals
    const m = data.mathematical_equations || {};
    const st = data.stylometrics || {};
    const num = (v, d) => (v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));
    $("mathValBurstiness").textContent = num(m.syntactic_burstiness_b, 3);
    $("mathValRichness").textContent = num(m.lexical_richness_omega, 3);
    $("mathValDiscourse").textContent = m.discourse_polarity_phi == null ? "—" : (m.discourse_polarity_phi > 0 ? "+" : "") + num(m.discourse_polarity_phi, 3);
    $("mathValBinoculars").textContent = num(m.binoculars_ratio_r, 2);
    $("mathValAffinity").textContent = m.authorial_affinity_lambda == null ? "—" : (m.authorial_affinity_lambda > 0 ? "+" : "") + num(m.authorial_affinity_lambda, 3);
    const rd = st.readability || {};
    $("valGrade").textContent = num(rd.flesch_kincaid_grade, 1);
    $("valWps").textContent = num(rd.words_per_sentence, 1);
    const dp = st.discourse_punctuation || {};
    $("valContractions").textContent = num(dp.contraction_rate, 1);

    const words = dp.detected_ai_samples || [];
    $("flaggedWords").hidden = words.length === 0;
    const list = $("flaggedWordsList");
    list.innerHTML = "";
    words.forEach((w) => {
      const span = document.createElement("span");
      span.textContent = w;
      list.appendChild(span);
    });

    $("telemetryLatency").textContent = summary.elapsed_seconds != null ? `${Number(summary.elapsed_seconds).toFixed(2)} s` : "—";

    // Document
    renderHeatmapSpans(sentences);
    switchToHeatmapMode();
    heatmapViewer.scrollTop = 0;
  }

  function renderHeatmapSpans(sentences) {
    heatmapViewer.innerHTML = "";
    heatmapViewer.classList.remove("is-revealing");
    const inner = document.createElement("div");
    inner.className = "doc-inner";

    sentences.forEach((s, idx) => {
      const span = document.createElement("span");
      span.className = `sentence-span ${s.highlight_class || "highlight-human"}`;
      span.textContent = s.text;
      span.dataset.index = idx;
      span.tabIndex = 0;
      span.setAttribute("role", "button");
      span.setAttribute("aria-label", `Sentence ${idx + 1}: ${displayLabel(s.class_label)}`);
      span.style.setProperty("--i", Math.min(idx, 30));
      span.addEventListener("click", () => selectSentence(idx));
      span.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectSentence(idx);
        }
      });
      inner.appendChild(span);
      inner.appendChild(document.createTextNode(" "));
    });

    heatmapViewer.appendChild(inner);
    // Restart the sweep animation on each new result.
    void heatmapViewer.offsetWidth;
    heatmapViewer.classList.add("is-revealing");

    if (sentences.length > 0) {
      // Start the inspector on the first marked sentence, if any.
      const firstMarked = sentences.findIndex((s) => s.highlight_class === "highlight-ai" || s.highlight_class === "highlight-ai-refined");
      selectSentence(firstMarked >= 0 ? firstMarked : 0, { scroll: false });
    } else {
      inspectorTitle.textContent = "No sentences";
      inspectorContent.textContent = "";
    }
  }

  function selectSentence(idx, opts = {}) {
    if (idx < 0 || idx >= currentSentences.length) return;
    activeSentence = idx;
    heatmapViewer.querySelectorAll(".sentence-span.active").forEach((el) => el.classList.remove("active"));
    const span = heatmapViewer.querySelector(`.sentence-span[data-index="${idx}"]`);
    if (span) {
      span.classList.add("active");
      if (opts.scroll !== false && !heatmapViewer.hidden) span.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    renderInspectorDetails(currentSentences[idx], idx);
  }

  btnPrevSentence.addEventListener("click", () => selectSentence(activeSentence - 1));
  btnNextSentence.addEventListener("click", () => selectSentence(activeSentence + 1));

  function renderInspectorDetails(sent, idx) {
    const tone = TONE_FROM_CLASS_KEY[sent.class_key] || "uncertain";
    inspectorTitle.textContent = `Sentence ${idx + 1} of ${currentSentences.length}`;
    btnPrevSentence.disabled = idx <= 0;
    btnNextSentence.disabled = idx >= currentSentences.length - 1;

    inspectorContent.innerHTML = "";
    inspectorContent.style.setProperty("--tone", toneVar(tone));

    const quote = document.createElement("blockquote");
    quote.className = "insp-quote";
    quote.textContent = sent.text;
    inspectorContent.appendChild(quote);

    const cls = document.createElement("div");
    cls.className = "insp-class";
    const strong = document.createElement("strong");
    strong.textContent = displayLabel(sent.class_label);
    const meta = document.createElement("span");
    meta.textContent = `${sent.ai_likelihood_pct}% AI likelihood`;
    cls.append(strong, meta);
    inspectorContent.appendChild(cls);

    const probs = sent.probabilities || {};
    const dl = document.createElement("dl");
    dl.className = "probs";
    ["AI-generated", "AI-generated & AI-refined", "Human-written & AI-refined", "Human-written"].forEach((k) => {
      if (!(k in probs)) return;
      const v = Math.max(0, Math.min(1, Number(probs[k]) || 0));
      const row = document.createElement("div");
      row.className = "prob";
      const dt = document.createElement("dt");
      dt.textContent = SHORT_LABEL[k];
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = `${(v * 100).toFixed(1)}%`;
      fill.style.background = toneVar(PROB_KEYS[k]);
      bar.appendChild(fill);
      const dd = document.createElement("dd");
      dd.textContent = `${Math.round(v * 100)}%`;
      row.append(dt, bar, dd);
      dl.appendChild(row);
    });
    inspectorContent.appendChild(dl);

    const reasons = sent.reasons && sent.reasons.length ? sent.reasons : ["Natural phrasing and varied vocabulary."];
    const ul = document.createElement("ul");
    ul.className = "reasons";
    reasons.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      ul.appendChild(li);
    });
    inspectorContent.appendChild(ul);
  }

  // Arrow keys step through sentences while the marked-up view is open.
  document.addEventListener("keydown", (e) => {
    if (heatmapViewer.hidden || !currentSentences.length) return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input" || tag === "select") return;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      if (activeSentence < currentSentences.length - 1) { e.preventDefault(); selectSentence(activeSentence + 1); }
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      if (activeSentence > 0) { e.preventDefault(); selectSentence(activeSentence - 1); }
    }
  });

  // ---------- copy summary ----------

  const copyLabel = btnCopyReport.querySelector(".btn-label");
  btnCopyReport.addEventListener("click", async () => {
    if (!currentAnalysisData) return;
    const s = currentAnalysisData.summary || {};
    const p = currentAnalysisData.percentages || {};
    const report = [
      "Veritas AI detection summary",
      "",
      `${s.quillbot_headline || ""}`,
      `Verdict: ${s.is_uncertain ? "Withheld (below uncertainty cutoff)" : s.verdict} (${s.confidence_pct}% confidence)`,
      "",
      "Breakdown",
      ...CLASSES.map((c) => `  ${c.label}: ${fmtPct(p[c.key])}`),
      "",
      `Words: ${s.word_count != null ? s.word_count : "—"}, sentences: ${s.sentence_count != null ? s.sentence_count : "—"}`,
      `Checked offline in ${s.elapsed_seconds != null ? Number(s.elapsed_seconds).toFixed(2) : "—"} s`,
    ].join("\n");
    try {
      await navigator.clipboard.writeText(report);
      copyLabel.textContent = "Copied";
    } catch (e) {
      copyLabel.textContent = "Copy blocked by browser";
    }
    setTimeout(() => { copyLabel.textContent = "Copy summary"; }, 1800);
  });

  // ---------- samples ----------

  function markLoadedChip(chip) {
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("is-loaded", c === chip));
    if (chip) sampleSelect.value = "";
  }

  function loadSampleText(text) {
    textInput.value = text;
    updateTextCounters();
    clearError();
    resetResults();
  }

  document.querySelectorAll(".chip[data-sample]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const sample = archetypeSamples[chip.dataset.sample];
      if (!sample) {
        showError("Samples are still loading. Try again in a moment.");
        return;
      }
      loadSampleText(sample.text);
      markLoadedChip(chip);
    });
  });

  sampleSelect.addEventListener("change", (e) => {
    const found = comparisonSheetData.find((s) => s.id === e.target.value);
    if (found) {
      loadSampleText(found.text);
      markLoadedChip(null);
      sampleSelect.value = found.id;
    }
  });

  async function loadSamples() {
    try {
      const resp = await fetch("/api/samples");
      if (resp.ok) archetypeSamples = await resp.json();
    } catch (e) { /* chips will report it on click */ }

    try {
      const compResp = await fetch("/api/comparison-sheet");
      if (!compResp.ok) return;
      comparisonSheetData = await compResp.json();
      comparisonSheetData.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = `${s.id}  ${s.expected_class}, ${s.word_count} words`;
        qbComparisonGroup.appendChild(opt);
      });
    } catch (e) { /* benchmark menu stays empty */ }
  }

  updateTextCounters();
  checkEngine();
  loadSamples();
});
