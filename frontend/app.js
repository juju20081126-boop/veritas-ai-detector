/**
 * Veritas — frontend logic.
 * Sends text to /api/detect (or a file to /api/upload), then fills the report:
 * score, 4-class breakdown, verdict, marked-up sentences, inspector and signals.
 * Includes complete i18n support for English and Traditional Chinese.
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
  const langToggleBtn = $("langToggleBtn");
  const langToggleLabel = $("langToggleLabel");
  const btnHelpShortcuts = $("btnHelpShortcuts");
  const btnCloseShortcuts = $("btnCloseShortcuts");
  const shortcutsModal = $("shortcutsModal");
  const btnCopyReport = $("btnCopyReport");
  const btnDownloadJson = $("btnDownloadJson");
  const btnDownloadCsv = $("btnDownloadCsv");
  const btnResetReport = $("btnResetReport");
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

  // ---------- i18n language management ----------

  let currentLang = "en";
  try {
    const saved = localStorage.getItem("veritas_lang");
    if (saved === "zh-TW" || saved === "en") {
      currentLang = saved;
    }
  } catch (e) { /* storage unavailable */ }

  const t = (key, replacements) => {
    if (window.VeritasI18N && window.VeritasI18N.getTranslation) {
      return window.VeritasI18N.getTranslation(currentLang, key, replacements);
    }
    return key;
  };

  // Below this many words, results get a short-text sensitivity notice.
  const SHORT_TEXT_WORDS = 150;
  let lastWordCount = null;
  function renderShortTextNotice() {
    const el = $("shortTextNotice");
    if (!el) return;
    const short = lastWordCount != null && lastWordCount < SHORT_TEXT_WORDS;
    el.hidden = !short;
    el.textContent = short ? t("short_text_notice", { count: lastWordCount }) : "";
  }

  // The backend's 4 classes, in spectrum order (most AI first).
  const getClasses = () => [
    { key: "ai_generated", label: t("cls_ai_generated"), short: t("cls_short_ai"), tone: "ai" },
    { key: "ai_ai_refined", label: t("cls_ai_refined"), short: t("cls_short_ai_refined"), tone: "ai-refined" },
    { key: "human_ai_refined", label: t("cls_human_refined"), short: t("cls_short_human_refined"), tone: "human-refined" },
    { key: "human", label: t("cls_human"), short: t("cls_short_human"), tone: "human" },
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

  function displayLabel(label) {
    if (currentLang === "zh-TW") {
      if (label === "AI-generated") return t("cls_ai_generated");
      if (label === "AI-generated & AI-refined" || label === "AI-generated, then paraphrased") return t("cls_ai_refined");
      if (label === "Human-written & AI-refined" || label === "Human-written, AI-refined") return t("cls_human_refined");
      if (label === "Human-written") return t("cls_human");
      if (label === "Uncertain") return t("verdict_uncertain");
      return label;
    }
    if (label === "AI-generated & AI-refined") return "AI-generated, then paraphrased";
    if (label === "Human-written & AI-refined") return "Human-written, AI-refined";
    return label;
  }

  function shortLabel(k) {
    if (currentLang === "zh-TW") {
      if (k === "AI-generated") return t("cls_short_ai");
      if (k === "AI-generated & AI-refined") return t("cls_short_ai_refined");
      if (k === "Human-written & AI-refined") return t("cls_short_human_refined");
      if (k === "Human-written") return t("cls_short_human");
    }
    const SHORT_LABEL = {
      "AI-generated": "AI",
      "AI-generated & AI-refined": "Paraphrased AI",
      "Human-written & AI-refined": "AI-refined human",
      "Human-written": "Human",
    };
    return SHORT_LABEL[k] || k;
  }

  // ---------- theme ----------

  function syncThemeButton() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    themeToggleBtn.setAttribute("aria-label", dark ? t("theme_toggle_light") : t("theme_toggle_dark"));
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
      engineStatus.classList.remove("is-offline");
      engineStatusText.textContent = t("engine_online", { threads: h.cpu_threads || 2 });
      engineStatus.setAttribute("title", t("engine_title"));
      if (h.cpu_threads) $("telemetryThreads").textContent = h.cpu_threads;
      if (h.process_ram_mb) $("telemetryMemory").textContent = `${Math.round(h.process_ram_mb)} MB of ${Number(h.target_ram_cap_mb || 1500).toLocaleString()} MB`;
      const offlineNotice = $("offlineNotice");
      if (offlineNotice) offlineNotice.hidden = true;
    } catch (e) {
      engineStatus.classList.add("is-offline");
      engineStatus.classList.remove("is-online");
      engineStatusText.textContent = t("engine_offline");
      engineStatus.setAttribute("title", t("engine_offline_title"));
      const offlineNotice = $("offlineNotice");
      if (offlineNotice) {
        offlineNotice.hidden = false;
        const bannerTitle = $("offlineBannerTitle");
        const bannerDesc = $("offlineBannerDesc");
        if (bannerTitle) bannerTitle.textContent = t("offline_banner_title");
        if (bannerDesc) bannerDesc.innerHTML = t("offline_banner_desc");
      }
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

    wordCountLabel.textContent = `${words.toLocaleString()} ${words === 1 ? t("word_single") : t("words_plural")}`;
    sentCountLabel.textContent = `${sents.toLocaleString()} ${sents === 1 ? t("sent_single") : t("sents_plural")}`;

    if (words === 0) {
      wordGuideBadge.className = "guide";
      wordGuideBadge.textContent = t("guide_rec");
    } else if (words < 80) {
      wordGuideBadge.className = "guide warn";
      wordGuideBadge.textContent = t("guide_add", { count: 80 - words });
    } else if (words < SHORT_TEXT_WORDS) {
      wordGuideBadge.className = "guide warn";
      wordGuideBadge.textContent = t("guide_short");
    } else if (words > 2000) {
      wordGuideBadge.className = "guide warn";
      wordGuideBadge.textContent = t("guide_over");
    } else {
      wordGuideBadge.className = "guide ok";
      wordGuideBadge.textContent = t("guide_ok");
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
    lastWordCount = null;
    renderShortTextNotice();
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
      showError(t("err_clipboard_blocked"));
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
    analyzeLabel.textContent = busy ? label : t("btn_check_text");
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
      showError(t("err_empty_input"));
      switchToEditMode();
      textInput.focus();
      return;
    }
    clearError();
    setBusy(true, t("btn_checking"));
    try {
      const resp = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, confidence_threshold: getThreshold() }),
      });
      if (!resp.ok) throw new Error(await readError(resp, t("err_engine_fallback")));
      const data = await resp.json();
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message === "Failed to fetch"
        ? t("err_cant_reach_engine")
        : err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleFileUpload(file) {
    const okType = /\.(pdf|docx|txt)$/i.test(file.name);
    if (!okType) {
      showError(t("err_unsupported_file", { name: file.name }));
      return;
    }
    clearError();
    const formData = new FormData();
    formData.append("file", file);
    setBusy(true, t("btn_reading_file"));
    try {
      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      if (!resp.ok) throw new Error(await readError(resp, t("err_file_fallback")));
      const data = await resp.json();
      const text = data.document_text || data.text || (data.sentences || []).map((s) => s.text).join(" ");
      textInput.value = text;
      updateTextCounters();
      markLoadedChip(null);
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message === "Failed to fetch"
        ? t("err_cant_reach_engine")
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
    scoreEl.dataset.tone = aiPct <= 0 ? "human"
      : (Number(pcts.ai_ai_refined) || 0) > (Number(pcts.ai_generated) || 0) ? "ai-refined" : "ai";
    $("scoreNumber").textContent = aiPct % 1 === 0 ? aiPct.toFixed(0) : aiPct.toFixed(1);
    $("qbHeadlineText").textContent = t("score_caption");

    // Spectrum + rows
    const barIds = { ai_generated: "stackBarAI", ai_ai_refined: "stackBarAIRefined", human_ai_refined: "stackBarHumanRefined", human: "stackBarHuman" };
    const pctIds = { ai_generated: "pctAIGen", ai_ai_refined: "pctAIRefined", human_ai_refined: "pctHumanRefined", human: "pctHuman" };
    const ariaParts = [];
    getClasses().forEach((c) => {
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
    $("verdictTitle").textContent = summary.is_uncertain ? t("verdict_uncertain") : displayLabel(summary.verdict || "—");
    $("confidenceTag").textContent = summary.confidence_pct != null ? `${summary.confidence_pct}% ${t("confidence_suffix")}` : "";
    $("verdictDescription").textContent = summary.verdict_description || "";
    $("uncertainAlertBanner").hidden = !summary.is_uncertain;
    const lw = $("lengthWarning");
    lw.hidden = !summary.length_warning;
    lw.textContent = summary.length_warning || "";
    lastWordCount = summary.word_count != null ? Number(summary.word_count) : null;
    renderShortTextNotice();

    const counts = { ai: 0, "ai-refined": 0, "human-refined": 0, human: 0 };
    sentences.forEach((s) => {
      const curTone = TONE_FROM_CLASS_KEY[s.class_key];
      if (curTone in counts) counts[curTone]++;
    });
    const total = sentences.length;
    const flaggedCount = counts.ai + counts["ai-refined"];
    $("sentenceCountsSummary").textContent = total
      ? t("sentence_counts_summary", { flagged: flaggedCount, total: total })
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
    if ($("telemetryDetector")) {
      const d = data.detector || {};
      $("telemetryDetector").textContent = d.model ? `${d.model} (${d.mode || "frontier"})` : "Frontier INT8";
    }

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
    void heatmapViewer.offsetWidth;
    heatmapViewer.classList.add("is-revealing");

    if (sentences.length > 0) {
      const firstMarked = sentences.findIndex((s) => s.highlight_class === "highlight-ai" || s.highlight_class === "highlight-ai-refined");
      selectSentence(firstMarked >= 0 ? firstMarked : 0, { scroll: false });
    } else {
      inspectorTitle.textContent = t("inspector_no_sentences");
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
    inspectorTitle.textContent = t("inspector_title_sentence", { idx: idx + 1, total: currentSentences.length });
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
    meta.textContent = `${sent.ai_likelihood_pct}% ${t("ai_likelihood_suffix")}`;
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
      dt.textContent = shortLabel(k);
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

  // ---------- keyboard shortcuts & modal ----------

  function openShortcutsModal() {
    if (shortcutsModal) shortcutsModal.hidden = false;
  }
  function closeShortcutsModal() {
    if (shortcutsModal) shortcutsModal.hidden = true;
  }

  if (btnHelpShortcuts) btnHelpShortcuts.addEventListener("click", openShortcutsModal);
  if (btnCloseShortcuts) btnCloseShortcuts.addEventListener("click", closeShortcutsModal);
  if (shortcutsModal) {
    shortcutsModal.addEventListener("click", (e) => {
      if (e.target === shortcutsModal) closeShortcutsModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const inInput = tag === "textarea" || tag === "input" || tag === "select";

    if (e.key === "Escape") {
      closeShortcutsModal();
      return;
    }

    if ((e.key === "?" || (e.key === "/" && e.shiftKey)) && !inInput) {
      e.preventDefault();
      if (shortcutsModal && !shortcutsModal.hidden) {
        closeShortcutsModal();
      } else {
        openShortcutsModal();
      }
      return;
    }

    if (heatmapViewer.hidden || !currentSentences.length || inInput) return;
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
      t("export_summary_title"),
      "",
      `${s.quillbot_headline || ""}`,
      `Verdict: ${s.is_uncertain ? t("export_verdict_withheld") : displayLabel(s.verdict)} (${s.confidence_pct}% ${t("confidence_suffix")})`,
      "",
      t("export_breakdown"),
      ...getClasses().map((c) => `  ${c.label}: ${fmtPct(p[c.key])}`),
      "",
      `${t("export_words")}: ${s.word_count != null ? s.word_count : "—"}, ${t("export_sentences")}: ${s.sentence_count != null ? s.sentence_count : "—"}`,
      `${t("export_checked_offline")} ${s.elapsed_seconds != null ? Number(s.elapsed_seconds).toFixed(2) : "—"} s`,
    ].join("\n");
    try {
      await navigator.clipboard.writeText(report);
      copyLabel.textContent = t("btn_copied");
    } catch (e) {
      copyLabel.textContent = t("btn_copy_blocked");
    }
    setTimeout(() => { copyLabel.textContent = t("btn_copy_summary"); }, 1800);
  });

  // ---------- download results & clear report ----------

  function downloadClientBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 150);
  }

  if (btnDownloadJson) {
    btnDownloadJson.addEventListener("click", () => {
      if (!currentAnalysisData) return;
      const fn = (currentAnalysisData.summary && currentAnalysisData.summary.filename)
        ? currentAnalysisData.summary.filename.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_")
        : "veritas_analysis";
      downloadClientBlob(JSON.stringify(currentAnalysisData, null, 2), `${fn}.json`, "application/json");
    });
  }

  if (btnDownloadCsv) {
    btnDownloadCsv.addEventListener("click", () => {
      if (!currentAnalysisData || !currentAnalysisData.sentences) return;
      const escapeCsv = (str) => `"${String(str != null ? str : "").replace(/"/g, '""')}"`;
      const header = ["index", "text", "dominant_class", "confidence_pct", "class_label", "ai_likelihood_pct"];
      const rows = currentAnalysisData.sentences.map((s) => [
        s.index,
        escapeCsv(s.text),
        escapeCsv(s.class_key != null ? s.class_key : s.class_label),
        (Number(s.confidence || 0) * 100).toFixed(1),
        escapeCsv(s.class_label),
        Number(s.ai_likelihood_pct || 0).toFixed(1)
      ].join(","));
      const csvContent = [header.join(","), ...rows].join("\r\n");
      const fn = (currentAnalysisData.summary && currentAnalysisData.summary.filename)
        ? currentAnalysisData.summary.filename.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_")
        : "veritas_analysis";
      downloadClientBlob(csvContent, `${fn}_sentences.csv`, "text/csv;charset=utf-8;");
    });
  }

  if (btnResetReport) {
    btnResetReport.addEventListener("click", () => {
      textInput.value = "";
      updateTextCounters();
      clearError();
      markLoadedChip(null);
      resetResults();
      textInput.focus();
    });
  }

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
        showError(t("err_samples_loading"));
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

  // ---------- Language application & toggle ----------

  function applyLanguage(lang) {
    currentLang = lang;
    document.documentElement.setAttribute("lang", lang === "zh-TW" ? "zh-TW" : "en");

    if (langToggleLabel) langToggleLabel.textContent = t("lang_toggle_btn_label");
    if (langToggleBtn) {
      langToggleBtn.setAttribute("aria-label", t("lang_toggle_aria"));
      langToggleBtn.setAttribute("title", t("lang_toggle_title"));
    }
    if (btnHelpShortcuts) {
      btnHelpShortcuts.setAttribute("aria-label", t("shortcuts_btn_aria"));
      btnHelpShortcuts.setAttribute("title", t("shortcuts_btn_title"));
    }
    syncThemeButton();

    if (engineStatus) engineStatus.setAttribute("title", t("engine_title"));
    if (engineStatus.classList.contains("is-online")) {
      const th = $("telemetryThreads") ? $("telemetryThreads").textContent : 2;
      engineStatusText.textContent = t("engine_online", { threads: th || 2 });
    } else if (engineStatus.classList.contains("is-offline")) {
      engineStatusText.textContent = t("engine_offline");
      engineStatus.setAttribute("title", t("engine_offline_title"));
      const offlineNotice = $("offlineNotice");
      if (offlineNotice && !offlineNotice.hidden) {
        const bannerTitle = $("offlineBannerTitle");
        const bannerDesc = $("offlineBannerDesc");
        if (bannerTitle) bannerTitle.textContent = t("offline_banner_title");
        if (bannerDesc) bannerDesc.innerHTML = t("offline_banner_desc");
      }
    } else {
      engineStatusText.textContent = t("engine_connecting");
    }

    const introH1 = $("introTitle");
    if (introH1) introH1.textContent = t("intro_title");
    const introP = $("introDesc");
    if (introP) introP.textContent = t("intro_desc");
    const sLabel = $("samplesLabel");
    if (sLabel) sLabel.textContent = t("samples_label");

    const chipKeys = {
      ai_pure: "chip_ai_pure",
      ai_refined_ai: "chip_ai_refined_ai",
      human_refined_ai: "chip_human_refined_ai",
      human_pure: "chip_human_pure",
      human_esl: "chip_human_esl",
    };
    Object.keys(chipKeys).forEach((sk) => {
      const chip = document.querySelector(`.chip[data-sample="${sk}"]`);
      if (chip && chip.childNodes.length > 1) {
        chip.childNodes[1].textContent = t(chipKeys[sk]);
      }
    });

    if (sampleSelect && sampleSelect.options.length > 0) {
      sampleSelect.options[0].textContent = t("sample_select_default");
    }
    if (qbComparisonGroup) {
      qbComparisonGroup.label = t("sample_select_optgroup");
    }

    if (btnModeEdit) btnModeEdit.textContent = t("tab_write");
    if (btnModeHeatmap && btnModeHeatmap.childNodes.length > 0) {
      btnModeHeatmap.childNodes[0].textContent = t("tab_highlights") + " ";
    }
    const upSpan = btnUpload ? btnUpload.querySelector("span") : null;
    if (upSpan) upSpan.textContent = t("btn_upload");
    const pasteSpan = btnPaste ? btnPaste.querySelector("span") : null;
    if (pasteSpan) pasteSpan.textContent = t("btn_paste");

    if (textInput) textInput.placeholder = t("text_placeholder");
    const dropStrong = document.querySelector("#dragDropOverlay strong");
    if (dropStrong) dropStrong.textContent = t("drop_title");
    const dropSpan = document.querySelector("#dragDropOverlay span");
    if (dropSpan) dropSpan.textContent = t("drop_desc");

    const cutoffSpan = document.querySelector(".cutoff span");
    if (cutoffSpan) cutoffSpan.textContent = t("uncertainty_cutoff");
    const cutoffLabel = document.querySelector(".cutoff");
    if (cutoffLabel) cutoffLabel.setAttribute("title", t("uncertainty_title"));

    if (btnClear) btnClear.textContent = t("btn_clear");
    if (analyzeLabel && !btnAnalyze.disabled) analyzeLabel.textContent = t("btn_check_text");

    const keyH2 = document.querySelector("#resultsPlaceholder h2");
    if (keyH2) keyH2.textContent = t("key_title");
    const keyLead = document.querySelector("#resultsPlaceholder .key-lead");
    if (keyLead) keyLead.textContent = t("key_lead");

    const smHuman = document.querySelector("#resultsPlaceholder .mk-human");
    if (smHuman) smHuman.textContent = t("mk_human_label");
    const smHumanRef = document.querySelector("#resultsPlaceholder .mk-human-refined");
    if (smHumanRef) smHumanRef.textContent = t("mk_human_refined_label");
    const smAiRef = document.querySelector("#resultsPlaceholder .mk-ai-refined");
    if (smAiRef) smAiRef.textContent = t("mk_ai_refined_label");
    const smAi = document.querySelector("#resultsPlaceholder .mk-ai");
    if (smAi) smAi.textContent = t("mk_ai_label");

    const keyTexts = document.querySelectorAll("#resultsPlaceholder .key-list li .key-text");
    if (keyTexts.length >= 4) {
      keyTexts[0].innerHTML = `<strong>${t("mk_human_title")}</strong> ${t("mk_human_desc")}`;
      keyTexts[1].innerHTML = `<strong>${t("mk_human_refined_title")}</strong> ${t("mk_human_refined_desc")}`;
      keyTexts[2].innerHTML = `<strong>${t("mk_ai_refined_title")}</strong> ${t("mk_ai_refined_desc")}`;
      keyTexts[3].innerHTML = `<strong>${t("mk_ai_title")}</strong> ${t("mk_ai_desc")}`;
    }
    const keyNote = document.querySelector("#resultsPlaceholder .key-note");
    if (keyNote) keyNote.textContent = t("key_note");

    const qbHead = $("qbHeadlineText");
    if (qbHead) qbHead.textContent = t("score_caption");

    const classDts = document.querySelectorAll("#activeResultsContent dl.classes dt");
    if (classDts.length >= 4) {
      if (classDts[0].childNodes.length > 1) classDts[0].childNodes[1].textContent = t("cls_ai_generated");
      if (classDts[1].childNodes.length > 1) classDts[1].childNodes[1].textContent = t("cls_ai_refined");
      if (classDts[2].childNodes.length > 1) classDts[2].childNodes[1].textContent = t("cls_human_refined");
      if (classDts[3].childNodes.length > 1) classDts[3].childNodes[1].textContent = t("cls_human");
    }

    const discl = $("verdictSignalDisclaimer");
    if (discl) discl.textContent = t("verdict_disclaimer");
    renderShortTextNotice();
    const uncertAlert = $("uncertainAlertBanner");
    if (uncertAlert) uncertAlert.innerHTML = `<strong>${t("uncertain_alert_lead")}</strong> ${t("uncertain_alert")}`;

    const mathSumSpan = document.querySelector("#mathEquationsAccordion summary span");
    if (mathSumSpan) mathSumSpan.textContent = t("signals_title");
    const mathH4 = document.querySelector("#flaggedWords h4");
    if (mathH4) mathH4.textContent = t("flagged_words_title");

    const sigDts = document.querySelectorAll("#mathEquationsAccordion dl.metrics dt");
    if (sigDts.length >= 8) {
      sigDts[0].innerHTML = `${t("sig_burstiness")} <small>${t("sig_burstiness_sub")}</small>`;
      sigDts[1].innerHTML = `${t("sig_richness")} <small>${t("sig_richness_sub")}</small>`;
      sigDts[2].innerHTML = `${t("sig_discourse")} <small>${t("sig_discourse_sub")}</small>`;
      sigDts[3].innerHTML = `${t("sig_binoculars")} <small>${t("sig_binoculars_sub")}</small>`;
      sigDts[4].innerHTML = `${t("sig_affinity")} <small>${t("sig_affinity_sub")}</small>`;
      sigDts[5].innerHTML = `${t("sig_grade")} <small>${t("sig_grade_sub")}</small>`;
      sigDts[6].textContent = t("sig_wps");
      sigDts[7].innerHTML = `${t("sig_contractions")} <small>${t("sig_contractions_sub")}</small>`;
    }

    const telemSumSpan = document.querySelector("#telemetryAccordion summary span");
    if (telemSumSpan) telemSumSpan.textContent = t("engine_info_title");

    const telemDts = document.querySelectorAll("#telemetryAccordion dl.metrics dt");
    if (telemDts.length >= 7) {
      telemDts[0].textContent = t("tel_time");
      telemDts[1].textContent = t("tel_memory");
      telemDts[2].textContent = t("tel_runtime");
      telemDts[3].textContent = t("tel_threads");
      telemDts[4].textContent = t("tel_size");
      telemDts[5].textContent = t("tel_detector");
      telemDts[6].textContent = t("tel_network");
    }

    const telemDivs = document.querySelectorAll("#telemetryAccordion dl.metrics div");
    if (telemDivs.length >= 7) {
      const ddRuntime = telemDivs[2].querySelector("dd");
      if (ddRuntime) ddRuntime.textContent = t("tel_runtime_val");
      const ddNetwork = telemDivs[6].querySelector("dd");
      if (ddNetwork) ddNetwork.textContent = t("tel_network_val");
    }

    const copyBtnLabel = btnCopyReport ? btnCopyReport.querySelector(".btn-label") : null;
    if (copyBtnLabel) copyBtnLabel.textContent = t("btn_copy_summary");
    const jsonBtnLabel = btnDownloadJson ? btnDownloadJson.querySelector(".btn-label") : null;
    if (jsonBtnLabel) jsonBtnLabel.textContent = t("btn_json");
    const csvBtnLabel = btnDownloadCsv ? btnDownloadCsv.querySelector(".btn-label") : null;
    if (csvBtnLabel) csvBtnLabel.textContent = t("btn_csv");
    const resetBtnLabel = btnResetReport ? btnResetReport.querySelector(".btn-label") : null;
    if (resetBtnLabel) resetBtnLabel.textContent = t("btn_clear_results");

    const rDiscl = document.querySelector("#activeResultsContent .results-disclaimer");
    if (rDiscl) rDiscl.textContent = t("results_disclaimer");

    const limitsSum = document.querySelector("#methodLimitsDetails summary span");
    if (limitsSum) limitsSum.textContent = t("limits_title");
    const limIntro = document.querySelector(".limits-intro");
    if (limIntro) limIntro.textContent = t("limits_intro");
    const limLis = document.querySelectorAll(".limits-list li");
    if (limLis.length >= 5) {
      limLis[0].innerHTML = `<strong>${t("limit_1_title")}</strong> ${t("limit_1_body")}`;
      limLis[1].innerHTML = `<strong>${t("limit_2_title")}</strong> ${t("limit_2_body")}`;
      limLis[2].innerHTML = `<strong>${t("limit_3_title")}</strong> ${t("limit_3_body")}`;
      limLis[3].innerHTML = `<strong>${t("limit_4_title")}</strong> ${t("limit_4_body")}`;
      limLis[4].innerHTML = `<strong>${t("limit_5_title")}</strong> ${t("limit_5_body")}`;
    }

    const colophonP = document.querySelector(".colophon p");
    if (colophonP) colophonP.textContent = t("colophon_text");

    const scTitle = $("shortcutsTitle");
    if (scTitle) scTitle.textContent = t("shortcuts_title");
    const scClose = $("btnCloseShortcuts");
    if (scClose) scClose.setAttribute("aria-label", t("sc_close_aria"));
    const scDds = document.querySelectorAll(".shortcuts-list dd");
    if (scDds.length >= 5) {
      scDds[0].textContent = t("sc_help_desc");
      scDds[1].textContent = t("sc_check_desc");
      scDds[2].textContent = t("sc_next_desc");
      scDds[3].textContent = t("sc_prev_desc");
      scDds[4].textContent = t("sc_esc_desc");
    }

    updateTextCounters();

    if (currentAnalysisData) {
      renderAnalysisResults(currentAnalysisData);
    }
  }

  if (langToggleBtn) {
    langToggleBtn.addEventListener("click", () => {
      const nextLang = currentLang === "en" ? "zh-TW" : "en";
      try { localStorage.setItem("veritas_lang", nextLang); } catch (e) {}
      applyLanguage(nextLang);
    });
  }

  applyLanguage(currentLang);
  checkEngine();
  loadSamples();
});
