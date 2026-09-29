/**
 * Veritas — frontend logic.
 * Sends text to /api/detect (or a file to /api/upload), then fills the report:
 * headline, verdict, 4-class breakdown, highlighted sentences and inspector.
 */

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);

  const appCard = $("appCard");
  const textInput = $("textInput");
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
  const wordCountLabel = $("wordCountLabel");
  const wordGuideBadge = $("wordGuideBadge");
  const formError = $("formError");

  const report = $("report");
  const heatmapViewer = $("heatmapViewer");
  const inspectorContent = $("inspectorContent");
  const btnCopyReport = $("btnCopyReport");

  const ANALYZE_LABEL = "Analyze";

  // The API returns generic badge names; map them onto the ink levels
  const BADGE_TO_FILL = {
    "badge-danger": "fill-ai",
    "badge-orange": "fill-ai-refined",
    "badge-warning": "fill-human-refined",
    "badge-success": "fill-human",
  };

  let currentAnalysisData = null;
  let archetypeSamples = {};
  let comparisonSheetData = [];

  // ---------- Input ----------

  function updateTextCounters() {
    const text = textInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCountLabel.textContent = `${words} ${words === 1 ? "word" : "words"}`;

    wordGuideBadge.classList.remove("warn");
    if (words === 0) {
      wordGuideBadge.textContent = "80 to 2,000 works best";
    } else if (words < 80) {
      wordGuideBadge.textContent = "Add more for a confident result";
      wordGuideBadge.classList.add("warn");
    } else if (words > 2000) {
      wordGuideBadge.textContent = "Only the first 2,000 are checked";
      wordGuideBadge.classList.add("warn");
    } else {
      wordGuideBadge.textContent = "Good length";
    }
  }

  function setText(value) {
    textInput.value = value;
    updateTextCounters();
    hideError();
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function hideError() {
    formError.hidden = true;
  }

  textInput.addEventListener("input", () => {
    updateTextCounters();
    hideError();
  });

  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runAnalysis();
    }
  });

  btnClear.addEventListener("click", () => {
    setText("");
    currentAnalysisData = null;
    report.hidden = true;
    appCard.classList.remove("has-results");
    textInput.focus();
  });

  btnPaste.addEventListener("click", async () => {
    try {
      setText(await navigator.clipboard.readText());
    } catch (e) {
      showError("Veritas can't read your clipboard. Paste into the text box with Ctrl+V instead.");
    }
  });

  // ---------- Files ----------

  btnUpload.addEventListener("click", () => fileUploadInput.click());

  fileUploadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) await handleFileUpload(file);
    fileUploadInput.value = "";
  });

  editorDropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dragDropOverlay.hidden = false;
  });

  editorDropZone.addEventListener("dragleave", (e) => {
    if (!editorDropZone.contains(e.relatedTarget)) dragDropOverlay.hidden = true;
  });

  editorDropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDropOverlay.hidden = true;
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) await handleFileUpload(file);
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    setBusy(true, "Reading file...");
    try {
      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Veritas couldn't read that file. Try a PDF, DOCX or TXT.");
      }
      const data = await resp.json();
      setText(data.document_text || data.text || "");
      renderAnalysisResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
    }
  }

  // ---------- Analysis ----------

  function setBusy(busy, label) {
    btnAnalyze.disabled = busy;
    btnAnalyze.textContent = busy ? label : ANALYZE_LABEL;
  }

  btnAnalyze.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
      showError("Add some text first, or load a sample.");
      textInput.focus();
      return;
    }
    hideError();
    setBusy(true, "Analyzing...");

    try {
      const resp = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          confidence_threshold: parseFloat(thresholdInput.value) || 0.4,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "The analysis didn't finish. Check that the Veritas server is running, then try again.");
      }
      renderAnalysisResults(await resp.json());
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
    }
  }

  // ---------- Report ----------

  const fmtPct = (n) => `${Number(n).toFixed(n % 1 === 0 ? 0 : 1)}%`;

  function renderAnalysisResults(data) {
    currentAnalysisData = data;
    const s = data.summary;
    const p = data.percentages;
    const sentences = data.sentences || [];

    appCard.classList.add("has-results");
    report.hidden = false;

    $("qbHeadlineText").textContent = s.quillbot_headline || s.verdict;
    $("verdictTitle").textContent = s.verdict;
    $("confidenceTag").textContent = `${s.confidence_pct}% confidence`;
    $("verdictDescription").textContent = s.verdict_description || "";
    $("verdictSwatch").className = `swatch ${BADGE_TO_FILL[s.badge] || "fill-human"}`;
    $("uncertainAlertBanner").hidden = !s.is_uncertain;

    const classes = [
      ["stackBarAI", "pctAIGen", p.ai_generated],
      ["stackBarAIRefined", "pctAIRefined", p.ai_ai_refined],
      ["stackBarHumanRefined", "pctHumanRefined", p.human_ai_refined],
      ["stackBarHuman", "pctHuman", p.human],
    ];
    classes.forEach(([barId, pctId, value]) => {
      $(barId).style.width = `${value}%`;
      $(pctId).textContent = fmtPct(value);
    });

    const m = data.mathematical_equations || {};
    const fix = (v, d) => (typeof v === "number" ? v.toFixed(d) : "—");
    $("mathValAffinity").textContent = fix(m.authorial_affinity_lambda, 4);
    $("mathValBurstiness").textContent = fix(m.syntactic_burstiness_b, 4);
    $("mathValDiscourse").textContent =
      typeof m.discourse_polarity_phi === "number"
        ? (m.discourse_polarity_phi > 0 ? "+" : "") + m.discourse_polarity_phi.toFixed(4)
        : "—";
    $("mathValBinoculars").textContent = fix(m.binoculars_ratio_r, 3);
    $("mathValRichness").textContent = fix(m.lexical_richness_omega, 4);
    $("telemetryLatency").textContent = `${fix(s.elapsed_seconds, 2)}s`;

    renderSentenceCounts(sentences);
    renderHeatmap(sentences);

    window.dispatchEvent(new CustomEvent("veritas:result", {
      detail: { aiShare: (p.ai_generated + p.ai_ai_refined) / 100 },
    }));

    // On a stacked (narrow) layout the report sits below the form
    if (window.matchMedia("(max-width: 900px)").matches) {
      report.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderSentenceCounts(sentences) {
    const ai = sentences.filter((x) => x.highlight_class === "highlight-ai" || x.highlight_class === "highlight-ai-refined").length;
    $("sentenceCountsSummary").textContent = `${ai} of ${sentences.length} sentences flagged`;
  }

  function renderHeatmap(sentences) {
    heatmapViewer.replaceChildren();

    sentences.forEach((sent, idx) => {
      const span = document.createElement("span");
      span.className = `sentence-span ${sent.highlight_class || "highlight-neutral"}`;
      span.textContent = sent.text;
      span.tabIndex = 0;
      span.setAttribute("role", "button");

      const select = () => {
        heatmapViewer.querySelectorAll(".sentence-span.active").forEach((el) => el.classList.remove("active"));
        span.classList.add("active");
        renderInspector(sent);
      };
      span.addEventListener("click", select);
      span.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });

      heatmapViewer.append(span, document.createTextNode(" "));
      if (idx === 0) select();
    });
  }

  function renderInspector(sent) {
    const quote = document.createElement("blockquote");
    quote.textContent = sent.text;

    const meta = document.createElement("p");
    meta.className = "meta";
    const addMeta = (label, value) => {
      const item = document.createElement("span");
      const strong = document.createElement("strong");
      strong.textContent = value;
      item.append(`${label} `, strong);
      meta.append(item);
    };
    addMeta("Class", sent.class_label);
    addMeta("Confidence", `${(sent.confidence * 100).toFixed(1)}%`);
    addMeta("AI likelihood", `${sent.ai_likelihood_pct}%`);

    const list = document.createElement("ul");
    const reasons = sent.reasons && sent.reasons.length ? sent.reasons : ["Natural phrasing and varied vocabulary"];
    reasons.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      list.append(li);
    });

    inspectorContent.replaceChildren(quote, meta, list);
  }

  btnCopyReport.addEventListener("click", async () => {
    if (!currentAnalysisData) return;
    const s = currentAnalysisData.summary;
    const p = currentAnalysisData.percentages;
    const text = [
      "Veritas AI detection report",
      `Headline: ${s.quillbot_headline || "N/A"}`,
      `Verdict: ${s.verdict} (${s.confidence_pct}% confidence)`,
      `Status: ${s.is_uncertain ? "Uncertain, below the cutoff" : "Confident"}`,
      `AI-generated: ${p.ai_generated.toFixed(1)}%`,
      `AI-edited: ${p.ai_ai_refined.toFixed(1)}%`,
      `Polished human: ${p.human_ai_refined.toFixed(1)}%`,
      `Human: ${p.human.toFixed(1)}%`,
      `Analysis time: ${s.elapsed_seconds.toFixed(2)}s on 2 CPU threads`,
    ].join("\n");

    try {
      await navigator.clipboard.writeText(text);
      btnCopyReport.textContent = "Copied";
      setTimeout(() => (btnCopyReport.textContent = "Copy summary"), 1800);
    } catch (e) {
      btnCopyReport.textContent = "Couldn't copy";
      setTimeout(() => (btnCopyReport.textContent = "Copy summary"), 1800);
    }
  });

  // ---------- Samples ----------

  sampleSelect.addEventListener("change", () => {
    const value = sampleSelect.value;
    if (value.startsWith("archetype:")) {
      const sample = archetypeSamples[value.slice("archetype:".length)];
      if (sample) setText(sample.text);
    } else {
      const found = comparisonSheetData.find((x) => x.id === value);
      if (found) setText(found.text);
    }
    sampleSelect.selectedIndex = 0;
    textInput.focus();
  });

  async function loadSamples() {
    try {
      const resp = await fetch("/api/samples");
      if (resp.ok) archetypeSamples = await resp.json();

      const compResp = await fetch("/api/comparison-sheet");
      if (compResp.ok) {
        comparisonSheetData = await compResp.json();
        comparisonSheetData.forEach((x) => {
          const opt = document.createElement("option");
          opt.value = x.id;
          opt.textContent = `${x.id}: ${x.expected_class} (${x.word_count} words)`;
          qbComparisonGroup.append(opt);
        });
      }
    } catch (e) {
      console.warn("Could not load samples:", e);
    }
  }

  updateTextCounters();
  loadSamples();
});
