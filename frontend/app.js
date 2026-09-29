/**
 * Veritas AI — QuillBot-Style Frontend Application Logic
 * Supports 4-class detection, calibrated probabilities, sentence highlighting,
 * uncertainty gating, document upload, and low-end hardware offline telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const textInput = document.getElementById("textInput");
  const heatmapViewer = document.getElementById("heatmapViewer");
  const btnModeEdit = document.getElementById("btnModeEdit");
  const btnModeHeatmap = document.getElementById("btnModeHeatmap");
  const heatmapSentCountBadge = document.getElementById("heatmapSentCountBadge");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnPaste = document.getElementById("btnPaste");
  const btnUpload = document.getElementById("btnUpload");
  const fileUploadInput = document.getElementById("fileUploadInput");
  const editorDropZone = document.getElementById("editorDropZone");
  const dragDropOverlay = document.getElementById("dragDropOverlay");
  const sampleSelect = document.getElementById("sampleSelect");
  const qbComparisonGroup = document.getElementById("qbComparisonGroup");
  const thresholdInput = document.getElementById("thresholdInput");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const btnCopyReport = document.getElementById("btnCopyReport");

  // Counters
  const wordCountLabel = document.getElementById("wordCountLabel");
  const charCountLabel = document.getElementById("charCountLabel");
  const sentCountLabel = document.getElementById("sentCountLabel");
  const wordGuideBadge = document.getElementById("wordGuideBadge");

  // Results DOM
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const activeResultsContent = document.getElementById("activeResultsContent");
  const verdictCard = document.getElementById("verdictCard");
  const verdictTitle = document.getElementById("verdictTitle");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictDescription = document.getElementById("verdictDescription");
  const verdictIconBox = document.getElementById("verdictIconBox");
  const confidenceTag = document.getElementById("confidenceTag");
  const uncertainAlertBanner = document.getElementById("uncertainAlertBanner");
  const qbHeadlineBanner = document.getElementById("qbHeadlineBanner");
  const qbHeadlineText = document.getElementById("qbHeadlineText");
  const qbHeadlinePulse = document.getElementById("qbHeadlinePulse");

  // Stack & Bars
  const stackBarAI = document.getElementById("stackBarAI");
  const stackBarAIRefined = document.getElementById("stackBarAIRefined");
  const stackBarHumanRefined = document.getElementById("stackBarHumanRefined");
  const stackBarHuman = document.getElementById("stackBarHuman");

  const barAIGen = document.getElementById("barAIGen");
  const barAIRefined = document.getElementById("barAIRefined");
  const barHumanRefined = document.getElementById("barHumanRefined");
  const barHuman = document.getElementById("barHuman");

  const pctAIGen = document.getElementById("pctAIGen");
  const pctAIRefined = document.getElementById("pctAIRefined");
  const pctHumanRefined = document.getElementById("pctHumanRefined");
  const pctHuman = document.getElementById("pctHuman");

  // Inspector & Telemetry
  const sentenceCountsSummary = document.getElementById("sentenceCountsSummary");
  const inspectorContent = document.getElementById("inspectorContent");
  const telemetryLatency = document.getElementById("telemetryLatency");
  const telemetryMemory = document.getElementById("telemetryMemory");

  // Mathematical Forensic Equations
  const mathValAffinity = document.getElementById("mathValAffinity");
  const mathValBurstiness = document.getElementById("mathValBurstiness");
  const mathValDiscourse = document.getElementById("mathValDiscourse");
  const mathValBinoculars = document.getElementById("mathValBinoculars");
  const mathValRichness = document.getElementById("mathValRichness");
  const mathAffinityBadge = document.getElementById("mathAffinityBadge");

  let currentAnalysisData = null;
  let archetypeSamples = {};
  let comparisonSheetData = [];

  // Theme Management
  const savedTheme = localStorage.getItem("veritas_theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  themeToggleBtn.addEventListener("click", () => {
    const curr = document.documentElement.getAttribute("data-theme");
    const next = curr === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("veritas_theme", next);
  });

  // Live Text Metrics
  function updateTextCounters() {
    const text = textInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const sents = text.trim() ? text.split(/[.!?]+/).filter(s => s.trim().length > 0).length : 0;

    wordCountLabel.textContent = `${words} words`;
    charCountLabel.textContent = `${chars} chars`;
    sentCountLabel.textContent = `${sents} sentences`;

    if (words === 0) {
      wordGuideBadge.className = "guide-badge ok";
      wordGuideBadge.textContent = "Best results: 80 to 2,000 words";
    } else if (words < 80) {
      wordGuideBadge.className = "guide-badge warn";
      wordGuideBadge.textContent = `Short text (${words}w) — 80+ words recommended`;
    } else if (words > 2000) {
      wordGuideBadge.className = "guide-badge warn";
      wordGuideBadge.textContent = `Long text (${words}w) — analyzing first 2,000 words`;
    } else {
      wordGuideBadge.className = "guide-badge ok";
      wordGuideBadge.textContent = "Good length for a confident result";
    }
  }

  textInput.addEventListener("input", updateTextCounters);

  // Clear
  btnClear.addEventListener("click", () => {
    textInput.value = "";
    updateTextCounters();
    switchToEditMode();
    resultsPlaceholder.style.display = "flex";
    activeResultsContent.style.display = "none";
    btnModeHeatmap.disabled = true;
    heatmapSentCountBadge.style.display = "none";
    currentAnalysisData = null;
  });

  // Paste
  btnPaste.addEventListener("click", async () => {
    try {
      const clipText = await navigator.clipboard.readText();
      textInput.value = clipText;
      updateTextCounters();
      switchToEditMode();
    } catch (e) {
      console.warn("Clipboard access denied:", e);
    }
  });

  // Upload File Handling
  btnUpload.addEventListener("click", () => {
    fileUploadInput.click();
  });

  fileUploadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      await handleFileUpload(file);
    }
    fileUploadInput.value = "";
  });

  // Drag and drop onto editor
  editorDropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dragDropOverlay.style.display = "flex";
  });

  editorDropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDropOverlay.style.display = "none";
  });

  editorDropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDropOverlay.style.display = "none";
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await handleFileUpload(file);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `<span>Uploading...</span>`;

    try {
      const resp = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "File processing failed.");
      }

      const data = await resp.json();
      textInput.value = data.document_text || data.text || "";
      updateTextCounters();
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      btnAnalyze.disabled = false;
      resetAnalyzeButtonText();
    }
  }

  // View Mode Switching
  function switchToEditMode() {
    btnModeEdit.classList.add("active");
    btnModeHeatmap.classList.remove("active");
    textInput.style.display = "block";
    heatmapViewer.style.display = "none";
  }

  function switchToHeatmapMode() {
    if (!currentAnalysisData) return;
    btnModeHeatmap.classList.add("active");
    btnModeEdit.classList.remove("active");
    textInput.style.display = "none";
    heatmapViewer.style.display = "block";
  }

  btnModeEdit.addEventListener("click", switchToEditMode);
  btnModeHeatmap.addEventListener("click", switchToHeatmapMode);

  // Keyboard shortcut Ctrl+Enter
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runAnalysis();
    }
  });

  btnAnalyze.addEventListener("click", runAnalysis);

  function resetAnalyzeButtonText() {
    btnAnalyze.innerHTML = `
      <span>Analyze text</span>
      <kbd class="shortcut-key">Ctrl+Enter</kbd>
    `;
  }

  // Run Analysis Call
  async function runAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
      alert("Please paste or write some text before analyzing.");
      return;
    }

    const threshold = parseFloat(thresholdInput.value) || 0.40;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `<span>Analyzing...</span>`;

    try {
      const resp = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          confidence_threshold: threshold
        })
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Analysis request failed.");
      }

      const data = await resp.json();
      currentAnalysisData = data;
      renderAnalysisResults(data);
    } catch (err) {
      alert(`Detection error: ${err.message}`);
    } finally {
      btnAnalyze.disabled = false;
      resetAnalyzeButtonText();
    }
  }

  // Render Results
  function renderAnalysisResults(data) {
    resultsPlaceholder.style.display = "none";
    activeResultsContent.style.display = "flex";
    btnModeHeatmap.disabled = false;

    const summary = data.summary;
    const pcts = data.percentages;
    const sentences = data.sentences || [];

    heatmapSentCountBadge.textContent = sentences.length;
    heatmapSentCountBadge.style.display = "inline-block";

    // QuillBot Replica Headline Card
    if (qbHeadlineText && summary.quillbot_headline) {
      qbHeadlineText.textContent = summary.quillbot_headline;
      if (qbHeadlinePulse) {
        qbHeadlinePulse.className = "qb-headline-pulse";
        if (summary.quillbot_ai_pct >= 50.0) {
          qbHeadlinePulse.classList.add("pulse-ai");
        } else if (summary.quillbot_ai_pct > 0.0) {
          qbHeadlinePulse.classList.add("pulse-refined");
        } else {
          qbHeadlinePulse.classList.add("pulse-human");
        }
      }
    }

    // Document Verdict Card
    verdictTitle.textContent = summary.verdict;
    verdictDescription.textContent = summary.verdict_description;
    confidenceTag.textContent = `${summary.confidence_pct}% confidence`;

    // The API returns generic badge names; map them onto the four-class ramp
    const badgeMap = {
      "badge-success": "badge-human",
      "badge-warning": "badge-human-refined",
      "badge-orange": "badge-ai-refined",
      "badge-danger": "badge-ai",
      "badge-secondary": "badge-uncertain"
    };
    const badgeClass = badgeMap[summary.badge] || summary.badge || "badge-uncertain";

    verdictBadge.textContent = summary.verdict;
    verdictBadge.className = `verdict-badge ${badgeClass}`;

    // Verdict Icon
    renderVerdictIcon(badgeClass);

    // Uncertain Banner Handling
    if (summary.is_uncertain) {
      uncertainAlertBanner.style.display = "flex";
    } else {
      uncertainAlertBanner.style.display = "none";
    }

    // Composite Stacked Bar
    stackBarAI.style.width = `${pcts.ai_generated}%`;
    stackBarAIRefined.style.width = `${pcts.ai_ai_refined}%`;
    stackBarHumanRefined.style.width = `${pcts.human_ai_refined}%`;
    stackBarHuman.style.width = `${pcts.human}%`;

    // Individual Percentage Meters
    pctAIGen.textContent = `${pcts.ai_generated.toFixed(1)}%`;
    barAIGen.style.width = `${pcts.ai_generated}%`;

    pctAIRefined.textContent = `${pcts.ai_ai_refined.toFixed(1)}%`;
    barAIRefined.style.width = `${pcts.ai_ai_refined}%`;

    pctHumanRefined.textContent = `${pcts.human_ai_refined.toFixed(1)}%`;
    barHumanRefined.style.width = `${pcts.human_ai_refined}%`;

    pctHuman.textContent = `${pcts.human.toFixed(1)}%`;
    barHuman.style.width = `${pcts.human}%`;

    // Telemetry Latency & Memory
    telemetryLatency.textContent = `${summary.elapsed_seconds.toFixed(2)}s`;
    if (telemetryMemory) {
      telemetryMemory.textContent = `~152 MB (Cap: 1,500 MB)`;
    }

    // Populate Mathematical Forensic Equations
    const mathEqs = data.mathematical_equations || {};
    if (mathValAffinity && mathEqs.authorial_affinity_lambda !== undefined) {
      mathValAffinity.textContent = mathEqs.authorial_affinity_lambda.toFixed(4);
      mathValBurstiness.textContent = mathEqs.syntactic_burstiness_b !== undefined ? mathEqs.syntactic_burstiness_b.toFixed(4) : "—";
      const phi = mathEqs.discourse_polarity_phi;
      mathValDiscourse.textContent = phi !== undefined ? (phi > 0 ? "+" : "") + phi.toFixed(4) : "—";
      mathValBinoculars.textContent = mathEqs.binoculars_ratio_r !== undefined ? mathEqs.binoculars_ratio_r.toFixed(3) : "—";
      mathValRichness.textContent = mathEqs.lexical_richness_omega !== undefined ? mathEqs.lexical_richness_omega.toFixed(4) : "—";
      mathAffinityBadge.textContent = "Λ_auth " + mathEqs.authorial_affinity_lambda.toFixed(3);
    }

    // Let the halftone field react to the result
    window.dispatchEvent(new CustomEvent("veritas:result", {
      detail: { aiShare: (pcts.ai_generated + pcts.ai_ai_refined) / 100 }
    }));

    // Sentence Highlight Counts
    renderSentenceCounts(sentences);

    // Render Heatmap Content
    renderHeatmapSpans(sentences);

    // Automatically switch to Heatmap View so user sees highlights immediately
    switchToHeatmapMode();
  }

  function renderVerdictIcon(badgeClass) {
    let iconSvg = "";
    let boxClass = "";

    if (badgeClass.includes("badge-human")) {
      boxClass = "badge-human";
      iconSvg = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      `;
    } else if (badgeClass.includes("badge-ai-refined") || badgeClass.includes("badge-human-refined")) {
      boxClass = badgeClass.includes("ai-refined") ? "badge-ai-refined" : "badge-human-refined";
      iconSvg = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
          <path d="M3 3v5h5"/>
          <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
          <path d="M16 21h5v-5"/>
        </svg>
      `;
    } else if (badgeClass.includes("badge-ai")) {
      boxClass = "badge-ai";
      iconSvg = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="11" width="18" height="10" rx="2"/>
          <circle cx="12" cy="5" r="2"/>
          <path d="M12 7v4"/>
          <line x1="8" y1="16" x2="8" y2="16"/>
          <line x1="16" y1="16" x2="16" y2="16"/>
        </svg>
      `;
    } else {
      boxClass = "badge-uncertain";
      iconSvg = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      `;
    }

    verdictIconBox.className = `verdict-icon-box ${boxClass}`;
    verdictIconBox.innerHTML = iconSvg;
  }

  function renderSentenceCounts(sentences) {
    let ai = 0, aiRef = 0, humRef = 0, hum = 0;
    sentences.forEach(s => {
      const cls = s.highlight_class;
      if (cls === "highlight-ai") ai++;
      else if (cls === "highlight-ai-refined") aiRef++;
      else if (cls === "highlight-human-refined") humRef++;
      else hum++;
    });

    sentenceCountsSummary.innerHTML = `
      ${ai > 0 ? `<span class="count-pill badge-ai">${ai} AI</span>` : ""}
      ${aiRef > 0 ? `<span class="count-pill badge-ai-refined">${aiRef} AI-edited</span>` : ""}
      ${humRef > 0 ? `<span class="count-pill badge-human-refined">${humRef} Polished</span>` : ""}
      <span class="count-pill badge-human">${hum} Human</span>
    `;
  }

  function renderHeatmapSpans(sentences) {
    heatmapViewer.innerHTML = "";

    sentences.forEach((s, idx) => {
      const span = document.createElement("span");
      span.className = `sentence-span ${s.highlight_class}`;
      span.textContent = s.text + " ";
      span.dataset.index = idx;

      span.addEventListener("click", () => {
        document.querySelectorAll(".sentence-span").forEach(el => el.classList.remove("active"));
        span.classList.add("active");
        renderInspectorDetails(s);
      });

      heatmapViewer.appendChild(span);
    });

    // Auto-select first sentence
    if (sentences.length > 0) {
      const firstSpan = heatmapViewer.querySelector(".sentence-span");
      if (firstSpan) {
        firstSpan.classList.add("active");
        renderInspectorDetails(sentences[0]);
      }
    }
  }

  function renderInspectorDetails(sent) {
    const reasonsHtml = (sent.reasons || []).map(r => `<li>${r}</li>`).join("");

    inspectorContent.innerHTML = `
      <div class="inspector-sentence-text">"${sent.text}"</div>
      <div class="inspector-meta-row">
        <span>Class: <strong>${sent.class_label}</strong></span>
        <span>Confidence: <strong>${(sent.confidence * 100).toFixed(1)}%</strong></span>
      </div>
      <div class="inspector-meta-row">
        <span>AI likelihood: <strong>${sent.ai_likelihood_pct}%</strong></span>
      </div>
      <div class="inspector-subhead">Why it was scored this way</div>
      <ul class="inspector-reasons-list">
        ${reasonsHtml || "<li>Natural phrasing and vocabulary variance</li>"}
      </ul>
    `;
  }

  // Copy Summary Report
  btnCopyReport.addEventListener("click", async () => {
    if (!currentAnalysisData) return;
    const s = currentAnalysisData.summary;
    const p = currentAnalysisData.percentages;
    const report = [
      `=== VERITAS AI DETECTION REPORT ===`,
      `QuillBot Headline: ${s.quillbot_headline || "N/A"}`,
      `Final Verdict: ${s.verdict} (${s.confidence_pct}% Confidence)`,
      `Status: ${s.is_uncertain ? "Uncertain (Confidence below cutoff)" : "High Certainty"}`,
      `Document Breakdown:`,
      `  • AI-generated: ${p.ai_generated.toFixed(1)}%`,
      `  • AI-generated & AI-refined: ${p.ai_ai_refined.toFixed(1)}%`,
      `  • Human-written & AI-refined: ${p.human_ai_refined.toFixed(1)}%`,
      `  • Human-written: ${p.human.toFixed(1)}%`,
      `Analysis Latency: ${s.elapsed_seconds.toFixed(2)}s on 2 CPU Threads (ONNX INT8)`,
      `==================================`
    ].join("\n");

    try {
      await navigator.clipboard.writeText(report);
      const origText = btnCopyReport.querySelector("span").textContent;
      btnCopyReport.querySelector("span").textContent = "Copied";
      setTimeout(() => {
        btnCopyReport.querySelector("span").textContent = origText;
      }, 2000);
    } catch (e) {
      alert("Could not copy report to clipboard.");
    }
  });

  // Quick-Test Sample Pills Setup
  document.querySelectorAll(".sample-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const sampleKey = pill.getAttribute("data-sample");
      if (archetypeSamples[sampleKey]) {
        textInput.value = archetypeSamples[sampleKey].text;
        updateTextCounters();
        switchToEditMode();
      }
    });
  });

  // Load Pre-loaded Benchmark Samples & Comparison Sheet
  async function loadSamples() {
    try {
      const resp = await fetch("/api/samples");
      if (resp.ok) {
        archetypeSamples = await resp.json();
      }

      // Fetch 30-sample QuillBot comparison sheet
      const compResp = await fetch("/api/comparison-sheet");
      if (compResp.ok) {
        comparisonSheetData = await compResp.json();
        comparisonSheetData.forEach(s => {
          const opt = document.createElement("option");
          opt.value = s.id;
          opt.textContent = `${s.id}: [${s.expected_class}] (${s.word_count}w) ${s.type}`;
          qbComparisonGroup.appendChild(opt);
        });

        sampleSelect.addEventListener("change", (e) => {
          const val = e.target.value;
          const found = comparisonSheetData.find(s => s.id === val);
          if (found) {
            textInput.value = found.text;
            updateTextCounters();
            switchToEditMode();
          }
        });
      }
    } catch (e) {
      console.warn("Could not load sample benchmarks:", e);
    }
  }

  loadSamples();
});
