/**
 * Veritas AI — QuillBot-Style Frontend Application Logic
 * Supports 4-class detection, calibrated probabilities, sentence highlighting,
 * uncertainty gating, and low-end hardware offline telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const textInput = document.getElementById("textInput");
  const heatmapViewer = document.getElementById("heatmapViewer");
  const btnModeEdit = document.getElementById("btnModeEdit");
  const btnModeHeatmap = document.getElementById("btnModeHeatmap");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnPaste = document.getElementById("btnPaste");
  const sampleSelect = document.getElementById("sampleSelect");
  const qbComparisonGroup = document.getElementById("qbComparisonGroup");
  const thresholdInput = document.getElementById("thresholdInput");
  const themeToggleBtn = document.getElementById("themeToggleBtn");

  // Counters
  const wordCountLabel = document.getElementById("wordCountLabel");
  const charCountLabel = document.getElementById("charCountLabel");
  const sentCountLabel = document.getElementById("sentCountLabel");
  const wordGuideBadge = document.getElementById("wordGuideBadge");

  // Results DOM
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const activeResultsContent = document.getElementById("activeResultsContent");
  const verdictTitle = document.getElementById("verdictTitle");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictDescription = document.getElementById("verdictDescription");
  const confidenceTag = document.getElementById("confidenceTag");
  const uncertainAlertBanner = document.getElementById("uncertainAlertBanner");

  // Bars
  const barAIGen = document.getElementById("barAIGen");
  const barAIRefined = document.getElementById("barAIRefined");
  const barHumanRefined = document.getElementById("barHumanRefined");
  const barHuman = document.getElementById("barHuman");

  const pctAIGen = document.getElementById("pctAIGen");
  const pctAIRefined = document.getElementById("pctAIRefined");
  const pctHumanRefined = document.getElementById("pctHumanRefined");
  const pctHuman = document.getElementById("pctHuman");

  // Inspector & Telemetry
  const inspectorContent = document.getElementById("inspectorContent");
  const telemetryLatency = document.getElementById("telemetryLatency");

  let currentAnalysisData = null;
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
      wordGuideBadge.textContent = "Recommended: 80–2,000 words";
    } else if (words < 80) {
      wordGuideBadge.className = "guide-badge warn";
      wordGuideBadge.textContent = `Short text (${words} words) — 80+ words recommended`;
    } else if (words > 2000) {
      wordGuideBadge.className = "guide-badge warn";
      wordGuideBadge.textContent = `Long text (${words} words) — analyzing first 2,000 words`;
    } else {
      wordGuideBadge.className = "guide-badge ok";
      wordGuideBadge.textContent = "Optimal length for forensic confidence";
    }
  }

  textInput.addEventListener("input", updateTextCounters);

  // Clear & Paste
  btnClear.addEventListener("click", () => {
    textInput.value = "";
    updateTextCounters();
    switchToEditMode();
    resultsPlaceholder.style.display = "flex";
    activeResultsContent.style.display = "none";
    btnModeHeatmap.disabled = true;
    currentAnalysisData = null;
  });

  btnPaste.addEventListener("click", async () => {
    try {
      const clipText = await navigator.clipboard.readText();
      textInput.value = clipText;
      updateTextCounters();
    } catch (e) {
      console.warn("Clipboard access denied or unsupported:", e);
    }
  });

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
      btnAnalyze.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <span>Analyze Text</span>
        <kbd class="shortcut-key">Ctrl+Enter</kbd>
      `;
    }
  }

  // Render Results
  function renderAnalysisResults(data) {
    resultsPlaceholder.style.display = "none";
    activeResultsContent.style.display = "flex";
    btnModeHeatmap.disabled = false;

    const summary = data.summary;
    const probs = data.calibrated_probabilities;
    const pcts = data.percentages;

    // Document Verdict Card
    verdictTitle.textContent = summary.verdict;
    verdictDescription.textContent = summary.verdict_description;
    confidenceTag.textContent = `${summary.confidence_pct}% Confidence`;

    verdictBadge.textContent = summary.verdict;
    verdictBadge.className = `verdict-badge ${summary.badge}`;

    // Uncertain Banner Handling
    if (summary.is_uncertain) {
      uncertainAlertBanner.style.display = "flex";
    } else {
      uncertainAlertBanner.style.display = "none";
    }

    // Probability Bars
    pctAIGen.textContent = `${pcts.ai_generated.toFixed(1)}%`;
    barAIGen.style.width = `${pcts.ai_generated}%`;

    pctAIRefined.textContent = `${pcts.ai_ai_refined.toFixed(1)}%`;
    barAIRefined.style.width = `${pcts.ai_ai_refined}%`;

    pctHumanRefined.textContent = `${pcts.human_ai_refined.toFixed(1)}%`;
    barHumanRefined.style.width = `${pcts.human_ai_refined}%`;

    pctHuman.textContent = `${pcts.human.toFixed(1)}%`;
    barHuman.style.width = `${pcts.human}%`;

    // Latency
    telemetryLatency.textContent = `${summary.elapsed_seconds.toFixed(2)}s`;

    // Render Heatmap Content
    renderHeatmapSpans(data.sentences);

    // Switch to Heatmap View automatically for clarity
    switchToHeatmapMode();
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
    const reasonsHtml = sent.reasons.map(r => `<li>${r}</li>`).join("");

    inspectorContent.innerHTML = `
      <div class="inspector-sentence-text">"${sent.text}"</div>
      <div class="inspector-meta-row">
        <span>Classification: <strong class="${sent.color_class}">${sent.class_label}</strong></span>
        <span>Confidence: <strong>${(sent.confidence * 100).toFixed(1)}%</strong></span>
      </div>
      <div class="inspector-meta-row">
        <span>AI Likelihood: <strong>${sent.ai_likelihood_pct}%</strong></span>
      </div>
      <div style="font-size: 0.78rem; font-weight: 700; margin-top: 6px; margin-bottom: 3px; color: var(--text-muted);">
        Forensic Indicators:
      </div>
      <ul class="inspector-reasons-list">
        ${reasonsHtml}
      </ul>
    `;
  }

  // Load Pre-loaded Benchmark Samples & Comparison Sheet
  async function loadSamples() {
    try {
      const resp = await fetch("/api/samples");
      if (resp.ok) {
        const samples = await resp.json();
        sampleSelect.addEventListener("change", (e) => {
          const val = e.target.value;
          if (samples[val]) {
            textInput.value = samples[val].text;
            updateTextCounters();
            switchToEditMode();
          } else {
            // Check in comparison sheet
            const found = comparisonSheetData.find(s => s.id === val);
            if (found) {
              textInput.value = found.text;
              updateTextCounters();
              switchToEditMode();
            }
          }
        });
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
      }
    } catch (e) {
      console.warn("Could not load sample benchmarks:", e);
    }
  }

  loadSamples();
});
