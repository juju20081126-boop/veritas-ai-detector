/**
 * Veritas AI — Frontend Interaction & Telemetry Controller
 */

// Application State
let currentAnalysis = null;
let selectedSentenceIndex = 0;
let currentFilter = 'all';
let sampleCache = {};

// DOM Elements
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const wordCount = document.getElementById('wordCount');
const sentenceCount = document.getElementById('sentenceCount');
const scanBtn = document.getElementById('scanBtn');
const clearBtn = document.getElementById('clearBtn');
const sampleSelect = document.getElementById('sampleSelect');
const dropZone = document.getElementById('dropZone');
const dropTrigger = document.getElementById('dropTrigger');
const fileInput = document.getElementById('fileInput');
const fileUploadSpinner = document.getElementById('fileUploadSpinner');

const scanLoadingOverlay = document.getElementById('scanLoadingOverlay');
const loadingStatusText = document.getElementById('loadingStatusText');
const resultsSection = document.getElementById('resultsSection');

const resultDocTitle = document.getElementById('resultDocTitle');
const resultDocMeta = document.getElementById('resultDocMeta');
const exportPdfBtn = document.getElementById('exportPdfBtn');

const gaugeProgress = document.getElementById('gaugeProgress');
const scoreValue = document.getElementById('scoreValue');
const verdictBadge = document.getElementById('verdictBadge');
const confidenceTag = document.getElementById('confidenceTag');
const verdictTitle = document.getElementById('verdictTitle');
const verdictDesc = document.getElementById('verdictDesc');

const cntHighAI = document.getElementById('cntHighAI');
const cntModAI = document.getElementById('cntModAI');
const cntMixed = document.getElementById('cntMixed');
const cntHuman = document.getElementById('cntHuman');

const heatmapBox = document.getElementById('heatmapBox');
const filterBtns = document.querySelectorAll('.filter-btn');

// Inspector Elements
const inspectSentenceNum = document.getElementById('inspectSentenceNum');
const inspectAiBadge = document.getElementById('inspectAiBadge');
const inspectSentenceText = document.getElementById('inspectSentenceText');
const inspectPpl = document.getElementById('inspectPpl');
const inspectTop10 = document.getElementById('inspectTop10');
const inspectNeural = document.getElementById('inspectNeural');
const inspectEvidenceList = document.getElementById('inspectEvidenceList');
const inspectClichesBox = document.getElementById('inspectClichesBox');
const inspectClicheTags = document.getElementById('inspectClicheTags');

// Telemetry Elements
const mAvgPpl = document.getElementById('mAvgPpl');
const barPpl = document.getElementById('barPpl');
const mBurstiness = document.getElementById('mBurstiness');
const barBurst = document.getElementById('barBurst');
const mTtr = document.getElementById('mTtr');
const barTtr = document.getElementById('barTtr');
const mClicheCount = document.getElementById('mClicheCount');
const barCliche = document.getElementById('barCliche');
const mUniformity = document.getElementById('mUniformity');
const barUniformity = document.getElementById('barUniformity');
const mGrade = document.getElementById('mGrade');
const mEase = document.getElementById('mEase');
const barGrade = document.getElementById('barGrade');

const themeToggleBtn = document.getElementById('themeToggleBtn');


// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupTheme();
  setupLiveCounters();
  setupEventListeners();
  fetchSamples();
  checkHealth();
});

// Theme Management
function setupTheme() {
  const savedTheme = localStorage.getItem('veritas_theme') || 'dark';
  if (savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }
}

themeToggleBtn.addEventListener('click', () => {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  if (isLight) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('veritas_theme', 'dark');
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('veritas_theme', 'light');
  }
});

// Live Counting
function setupLiveCounters() {
  textInput.addEventListener('input', updateTextCounters);
}

function updateTextCounters() {
  const text = textInput.value.trim();
  const chars = text.length;
  const words = text ? text.split(/\s+/).filter(w => w.length > 0).length : 0;
  
  // Quick sentence estimate
  const sentences = text ? text.split(/[.!?]+/).filter(s => s.trim().length > 2).length : 0;

  charCount.textContent = `${chars} characters`;
  wordCount.textContent = `${words} words`;
  sentenceCount.textContent = `${sentences} sentences`;
}

// Fetch Preloaded Samples from Backend
async function fetchSamples() {
  try {
    const res = await fetch('/api/samples');
    if (res.ok) {
      sampleCache = await res.json();
    }
  } catch (err) {
    console.warn('Could not load samples cache:', err);
  }
}

// Check Backend Health
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('engineStatusText').textContent = `Engine: Online (${data.device.toUpperCase()})`;
    }
  } catch (err) {
    document.getElementById('engineStatusText').textContent = 'Engine: Offline';
    document.getElementById('engineStatusPill').style.borderColor = 'red';
  }
}

// Event Listeners
function setupEventListeners() {
  // Clear button
  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    updateTextCounters();
    resultsSection.style.display = 'none';
    currentAnalysis = null;
    sampleSelect.selectedIndex = 0;
  });

  // Sample Selection
  sampleSelect.addEventListener('change', (e) => {
    const key = e.target.value;
    if (sampleCache[key]) {
      textInput.value = sampleCache[key].text;
      updateTextCounters();
      // Auto scan sample for instant experience
      triggerScan(sampleCache[key].title);
    }
  });

  // Scan Button & Keyboard Shortcut
  scanBtn.addEventListener('click', () => triggerScan());
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      triggerScan();
    }
  });

  // DropZone & File Upload
  dropTrigger.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileUpload);

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Filter Buttons
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      renderHeatmap();
    });
  });

  // Export PDF Report
  exportPdfBtn.addEventListener('click', exportPdf);
}

// File Upload Handler
function handleFileUpload(e) {
  if (e.target.files && e.target.files[0]) {
    uploadFile(e.target.files[0]);
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  dropTrigger.style.display = 'none';
  fileUploadSpinner.style.display = 'flex';
  showLoading('Ingesting and scanning document...');

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'File upload failed');
    }

    const data = await res.json();
    textInput.value = data.sentences.map(s => s.sentence).join(' ');
    updateTextCounters();
    displayResults(data);
  } catch (err) {
    alert(`Upload Error: ${err.message}`);
  } finally {
    hideLoading();
    dropTrigger.style.display = 'flex';
    fileUploadSpinner.style.display = 'none';
    fileInput.value = '';
  }
}

// Trigger Scan
async function triggerScan(customTitle = null) {
  const text = textInput.value.trim();
  if (!text) {
    alert('Please enter or paste text to analyze.');
    textInput.focus();
    return;
  }

  const wordCountVal = text.split(/\s+/).filter(w => w.length > 0).length;
  if (wordCountVal < 10) {
    alert('Please provide at least 15 words for statistically reliable AI detection.');
    return;
  }

  showLoading('Running neural RoBERTa classification, GPT-2 perplexity analysis, and forensic stylometrics...');

  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        filename: customTitle || (text.slice(0, 32) + '...')
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Detection failed');
    }

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    alert(`Scan Error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

function showLoading(msg) {
  loadingStatusText.textContent = msg;
  scanLoadingOverlay.style.display = 'flex';
}

function hideLoading() {
  scanLoadingOverlay.style.display = 'none';
}

// Display Results
function displayResults(data) {
  currentAnalysis = data;
  resultsSection.style.display = 'flex';

  const sum = data.summary;
  const met = data.metrics;
  const cnt = data.counts;

  // Header info
  resultDocTitle.textContent = sum.filename;
  resultDocMeta.textContent = `${sum.word_count} words • ${sum.sentence_count} sentences • ${sum.reading_time_minutes} min read • Processed in ${sum.elapsed_seconds}s`;

  // Circular Score Gauge Animation
  const aiPct = sum.overall_ai_percentage;
  scoreValue.textContent = `${Math.round(aiPct)}%`;

  // Circumference = 2 * PI * 52 = 326.7
  const circumference = 326.7;
  const offset = circumference - (circumference * (aiPct / 100));
  gaugeProgress.style.strokeDashoffset = offset;

  // Gauge Progress Color
  if (aiPct >= 75) {
    gaugeProgress.style.stroke = '#ef4444';
  } else if (aiPct >= 50) {
    gaugeProgress.style.stroke = '#f97316';
  } else if (aiPct >= 25) {
    gaugeProgress.style.stroke = '#eab308';
  } else {
    gaugeProgress.style.stroke = '#22c55e';
  }

  // Verdict Card
  verdictBadge.textContent = sum.verdict;
  verdictBadge.className = `badge ${sum.verdict_badge}`;
  confidenceTag.textContent = `${sum.confidence_percentage}% Statistical Confidence`;
  verdictTitle.textContent = sum.verdict;
  verdictDesc.textContent = sum.verdict_description;

  // Breakdown Counters
  cntHighAI.textContent = cnt.highly_likely_ai;
  cntModAI.textContent = cnt.likely_ai;
  cntMixed.textContent = cnt.mixed;
  cntHuman.textContent = cnt.human;

  // Render Telemetry
  mAvgPpl.textContent = met.average_perplexity;
  barPpl.style.width = `${Math.min(100, (met.average_perplexity / 80) * 100)}%`;

  mBurstiness.textContent = `${met.burstiness_index} (${met.burstiness_label})`;
  barBurst.style.width = `${Math.min(100, (met.burstiness_index / 0.8) * 100)}%`;

  mTtr.textContent = met.lexical_diversity.ttr;
  barTtr.style.width = `${Math.min(100, met.lexical_diversity.ttr * 100)}%`;

  mClicheCount.textContent = `${met.total_ai_markers} markers`;
  barCliche.style.width = `${Math.min(100, (met.total_ai_markers / 12) * 100)}%`;

  mUniformity.textContent = met.syntax_variance.uniformity_score;
  barUniformity.style.width = `${Math.min(100, met.syntax_variance.uniformity_score * 100)}%`;

  mGrade.textContent = `Grade ${met.readability.flesch_kincaid_grade}`;
  mEase.textContent = met.readability.flesch_reading_ease;
  barGrade.style.width = `${Math.min(100, (met.readability.flesch_kincaid_grade / 18) * 100)}%`;

  // Render Sentence Heatmap
  renderHeatmap();

  // Auto-select first or most AI-intensive sentence
  let bestIdx = 0;
  let maxP = -1;
  data.sentences.forEach((s, idx) => {
    if (s.ai_probability > maxP) {
      maxP = s.ai_probability;
      bestIdx = idx;
    }
  });
  selectSentence(bestIdx);

  // Smooth scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Render Document Sentence Heatmap
function renderHeatmap() {
  if (!currentAnalysis) return;

  heatmapBox.innerHTML = '';
  const sentences = currentAnalysis.sentences;

  sentences.forEach((s) => {
    // Filter conditions
    if (currentFilter === 'ai-only' && s.ai_percentage < 45) {
      return;
    }
    if (currentFilter === 'cliches' && (!s.cliches || s.cliches.length === 0)) {
      return;
    }

    const span = document.createElement('span');
    span.className = `sent-span ${s.color_class}`;
    span.dataset.index = s.index;

    // Sentence Number Pill
    const tag = document.createElement('span');
    tag.className = 'sent-num-tag';
    tag.textContent = `[${s.index + 1}]`;
    span.appendChild(tag);

    // Text with highlighted clichés if filtered
    if (currentFilter === 'cliches' && s.cliches && s.cliches.length > 0) {
      let sentenceHtml = escapeHtml(s.sentence);
      s.cliches.forEach(cliche => {
        const regex = new RegExp(`\\b${escapeRegExp(cliche.term)}\\b`, 'gi');
        sentenceHtml = sentenceHtml.replace(regex, `<span class="cliche-highlight-word">${escapeHtml(cliche.term)}</span>`);
      });
      const textContainer = document.createElement('span');
      textContainer.innerHTML = sentenceHtml;
      span.appendChild(textContainer);
    } else {
      const textNode = document.createTextNode(s.sentence + ' ');
      span.appendChild(textNode);
    }

    // Click handler
    span.addEventListener('click', () => {
      selectSentence(s.index);
    });

    heatmapBox.appendChild(span);
  });
}

// Select Sentence for Inspector Card
function selectSentence(index) {
  if (!currentAnalysis || !currentAnalysis.sentences[index]) return;
  selectedSentenceIndex = index;
  const s = currentAnalysis.sentences[index];

  // Update active styling in heatmap
  document.querySelectorAll('.sent-span').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.index) === index);
  });

  // Populate Inspector
  inspectSentenceNum.textContent = `Sentence #${s.index + 1} Selected`;
  inspectAiBadge.textContent = `${s.ai_percentage}% AI`;

  if (s.ai_percentage >= 75) {
    inspectAiBadge.className = 'badge-mini badge-danger';
  } else if (s.ai_percentage >= 55) {
    inspectAiBadge.className = 'badge-mini badge-warning';
  } else if (s.ai_percentage >= 35) {
    inspectAiBadge.className = 'badge-mini badge-info';
  } else {
    inspectAiBadge.className = 'badge-mini badge-success';
  }

  inspectSentenceText.textContent = `"${s.sentence}"`;
  inspectPpl.textContent = s.perplexity.toFixed(1);
  inspectTop10.textContent = `${Math.round(s.top10_ratio * 100)}%`;
  inspectNeural.textContent = `${Math.round(s.neural_score * 100)}%`;

  // Diagnostic Evidence List
  inspectEvidenceList.innerHTML = '';
  s.reasons.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    inspectEvidenceList.appendChild(li);
  });

  // Clichés Box
  if (s.cliches && s.cliches.length > 0) {
    inspectClichesBox.style.display = 'block';
    inspectClicheTags.innerHTML = '';
    s.cliches.forEach(c => {
      const chip = document.createElement('span');
      chip.className = 'cliche-chip';
      chip.textContent = c.term;
      chip.title = c.explanation;
      inspectClicheTags.appendChild(chip);
    });
  } else {
    inspectClichesBox.style.display = 'none';
  }
}

// Export PDF Report
async function exportPdf() {
  if (!currentAnalysis) {
    alert('Please scan a document before exporting an audit report.');
    return;
  }

  const prevText = exportPdfBtn.innerHTML;
  exportPdfBtn.innerHTML = `<span>Generating PDF Report...</span>`;
  exportPdfBtn.disabled = true;

  try {
    const res = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis: currentAnalysis })
    });

    if (!res.ok) {
      throw new Error('Failed to generate PDF report from server.');
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `veritas_ai_originality_report_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  } catch (err) {
    alert(`Export Error: ${err.message}`);
  } finally {
    exportPdfBtn.innerHTML = prevText;
    exportPdfBtn.disabled = false;
  }
}

// Helpers
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
