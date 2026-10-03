// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Veritas AI Detector — Frontend Playwright Smoke Tests
 * Validates:
 * 1. Page loads successfully with no browser console or runtime errors.
 * 2. All 74 DOM element IDs queried by app.js exist in the document.
 * 3. Pasting a 100-word sample and clicking analyze renders a valid 4-class verdict.
 * 4. Client-side download buttons (#btnDownloadJson, #btnDownloadCsv, #btnResetReport) exist.
 * 5. Responsive mobile viewport (375px width) renders without horizontal scroll overflow.
 * 6. Language toggle switches between English and Traditional Chinese cleanly.
 */

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:8003';

// Exhaustive catalog of all 74 element IDs referenced by frontend/app.js
const REQUIRED_APP_IDS = [
  'activeResultsContent',
  'btnAnalyze',
  'btnClear',
  'btnCloseShortcuts',
  'btnCopyReport',
  'btnDownloadCsv',
  'btnDownloadJson',
  'btnHelpShortcuts',
  'btnModeEdit',
  'btnModeHeatmap',
  'btnNextSentence',
  'btnPaste',
  'btnPrevSentence',
  'btnResetReport',
  'btnUpload',
  'confidenceTag',
  'dragDropOverlay',
  'editorDropZone',
  'engineStatus',
  'engineStatusText',
  'fileUploadInput',
  'flaggedWords',
  'flaggedWordsList',
  'formError',
  'heatmapSentCountBadge',
  'heatmapViewer',
  'inspectorContent',
  'inspectorTitle',
  'introDesc',
  'introTitle',
  'langToggleBtn',
  'langToggleLabel',
  'lengthWarning',
  'mathValAffinity',
  'mathValBinoculars',
  'mathValBurstiness',
  'mathValDiscourse',
  'mathValRichness',
  'pctAIGen',
  'pctAIRefined',
  'pctHuman',
  'pctHumanRefined',
  'qbComparisonGroup',
  'qbHeadlineBanner',
  'qbHeadlineText',
  'resultsPlaceholder',
  'sampleSelect',
  'samplesLabel',
  'scoreNumber',
  'sentCountLabel',
  'sentenceCountsSummary',
  'shortcutsModal',
  'shortcutsTitle',
  'spectrumBar',
  'stackBarAI',
  'stackBarAIRefined',
  'stackBarHuman',
  'stackBarHumanRefined',
  'telemetryLatency',
  'telemetryMemory',
  'telemetryThreads',
  'textInput',
  'themeToggleBtn',
  'thresholdInput',
  'uncertainAlertBanner',
  'valContractions',
  'valGrade',
  'valWps',
  'verdictDescription',
  'verdictIconBox',
  'verdictSignalDisclaimer',
  'verdictTitle',
  'wordCountLabel',
  'wordGuideBadge',
];

const SAMPLE_100_WORDS = `The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance between centralized state regulation and private merchant enterprise. The Senate maintained rigorous oversight of the state galley fleets, which operated along fixed commercial routes connecting the eastern Mediterranean with Western Europe. Auctioned to private syndicates for individual voyages, these state-owned vessels ensured military defense while offering reliable cargo space for precious spices, silk, and silver bullion. Simultaneously, privately built round ships transported bulk commodities such as grain, timber, and wine under comprehensive maritime statutes that regulated crew sizes and shipboard safety standards across centuries.`;

test.describe('Veritas AI Detector — Frontend Smoke Suite', () => {
  test('1. Page loads cleanly with zero console or uncaught errors', async ({ page }) => {
    const consoleErrors = [];
    const pageErrors = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    page.on('pageerror', (err) => {
      pageErrors.push(err.message);
    });

    const response = await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    expect(response).not.toBeNull();
    expect(response.status()).toBe(200);

    const title = await page.title();
    expect(title).toContain('Veritas');

    // Wait 500ms for health check to resolve
    await page.waitForTimeout(500);

    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('2. Every element ID used by app.js exists in the DOM', async ({ page }) => {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });

    for (const id of REQUIRED_APP_IDS) {
      const locator = page.locator('#' + id);
      const count = await locator.count();
      expect(count, `Element #${id} should exist in the DOM`).toBe(1);
    }
  });

  test('3. Pasting a 100-word sample and clicking analyze displays a verdict and renders heatmap', async ({ page }) => {
    await page.goto(BASE_URL, { waitUntil: 'networkidle' });

    // Fill the editor with 100+ words
    const textarea = page.locator('#textInput');
    await textarea.fill(SAMPLE_100_WORDS);

    // Verify word counter reflects input
    const wordCount = page.locator('#wordCountLabel');
    await expect(wordCount).toContainText('words');

    // Click Analyze button
    const btnAnalyze = page.locator('#btnAnalyze');
    await expect(btnAnalyze).toBeEnabled();
    await btnAnalyze.click();

    // Results container should become visible
    const resultsContainer = page.locator('#activeResultsContent');
    await expect(resultsContainer).toBeVisible({ timeout: 10000 });

    // Verdict title should be rendered with legitimate content
    const verdictTitle = page.locator('#verdictTitle');
    await expect(verdictTitle).toBeVisible();
    const verdictText = await verdictTitle.textContent();
    expect(verdictText.trim().length).toBeGreaterThan(1);
    expect(verdictText.trim()).not.toBe('—');

    // Score number should be present
    const scoreNum = page.locator('#scoreNumber');
    await expect(scoreNum).toBeVisible();

    // Highlights view should be active
    const heatmapViewer = page.locator('#heatmapViewer');
    await expect(heatmapViewer).toBeVisible();

    // Inspector should show first sentence
    const inspectorTitle = page.locator('#inspectorTitle');
    await expect(inspectorTitle).toContainText('Sentence');
  });

  test('4. Client-side download buttons and clear action exist and are visible in results', async ({ page }) => {
    await page.goto(BASE_URL, { waitUntil: 'networkidle' });

    // Run quick analysis
    await page.locator('#textInput').fill(SAMPLE_100_WORDS);
    await page.locator('#btnAnalyze').click();
    await expect(page.locator('#activeResultsContent')).toBeVisible({ timeout: 10000 });

    // Check JSON, CSV, Copy, and Reset buttons
    const btnJson = page.locator('#btnDownloadJson');
    const btnCsv = page.locator('#btnDownloadCsv');
    const btnCopy = page.locator('#btnCopyReport');
    const btnReset = page.locator('#btnResetReport');

    await expect(btnJson).toBeVisible();
    await expect(btnCsv).toBeVisible();
    await expect(btnCopy).toBeVisible();
    await expect(btnReset).toBeVisible();

    // Test clear action
    await btnReset.click();
    await expect(page.locator('#activeResultsContent')).toBeHidden();
    await expect(page.locator('#resultsPlaceholder')).toBeVisible();
    const editorVal = await page.locator('#textInput').inputValue();
    expect(editorVal).toBe('');
  });

  test('5. Responsive layout functions cleanly at 375px mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });

    // Ensure critical interactive elements are visible
    await expect(page.locator('#textInput')).toBeVisible();
    await expect(page.locator('#btnAnalyze')).toBeVisible();
    await expect(page.locator('#langToggleBtn')).toBeVisible();
    await expect(page.locator('#themeToggleBtn')).toBeVisible();

    // Verify no horizontal overflow beyond viewport
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    expect(bodyWidth).toBeLessThanOrEqual(376);
  });

  test('6. Language toggle switches between English and Traditional Chinese UI copy', async ({ page }) => {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });

    const langBtn = page.locator('#langToggleBtn');
    const langLabel = page.locator('#langToggleLabel');
    const analyzeBtn = page.locator('#btnAnalyze .btn-label');

    // Default English state
    await expect(langLabel).toHaveText('繁中');
    await expect(analyzeBtn).toHaveText('Check text');

    // Click to toggle to Traditional Chinese
    await langBtn.click();
    await expect(langLabel).toHaveText('EN');
    await expect(analyzeBtn).toHaveText('開始檢測');

    const htmlLang = await page.getAttribute('html', 'lang');
    expect(htmlLang).toBe('zh-TW');

    // Click again to toggle back to English
    await langBtn.click();
    await expect(langLabel).toHaveText('繁中');
    await expect(analyzeBtn).toHaveText('Check text');
  });
});
