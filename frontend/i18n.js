/**
 * Veritas AI Detector — Internationalization (i18n) Module
 * Supports English (en) and Traditional Chinese (zh-TW).
 * Only UI copy/labels are translated; element IDs and API contracts are preserved.
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.VeritasI18N = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var translations = {
    en: {
      lang_code: "en",
      lang_toggle_btn_label: "繁中",
      lang_toggle_aria: "切換為繁體中文 / Switch to Traditional Chinese",
      lang_toggle_title: "切換語言 (Language: EN / 繁體中文)",

      // Topbar
      brand_name: "Veritas",
      engine_title: "Detection runs on this computer. Nothing is sent online.",
      engine_connecting: "Connecting to engine",
      engine_online: "Offline engine ready, {threads} CPU threads",
      engine_offline: "Offline: engine stopped",
      engine_offline_title: "Detection engine is offline. Start the server with: python run.py --port 8003",
      engine_offline_error: "Cannot analyze text: Local detection engine is offline. Start the server with 'python run.py --port 8003' in your terminal.",
      offline_banner_title: "Local detection engine is offline",
      offline_banner_desc: "Veritas runs entirely on your local machine. Please start the engine in your terminal with: <code>python run.py --port 8003</code>",
      shortcuts_btn_aria: "Keyboard shortcuts",
      shortcuts_btn_title: "Keyboard shortcuts (?)",
      theme_toggle_dark: "Switch to dark theme",
      theme_toggle_light: "Switch to light theme",

      // Intro
      intro_title: "Who wrote this?",
      intro_desc: "Paste an essay, article or report. Veritas marks each sentence as human, AI-refined or AI-generated, and never sends your text online.",

      // Samples
      samples_label: "Try a sample",
      chip_ai_pure: "AI essay",
      chip_ai_refined_ai: "Paraphrased AI",
      chip_human_refined_ai: "Polished human",
      chip_human_pure: "Human",
      chip_human_esl: "ESL student",
      sample_select_default: "Benchmark set (30)",
      sample_select_optgroup: "QuillBot comparison set",

      // Workspace & Sheet tools
      tab_write: "Write",
      tab_highlights: "Highlights",
      btn_upload: "Upload file",
      btn_paste: "Paste",
      text_placeholder: "Paste or type your text here. 80 to 2,000 words gives the most reliable result.\n\nYou can also drop a PDF, Word or text file onto this page.",
      drop_title: "Drop to check this file",
      drop_desc: "PDF, Word (.docx) or plain text",

      // Counts & Footer actions
      word_single: "word",
      words_plural: "words",
      sent_single: "sentence",
      sents_plural: "sentences",
      guide_rec: "80–2,000 words recommended",
      guide_add: "Add {count} more for a reliable result",
      guide_over: "Only the first 2,000 words are checked",
      guide_ok: "Good length",
      uncertainty_cutoff: "Uncertainty cutoff",
      uncertainty_title: "Below this confidence, Veritas withholds a verdict instead of guessing",
      btn_clear: "Clear",
      btn_check_text: "Check text",
      btn_checking: "Checking",
      btn_reading_file: "Reading file",

      // How to read the marks (Empty state)
      key_title: "How to read the marks",
      key_lead: "After you check a text, each sentence in the document is marked in one of four ways.",
      mk_human_label: "Plain ink",
      mk_human_desc: "Natural variation in rhythm and word choice.",
      mk_human_title: "Human-written.",
      mk_human_refined_label: "Teal underline",
      mk_human_refined_desc: "A human draft polished by an AI tool.",
      mk_human_refined_title: "Human, AI-refined.",
      mk_ai_refined_label: "Amber highlight",
      mk_ai_refined_desc: "AI text run through a rewriter or humanizer.",
      mk_ai_refined_title: "AI, then paraphrased.",
      mk_ai_label: "Pink highlight",
      mk_ai_desc: "Reads as direct output from a language model.",
      mk_ai_title: "AI-generated.",
      key_note: "Detection is a statistical estimate, not proof. When the signals are borderline, Veritas says so instead of guessing.",

      // Active results & Classes
      score_caption: "of the text is likely AI",
      cls_ai_generated: "AI-generated",
      cls_ai_refined: "AI-generated, then paraphrased",
      cls_human_refined: "Human-written, AI-refined",
      cls_human: "Human-written",
      cls_short_ai: "AI",
      cls_short_ai_refined: "Paraphrased AI",
      cls_short_human_refined: "AI-refined human",
      cls_short_human: "Human",

      // Verdict card
      verdict_uncertain: "Not enough signal to decide",
      confidence_suffix: "confidence",
      sentence_counts_summary: "{flagged} of {total} sentences marked as AI",
      verdict_disclaimer: "Verdicts are statistical signals, not proof of authorship. Results are unreliable for text under 80 words (150+ recommended).",
      uncertain_alert: "The signals fall below your uncertainty cutoff, so Veritas won't call this one. Try a longer passage or lower the cutoff.",
      uncertain_alert_lead: "Verdict withheld.",

      // Inspector
      inspector_title_sentence: "Sentence {idx} of {total}",
      inspector_no_sentences: "No sentences",
      ai_likelihood_suffix: "AI likelihood",

      // Writing signals
      signals_title: "Writing signals",
      flagged_words_title: "Words often overused by AI",
      sig_burstiness: "Sentence rhythm",
      sig_burstiness_sub: "burstiness",
      sig_richness: "Vocabulary range",
      sig_richness_sub: "lexical richness",
      sig_discourse: "Connective tone",
      sig_discourse_sub: "discourse polarity",
      sig_binoculars: "Predictability ratio",
      sig_binoculars_sub: "binoculars",
      sig_affinity: "Authorial affinity",
      sig_affinity_sub: "combined score",
      sig_grade: "Reading grade",
      sig_grade_sub: "Flesch–Kincaid",
      sig_wps: "Words per sentence",
      sig_contractions: "Contractions",
      sig_contractions_sub: "per 100 words",

      // Telemetry
      engine_info_title: "Engine",
      tel_time: "Time to check",
      tel_memory: "Memory in use",
      tel_runtime: "Runtime",
      tel_runtime_val: "ONNX, INT8 on CPU",
      tel_threads: "CPU threads",
      tel_size: "Model size on disk",
      tel_size_val: "22 MB",
      tel_network: "Network access",
      tel_network_val: "None",

      // Action buttons
      btn_copy_summary: "Copy summary",
      btn_copied: "Copied",
      btn_copy_blocked: "Copy blocked by browser",
      btn_json: "JSON",
      btn_csv: "CSV",
      btn_clear_results: "Clear",
      results_disclaimer: "Results are unreliable for text under 80 words (150+ recommended) and are signals, not proof of authorship.",

      // Method & limits
      limits_title: "Method & limits",
      limits_intro: "Documented limits from peer-reviewed literature (see README Limitations for S-numbers):",
      limit_1_title: "Non-native English false positives:",
      limit_1_body: "Non-native English writers can be wrongly flagged as AI. Liang et al. (Patterns 2023, S11) found 7 detectors averaged 61.22% false positives on 91 TOEFL essays vs ~5.19% on native essays due to lower perplexity and simpler vocabulary.",
      limit_2_title: "Short text is unreliable:",
      limit_2_body: "Texts under 80 words lack reliable statistical signatures (Ghostbuster S10 degrades on \u2264100 tokens; Fast-DetectGPT S7 accuracy scales with length; Pangram S1 drops to 73.32% under 50 words). 150+ words is recommended.",
      limit_3_title: "Paraphrasing and humanizers:",
      limit_3_body: "Paraphrased/humanized text is the hardest case. In DAMAGE (S16), detection of humanized text fell to 60.04% for GPTZero and 28.23% for Binoculars. Iterative paraphrasing (PADBen S3) breaks statistical detectors.",
      limit_4_title: "Vendor figures are self-reported:",
      limit_4_body: "Commercial accuracy claims (e.g. Pangram S1/S13) are self-reported. Independent RAID benchmarks (S14) show detectors are domain-biased and not robust enough for high-stakes sole-arbiter use.",
      limit_5_title: "Signals, not proof of authorship:",
      limit_5_body: "A score is a probabilistic signal, not proof of authorship (Pangram S1/S13). Memorized texts like the US Constitution can be flagged as machine-generated (Binoculars S6).",

      // Colophon footer
      colophon_text: "Veritas runs entirely on this computer. Results are probabilities, so use them as one input alongside your own judgment.",

      // Keyboard shortcuts modal
      shortcuts_title: "Keyboard shortcuts",
      sc_close_aria: "Close shortcuts dialog",
      sc_help_desc: "Open or close this shortcuts help",
      sc_check_desc: "Check document text",
      sc_next_desc: "Next sentence in highlights view",
      sc_prev_desc: "Previous sentence in highlights view",
      sc_esc_desc: "Close help dialog / dismiss overlay",

      // Dynamic messages
      err_empty_input: "Paste or type some text first. 80 words or more gives a reliable result.",
      err_unsupported_file: '"{name}" isn\'t supported. Upload a PDF, Word (.docx) or .txt file.',
      err_cant_reach_engine: "Can't reach the Veritas engine. Make sure run.py is still running, then try again.",
      err_engine_fallback: "The engine couldn't check this text.",
      err_file_fallback: "That file couldn't be read.",
      err_clipboard_blocked: "Your browser blocked clipboard access. Press Ctrl+V in the text box instead.",
      err_samples_loading: "Samples are still loading. Try again in a moment.",
      export_summary_title: "Veritas AI detection summary",
      export_verdict_withheld: "Withheld (below uncertainty cutoff)",
      export_breakdown: "Breakdown",
      export_words: "Words",
      export_sentences: "sentences",
      export_checked_offline: "Checked offline in",
    },

    "zh-TW": {
      lang_code: "zh-TW",
      lang_toggle_btn_label: "EN",
      lang_toggle_aria: "切換為英文 / Switch to English",
      lang_toggle_title: "切換語言 (Language: 繁體中文 / English)",

      // Topbar
      brand_name: "Veritas",
      engine_title: "偵測完全在本機電腦端執行，絕不上傳任何文字。",
      engine_connecting: "正在連線至推論引擎",
      engine_online: "離線推論引擎就緒，{threads} 個 CPU 執行緒",
      engine_offline: "離線：請啟動伺服器",
      engine_offline_title: "離線引擎未連線。請在終端機啟動：python run.py --port 8003",
      engine_offline_error: "無法分析文本：本機偵測引擎尚未啟動。請在終端機執行「python run.py --port 8003」。",
      offline_banner_title: "本機偵測引擎尚未啟動",
      offline_banner_desc: "Veritas 偵測完全於您的本機電腦離線執行。請於終端機執行以下指令啟動引擎：<code>python run.py --port 8003</code>",
      shortcuts_btn_aria: "鍵盤快捷鍵",
      shortcuts_btn_title: "鍵盤快捷鍵 (?)",
      theme_toggle_dark: "切換為深色主題",
      theme_toggle_light: "切換為淺色主題",

      // Intro
      intro_title: "這篇文章是誰寫的？",
      intro_desc: "貼上文章、報告或論文。Veritas 逐句分析標記為真人撰寫、AI 潤飾或 AI 生成，全程離線不送出任何資料。",

      // Samples
      samples_label: "載入範例文字",
      chip_ai_pure: "純 AI 文章",
      chip_ai_refined_ai: "AI 改寫換詞",
      chip_human_refined_ai: "真人草稿經 AI 潤飾",
      chip_human_pure: "真人寫作",
      chip_human_esl: "非母語學生習作",
      sample_select_default: "基準測試集 (30 篇)",
      sample_select_optgroup: "QuillBot 對照驗證集",

      // Workspace & Sheet tools
      tab_write: "編輯文字",
      tab_highlights: "重點標記",
      btn_upload: "上傳檔案",
      btn_paste: "貼上",
      text_placeholder: "在此貼上或輸入欲分析的文字。建議長度為 80 至 2,000 字可得最可靠之結果。\n\n您也可以直接將 PDF、Word 或純文字檔拖曳至此處。",
      drop_title: "拖放檔案以開始檢測",
      drop_desc: "支援 PDF、Word (.docx) 或純文字 (.txt)",

      // Counts & Footer actions
      word_single: "字",
      words_plural: "字",
      sent_single: "句",
      sents_plural: "句",
      guide_rec: "建議 80–2,000 字",
      guide_add: "請再增加 {count} 字以達可靠分析",
      guide_over: "僅檢測前 2,000 字",
      guide_ok: "字數適中",
      uncertainty_cutoff: "不確定性門檻",
      uncertainty_title: "低於此信心度時，Veritas 將保留判定而非憑空猜測",
      btn_clear: "清除",
      btn_check_text: "開始檢測",
      btn_checking: "檢測中",
      btn_reading_file: "讀取檔案中",

      // How to read the marks (Empty state)
      key_title: "標記判讀指南",
      key_lead: "完成分析後，文檔中的每一句話將標記為四種類別之一。",
      mk_human_label: "純色無標記",
      mk_human_desc: "句式節奏與用詞具備自然的豐富變化。",
      mk_human_title: "真人撰寫。",
      mk_human_refined_label: "藍綠色底線",
      mk_human_refined_desc: "由真人起草後經語言模型修改潤色。",
      mk_human_refined_title: "真人寫作，經 AI 潤飾。",
      mk_ai_refined_label: "琥珀色高亮",
      mk_ai_refined_desc: "由模型生成後再經同義詞替換或降 AI 工具改寫。",
      mk_ai_refined_title: "AI 生成後經改寫。",
      mk_ai_label: "粉紅色高亮",
      mk_ai_desc: "呈現語言模型典型之高度可預測輸出模式。",
      mk_ai_title: "AI 直接生成。",
      key_note: "偵測結果為統計機率估算而非絕對證明。當特徵處於邊界模糊帶時，Veritas 將明示信號不足而非憑空猜測。",

      // Active results & Classes
      score_caption: "的文字內容疑似由 AI 生成或改寫",
      cls_ai_generated: "純 AI 生成",
      cls_ai_refined: "AI 生成後再改寫",
      cls_human_refined: "真人撰寫、AI 潤飾",
      cls_human: "純真人撰寫",
      cls_short_ai: "AI 生成",
      cls_short_ai_refined: "AI 改寫",
      cls_short_human_refined: "AI 潤飾",
      cls_short_human: "真人",

      // Verdict card
      verdict_uncertain: "信號不足以做出確切判定",
      confidence_suffix: "信心度",
      sentence_counts_summary: "共 {total} 句中，有 {flagged} 句標記為 AI 相關",
      verdict_disclaimer: "判定為統計機率信號而非著作權歸屬之證明。少於 80 字之短文本結果不可靠（建議 150 字以上）。",
      uncertain_alert: "特徵信心度低於您設定的不確定性門檻，Veritas 選擇不貿然判斷。請嘗試更長的篇幅或調低門檻。",
      uncertain_alert_lead: "保留判定結果。",

      // Inspector
      inspector_title_sentence: "第 {idx} 句（共 {total} 句）",
      inspector_no_sentences: "尚無句子分析",
      ai_likelihood_suffix: "AI 可能性",

      // Writing signals
      signals_title: "寫作統計特徵",
      flagged_words_title: "AI 常見過度使用詞彙",
      sig_burstiness: "句式節奏感",
      sig_burstiness_sub: "突發變異 (burstiness)",
      sig_richness: "詞彙多樣度",
      sig_richness_sub: "詞彙豐富度",
      sig_discourse: "轉折與語氣銜接",
      sig_discourse_sub: "語篇極性",
      sig_binoculars: "預測難度比率",
      sig_binoculars_sub: "雙眼比率 (binoculars)",
      sig_affinity: "作者特徵綜合評分",
      sig_affinity_sub: "綜合特徵分數",
      sig_grade: "可讀性年級指標",
      sig_grade_sub: "Flesch–Kincaid",
      sig_wps: "平均每句詞數",
      sig_contractions: "縮寫使用率",
      sig_contractions_sub: "每百詞出現率",

      // Telemetry
      engine_info_title: "推論引擎資訊",
      tel_time: "分析耗時",
      tel_memory: "記憶體使用量",
      tel_runtime: "執行環境",
      tel_runtime_val: "ONNX, INT8 CPU 推論",
      tel_threads: "CPU 執行緒數",
      tel_size: "模型磁碟大小",
      tel_size_val: "22 MB",
      tel_network: "聯網請求",
      tel_network_val: "無 (完全離線)",

      // Action buttons
      btn_copy_summary: "複製摘要",
      btn_copied: "已複製",
      btn_copy_blocked: "瀏覽器封鎖剪貼簿權限",
      btn_json: "JSON",
      btn_csv: "CSV",
      btn_clear_report: "清除結果",
      results_disclaimer: "少於 80 字之短文本結果不可靠（建議 150 字以上）；檢測指標僅為機率信號，非作者身分之法律證明。",

      // Method & limits
      limits_title: "方法與限制說明",
      limits_intro: "引述自同儕審查學術文獻之已確認限制（編號參閱 README Limitations 之 S 系列文獻）：",
      limit_1_title: "非英語母語者的高偽陽性率：",
      limit_1_body: "非母語英文寫作者容易被誤判為 AI。Liang 等人（Patterns 2023, S11）針對 91 篇托福作文進行測試，發現 7 款商用偵測器之偽陽性率高達 61.22%（母語者對照組僅 ~5.19%），主因在於非母語者困惑度較低且詞彙較單純。",
      limit_2_title: "短文本統計特徵不可靠：",
      limit_2_body: "低於 80 字的文字缺乏足夠的統計特徵（Ghostbuster S10 在 \u2264100 tokens 時顯著衰退；Fast-DetectGPT S7 準確率隨長度遞減；Pangram S1 在 50 字以下降至 73.32%）。建議輸入 150 字以上。",
      limit_3_title: "改寫與降 AI 人工化攻擊：",
      limit_3_body: "經改寫或 Humanizer 處理之文本是最嚴峻的挑戰。DAMAGE（S16）測試中，對降 AI 人工化文本之檢測準確率，GPTZero 跌落至 60.04%，Binoculars 跌至 28.23%。反覆多輪改寫（PADBen S3）會瓦解統計型偵測器。",
      limit_4_title: "廠商宣稱準確率多為自行回報：",
      limit_4_body: "商業產品宣稱的準確率（如 Pangram S1/S13）多屬廠商自行報導。獨立的 RAID 基準測試（S14）顯示現有偵測器具顯著領域偏差，不足以在關鍵高風險場景中單獨作為裁決證據。",
      limit_5_title: "機率信號而非著作身分證明：",
      limit_5_body: "評分純屬機率統計信號，絕非原創身分之法律證明（Pangram S1/S13）。被模型高度記憶之經典文本（如美國憲法）極易被誤判為機器生成（Binoculars S6）。",

      // Colophon footer
      colophon_text: "Veritas 完全在本機電腦端離線運行。檢測結果皆為統計機率，請僅將其作為您獨立判斷之參考依據之一。",

      // Keyboard shortcuts modal
      shortcuts_title: "鍵盤快捷鍵",
      sc_close_aria: "關閉快捷鍵對話框",
      sc_help_desc: "開啟或關閉此快捷鍵說明",
      sc_check_desc: "開始檢測文件",
      sc_next_desc: "在重點標記檢視中切換至下一句",
      sc_prev_desc: "在重點標記檢視中切換至上一句",
      sc_esc_desc: "關閉對話框 / 關閉浮層",

      // Dynamic messages
      err_empty_input: "請先貼上或輸入文字。80 字以上可獲得較可靠之檢測結果。",
      err_unsupported_file: "不支援「{name}」檔案格式。請上傳 PDF、Word (.docx) 或純文字 (.txt) 檔案。",
      err_cant_reach_engine: "無法連線至 Veritas 離線推論引擎。請確認 run.py 仍在執行中，然後重試。",
      err_engine_fallback: "推論引擎無法檢測此段文字。",
      err_file_fallback: "無法讀取該檔案。",
      err_clipboard_blocked: "您的瀏覽器限制了剪貼簿存取權限。請在文字框內直接按 Ctrl+V 貼上。",
      err_samples_loading: "範例文字載入中，請稍候片刻再試。",
      export_summary_title: "Veritas AI 偵測分析摘要",
      export_verdict_withheld: "保留判定（低於不確定性門檻）",
      export_breakdown: "類別分佈佔比",
      export_words: "總詞數",
      export_sentences: "句子數",
      export_checked_offline: "離線推論耗時",
    }
  };

  return {
    translations: translations,
    getTranslation: function (lang, key, replacements) {
      var dict = translations[lang] || translations.en;
      var str = dict[key] !== undefined ? dict[key] : (translations.en[key] !== undefined ? translations.en[key] : key);
      if (replacements && typeof replacements === "object") {
        Object.keys(replacements).forEach(function (rKey) {
          str = str.replace(new RegExp("\\{" + rKey + "\\}", "g"), replacements[rKey]);
        });
      }
      return str;
    },
    supportedLanguages: ["en", "zh-TW"]
  };
});
