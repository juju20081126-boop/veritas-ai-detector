# 🛡️ Veritas AI — QuillBot-Style Offline AI Writing Detector

> [!WARNING]
> **舊版評估指標警語（於合成且存在資料外洩之資料集上測得；教師模型數據從未實際訓練測量）—— 已廢止並由真實資料報告取代；請參閱 `data/reports/FRONTIER_DETECTION_REPORT.md`（生成後生效）。**

Veritas AI 是一款完全離線運行、達到生產級品質的 4 類別 AI 寫作偵測系統。本專案之設計旨在重現 **QuillBot AI Detector** 的使用者體驗、數位鑑識方法論以及細緻的四分類檢測能力。本系統專為**低階個人電腦硬體**打造，完全無須依賴 GPU 顯示卡加速或任何雲端連線，透過 INT8 動態量化技術與輕量級文體計量元分類器（Stylometric Meta-Classifier）實現極致的高效推論。

---

## Table of Contents

- [Key Highlights & Capabilities](#key-highlights--capabilities) — 核心亮點與功能特點
- [Project Structure](#project-structure) — 專案結構與模組歸屬
- [How It Works: Current Engine Architecture](#how-it-works-current-engine-architecture) — 運作原理：現行推論引擎架構
  - [1. Quantized ONNX INT8 Neural Student](#1-quantized-onnx-int8-neural-student) — 量化 ONNX INT8 神經學生模型
  - [2. 20 Tabular Stylometric Features](#2-20-tabular-stylometric-features) — 20 維表格文體計量特徵
  - [3. Meta-Classifier Fusion](#3-meta-classifier-fusion) — 元分類器融合
  - [4. Temperature Calibration & Confidence Gating](#4-temperature-calibration--confidence-gating) — 溫度校準與不確定性門檻判定
  - [5. Hierarchical Chunking & Sentence-Level Smoothing](#5-hierarchical-chunking--sentence-level-smoothing) — 分層語塊切分與句級平滑機制
  - [6. FastAPI Server Endpoints & Response Schema](#6-fastapi-server-endpoints--response-schema) — FastAPI 伺服器端點與回應綱要
- [Quick Start Guide](#quick-start-guide) — 快速入門指南
  - [1. Installation on Low-End PC (Offline Ready)](#1-installation-on-low-end-pc-offline-ready) — 低階電腦安裝（完全離線就緒）
  - [2. Launch Local Web UI](#2-launch-local-web-ui) — 啟動本機 Web 介面
  - [3. Command-Line Interface (CLI)](#3-command-line-interface-cli) — 命令列介面 (CLI)
- [Verification & Empirical Evaluation](#verification--empirical-evaluation) — 驗證與實證評估
- [Retraining & Data-Refresh Pipeline](#retraining--data-refresh-pipeline) — 重訓與資料更新管線
  - [How to Run the Real-Data Pipeline](#how-to-run-the-real-data-pipeline) — 如何執行真實資料處理管線
  - [Cloud Jupyter Notebooks (Kaggle / Google Colab)](#cloud-jupyter-notebooks-kaggle--google-colab) — 雲端 Jupyter 筆記本
- [State-of-the-Art Research & Mathematical Formulations (2024–2026)](#state-of-the-art-research--mathematical-formulations-20242026) — 前沿研究與數學公式 (2024–2026)
- [QuillBot Comparison Sheet (30 Hand-Check Samples)](#quillbot-comparison-sheet-30-hand-check-samples) — QuillBot 對照工作表（30 篇人工檢核範例）
- [Hand-collecting detector verdicts (no automation, ToS-safe, <=200 texts)](#hand-collecting-detector-verdicts-no-automation-tos-safe-200-texts) — 手動收集偵測器判定（無自動化、符合服務條款、<=200 篇）
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq) — 常見問題解答 (FAQ)
- [Glossary](#glossary) — 專有名詞辭典 (Glossary)
- [Limitations](#limitations) — 限制與已知邊界
- [License & Acknowledgments](#license--acknowledgments) — 授權條款與致謝

---

## ⚡ Key Highlights & Capabilities

**核心亮點與功能特點：**

- **QuillBot 等級之 4 類別分類體系**：
  1. 🟥 **AI-generated（純 AI 生成）**：純粹由自回歸語言模型直接生成之文字（如 GPT-4o、Claude 3.5、Gemini 1.5、Llama 3.3、Qwen 2.5、DeepSeek-V3）。
  2. 🟧 **AI-generated & AI-refined（AI 生成後經改寫）**：由 AI 起草後，再透過自動化改寫工具或「降 AI 人工化（Humanizer）」重寫之文本。
  3. 🟨 **Human-written & AI-refined（真人撰寫、AI 潤飾）**：由真實人類撰寫原作，後續經由大語言模型進行潤飾、文法修正或風格修飾。
  4. 🟩 **Human-written（純真人撰寫）**：真實人類的原創學術論文、個人敘事、隨筆及各類日常寫作。
  5. ⚠️ **「不確定 (Uncertain)」判定保留機制**：當校準後的決策信心度低於門檻時，偵測器將主動保留判斷，避免在特徵模糊時強行給出不精準之結論。

- **專為低階目標硬體精心最佳化**：
  - **記憶體佔用極低**：程序常駐記憶體 **&le;150 MB 峰值 RAM**（硬體上限：&le;1.5 GB，可在僅有 4GB RAM 之系統流暢運作）。
  - **計算資源受限環境運作**：僅需 **2 個 CPU 執行緒**（`intra_op_num_threads=2`），完全無須 GPU、CUDA 或 ROCm 環境。
  - **磁碟儲存空間精巧**：內建模型與執行時期資產僅佔約 **25 MB**（硬體上限：&le;500 MB）。
  - **極速推論效能**：在 2 個 CPU 執行緒下，分析 500 字篇幅之文本僅需約 **0.34 秒**（目標：&le;15 秒）。
  - **執行時期完全零 PyTorch 依賴**：發行環境僅使用 `onnxruntime` CPU 版本與 Rust 編寫之 `tokenizers`。

- **公平性與非英語母語者 (ESL) 穩健性**：
  - 專門針對國際學習者語料庫（如 TOEFL / IELTS 作文）進行最佳化與評估。
  - *（舊版未驗證宣稱：在合成資料上測得 ESL 偽陽性率為 1.2% / 0.00%，已廢止；正等待真實資料重新驗證）*。

- **多介面完整支援**：
  - **本機 Web UI**：採用純 HTML/CSS/JS 開發的響應式雙欄儀表板（零大型 npm / Node 繁重依賴）。
  - **互動式命令列 (CLI)**：提供終端機彩色文字高亮標記、各類別百分比長條圖與詳細延遲效能分析。
  - **REST API**：基於 FastAPI 構建之高效後端，便於本機整合與微服務介接。

---

## 📁 Project Structure

**專案結構與模組歸屬：**

Veritas AI 專案依循嚴格的模組化職責與代碼所有權劃分，由兩位自主代理（**Claude Code** 與 **Antigravity CLI**）依據 [`AGENTS.md`](AGENTS.md) 共同維護：

```text
veritas-ai-detector/
├── .github/                   # GitHub Actions CI 工作流程、議題範本與自動化檢核
│   └── workflows/             # CI 管線（pytest、ruff、nbformat、連結檢查、Playwright 冒煙測試）
├── backend/                   # 離線推論核心引擎、20 維文體特徵抽取器與 FastAPI 伺服器
│   ├── engine.py              # 備用啟發式偵測引擎
│   ├── runtime_engine.py      # 雙分支推論引擎（ONNX INT8 + 文體計量 + 機率校準）
│   ├── server.py              # FastAPI REST 端點實作
│   └── stylometrics.py        # 20 維表格文體特徵向量抽取器
├── data/                      # 前沿資料集、對抗攻擊套件、研究筆記與評估工作表
│   ├── _quarantine_synthetic/ # 隔離之早期合成訓練資料
│   ├── esl/                   # 國際英語學習者作文語料庫
│   ├── eval/                  # 審計結果與歷史評估數據
│   ├── research/              # 文獻回顧、攻擊目錄與商業拆解筆記
│   └── quillbot_comparison_sheet.json # 30 篇人工檢核參照範例
├── frontend/                  # 單頁 Web 儀表板、UI 樣式、國際化與冒煙測試
│   ├── app.js                 # 前端控制器與 API 通訊
│   ├── i18n.js                # 雙語辭典（英文 / 繁體中文）
│   ├── index.html             # 主儀表板標記
│   ├── style.css              # 響應式佈局、深色模式與列印樣式表
│   └── tests/                 # Playwright 端對端冒煙測試套件
├── models/                    # 蒸餾量化 ONNX 學生模型、權重與分詞器資產
│   ├── student_model_int8.onnx # 內建量化學生模型（約 22 MB）
│   ├── meta_classifier.json   # 校準之表格/神經融合權重
│   └── tokenizer/             # 本機快速分詞器配置與詞彙表
├── notebooks/                 # 互動式 Jupyter 研究、教師標註與視覺化範本
│   ├── 01_teacher_ensemble_and_labeling.ipynb
│   ├── 02_student_distillation_and_onnx_export.ipynb
│   ├── 03_gpu_finetune.ipynb
│   └── 04_results_figures.ipynb
├── samples/                   # 具完整來源證明之評估文章與參照範例文本
│   ├── algorithmic_commons_essay.md
│   ├── real_samples.py        # 具備明確來源記錄之真實前沿生成文本
│   └── sample_data.py         # 人工撰寫之示範展示文本
├── scratch/                   # 臨時診斷、單一文本檢驗與探勘腳本
└── scripts/                   # 語料庫構建、LLM 對抗管線、偵測器評估與訓練腳本
    ├── common/                # 共用資料公用函式與文本標準化
    ├── corpus/                # 前沿資料生成、提示詞收集與對抗管線
    ├── detectors/             # 學生模型與基準偵測器包裝
    ├── tests/                 # 後端與文體特徵單元測試
    ├── check_integrity.py     # 資料集雜湊與綱要完整性閘門
    ├── eval_frontier.py       # 鎖定測試集評估執行器
    └── train_detector.py      # 前沿模型訓練與知識蒸餾腳本
```

### Folder Descriptions & Ownership

**頂層目錄職責與權限劃分：**

| 目錄路徑 | 負責代理（依據 [`AGENTS.md`](AGENTS.md)） | 職責與說明 |
|---|---|---|
| `.github/` | **Antigravity** | CI/CD 自動化工作流程、Issue/PR 範本與品質檢驗。 |
| `backend/` | **Claude** | 離線推論核心引擎、20 維文體特徵抽取器與 FastAPI REST 端點。 |
| `data/` | **Claude** | 前沿資料集、對抗攻擊樣本、文獻筆記與對照評估工作表。 |
| `frontend/` | **Antigravity** | 單頁 Web 儀表板、雙語辭典 (EN / zh-TW) 與 Playwright 冒煙測試。 |
| `models/` | **Claude** | 量化 INT8 ONNX 學生模型 (~22 MB)、元分類器權重與分詞器資產。 |
| `notebooks/` | **Antigravity** | 研究、教師標註、GPU 微調與評估結果繪圖之 Jupyter 筆記本範本。 |
| `samples/` | **Antigravity** | 具備完整來源記錄之評估文章與真實前沿文本範本。 |
| `scratch/` | **Claude** | 探勘性診斷、單篇文章分析與即時除錯腳本。 |
| `scripts/` | **Claude** | 語料庫管線、LLM 對抗重寫、評估框架與模型訓練腳本。 |
| 根目錄共用檔案 | **Shared（共同維護）** | `cli.py`、`run.py`、`requirements*.txt`、`AGENTS.md` 與 `HANDOFF.md`。 |

---

## 🔬 How It Works: Current Engine Architecture

`backend/runtime_engine.py` 中的執行引擎與 `backend/server.py` 的伺服器實現了一套完全離線、極低資源消耗的偵測架構，專為一般大眾之個人電腦硬體打造（2 CPU 執行緒、&le;150 MB RAM、零 PyTorch 依賴）。

```
原始輸入文件（貼上文字 / 上傳檔案）
  │
  ├─► 正規表達式斷句器與 ~100 字段落分塊（Paragraph Chunking）
  │
  ├─► 20 維文體計量特徵擷取 (backend/stylometrics.py)
  │     （句長突發變異係數、TTR、ARI、音節常態分佈、資訊熵、連續節奏差值、DEFLATE 壓縮率、語篇銜接標記）
  │
  ├─► ONNX INT8 學生神經模型 (models/student_model_int8.onnx，透過 onnxruntime 推論)
  │     （Rust tokenizers，最大序列長度 512，限制 2 個 CPU 執行緒）
  │
  ├─► 元分類器融合 (models/meta_classifier.json)
  │     （Z 分數標準化文體特徵 + 神經輸出 Logits 藉由權重融合矩陣結合）
  │
  ├─► 溫度校準與信心度門檻控制 (Temperature Calibration & Confidence Gating)
  │     （量化後機率縮放，邊界模糊帶指派 Uncertain）
  │
  ├─► 分層句級平滑機制 (Hierarchical Sentence Smoothing，40% 當句 + 60% 語塊上下文)
  │
  ▼
QuillBot 風格之 4 類別佔比長條圖、字數加權 AI 百分比與逐句視覺化色彩標記
```

### 1. Quantized ONNX INT8 Neural Student
- **模型檔案**：`models/student_model_int8.onnx`（約 22 MB）。
- **推論執行時期**：由 `onnxruntime.InferenceSession` 載入執行，並固定設定 `intra_op_num_threads=2`。執行時期完全不載入任何 PyTorch 或 CUDA 函式庫。
- **分詞器 (Tokenizer)**：採用 Hugging Face 由 Rust 編寫之 `tokenizers`，自 `models/tokenizer/tokenizer.json` 載入。序列截斷與填補長度上限為 512 tokens。
- **神經網路輸出**：輸出 4 個類別之未歸一化對數幾率 (Logits)，對應於：
  1. `human`（純真人，索引 0）
  2. `human_ai_refined`（真人草稿、AI 潤飾，索引 1）
  3. `ai_ai_refined`（AI 生成後改寫，索引 2）
  4. `ai_generated`（純 AI 生成，索引 3）

### 2. 20 Tabular Stylometric Features
推論引擎自文本中擷取 20 維文體計量特徵向量（定義於 `backend/stylometrics.py`），這些統計特徵對文章討論之主題具備高度不變性：
1. **平均句子長度 (Sentence length mean)**：每句平均詞數。
2. **句子長度變異數 (Sentence length variance)**：全篇各句子長度之分散程度。
3. **句長變異係數 ($\lambda_{	ext{auth}}$)**：變異係數 ($\sigma / \mu$)，用以捕捉人類寫作自然的句式突發性（Burstiness）對比 LLM 規律一致的節奏。
4. **相異詞比率 (Type-Token Ratio, TTR)**：不重複相異詞數除以總詞數（詞彙豐富度指標）。
5. **平方根相異詞比率 (Root TTR / Guiraud's Index)**：$V / \sqrt{N}$，大幅消除篇幅長短造成的偏差。
6. **自動化可讀性指數 (Automated Readability Index, ARI)**：基於字元數與句子結構之客觀可讀性年級指標。
7. **Flesch 閱讀難易度 (Flesch Reading Ease)**：基於音節數與句長之經典可讀性評分。
8. **每詞平均音節數 (Mean syllables per word)**：衡量平均詞彙複雜度。
9. **音節數分散變異數 (Syllable count variance)**：多音節詞彙在文章中的分佈情況。
10. **標點符號密度 (Punctuation density)**：每 100 字中各類標點符號之出現頻率。
11. **逗號頻率 (Comma frequency)**：每句平均逗號數量。
12. **分號與冒號頻率 (Semicolon and colon frequency)**：複合子句連接符號之密度。
13. **問號頻率 (Question mark frequency)**：修辭性問句或反問句之出現密度。
14. **香農 Token 資訊熵率 (Shannon token entropy rate)**：衡量詞彙序列位置之資訊密度與多樣性。
15. **相鄰節奏變化量 ($\Delta_{	ext{rhythm}}$)**：相鄰兩句長度之平均絕對差值。
16. **DEFLATE 壓縮比率 (DEFLATE compression ratio)**：在 Lempel-Ziv 演算法下的演算法資訊複雜度（Normalized Compression Distance 代用指標）。
17. **AI 轉折標記詞密度 (AI discourse transition density)**：大語言模型典型之套路化銜接詞頻率（如 *furthermore*, *delve*, *moreover*, *testament*, *in summary*）。
18. **個人主觀語態詞頻率 (Personal voice marker frequency)**：第一人稱代名詞與親身經歷表達詞頻率（如 *I*, *my*, *we*, *personally*）。
19. **長子句比率 (Long clause ratio)**：詞數 $\ge 35$ 字之超長句子比例。
20. **短子句比率 (Short clause ratio)**：詞數 $\le 6$ 字之短句比例。

*法證鑑識防護機制 (Forensic Guardrail)*：系統內建真實主觀語態防護邏輯，用以保護文學性人類散文。當 $\lambda_{	ext{auth}} \ge 0.70$、存在顯著個人主觀標記且完全無 AI 轉折套路詞時，偵測器將防止誤判升級。

### 3. Meta-Classifier Fusion
- **參數設定**：`models/meta_classifier.json` 儲存了 `scaler_mean`、`scaler_std`、`meta_weights` 與 `meta_intercept`。
- **Z 分數標準化**：每一項文體計量特徵均透過訓練集均值與標準差進行標準化：
  $$	ilde{x}_i = rac{x_i - \mu_i}{\sigma_i}$$
- **線性 Logits 融合**：將 ONNX 學生模型產生的 4 類別神經 Logits 與 20 維標準化文體特徵進行線性融合：
  $$\mathbf{z}_{	ext{fused}} = \mathbf{W}_{	ext{neural}} \mathbf{z}_{	ext{onnx}} + \mathbf{W}_{	ext{style}} 	ilde{\mathbf{x}}_{	ext{style}} + \mathbf{b}$$

### 4. Temperature Calibration & Confidence Gating
- **溫度縮放 (Temperature Scaling)**：採用 Platt 風格之量化後機率校準模型對融合 Logits 進行縮放調整：
  $$p_c = rac{\exp(z_c / T)}{\sum_{k=1}^4 \exp(z_k / T)}$$
  其中 $T$ 為在驗證集上實證擬合之 `calibration_temperature`（儲存於 `models/meta_classifier.json`）。
- **信心度門檻控制與不確定判定**：
  若最高之校準後機率低於決策門檻（預設值為 $0.40$），或者最高機率 $< 0.45$ 且前兩大類別之機率差距 $< 0.04$，推論引擎將保留判定，將整體結果指派為 **Uncertain（不確定）**。

### 5. Hierarchical Chunking & Sentence-Level Smoothing
- **段落階層式語塊切分**：針對長文本，系統在嚴格保持句子邊界完整的前提下，將文本切分為多個約 100 字之階層式上下文語塊（Context Chunks）。
- **上下文加權平滑**：為避免極短子句出現不穩定的類別跳動，每個單句的最終預測將融合其單句局部 Logits（佔比 $40\%$）與所屬段落語塊上下文（佔比 $60\%$）：
  $$\mathbf{p}_{	ext{sentence}}^{	ext{blended}} = 0.40 \cdot \mathbf{p}_{	ext{local}} + 0.60 \cdot \mathbf{p}_{	ext{chunk}}$$
- **QuillBot 風格標題與覆蓋率計算**：
  整篇文檔之 AI 佔比，係依照判定為 `ai_generated` 或 `ai_ai_refined` 之各句子字數加權計算：
  $$	ext{AI Coverage \%} = rac{\sum_{s \in 	ext{AI Sentences}} 	ext{Words}(s)}{\sum_{s \in 	ext{All Sentences}} 	ext{Words}(s)} 	imes 100$$
  藉此生成如 `"82% of text is likely AI"` 或 `"100% of text is likely Human"` 之經典 QuillBot 風格標題。

### 6. FastAPI Server Endpoints & Response Schema
後端伺服器（`backend/server.py`）提供 5 個 REST API 端點。以下為完整規範，包含請求參數、具備明確資料型態之完整回應綱要、HTTP 錯誤狀態碼，以及在本機實體實例（`http://127.0.0.1:8003`）上驗證執行之實際 `curl` 範例。

---

#### 1. `GET /api/health`
檢查後端伺服器存活動態、模型推論執行時期資訊與程序記憶體遙測數據。

- **HTTP 方法**：`GET`
- **請求參數**：無
- **回應欄位**：
  - `status` (`str`)：伺服器運作狀態（例如 `"online"`）。
  - `architecture` (`str`)：模型架構描述（`"Distilled Student ONNX INT8 + Stylometric Meta-Classifier"`）。
  - `engine_runtime` (`str`)：推論執行時期環境（`"onnxruntime (Zero PyTorch)"`）。
  - `device` (`str`)：硬體推論裝置（`"cpu"`）。
  - `cpu_threads` (`int`)：配置之 intra-op 執行緒數（`2`）。
  - `process_ram_mb` (`float`)：目前程序常駐記憶體 (RSS) 使用量（以 MB 為單位）。
  - `target_ram_cap_mb` (`float`)：記憶體上限預算指標（`1500.0`）。
  - `timestamp` (`float`)：回應生成之 Unix Epoch 時間戳記。
- **錯誤代碼**：若遙測檢查失敗則回傳 `500 Internal Server Error`。
- **實測 `curl` 範例**：
  ```bash
  curl.exe -s http://127.0.0.1:8003/api/health
  ```
  **伺服器真實輸出**：
  ```json
  {
    "status": "online",
    "architecture": "Distilled Student ONNX INT8 + Stylometric Meta-Classifier",
    "engine_runtime": "onnxruntime (Zero PyTorch)",
    "device": "cpu",
    "cpu_threads": 2,
    "process_ram_mb": 76.0,
    "target_ram_cap_mb": 1500.0,
    "timestamp": 1791033220.3387377
  }
  ```

---

#### 2. `GET /api/samples`
取得預先載入之 4 類別代表性參考文本範例，涵蓋各種典型作者風格，供介面示範與即時測試使用。

- **HTTP 方法**：`GET`
- **請求參數**：無
- **回應欄位**：
  - 最外層為以原型標籤（`"ai_pure"`, `"ai_refined_ai"`, `"human_refined_ai"`, `"human_pure"`, `"human_esl"`）為鍵之物件，各包含：
    - `title` (`str`)：範例文字的人類可讀標題。
    - `expected_class` (`str`)：預期的標準 4 分類標籤。
    - `text` (`str`)：完整範例文字內容。
- **錯誤代碼**：若原型資料載入失敗則回傳 `500 Internal Server Error`。
- **實測 `curl` 範例**：
  ```bash
  curl.exe -s http://127.0.0.1:8003/api/samples
  ```
  **伺服器真實輸出（節錄）**：
  ```json
  {
    "ai_pure": {
      "title": "1. Pure AI-generated (GPT-4o Academic Essay)",
      "expected_class": "AI-generated",
      "text": "In the contemporary era, the rapid proliferation of artificial intelligence technologies has fundamentally reconstituted the landscape of higher education..."
    },
    "human_pure": {
      "title": "4. Human-written (Venetian Maritime Commerce)",
      "expected_class": "Human-written",
      "text": "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance between centralized state regulation..."
    }
  }
  ```

---

#### 3. `GET /api/comparison-sheet`
回傳載自 `data/quillbot_comparison_sheet.json` 之 30 篇雙欄 QuillBot 對照工作表資料，用於基準校準與行為比對。

- **HTTP 方法**：`GET`
- **請求參數**：無
- **回應欄位**：
  - 包含 30 個測試案例物件之陣列，每個物件包含：
    - `id` (`str`)：唯一識別碼（`"QB-01"` 至 `"QB-30"`）。
    - `text` (`str`)：用於橫向比對評估的文本內容。
    - `expected_class` (`str`)：真實類別標籤（`"Human-written"`, `"Human-written & AI-refined"`, `"AI-generated & AI-refined"`, 或 `"AI-generated"`）。
    - `class_id` (`int`)：整數類別索引（0 至 3）。
    - `type` (`str`)：具體的生成或作者子分類標籤。
    - `domain` (`str`)：文體或來源領域（`"academic"`, `"creative"`, `"email"`, `"technical"`, `"story"`）。
    - `word_count` (`int`)：文本詞數。
    - `notes` (`str`)：記錄生成模型、提示詞或潤飾來源之詳細脈絡備註。
- **錯誤代碼**：若資料集檔案遺失或損毀則回傳 `500 Internal Server Error`。
- **實測 `curl` 範例**：
  ```bash
  curl.exe -s http://127.0.0.1:8003/api/comparison-sheet
  ```
  **伺服器真實輸出（第一筆項目節錄）**：
  ```json
  [
    {
      "id": "QB-01",
      "text": "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance...",
      "expected_class": "Human-written",
      "class_id": 0,
      "type": "human_native",
      "domain": "academic",
      "word_count": 87,
      "notes": "Authentic academic prose with historical domain vocabulary."
    }
  ]
  ```

---

#### 4. `POST /api/detect`
對傳入之文字進行全面法證鑑識分析，回傳整篇文件之判定結果、QuillBot 風格覆蓋比例、逐句預測詳情以及 20 維文體計量特徵。

- **HTTP 方法**：`POST`
- **請求標頭**：`Content-Type: application/json`
- **請求主體欄位**：
  - `text` (`str`，必填)：欲分析之字串文本（最少需 5 個詞）。
  - `confidence_threshold` (`float`，選填)：用於保留不確定判定之校準信心度門檻（預設值為 `0.40`）。
  - `filename` (`str`，選填)：客戶端匯出檔案之來源標籤（預設值為 `"Pasted Text"`）。
- **回應欄位**：
  - `summary` (`dict`)：
    - `verdict` (`str`)：最終離散分類判定（`"Human-written"`, `"Human-written & AI-refined"`, `"AI-generated & AI-refined"`, `"AI-generated"`, 或 `"Uncertain"`）。
    - `verdict_description` (`str`)：整篇判定之鑑識說明文字。
    - `badge` (`str`)：CSS 徽章樣式識別碼（例如 `"badge-success"`）。
    - `is_uncertain` (`bool`)：若校準信心度低於門檻或類別間距過小則為 `true`。
    - `confidence` (`float`)：在 $[0.0, 1.0]$ 區間之校準決策信心度。
    - `confidence_pct` (`float`)：格式化為百分比 $[0.0, 100.0]$ 之校準信心度。
    - `quillbot_headline` (`str`)：QuillBot 風格總評標題（例如 `"100% of text is likely Human"`）。
    - `quillbot_headline_class` (`str`)：標題樣式類別（`"badge-human"`, `"badge-ai"`, `"badge-ai-refined"`）。
    - `quillbot_ai_pct` (`float`)：被判定為 AI 生成或改寫之字數加權百分比。
    - `quillbot_human_pct` (`float`)：被判定為真人撰寫或潤飾之字數加權百分比。
    - `word_count` (`int`)：輸入文本總詞數。
    - `character_count` (`int`)：總字元數。
    - `sentence_count` (`int`)：剖析之句子總數。
    - `length_warning` (`str` 或 `null`)：若詞數少於 80 字則回傳長度警語字串，否則為 `null`。
    - `elapsed_seconds` (`float`)：總推論耗時（秒）。
  - `calibrated_probabilities` (`dict[str, float]`)：4 個標準類別經校準之 Softmax 事後機率：
    - `"Human-written"` (`float`)：純真人撰寫之機率。
    - `"Human-written & AI-refined"` (`float`)：真人寫作、AI 潤飾之機率。
    - `"AI-generated & AI-refined"` (`float`)：AI 生成後再改寫之機率。
    - `"AI-generated"` (`float`)：純機器生成之機率。
  - `percentages` (`dict[str, float]`)：`"ai_generated"`, `"ai_ai_refined"`, `"human_ai_refined"`, `"human"` 之字數加權覆蓋百分比。
  - `quillbot_breakdown` (`dict`)：包含 `headline`, `headline_class`, `ai_percentage`, `human_percentage`, `segments` 之複製品結構元數據。
  - `sentences` (`list[dict]`)：逐句分析物件清單：
    - `index` (`int`)：自 0 起算之句子索引。
    - `text` (`str`)：句子原始文字。
    - `class_label` (`str`)：機率最高之類別名稱。
    - `class_key` (`str`)：機器代碼（`"human"`, `"human_ai_refined"`, `"ai_ai_refined"`, `"ai_generated"`）。
    - `color_class` (`str`)：CSS 顏色類別。
    - `highlight_class` (`str`)：文字高亮 CSS 類別（`"highlight-human"`, `"highlight-ai"` 等）。
    - `confidence` (`float`)：該句所獲類別之加權平滑信心度。
    - `ai_likelihood_pct` (`float`)：該句之 AI 總可能性百分比 ($p_{	ext{ai\_gen}} + p_{	ext{ai\_ref}}$)。
    - `probabilities` (`dict[str, float]`)：該句在 4 類別之平滑機率分佈。
    - `reasons` (`list[str]`)：鑑識判定原因標籤（例如偵測到之 AI 標記、個人語態、句式節奏特徵等）。
  - `stylometrics` (`dict`)：完整 20 維文體計量特徵向量與語言學分析細項：
    - `word_count`, `character_count`, `sentence_count` (`int`)：結構詞彙統計。
    - `readability` (`dict`)：`flesch_reading_ease` (`float`), `flesch_kincaid_grade` (`float`), `words_per_sentence` (`float`)。
    - `lexical_diversity` (`dict`)：`ttr` (`float`), `root_ttr` (`float`), `hapax_ratio` (`float`), `yule_k` (`float`), `simpsons_d` (`float`), `honore_r` (`float`)。
    - `syntax_variance` (`dict`)：`mean_length` (`float`), `std_length` (`float`), `cv_length` (`float`), `rhythm_delta` (`float`), `rhythm_curvature` (`float`), `uniformity_score` (`float`)。
    - `syllable_dispersion` (`dict`)：`mean_syllables` (`float`), `std_syllables` (`float`), `dispersion_cv` (`float`)。
    - `hyphenation` (`dict`)：`hyphenated_count` (`int`), `hyphen_rate_per_100w` (`float`), `hyphenated_samples` (`list[str]`)。
    - `entropy` (`dict`)：`shannon_entropy` (`float`), `vocab_richness_bits` (`float`)。
    - `compression_ratio` (`float`)：Zlib deflate 壓縮比率（NCD 近似代用值）。
    - `binoculars_proxy` (`float`)：Token 資訊熵除以壓縮比率。
    - `discourse_punctuation` (`dict`)：AI 標記率、人類主觀語態率、縮寫/逗號/分號比率及匹配詞彙清單。
    - `mathematical_equations` (`dict`)：解析形式之鑑識數值：`lexical_richness_omega` (`float`), `syntactic_burstiness_b` (`float`), `discourse_polarity_phi` (`float`), `binoculars_ratio_r` (`float`), `authorial_affinity_lambda` (`float`)。
    - `stylometric_ai_score` (`float`)：線性文體計量 AI 綜合評分 $[0.0, 1.0]$。
  - `mathematical_equations` (`dict`)：最外層數學公式數值複本，供儀表板快速渲染。
- **錯誤代碼**：
  - `400 Bad Request`：當 `text` 為空（`{"detail":"Text cannot be empty."}`）或詞數少於 5 個詞（`{"detail":"Text is too brief. Please enter at least 5 words."}`）時回傳。
  - `422 Unprocessable Entity`：當請求主體未通過 Schema 驗證時（例如非數值型態之 `confidence_threshold`）回傳。
  - `500 Internal Server Error`：當神經推論或特徵擷取發生未預期例外時回傳。
- **實測 `curl` 範例**：
  ```bash
  curl.exe -s -X POST http://127.0.0.1:8003/api/detect     -H "Content-Type: application/json"     -d '{"text": "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance between state regulation and private merchant enterprise. The Senate maintained rigorous oversight of the state galley fleets which operated along fixed routes.", "confidence_threshold": 0.4}'
  ```
  **伺服器真實輸出**：
  ```json
  {
    "summary": {
      "verdict": "Human-written",
      "verdict_description": "Text displays natural syntactic cadence, authentic idiosyncratic phrasing, and human burstiness.",
      "badge": "badge-success",
      "is_uncertain": false,
      "confidence": 1.0,
      "confidence_pct": 100.0,
      "quillbot_headline": "100% of text is likely Human",
      "quillbot_headline_class": "badge-human",
      "quillbot_ai_pct": 0.0,
      "quillbot_human_pct": 100.0,
      "word_count": 38,
      "character_count": 274,
      "sentence_count": 2,
      "length_warning": "Input contains 38 words. QuillBot recommends 80–2,000 words for optimal forensic accuracy.",
      "elapsed_seconds": 0.017
    },
    "calibrated_probabilities": {
      "Human-written": 0.9648,
      "Human-written & AI-refined": 0.0245,
      "AI-generated & AI-refined": 0.0081,
      "AI-generated": 0.0026
    },
    "percentages": {
      "ai_generated": 0.0,
      "ai_ai_refined": 0.0,
      "human_ai_refined": 0.0,
      "human": 100.0
    },
    "quillbot_breakdown": {
      "headline": "100% of text is likely Human",
      "headline_class": "badge-human",
      "ai_percentage": 0.0,
      "human_percentage": 100.0,
      "segments": {
        "human": 100.0,
        "human_ai_refined": 0.0,
        "ai_ai_refined": 0.0,
        "ai_generated": 0.0
      }
    },
    "sentences": [
      {
        "index": 0,
        "text": "The historical development of maritime trade during the Venetian Republic was characterized by a delicate balance between state regulation and private merchant enterprise.",
        "class_label": "Human-written",
        "class_key": "human",
        "color_class": "badge-human",
        "highlight_class": "highlight-human",
        "confidence": 0.82,
        "ai_likelihood_pct": 5.1,
        "probabilities": {
          "Human-written": 0.82,
          "Human-written & AI-refined": 0.129,
          "AI-generated & AI-refined": 0.032,
          "AI-generated": 0.018
        },
        "reasons": [
          "Natural stylistic variation and human syntactic burstiness."
        ]
      },
      {
        "index": 1,
        "text": "The Senate maintained rigorous oversight of the state galley fleets which operated along fixed routes.",
        "class_label": "Human-written",
        "class_key": "human",
        "color_class": "badge-human",
        "highlight_class": "highlight-human",
        "confidence": 0.804,
        "ai_likelihood_pct": 6.9,
        "probabilities": {
          "Human-written": 0.804,
          "Human-written & AI-refined": 0.127,
          "AI-generated & AI-refined": 0.044,
          "AI-generated": 0.025
        },
        "reasons": [
          "Natural stylistic variation and human syntactic burstiness."
        ]
      }
    ],
    "stylometrics": {
      "word_count": 38,
      "character_count": 274,
      "sentence_count": 2,
      "readability": {
        "flesch_reading_ease": 20.6,
        "flesch_kincaid_grade": 15.1,
        "words_per_sentence": 19.0
      },
      "lexical_diversity": {
        "ttr": 0.868,
        "root_ttr": 5.35,
        "hapax_ratio": 0.909,
        "yule_k": 110.8,
        "simpsons_d": 0.9886,
        "honore_r": 4001.3
      },
      "syntax_variance": {
        "mean_length": 19.0,
        "std_length": 4.0,
        "cv_length": 0.211,
        "rhythm_delta": 8.0,
        "rhythm_curvature": 0.0,
        "uniformity_score": 0.9
      },
      "syllable_dispersion": {
        "mean_syllables": 1.97,
        "std_syllables": 1.04,
        "dispersion_cv": 0.526
      },
      "hyphenation": {
        "hyphenated_count": 0,
        "hyphen_rate_per_100w": 0.0,
        "hyphenated_samples": []
      },
      "entropy": {
        "shannon_entropy": 4.93,
        "vocab_richness_bits": 0.978
      },
      "compression_ratio": 0.6788,
      "binoculars_proxy": 7.263,
      "discourse_punctuation": {
        "base_ai_marker_rate": 0.0,
        "base_human_marker_rate": 0.0,
        "ai_marker_rate": 0.0,
        "human_marker_rate": 0.0,
        "contraction_rate": 0.0,
        "comma_rate": 0.0,
        "semi_rate": 0.0,
        "dash_rate": 0.0,
        "ai_count": 0,
        "human_count": 0,
        "detected_ai_samples": [],
        "detected_human_samples": []
      },
      "mathematical_equations": {
        "lexical_richness_omega": 0.8846,
        "syntactic_burstiness_b": 0.2317,
        "discourse_polarity_phi": 0.0,
        "binoculars_ratio_r": 7.263,
        "authorial_affinity_lambda": 0.2284
      },
      "stylometric_ai_score": 0.465
    },
    "mathematical_equations": {
      "lexical_richness_omega": 0.8846,
      "syntactic_burstiness_b": 0.2317,
      "discourse_polarity_phi": 0.0,
      "binoculars_ratio_r": 7.263,
      "authorial_affinity_lambda": 0.2284
    }
  }
  ```

---

#### 5. `POST /api/upload`
接收常見文檔檔案（`.txt`, `.pdf`, `.docx`），透過 `backend/document_parser.py` 解析文字內容與元數據，並執行完整法證偵測分析。

- **HTTP 方法**：`POST`
- **請求標頭**：`Content-Type: multipart/form-data`
- **請求表單欄位**：
  - `file` (`UploadFile`，二進位檔案，必填)：欲剖析與檢測之文檔檔案。支援副檔名：`.txt`, `.pdf`, `.docx`。
- **回應欄位**：
  - 包含與 `POST /api/detect` 完全相同之頂層欄位（`summary`, `calibrated_probabilities`, `percentages`, `quillbot_breakdown`, `sentences`, `stylometrics`, `mathematical_equations`），並額外包含：
    - `summary.filename` (`str`)：原始上傳檔案之檔名。
    - `metadata` (`dict`)：自檔案剖析擷取之元數據（例如 `{"format": "Plain Text"}` 或 PDF 頁數/屬性）。
- **錯誤代碼**：
  - `400 Bad Request`：若上傳檔案大小為 0 位元組（`{"detail":"Uploaded file is empty."}`）或無法自檔案擷取任何可讀文字（`{"detail":"No readable text extracted from document."}`）時回傳。
  - `500 Internal Server Error`：若文檔剖析流程發生嚴重例外時（`{"detail":"File parsing error: ..."}`）回傳。
- **實測 `curl` 範例**：
  ```bash
  curl.exe -s -X POST http://127.0.0.1:8003/api/upload \
    -F "file=@sample_essay.txt"
  ```
  **伺服器真實輸出（節錄）**：
  ```json
  {
    "summary": {
      "verdict": "AI-generated",
      "verdict_description": "Text exhibits direct machine-generation signatures, uniform token predictability, and canonical structures.",
      "badge": "badge-danger",
      "is_uncertain": false,
      "confidence": 1.0,
      "confidence_pct": 100.0,
      "quillbot_headline": "100% of text is likely AI",
      "quillbot_headline_class": "badge-ai",
      "quillbot_ai_pct": 100.0,
      "quillbot_human_pct": 0.0,
      "word_count": 50,
      "character_count": 427,
      "sentence_count": 3,
      "length_warning": "Input contains 50 words. QuillBot recommends 80–2,000 words for optimal forensic accuracy.",
      "elapsed_seconds": 0.018,
      "filename": "sample_essay.txt"
    },
    "calibrated_probabilities": {
      "Human-written": 0.0,
      "Human-written & AI-refined": 0.0259,
      "AI-generated & AI-refined": 0.0731,
      "AI-generated": 0.901
    },
    "metadata": {
      "format": "Plain Text"
    }
  }
  ```

---

## 🚀 Quick Start Guide

### 1. Installation on Low-End PC (Offline Ready)

複製專案儲存庫並安裝輕量級執行時期相依套件（完全無需安裝 PyTorch）：

```bash
git clone https://github.com/juju20081126-boop/veritas-ai-detector.git
cd veritas-ai-detector

# Install runtime dependencies (no PyTorch, installs in seconds)
pip install -r requirements.txt
```

### 2. Launch Local Web UI

透過單一指令啟動本機伺服器：

```bash
python run.py
```

該腳本將自動限制 CPU 執行緒上限為 2、預熱 ONNX 推論工作階段，並自動以預設瀏覽器開啟 `http://localhost:8000`。

若欲在不自動開啟瀏覽器的情況下指定連接埠啟動：
```bash
python run.py --no-browser --port 8080
```

### 3. Command-Line Interface (CLI)

直接於終端機命令列分析文章文本：

```bash
# Analyze a direct string
python cli.py --text "In the contemporary era, the rapid proliferation of artificial intelligence..."

# Analyze a document file with hardware benchmark telemetry
python cli.py --file samples/algorithmic_commons_essay.md --benchmark --threads 2

# Output raw JSON
python cli.py --text "This is a brief text." --json
```

---

## 📊 Verification & Empirical Evaluation

> [!WARNING]
> **舊版評估聲明**：以下指標係在合成且具資料外洩缺陷的驗證劃分集上測得，且教師模型數據為寫死未實測之佔位數值（詳情參見 `data/eval/legacy_audit.json`）。此批數據已全數撤回並作廢。基於真實資料管線測得之基準數據將發表於 `data/reports/FRONTIER_DETECTION_REPORT.md`。

| 評估指標 | 系統設計目標 | Veritas AI 內建模型 (舊版數據) | 當前狀態 / 撤回說明 |
|---|---|---|---|
| 500 字推論延遲（模擬 2 執行緒） | ≤ 15.0 秒 | ~~0.167 秒~~ | *舊版基準測試* |
| 程序記憶體峰值（目標硬體規格） | ≤ 1,500 MB | ~~178.5 MB~~ | *舊版基準測試* |
| 發行模型佔用磁碟空間 | ≤ 500 MB | **21.96 MB**（資產總計 196 MB） | 已驗證 |
| 執行時期 PyTorch 依賴 | 完全零 PyTorch | **完全無** (`onnxruntime` CPU + `tokenizers`) | 已驗證 |
| ESL 非母語寫作者偽陽性率 | ≤ 2.0× 母語者比率 | ~~0.00%~~ | *已撤回（合成/資料外洩）* |
| 4 類別 Macro-F1 分數 | 均衡 4 類別 F1 | ~~0.6392~~ | *已撤回（合成/資料外洩）* |
| 分佈內 TPR (@ ≤1% FPR) | 學生模型 TPR | ~~85.00%（教師：96.2%）~~ | *已撤回（教師從未訓練）* |
| 未見過模型 (Qwen-2.5-72B) | 誠實 TPR (@ 1% FPR) | ~~80.0% (12/15 檢出)~~ | *已撤回（合成測試）* |
| 未見過模型 (DeepSeek-V3) | 誠實 TPR (@ 1% FPR) | ~~73.3% (11/15 檢出)~~ | *已撤回（合成測試）* |
| 改寫 AI 檢出率 | 誠實檢出率 | ~~80.0% (40/50 檢出)~~ | *已撤回（僅為正規詞替換，非真實改寫）* |
| 校準 ECE | ECE < 0.05 | ~~0.0306~~ | *已撤回（合成測試）* |

---

## 🔄 Retraining & Data-Refresh Pipeline

*說明：舊版合成資料生成腳本 `scripts/refresh_pipeline.py` 因基於無金鑰的合成樣板運作且存在嚴重的訓練-測試集外洩問題，現已被隔離至 `scripts/legacy_synthetic/`。目前已由全新資料管線架構（`scripts/corpus/`, `scripts/eval_frontier.py`, `scripts/check_integrity.py`）取代，以支援經過嚴格驗證的前沿模型資料生成與公正評估。*

### How to Run the Real-Data Pipeline

> [!NOTE]
> **進行中：** 處理管線正在執行中；執行完成後各項指標將儲存於 `data/reports/`。

各項指令必須自儲存庫根目錄依序執行：

#### Step 1: Collect Prompts and Human Datasets
```bash
# Build the generation prompt set and the matched human documents (seeded, reproducible)
python scripts/corpus/build_prompts.py

# Human student essays from W&I+LOCNESS (BEA-2019 shared task): non-native learner and native university essays
python scripts/corpus/build_esl.py

# Sample REAL public corpora for training/dev breadth (RAID, MAGE, HC3)
python scripts/corpus/build_public_ai.py [--only raid|mage|hc3]

# Extra HUMAN texts to balance the genres that public AI corpora over-represent (arXiv abstracts and CNN news)
python scripts/corpus/build_human_extra.py
```

#### Step 2: Frontier Model Generation Batches & Ingestion
```bash
# Split data/corpus/prompts.jsonl into generation batches for real frontier-model generation
python scripts/corpus/make_generation_batches.py [--batch-size 20] [--round r2]

# Validate and ingest raw generation files written by Claude Code subagents
python scripts/corpus/ingest_generations.py            # 攝入所有原始檔案已存在之批次
python scripts/corpus/ingest_generations.py --batch opus__train__018
```

#### Step 3: Adversarial & Paraphrase Attacks
```bash
# Apply local/programmatic attack families (A4, A5, A6, A7) to ingested frontier texts and human controls
python scripts/corpus/make_attacks.py --gen claude-opus-5-5 --split locked --family A7 --n 40
python scripts/corpus/make_attacks.py --gen human --split locked --family A5 --n 20

# Create subagent task batches for the LLM attack families A1 (paraphrase), A2 (iterative) and A3 (humanizer prompt)
python scripts/corpus/make_llm_attack_batches.py --split locked --stage 1 --n1 40 --n2 40 --n3 40
python scripts/corpus/make_llm_attack_batches.py --split locked --stage 2          # A2 第二輪改寫（於第一輪攝入後執行）

# Validate and ingest the rewrites produced by subagents for the LLM attack families (A1, A2, A3)
python scripts/corpus/ingest_llm_attacks.py            # 攝入所有原始檔案已存在之批次
```

#### Step 4: Split Assembly & Integrity Verification
```bash
# Assemble the final train / dev / locked-test splits from the corpus pieces (deterministic group splits)
python scripts/corpus/build_splits.py [--lock]

# Integrity gates for the Veritas real-data pipeline (FAIL-CLOSED)
python scripts/check_integrity.py                  # 檢查 data/splits 與 data/locked
python scripts/check_integrity.py --legacy-demo    # 在隔離的舊版合成資料上執行關卡（預期失敗 FAIL）
python scripts/check_integrity.py --json out.json  # 同步儲存結構化檢驗報告
```

#### Step 5: Honest Detector Evaluation
```bash
# Honest evaluation of AI-text detectors on the dev or locked split
python scripts/eval_frontier.py --split dev    --detectors shipped hc3_roberta binoculars [--max-per-cell N]
python scripts/eval_frontier.py --split locked --detectors shipped hc3_roberta ... --out data/eval/results/locked_baselines.json
```

### Cloud Jupyter Notebooks (Kaggle / Google Colab)
- [`notebooks/01_teacher_ensemble_and_labeling.ipynb`](notebooks/01_teacher_ensemble_and_labeling.ipynb)：教師模型集成與偽標註範本（未執行；筆記本中之數據為未實測佔位符）。
- [`notebooks/02_student_distillation_and_onnx_export.ipynb`](notebooks/02_student_distillation_and_onnx_export.ipynb)：學生模型蒸餾、文體計量元分類器、INT8 ONNX 匯出與校準範本（未執行範本；不宣稱任何實測成果）。
- [`notebooks/03_gpu_finetune.ipynb`](notebooks/03_gpu_finetune.ipynb)：在真實訓練資料上微調 DeBERTa-v3-small 並匯出 ONNX INT8（未執行範本；不宣稱任何實測成果）。
- [`notebooks/04_results_figures.ipynb`](notebooks/04_results_figures.ipynb)：自 `data/eval/results/*.json` 產出評估圖表與彙整表（未執行範本；不宣稱任何實測成果）。

---

## 📚 State-of-the-Art Research & Mathematical Formulations (2024–2026)

欲詳閱涵蓋當代最新偵測數學公式、零樣本曲率指標、4 類別法證分類學與資料集基準測試之完整技術專論，請參閱 **[RESEARCH_COMPENDIUM.md](RESEARCH_COMPENDIUM.md)**：

- **Binoculars Zero-Shot Cross-Ratio（零樣本雙眼交叉比）**：$	ext{Score}(x) = rac{\log 	ext{PPL}_{M_1}(x)}{\log 	ext{xPPL}_{M_1, M_2}(x)}$（Hans et al., ICML 2024）
- **Fast-DetectGPT 條件機率曲率 (Conditional Probability Curvature)**：$	ilde{d}(x) = rac{\sum_t (\log p(x_t) + \mathcal{H}(p))}{\sqrt{\sum_t 	ext{Var}[\log p]}}$（Bao et al., ICLR 2024）
- **RADAR 對抗改寫不變性 (Adversarial Paraphrase Invariance)**：防範自動化規避之極小極大賽局公式（Hu et al., NeurIPS 2024）
- **長度不變法證文體特徵 (Length-Invariant Forensic Stylometrics)**：Yule's 特徵係數 $K$、香農 Token 資訊熵率、連續句法節奏差值 ($\Delta_{	ext{rhythm}}$) 與 DEFLATE 壓縮複雜度
- **14 大權威基準語料庫**：詳細收錄 RAID (ACL 2024)、M4 (EACL 2024)、HC3、DetectRL、MAGE 以及 ESL 學習者語料庫（PELIC, TOEFL11, ICNALE）之綱要
- **非英語母語者 (ESL) 公平性審核**：演算法緩解技術，避免因用詞簡潔清晰而受到偽陽性懲罰

---

## 📋 QuillBot Comparison Sheet (30 Hand-Check Samples)

在嚴格遵循第三方服務條款（禁止爬蟲或自動化呼叫）的前提下，本專案提供恰好 30 篇具備代表性的文本，涵蓋 4 大類別、英語母語人類、ESL 人類與前沿模型，以供人工對照檢核：
- JSON 格式：[`data/quillbot_comparison_sheet.json`](data/quillbot_comparison_sheet.json)
- Markdown 表格：[`data/quillbot_comparison_sheet.md`](data/quillbot_comparison_sheet.md)

---

## Hand-collecting detector verdicts (no automation, ToS-safe, <=200 texts)

為在嚴格遵循服務條款的前提下安全地對第三方商業偵測器進行基準比對：
1. Claude 產生包含待測文章之編號 CSV 檔案（欄位包含 `id` 與 `text`）。
2. 使用者**以手動方式（BY HAND）**將各篇文字逐一貼入偵測器的公開網頁。
3. 使用者將網頁顯示之判定結果謄寫記錄於結果欄位中。

**守則：**
- 每個偵測器最多手動測試約 200 篇文字，以避免過度使用並維持人工操作可行性。
- 嚴禁對任何偵測器或改寫網站進行自動化程式呼叫、嚴禁爬蟲，且絕不透過腳本傳送請求。

---

## Frequently Asked Questions (FAQ)

### 1. Can Veritas prove that a student or writer used AI?
**繁體中文：Veritas 能否確鑿證明某位學生或作者使用了 AI？**
不行。AI 偵測分數純屬基於文體與句法模式的機率統計估算，絕非作者身分的法證確鑿證明（S1, S13）。頂尖偵測系統之開發學者皆明確建議，切勿將自動化偵測器作為處分或懲戒決策的唯一裁決依據（S13, S14）。Veritas 旨在提供透明、多維度的特徵佐證以輔助人工審核，而非取代人類的獨立判斷。

### 2. Why does text under 80 words produce an uncertain or unreliable score?
**繁體中文：為何 80 字以下的短文本容易產生不確定或不可靠的分數？**
突發變異性 (Burstiness)、資訊熵率及詞彙豐富度等統計特徵，皆需足夠之文字篇幅才能收斂為具代表性之分佈（S7, S10）。在 50 至 100 字以下的極短文本中，所有已評估偵測器之敏感度皆會大幅下滑（例如 Pangram 之 TPR 在小於 50 字時由 100% 暴跌至 73.32%，S1；Ghostbuster 在 100 token 以下顯著衰退，S10）。對於少於 80 字之輸入，Veritas 會顯示篇幅警語並建議提供 150 字以上文本以確保精準度。

### 3. How does Veritas prevent false accusations against non-native (ESL) writers?
**繁體中文：Veritas 如何防止對非英語母語 (ESL) 寫作者產生不實指控？**
單純依賴語言模型困惑度 (Perplexity) 之偵測器，因非母語者用詞較簡潔且句式單純，極易對其產生嚴重的偏見誤判（Liang 等人指出托福作文之偽陽性率平均高達 61.22%，S11）。Veritas 結合了 20 維長度不變文體特徵、真實主觀語態防護關卡，並在 ESL 學習者語料庫上進行嚴格校準（S1, S11, S13）。此外，我們的評估管線獨立審核 ESL 偽陽性率，確保演算法符合公平性標準。

### 4. Does Veritas transmit my text or documents to any cloud server?
**繁體中文：Veritas 是否會將我的文章或文件傳送至任何雲端伺服器？**
絕不傳送。Veritas 完全在本機個人電腦上運行，採用 INT8 量化之 ONNX 學生模型與本機 Python 特徵擷取模組。執行時期零網路連線需求、零遙測數據上傳，亦不進行任何雲端 API 呼叫。所有貼入文字與上傳檔案僅暫存於本機記憶體中，絕不上傳遠端伺服器，亦不儲存於 localStorage。

### 5. How does Veritas handle text rewritten by paraphrasers or AI humanizers?
**繁體中文：Veritas 如何處理經由改寫工具或「降 AI 人工化 (Humanizer)」處理之文字？**
改寫與 Humanizer 工具透過打亂規律的 n-gram 模式重構 AI 文章，大幅增加偵測難度（PADBen S3, DAMAGE S16）。指標型偵測器在此類攻擊下表現驟降（在 DAMAGE 測試中準確率降至 28.23%，S16），而 Veritas 針對多輪改寫攻擊批次（A1–A3）進行對抗訓練，並細緻區分純 AI (`ai_generated`) 與經改寫文字 (`ai_ai_refined`)。此 4 類別體系能有效辨識中間改寫狀態並揭露殘留之文體特徵。

---

## Glossary

### AUROC (Area Under the Receiver Operating Characteristic Curve)
**繁體中文：受試者操作特徵曲線下面積 (AUROC)**
AUROC 衡量隨機抽選之 AI 文本其異常分數高於隨機抽選之人類文本的整體機率。不同於單一準確率，AUROC 與決策門檻無關，能全面反映分類器在整體操作曲線上的類別分離能力（S7, S8, S12）。然而，AUROC 往往會掩蓋實際部署時嚴格低偽陽性率運作點上的嚴重效能衰減（S14）。

### TPR@1%FPR (True Positive Rate at 1% False Positive Rate)
**繁體中文：在 1% 偽陽性率下之真陽性率 (TPR@1%FPR)**
TPR@1%FPR 代表當決策門檻被嚴格校準為在純真人文本上僅容許至多 1% 誤判率時，能成功檢出真實 AI 文本的比例（S6, S9, S14）。這是高風險部署環境下的首要基準標準，因為防止對無辜真人作者產生不實指控至關重要（S1, S13）。雖然寬鬆門檻下的表面整體準確率看似極高，但 TPR@1%FPR 能如實揭露改寫與逃逸攻擊下的顯著效能滑落（S4, S9, S16）。

### FPR (False Positive Rate)
**繁體中文：偽陽性率 (FPR)**
偽陽性率為真實人類原創文本被錯誤標記為 AI 生成或 AI 潤飾之比例。在學術與專業工作場合中，偏高的偽陽性率將引發不公正的懲處並瓦解人際與體制信任（S11, S13）。系統必須在母語與非母語等不同背景群體間獨立審核實際測得之 FPR，以消除系統性群體偏見（S11, S13）。

### ESL (English as a Second Language / Non-Native Writers)
**繁體中文：非英語母語寫作者 (ESL)**
ESL 指由英語非母語者與國際語言學習者撰寫的文章（如 TOEFL, IELTS, 以及 W&I+LOCNESS 基準語料庫）。僅依賴困惑度之偵測器具備明顯的族群偏差，因非母語者用詞較為受限且困惑度較低，導致誤判為 AI 的機率高達 61.22%（母語者僅 5.19%，S11）。Veritas 明確將 ESL FPR 列為獨立評估指標，確保公平性防護生效（S1, S11, S13）。

### Group Split
**繁體中文：確定性分組劃分 (Group Split)**
分組劃分是一種嚴謹的資料集切分方法，確保源自同一提示詞或母篇文本的所有生成、改寫與對照版本，皆被嚴格指派至同一資料劃分集中（訓練集、驗證集或鎖定測試集）。此機制能徹底防止模型因記憶特定主題用詞而產生「資料外洩」，使其必須學習真正泛化的法證特徵（S1, S10）。分組劃分確保回報之基準測試能如實反映跨領域偵測能力（S10, S14）。

### Locked Test Split
**繁體中文：鎖定測試劃分集 (Locked Test Split)**
鎖定測試集是封存在 `data/locked/` 的不可變保留評估語料庫，在模型蒸餾、訓練或門檻調整過程中嚴禁存取。為防止因反覆試探而對測試集過擬合，系統透過密碼學簽章稽核每次存取，且每個模型家族終生至多僅能評估三次（記錄於 `ACCESS_LOG.md`）。它是驗證對未見過前沿模型與新形態攻擊具備真實泛化能力的最高評判準則（S1, S14）。

### Humanizer
**繁體中文：降 AI 人工化改寫工具 (Humanizer)**
Humanizer 係指專為逃逸 AI 偵測而設計之對抗性改寫工具、提示詞樣板或線上服務（如 DIPPER, BypassGPT, Undetectable AI 等，S9, S16）。Humanizer 刻意引入詞彙擾動、同義詞替換與句法抖動，可將傳統偵測器召回率由 90% 以上削弱至 30% 以下（S4, S16）。構建穩健防線必須納入結構化對抗攻擊家族（A1–A3, A8）之專門訓練（S3, S16）。

### Hybrid / Mixed Authorship
**繁體中文：人機協作 / 混合署名 (Hybrid / Mixed Authorship)**
混合署名指人類與 AI 深度協作之文本，包括由大語言模型潤飾的人類草稿，以及經人類作者實質修訂重組的機器生成段落（S1, S2）。傳統二元分類器往往強行對整篇混合文本做出非黑即白的錯誤判斷（S1, S2）。Veritas 透過 4 類別體系（區分 `human_ai_refined` 與 `ai_ai_refined`）及逐句分析模型，如實呈現當代真實的協同寫作風貌（S1, S2）。

---

## Limitations

以下每一項論點皆為 `data/research/sources.md` 中標記為 [documented] 之已確認事實。S 編號對應於該檔案中的文獻索引條目。

- 非英語母語寫作者可能被錯誤標記為 AI。Liang 等人（Patterns 2023, S11）針對非母語寫作者的 91 篇托福 (TOEFL) 作文以及美國八年級學生的 88 篇作文測試了 7 款偵測器。平均而言，高達 61.22% 的托福作文被標記為 AI 撰寫，而母語學生作文僅約 5.19% 被標記。研究作者指出此現象主因在於非母語文章的困惑度 (perplexity) 較低且詞彙多樣度較侷限。儘管該研究採用的是 2023 年之偵測器，但此結果仍為目前已知之最嚴峻最差情況。
- 短文本之檢測結果不可靠。Ghostbuster（S10）在少於或等於 100 個 token 之文本上準確率出現顯著衰退。Fast-DetectGPT（S7）報告指出偵測準確率隨文章篇幅增長而穩定上升。Pangram（S1）指出短文本更難以判讀：在其降 AI 工具測試中，50 字以下文本在 1% FPR 下的真陽性率 (TPR) 僅達 73.32%，而完整長度文本則可達 100%。
- 改寫與降 AI 人工化處理文本為最困難之情境。PADBen（S3）發現多輪反覆改寫是最難以防範的攻擊手法，偵測器在各階段的中間改寫步驟便會失效。在 DAMAGE（S16）測試中，對降 AI 人工化文本之檢測，GPTZero 之 TPR 降至 60.04%，Binoculars 則降至 28.23%（皆在 5% FPR 下測得）。DIPPER 改寫（S9）使 DetectGPT 在 1% FPR 下之 TPR 由 70.3% 暴跌至 4.6%。
- 商業廠商公佈之準確率數據多屬自行回報。例如 Pangram 之各項數據（S1, S13）皆源自 Pangram 自行發佈之技術報告。而獨立的 RAID 基準測試（S14）則發現，現有偵測器對於其訓練所用的領域與模型存在顯著偏差，且「目前仍不足以穩健地在重大高風險決策中單獨作為裁決依據」。
- 偵測分數純屬機率統計信號，非作者身分之確鑿證明。Pangram（S1）表明其偵測器係基於統計機率運作，同一篇文章在不同脈絡環境下可能得出不同評分。Pangram 作者亦嚴正建議切勿將偵測器作為做出判斷的唯一依據（S13）。Binoculars（S6）亦曾將被模型高度記憶之著名歷史文本（如美國憲法）誤判為機器生成。

---

## 📜 License & Acknowledgments

- 本專案引用的公開研究資料集分別遵循其個別開源授權條款：RAID (CC-BY 4.0), M4 (Apache-2.0), HC3 (CC-BY-SA 4.0), DetectRL (Apache-2.0)。詳情請參閱 [`data/public_licenses.md`](data/public_licenses.md)。
- Veritas AI 依據 **MIT License** 條款發佈。
