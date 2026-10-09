# 🛡️ Veritas AI — Offline AI Writing Detector

Paste some text or upload a file, and Veritas estimates whether a person or an AI wrote it. Everything runs on your own computer: no internet connection, no GPU, and nothing you paste leaves your machine.

[Developer guide](DEVELOPERS.md) · [Evaluation report](EVAL_REPORT.md)

> [!WARNING]
> **A score is a clue, not proof.** Veritas misses many AI texts, and it wrongly flags about 1 in 100 human texts. Never use it as the only evidence against someone.

## Quick start

You need Python 3 (tested on 3.11).

```bash
git clone https://github.com/juju20081126-boop/veritas-ai-detector.git
cd veritas-ai-detector
pip install -r requirements.txt
python run.py
```

The app opens in your browser at `http://localhost:8000`. Paste text (or upload a `.txt`, `.pdf` or `.docx` file) and click **Check text**.

Prefer the terminal?

```bash
python cli.py --text "Paste your text here..."
python cli.py --file essay.txt
```

## What you get

- **A verdict** in one of four groups: **AI-generated**, **AI-generated & AI-refined**, **Human-written & AI-refined** or **Human-written**. When the evidence is weak, it says **Uncertain** instead of guessing.
- **Sentence highlights** that show which parts look most AI-like.
- **English and Traditional Chinese** interface.
- **Runs on a low-end PC:** 2 CPU threads, about 200 MB of memory, about 25 MB of model files, and under 0.2 seconds for 500 words.

## How accurate is it?

| Detector | Claude 5.5 text caught | GPT-4 text caught | Human writing wrongly flagged |
|---|---|---|---|
| **`ensemble` (default)** | **62%** | **11%** | **1.3%** |
| `tfidf` | 67% | 1% | 0.8% |
| `frontier` (default until Oct 2026) | 0.5% | 20% | 7.3% |

- `ensemble` runs `tfidf` and `frontier` together and flags a text if either one is confident it is AI. `tfidf` is best at Claude; `frontier` is better at older models such as GPT-4.
- On 230 essays by non-native English writers, `ensemble` and `tfidf` wrongly flagged none; `frontier` flagged 3.
- **Where the numbers come from:** AI catch rates come from the development set (185 Claude Opus/Sonnet 5.5 texts, 1,525 GPT-4 texts). False-alarm rates come from 1,100 modern human texts that no model was trained on. For `ensemble`, that is the 532 texts not used to set its thresholds.
- The final locked test set was used up before `ensemble` existed, so these are not locked-test results. Older locked-test results are in [EVAL_REPORT.md](EVAL_REPORT.md).

To use a different detector, set `VERITAS_DETECTOR` before starting the app:

```bash
VERITAS_DETECTOR=tfidf python run.py            # macOS / Linux
$env:VERITAS_DETECTOR="tfidf"; python run.py    # Windows PowerShell
```

Options: `ensemble` (default), `tfidf`, `frontier`, `multi_teacher`, `shipped` (the original pre-2.0 model).

## Limitations

- **It misses a lot.** About 4 in 10 Claude 5.5 texts and most text from older AI models pass as human. A "Human-written" verdict does not prove a person wrote it.
- **Short texts are unreliable.** Under about 150 words, results get shaky, and the app shows a warning.
- **Paraphrasing and "humanizer" tools** make AI text harder to catch.
- **The false-alarm rate was checked mostly on everyday Q&A-style writing** (Dolly and OpenAssistant). Other kinds of writing, such as fiction, may be flagged more often.
- **Built and tested on English text.**
- Earlier versions of this README reported much higher accuracy based on synthetic, leaky test data. Those numbers are withdrawn.

## Privacy

Veritas works fully offline. It makes no network calls and sends no telemetry. Text you paste or upload is analyzed in memory and is not saved. The browser only remembers your language and theme choice.

## More information

- [DEVELOPERS.md](DEVELOPERS.md): API reference, project layout, how the models work, and the training and evaluation pipeline
- [EVAL_REPORT.md](EVAL_REPORT.md): full test results
- [RESEARCH_COMPENDIUM.md](RESEARCH_COMPENDIUM.md): research background
- [HANDOFF.md](HANDOFF.md): change history

## License

MIT. Public datasets used for training and testing keep their own licenses; see [data/public_licenses.md](data/public_licenses.md).
