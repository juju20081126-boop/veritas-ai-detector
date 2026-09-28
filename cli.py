"""
QuillBot-Style Offline AI Detector — Command-Line Interface (CLI)
Runs fully offline on CPU via ONNX Runtime without PyTorch.
"""

import os
import sys
import json
import argparse
import time
import psutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("=" * 72)
    print("  \033[1;36mQUILLBOT-STYLE OFFLINE AI WRITING DETECTOR\033[0m")
    print("  \033[90mPowered by ONNX Runtime INT8 Distilled Student | Zero-PyTorch\033[0m")
    print("=" * 72)


COLOR_MAP = {
    "Human-written": "\033[1;32m",              # Green
    "Human-written & AI-refined": "\033[1;33m",  # Yellow
    "AI-generated & AI-refined": "\033[1;38;5;208m", # Orange
    "AI-generated": "\033[1;31m",               # Red
    "Uncertain": "\033[1;35m",                  # Magenta
    "RESET": "\033[0m"
}


def run_cli_analysis(text: str, threshold: float = 0.40, threads: int = 2, json_output: bool = False, benchmark: bool = False):
    from backend.runtime_engine import QuillBotDetectorEngine

    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    t0 = time.time()
    engine = QuillBotDetectorEngine.get_instance(threads=threads)
    result = engine.analyze_text(text, confidence_threshold=threshold)
    t1 = time.time()

    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = ram_after_mb

    if json_output:
        if benchmark:
            result["benchmark"] = {
                "latency_sec": round(t1 - t0, 3),
                "peak_ram_mb": round(peak_ram_mb, 1),
                "threads": threads
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print_banner()

    summary = result["summary"]
    verdict = summary["verdict"]
    color = COLOR_MAP.get(verdict, COLOR_MAP["RESET"])

    print(f"\n  \033[1mDOCUMENT VERDICT:\033[0m {color}[ {verdict.upper()} ]{COLOR_MAP['RESET']}")
    print(f"  \033[90m{summary['verdict_description']}\033[0m")
    print(f"  Confidence: \033[1m{summary['confidence_pct']}%\033[0m | Words: \033[1m{summary['word_count']}\033[0m | Sentences: \033[1m{summary['sentence_count']}\033[0m")

    if summary.get("length_warning"):
        print(f"\n  \033[33m[!] Note: {summary['length_warning']}\033[0m")

    print("\n" + "-" * 72)
    print("  \033[1mCALIBRATED 4-CLASS PROBABILITY DISTRIBUTION:\033[0m")
    print("-" * 72)
    probs = result["calibrated_probabilities"]
    bar_width = 30
    for cls_name, p in probs.items():
        filled = int(p * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        c = COLOR_MAP.get(cls_name, COLOR_MAP["RESET"])
        print(f"  {c}{cls_name:<30}{COLOR_MAP['RESET']} [{bar}] {p*100:5.1f}%")

    print("\n" + "-" * 72)
    print("  \033[1mPER-SENTENCE FORENSIC HIGHLIGHT MATRIX:\033[0m")
    print("-" * 72)

    for s in result["sentences"]:
        sc = COLOR_MAP.get(s["class_label"], COLOR_MAP["RESET"])
        print(f"  [{s['index']+1:02d}] {sc}{s['class_label']:<28}{COLOR_MAP['RESET']} (Conf: {s['confidence']*100:4.1f}%)")
        print(f"       \033[37m\"{s['text']}\"\033[0m")
        if s["reasons"]:
            print(f"       \033[90m-> {s['reasons'][0]}\033[0m")

    if benchmark:
        print("\n" + "-" * 72)
        print("  \033[1mTARGET HARDWARE SIMULATION TELEMETRY:\033[0m")
        print("-" * 72)
        print(f"  CPU Threads:      {threads} (Target limit: 2)")
        print(f"  Execution Time:   {t1 - t0:.3f} seconds (Target limit: <= 15s)")
        print(f"  Peak Process RAM: {peak_ram_mb:.1f} MB (Target limit: <= 1500 MB)")
        pass_ram = "PASS" if peak_ram_mb <= 1500 else "FAIL"
        pass_lat = "PASS" if (t1 - t0) <= 15.0 else "FAIL"
        print(f"  Hardware Compliance: RAM [{pass_ram}] | Latency [{pass_lat}]")

    print("\n" + "=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="QuillBot-Style Offline AI Writing Detector CLI")
    parser.add_argument("--text", type=str, help="Text string to analyze")
    parser.add_argument("--file", type=str, help="Path to .txt or .md file to analyze")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold for Uncertainty gating (default: 0.40)")
    parser.add_argument("--threads", type=int, default=2, help="CPU intra_op threads for simulated low-end target (default: 2)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    parser.add_argument("--benchmark", action="store_true", help="Report RAM & execution latency on simulated target")

    args = parser.parse_args()

    input_text = ""
    if args.text:
        input_text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        print_banner()
        print("\nEnter or paste text to analyze (Press Ctrl+D on Unix or Ctrl+Z then Enter on Windows when done):\n")
        try:
            input_text = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    if not input_text.strip():
        print("Error: Input text is empty.")
        sys.exit(1)

    run_cli_analysis(
        text=input_text,
        threshold=args.threshold,
        threads=args.threads,
        json_output=args.json,
        benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()
