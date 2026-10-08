"""
Simulated Low-End Target Hardware Benchmark for Veritas AI
Simulates:
- 2-core CPU constraint (ONNX intra_op_num_threads=2, inter_op_num_threads=1)
- Peak process RAM cap: <= 1.5 GB (measured via psutil RSS)
- Model & asset disk footprint: <= 500 MB
- Latency target: <= 15 seconds per 500 words
"""

import os
import sys
import time
import json
import argparse
import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.runtime_engine import DETECTOR_MODES, RUNTIME_CONFIGS, STUDENT_MODES, TFIDF_DIR  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def get_dir_size_mb(path: str) -> float:
    """Calculates total directory size in megabytes."""
    total_bytes = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def runtime_assets_mb(mode: str) -> float:
    """Size of the files the runtime engine actually loads in this mode (not training checkpoints or the FP32 export)."""
    if mode == "tfidf":
        return sum(os.path.getsize(f) for f in (os.path.join(TFIDF_DIR, "tfidf_model.json.gz"), RUNTIME_CONFIGS[mode])) / (1024 * 1024)
    base, cfg_name = (MODELS_DIR, "meta_classifier.json") if mode == "shipped" else STUDENT_MODES[mode]
    files = [os.path.join(base, "student_model_int8.onnx"), os.path.join(base, cfg_name)]
    size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    return size / (1024 * 1024) + get_dir_size_mb(os.path.join(base, "tokenizer"))


def generate_benchmark_prose(target_words: int) -> str:
    """Generates synthetic multi-sentence passage matching target word count."""
    sentence_bank = [
        "The historical progression of maritime commerce during the Renaissance relied on complex institutional partnerships.",
        "While traditional economic models assume perfect market transparency, empirical observations reveal persistent informational asymmetries.",
        "Rather than treating technological disruptions as isolated occurrences, scholars increasingly analyze them as systemic structural shifts.",
        "Consequently, establishing robust regulatory frameworks remains an essential requirement for long-term institutional stability.",
        "In particular, localized community practices frequently demonstrate greater resilience than centralized bureaucratic mandates.",
        "Modern algorithmic systems systematically reorganize communicative interaction, converting behavioral traces into predictive assets.",
        "Furthermore, sustainable ecological transitions require moving beyond purely technical interventions to address social equity.",
        "Careful examination of primary archival records corroborates the hypothesis that decentralized guilds preserved technical expertise."
    ]
    
    words = []
    idx = 0
    while len(words) < target_words:
        s = sentence_bank[idx % len(sentence_bank)]
        idx += 1
        words.extend(s.split())
    
    return " ".join(words[:target_words])


def run_target_benchmark(runs_per_tier: int = 3, mode: str = "shipped"):
    print("=" * 76)
    print("  \033[1;36mVERITAS AI — SIMULATED LOW-END TARGET HARDWARE BENCHMARK\033[0m")
    print("  \033[90mConstraints: 2 CPU Threads | <= 1.5 GB Peak RAM | <= 500 MB Disk | <= 15s Latency\033[0m")
    print("=" * 76)

    # 1. Disk Size Audit
    models_size_mb = runtime_assets_mb(mode)
    disk_target_mb = 500.0
    disk_passed = models_size_mb <= disk_target_mb

    print(f"\n[Disk Footprint Audit]")
    print(f"  Detector mode:             {mode} (runtime-loaded files only)")
    print(f"  Total Asset Size on Disk:  {models_size_mb:.2f} MB")
    print(f"  Disk Target Threshold:     <= {disk_target_mb:.0f} MB")
    print(f"  Status:                    [\033[1;32mPASS\033[0m]" if disk_passed else f"  Status: [\033[1;31mFAIL\033[0m]")

    # 2. Process Memory Baseline
    process = psutil.Process(os.getpid())
    ram_baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n[Process Baseline]")
    print(f"  Python Process RSS: {ram_baseline_mb:.1f} MB")

    # 3. Initialize Runtime Engine under simulated 2-thread cap
    from backend.runtime_engine import QuillBotDetectorEngine
    print("\n[Engine Initialization]")
    t_init_start = time.time()
    engine = QuillBotDetectorEngine(threads=2, mode=mode)
    t_init_end = time.time()
    ram_after_init_mb = process.memory_info().rss / (1024 * 1024)
    init_duration = t_init_end - t_init_start

    print(f"  Initialization Time: {init_duration:.3f} s")
    print(f"  Post-Init Process RSS: {ram_after_init_mb:.1f} MB")

    # 4. Latency & Memory Tiers: 80, 500, 1000, 2000 words
    word_tiers = [80, 500, 1000, 2000]
    tier_results = []
    overall_peak_ram_mb = ram_after_init_mb

    print("\n[Inference Latency & Memory Measurements]")
    print("-" * 76)
    print(f"  {'Word Count':<12} | {'Sentences':<10} | {'Mean Latency':<14} | {'Std Dev':<10} | {'Peak RAM':<12} | {'Compliance'}")
    print("-" * 76)

    for wc in word_tiers:
        test_text = generate_benchmark_prose(wc)
        latencies = []
        tier_peak_ram = 0.0

        for r in range(runs_per_tier):
            t0 = time.time()
            res = engine.analyze_text(test_text)
            t1 = time.time()
            elapsed = t1 - t0
            latencies.append(elapsed)

            current_rss = process.memory_info().rss / (1024 * 1024)
            tier_peak_ram = max(tier_peak_ram, current_rss)
            overall_peak_ram_mb = max(overall_peak_ram_mb, current_rss)

        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))
        sent_count = res["summary"]["sentence_count"]

        # Check compliance on 500 words
        compliance = "PASS"
        if wc == 500:
            if mean_lat > 15.0 or tier_peak_ram > 1500.0:
                compliance = "FAIL"

        color_status = "\033[1;32mPASS\033[0m" if compliance == "PASS" else "\033[1;31mFAIL\033[0m"

        print(f"  {wc:<12} | {sent_count:<10} | {mean_lat:6.3f} s      | {std_lat:6.3f} s   | {tier_peak_ram:6.1f} MB   | [{color_status}]")

        tier_results.append({
            "word_count": wc,
            "sentence_count": sent_count,
            "mean_latency_sec": round(mean_lat, 3),
            "std_latency_sec": round(std_lat, 3),
            "peak_ram_mb": round(tier_peak_ram, 1),
            "compliance": compliance
        })

    # Summary
    p500 = next(t for t in tier_results if t["word_count"] == 500)
    ram_passed = overall_peak_ram_mb <= 1500.0
    lat_passed = p500["mean_latency_sec"] <= 15.0

    print("-" * 76)
    print("\n[Target Compliance Final Summary]")
    print(f"  1. 500-Word Latency: {p500['mean_latency_sec']:.3f} s (Target: <= 15.0s) -> [\033[1;32m{'PASS' if lat_passed else 'FAIL'}\033[0m]")
    print(f"  2. Peak Process RAM: {overall_peak_ram_mb:.1f} MB (Target: <= 1500 MB) -> [\033[1;32m{'PASS' if ram_passed else 'FAIL'}\033[0m]")
    print(f"  3. Disk Footprint:   {models_size_mb:.2f} MB (Target: <= 500 MB)   -> [\033[1;32m{'PASS' if disk_passed else 'FAIL'}\033[0m]")
    print(f"  4. CPU Threads Used: 2 Threads (Target simulated exactly)")

    report = {
        "timestamp": time.time(),
        "detector_mode": mode,
        "hardware_simulation": {
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "provider": "CPUExecutionProvider"
        },
        "disk_footprint_mb": round(models_size_mb, 2),
        "disk_target_mb": disk_target_mb,
        "disk_passed": disk_passed,
        "overall_peak_ram_mb": round(overall_peak_ram_mb, 1),
        "ram_target_mb": 1500.0,
        "ram_passed": ram_passed,
        "p500_mean_latency_sec": p500["mean_latency_sec"],
        "p500_target_latency_sec": 15.0,
        "latency_passed": lat_passed,
        "tier_benchmarks": tier_results
    }

    name = "benchmark_target_results.json" if mode == "shipped" else f"benchmark_target_results_{mode}.json"
    report_path = os.path.join(os.path.dirname(MODELS_DIR), "data", "eval", "results", name)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Saved detailed benchmark report -> {report_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate target hardware and measure latency & RAM.")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark repetitions per word tier")
    parser.add_argument("--mode", choices=list(DETECTOR_MODES), default="shipped", help="Detector mode to benchmark")
    args = parser.parse_args()
    run_target_benchmark(runs_per_tier=args.runs, mode=args.mode)
