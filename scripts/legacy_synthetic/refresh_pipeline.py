# LEGACY / QUARANTINED 2026-10-01 -- DO NOT RUN.
# Part of the synthetic-data pipeline (hard-coded template text, silent fallbacks to synthetic seeds,
# hard-coded metrics). See scripts/legacy_synthetic/README.md and data/eval/legacy_audit.json.
raise SystemExit("scripts/legacy_synthetic/refresh_pipeline.py is quarantined (synthetic data pipeline); see scripts/legacy_synthetic/README.md")

"""
Veritas AI — Automated Data-Refresh & Retraining Pipeline
Executes end-to-end retraining with a single command:
  Add new generator models -> Regenerate data -> Teacher scoring -> Re-distill student -> INT8 ONNX Export -> Target Benchmark
"""

import os
import sys
import time
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_step(step_num: int, title: str):
    print("\n" + "=" * 76)
    print(f"  \033[1;34m[STEP {step_num}] {title.upper()}\033[0m")
    print("=" * 76 + "\n")


def run_command(cmd_args: list, desc: str):
    print(f"[Pipeline] Running: {' '.join(cmd_args)}")
    t0 = time.time()
    res = subprocess.run(cmd_args, cwd=REPO_ROOT, capture_output=True, text=True)
    t1 = time.time()
    
    if res.returncode != 0:
        print(f"\033[1;31m[Pipeline] ERROR during {desc} (Exit code {res.returncode}):\033[0m")
        print(res.stderr)
        print(res.stdout)
        sys.exit(res.returncode)
    else:
        print(f"[Pipeline] \033[1;32mSUCCESS\033[0m ({t1 - t0:.2f}s): {desc}")
        if res.stdout.strip():
            lines = res.stdout.strip().split("\n")
            for l in lines[-6:]:
                print(f"    {l}")


def main():
    parser = argparse.ArgumentParser(description="Veritas AI Complete Data-Refresh & Student Retraining Pipeline")
    parser.add_argument("--new_models", nargs="+", default=["deepseek-r1", "gemma-2-9b"], help="New generator models to add")
    parser.add_argument("--count_per_genre", type=int, default=25, help="Number of new samples per genre")
    parser.add_argument("--epochs", type=int, default=4, help="Student training epochs")
    parser.add_argument("--skip_generate", action="store_true", help="Skip data generation and only retrain/export")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 76)
    print("  \033[1;36mVERITAS AI — END-TO-END DATA-REFRESH & RETRAINING PIPELINE\033[0m")
    print(f"  Models to refresh: {', '.join(args.new_models)}")
    print(f"  Samples per genre: {args.count_per_genre} across 5 genres")
    print("=" * 76)

    # Step 1: Ingest Public Datasets
    log_step(1, "Public Detection Corpora Verification")
    run_command([sys.executable, "scripts/download_datasets.py", "--max_samples", "150"], "Ingesting public research datasets")

    # Step 2: Fresh Frontier & Open Model Generation
    if not args.skip_generate:
        log_step(2, f"Frontier Text Generation ({', '.join(args.new_models)})")
        run_command([sys.executable, "scripts/generate_frontier_data.py", "--count_per_genre", str(args.count_per_genre)],
                    "Generating fresh frontier model data")

    # Step 3: Refinement Pipeline (Human Polish & AI Paraphrase)
    log_step(3, "Refinement Pipeline: Human Polish & AI Paraphrasing")
    run_command([sys.executable, "scripts/refine_data.py"], "Creating 4-class refined datasets")

    # Step 4: Assemble Dataset & Leave-One-Model-Out Partitions
    log_step(4, "Assembling Dataset & Partitions")
    run_command([sys.executable, "scripts/build_dataset.py"], "Building train/val/test/LOMO splits")

    # Step 5: Student Knowledge Distillation & Meta-Classifier Training
    log_step(5, "Student Knowledge Distillation & Stylometric Meta-Classifier")
    run_command([sys.executable, "scripts/train_student.py", "--epochs", str(args.epochs)],
                "Training student model & fitting meta-classifier")

    # Step 6: ONNX Export & Dynamic INT8 Quantization
    log_step(6, "INT8 Dynamic Quantization & ONNX Export")
    weights_path = os.path.join(REPO_ROOT, "models", "student_pytorch.pt")
    run_command([sys.executable, "scripts/export_onnx.py", "--weights", weights_path],
                "Exporting and quantizing ONNX student graph")

    # Step 7: Target Hardware Simulation Benchmark
    log_step(7, "Simulated Target Hardware Benchmark (2 Threads, <=1.5GB RAM)")
    run_command([sys.executable, "scripts/benchmark_target.py"], "Running target hardware simulation")

    # Step 8: Comprehensive Model Evaluation
    log_step(8, "Comprehensive Model Evaluation Suite")
    run_command([sys.executable, "scripts/evaluate_models.py"], "Evaluating all success criteria")

    t_total = time.time() - t_start
    print("\n" + "=" * 76)
    print(f"  \033[1;32mPIPELINE EXECUTION COMPLETED IN {t_total:.1f} SECONDS\033[0m")
    print("  All artifacts updated, quantized, benchmarked, and evaluated.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
