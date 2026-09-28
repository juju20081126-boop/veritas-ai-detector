"""
Launcher Script for Veritas AI — QuillBot-Style Offline AI Detector
Starts the FastAPI application and automatically launches your web browser.
Fully offline execution on low-end hardware (<= 1.5GB RAM, 2 CPU threads).
"""

import sys
import os
import socket
import webbrowser
import threading
import time
import argparse
import uvicorn

# Restrict threading for low-end hardware simulation
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"


def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def open_browser(url: str, delay_seconds: float = 1.8):
    """Waits for the server to start, then opens the browser."""
    def _open():
        time.sleep(delay_seconds)
        print(f"[Launcher] Opening Veritas AI Dashboard at {url} ...")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Veritas AI Offline Detector Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    print("=" * 72)
    print("  \033[1;36mVERITAS AI — QUILLBOT-STYLE OFFLINE AI DETECTOR\033[0m")
    print("  \033[90mTarget: <= 1.5GB RAM | 2 CPU Threads | Zero-PyTorch Runtime\033[0m")
    print("=" * 72)

    # Pre-check model files
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    onnx_path = os.path.join(models_dir, "student_model_int8.onnx")
    if not os.path.exists(onnx_path):
        print("\n\033[33m[Notice] Shipped INT8 ONNX model not found.\033[0m")
        print("Exporting ONNX student model from checkpoint...")
        from scripts.export_onnx import export_student_to_onnx_int8
        export_student_to_onnx_int8()

    port = find_free_port(args.port)
    url = f"http://localhost:{port}"

    print(f"\n[Launcher] Starting local offline server at {url}")
    print("[Launcher] Engine initialized: ONNX INT8 + Stylometric Meta-Classifier")

    if not args.no_browser:
        open_browser(url, delay_seconds=2.0)

    try:
        uvicorn.run(
            "backend.server:app",
            host="127.0.0.1",
            port=port,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down Veritas AI server. Goodbye!")


if __name__ == "__main__":
    main()
