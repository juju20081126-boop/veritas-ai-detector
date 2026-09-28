"""
Launcher Script for Veritas AI Originality & AI Writing Detector
Starts the FastAPI application and automatically launches your web browser.
"""

import sys
import os
import socket
import webbrowser
import threading
import time
import uvicorn


def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def open_browser(url: str, delay_seconds: float = 2.0):
    """Waits for the server to start, then opens the browser."""
    def _open():
        time.sleep(delay_seconds)
        print(f"[Launcher] Opening Veritas AI Dashboard at {url} ...")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    print("=" * 70)
    print("          VERITAS AI — INSTITUTIONAL AI WRITING DETECTOR          ")
    print("        Turnitin-Grade Ensemble Forensic Originality Audit       ")
    print("=" * 70)

    port = find_free_port(8000)
    url = f"http://localhost:{port}"

    print(f"\n[Launcher] Starting server on {url}")
    print("[Launcher] Pre-warming transformer models (RoBERTa & GPT-2)...")

    # Open browser once server starts
    if "--no-browser" not in sys.argv:
        open_browser(url, delay_seconds=2.5)

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
