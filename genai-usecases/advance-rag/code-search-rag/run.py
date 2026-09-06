"""Start the API and the UI together.

    python run.py

Runs uvicorn (FastAPI) and Streamlit as child processes, waits for the API to
answer /health before starting the UI, and shuts both down on Ctrl+C. Needs no
install of this project: it puts the project root on PYTHONPATH itself.

Run the two halves separately if you prefer:

    uvicorn api.main:app --reload --port 8000
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))
API_BASE_URL = f"http://127.0.0.1:{API_PORT}"

processes: list[subprocess.Popen] = []


def child_env() -> dict:
    """Environment for the children: project root importable, API URL known."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    )
    env["API_BASE_URL"] = API_BASE_URL
    return env


def wait_for_api(timeout: float | None = None) -> bool:
    """Poll /health until the API answers, so the UI never starts too early.

    The wait is generous and configurable because a cold first start can be
    slow: importing heavy agent frameworks, or downloading a local embedding
    model, can take minutes the first time. Timing out early kills a perfectly
    healthy API and reports that it did not start, which sends the reader
    hunting for a bug that is not there.
    """
    timeout = timeout if timeout is not None else float(
        os.getenv("API_START_TIMEOUT", "300")
    )
    deadline = time.time() + timeout
    started = time.time()
    notified = False
    while time.time() < deadline:
        if processes and processes[0].poll() is not None:
            return False  # the API died; no point waiting out the timeout
        try:
            with urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=10) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            if not notified and time.time() - started > 20:
                print("  still starting (first-run imports can be slow)...", flush=True)
                notified = True
            time.sleep(0.5)
    return False


def shutdown(*_args) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    env = child_env()

    signal.signal(signal.SIGINT, lambda *a: (shutdown(), sys.exit(0)))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: (shutdown(), sys.exit(0)))

    print(f"Starting API on {API_BASE_URL} ...", flush=True)
    processes.append(subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=PROJECT_ROOT, env=env,
    ))

    if not wait_for_api():
        print("The API did not start. Its output is above.\n"
              "If it was still loading, raise the wait with "
              "API_START_TIMEOUT=600 python run.py", file=sys.stderr)
        shutdown()
        return 1

    print(f"API ready.  Docs: {API_BASE_URL}/docs", flush=True)
    print(f"Starting UI on http://localhost:{UI_PORT} ...", flush=True)
    processes.append(subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py",
         "--server.port", str(UI_PORT)],
        cwd=PROJECT_ROOT, env=env,
    ))

    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    print("A service exited; shutting down the other.", file=sys.stderr)
                    shutdown()
                    return process.returncode or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
