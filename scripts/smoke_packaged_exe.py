"""Smoke-test the PyInstaller onedir build by launching LocalRAG.exe briefly."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "LocalRAG" / "LocalRAG.exe"


def main() -> int:
    if not EXE.is_file():
        print(f"FAIL - {EXE} does not exist", flush=True)
        return 1
    url_file = EXE.parent / "data" / "desktop.url"
    url_file.unlink(missing_ok=True)
    proc = subprocess.Popen([str(EXE)], cwd=str(EXE.parent))
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if url_file.exists() and url_file.read_text(encoding="utf-8").strip():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        url = url_file.read_text(encoding="utf-8").strip() if url_file.exists() else ""
        time.sleep(5)
        alive = proc.poll() is None
        print(f"packaged url  : {url or 'MISSING'}", flush=True)
        print(f"packaged alive: {alive}", flush=True)
        if not url or not alive:
            log = EXE.parent / "data" / "logs" / "localrag.log"
            if log.is_file():
                print("--- packaged log tail ---", flush=True)
                print("\n".join(log.read_text(encoding="utf-8").splitlines()[-40:]), flush=True)
            return 1
        print("PASS - packaged LocalRAG.exe launched and stayed open", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
