"""Short GUI smoke test: open the real pywebview window, confirm it stays up."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import load_config  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def main() -> int:
    isolate_verification_data(ROOT)
    cfg = load_config()
    url_file = cfg.path("paths", "data_dir") / "desktop.url"
    url_file.unlink(missing_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        ["cmd", "/c", "run.bat"],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    output = ""
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if url_file.exists() and url_file.read_text(encoding="utf-8").strip():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        url = url_file.read_text(encoding="utf-8").strip() if url_file.exists() else ""
        print(f"GUI url      : {url or 'MISSING'}", flush=True)
        time.sleep(5)
        alive = proc.poll() is None
        print(f"GUI alive    : {alive}", flush=True)
        if not url or not alive:
            print(output or "", flush=True)
            return 1
        print("PASS - pywebview desktop window launched and stayed open", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                out, _ = proc.communicate(input="Y\r\n", timeout=15)
                output = out or ""
            except Exception:  # noqa: BLE001
                if os.name == "nt":  # pragma: no cover - Windows target
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                proc.wait(timeout=15)
        else:
            out, _ = proc.communicate(timeout=5)
            output = out or ""
        if output:
            print(output[-1600:], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
