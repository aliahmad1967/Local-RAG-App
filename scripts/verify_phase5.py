"""Real verification for Phase 5: desktop shell, free port, lock, run.bat.

Starts the actual ``run.bat`` in headless mode, probes the live FastAPI app,
confirms a second instance is rejected, then shuts everything down. GUI windows
are not opened so this can run unattended.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import load_config  # noqa: E402
from localrag.desktop.preflight import StartupCheck, format_log  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def heading(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(result.stdout.strip()[-6000:], flush=True)
    return result.returncode == 0


def stop_process(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is not None:
        try:
            stdout, _ = proc.communicate(timeout=5)
            return stdout or ""
        except Exception:  # noqa: BLE001
            return ""
    try:
        if os.name == "nt":  # pragma: no cover - Windows target
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:  # pragma: no cover - portable fallback
            proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(input="Y\r\n", timeout=15)
    except Exception:  # noqa: BLE001 - fallback to a hard kill
        proc.kill()
        stdout, _ = proc.communicate(input="Y\r\n", timeout=15)
    return stdout or ""


def main() -> int:
    isolate_verification_data(ROOT)
    cfg = load_config()
    data_dir = cfg.path("paths", "data_dir")
    url_file = data_dir / "desktop.url"
    failures: list[str] = []

    heading("1. ENVIRONMENT")
    print(f"python       : {sys.version.split()[0]}")
    for package in ("pywebview", "pystray", "pyinstaller"):
        try:
            print(f"{package:13s}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            failures.append(f"{package} is not installed")
            print(f"{package:13s}: MISSING")
    for path in ("run.bat", "LocalRAG.spec", "launcher.py"):
        exists = (ROOT / path).is_file()
        print(f"{path:13s}: {exists}")
        if not exists:
            failures.append(f"Missing {path}")
    for asset in ("index.html", "css/app.css", "js/app.js"):
        exists = (ROOT / "localrag" / "ui" / "static" / asset).is_file()
        print(f"ui/{asset:10s}: {exists}")
        if not exists:
            failures.append(f"Missing ui/{asset}")

    check = StartupCheck.from_config(cfg)
    print(format_log(check), flush=True)
    if not check.ok:
        failures.append("Startup preflight is not OK: " + ", ".join(check.missing_models))

    heading("2. UNIT TESTS")
    if not run_tests():
        failures.append("pytest failed")

    heading("3. LIVE DESKTOP SERVER VIA run.bat")
    url_file.unlink(missing_ok=True)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        ["cmd", "/c", "run.bat", "desktop", "--headless"],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    url = ""
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if url_file.exists():
                url = url_file.read_text(encoding="utf-8").strip()
                if url:
                    break
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        print(f"desktop url  : {url or 'MISSING'}", flush=True)
        if not url:
            failures.append("run.bat did not publish a desktop URL")
        else:
            port = int(urlparse(url).port or 0)
            print(f"auto port    : {port} (configured 8765)", flush=True)
            if port <= 0 or port == int(cfg.get("server", "port", default=8765)):
                failures.append("Desktop shell did not pick a distinct free port")
            with httpx.Client(timeout=15) as client:
                health = client.get(f"{url}api/health")
                ui = client.get(f"{url}ui/index.html")
                books = client.get(f"{url}api/books")
            print(f"GET /api/health  -> {health.status_code}", flush=True)
            print(f"GET /ui/index    -> {ui.status_code} bytes={len(ui.content)}", flush=True)
            print(f"GET /api/books   -> {books.status_code} books={len(books.json())}", flush=True)
            if health.status_code != 200:
                failures.append("/api/health failed")
            if ui.status_code != 200 or "LocalRAG" not in ui.text:
                failures.append("/ui/index.html failed")
            if books.status_code != 200 or not isinstance(books.json(), list):
                failures.append("/api/books failed")

            heading("4. SINGLE-INSTANCE LOCK")
            second = subprocess.run(
                ["cmd", "/c", "run.bat", "desktop", "--headless"],
                cwd=str(ROOT), text=True, encoding="utf-8",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
            )
            print(second.stdout.strip()[-1200:], flush=True)
            print(f"second exit  : {second.returncode}", flush=True)
            if second.returncode != 2 or "already running" not in second.stdout:
                failures.append("Second desktop instance was not rejected cleanly")
    finally:
        output = stop_process(proc)
        if output:
            print(output.strip()[-1200:], flush=True)

    heading("5. PACKAGING SPEC")
    spec = (ROOT / "LocalRAG.spec").read_text(encoding="utf-8")
    required_tokens = ("launcher.py", "config.yaml", "localrag/ui/static", "schema.sql", "COLLECT")
    missing_tokens = [token for token in required_tokens if token not in spec]
    print(f"spec tokens  : {'all present' if not missing_tokens else missing_tokens}", flush=True)
    if missing_tokens:
        failures.append("LocalRAG.spec is incomplete: " + ", ".join(missing_tokens))
    if "--build" in sys.argv:
        print("Running full PyInstaller build (this can take several minutes)...", flush=True)
        build = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "LocalRAG.spec"],
            cwd=str(ROOT), text=True, encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600,
        )
        print(build.stdout.strip()[-3000:], flush=True)
        if build.returncode != 0:
            failures.append("PyInstaller build failed")
        else:
            exe = ROOT / "dist" / "LocalRAG" / "LocalRAG.exe"
            print(f"built exe    : {exe} exists={exe.exists()}", flush=True)
            if not exe.exists():
                failures.append("PyInstaller build did not produce LocalRAG.exe")

    heading("6. RESULT")
    if failures:
        for failure in failures:
            print(f"FAIL - {failure}", flush=True)
        return 1
    print("PASS - Phase 5 verified end to end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
