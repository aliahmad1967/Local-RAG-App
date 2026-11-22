"""End-to-end desktop entry point: preflight, server, pywebview, cleanup."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ..api.app import create_app
from ..config import ensure_data_dirs, load_config
from ..logging_conf import get_logger, setup_logging
from ..paths import ensure_utf8_console
from .lock import InstanceAlreadyRunning, SingleInstanceLock
from .preflight import StartupCheck, format_log, format_startup_html
from .server import LocalServer, find_free_port
from .tray import TrayController

log = get_logger("desktop")


def run_desktop(*, headless: bool = False, port: int | None = None) -> int:
    """Start LocalRAG as a desktop app (or a headless server for verification)."""
    ensure_utf8_console()
    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)

    data_dir: Path = cfg.path("paths", "data_dir")
    lock = SingleInstanceLock(data_dir / ".localrag-desktop.lock")
    try:
        lock.acquire()
    except InstanceAlreadyRunning as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    server: LocalServer | None = None
    tray: TrayController | None = None
    url_file = data_dir / "desktop.url"
    try:
        check = StartupCheck.from_config(cfg)
        print(format_log(check), flush=True)

        host = str(cfg.get("server", "host", default="127.0.0.1"))
        chosen = int(port) if port is not None else int(cfg.get("desktop", "port", default=0))
        if chosen <= 0:
            chosen = find_free_port(host)

        server = LocalServer(
            create_app(), host=host, port=chosen, ready_path="/ui/index.html"
        )
        server.start()
        url = server.url
        url_file.write_text(url, encoding="utf-8")
        print(f"LOCALRAG_READY {url}", flush=True)

        if headless:
            print("Headless mode: press Ctrl+C to stop.", flush=True)
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:  # pragma: no cover - interactive only
                return 0

        try:
            import webview
        except ImportError as exc:
            print(
                "ERROR: pywebview is missing. Install dependencies with "
                "`python -m pip install -r requirements.txt`. "
                f"({exc})",
                flush=True,
            )
            return 1

        html = None if check.ok else format_startup_html(
            check, url, str(cfg.get("ui", "language", default="ar"))
        )
        window = webview.create_window(
            "LocalRAG — offline books",
            url=None if html else url,
            html=html,
            width=int(cfg.get("desktop", "window_width", default=1440)),
            height=int(cfg.get("desktop", "window_height", default=900)),
            min_size=(900, 620),
            background_color="#14110e",
        )
        if window is None:  # pragma: no cover - defensive
            print("ERROR: pywebview could not create a window.", flush=True)
            return 1

        def on_closing() -> None:
            log.info("Desktop window is closing")
            if server is not None:
                threading.Thread(target=server.stop, name="localrag-stop", daemon=True).start()

        window.events.closing += on_closing

        if bool(cfg.get("desktop", "system_tray", default=False)):
            tray = TrayController(
                on_show=lambda: window.show(),
                on_quit=lambda: (window.destroy(), on_closing()),
            )
            if not tray.start():
                tray = None

        webview.start(debug=False, private_mode=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - surfaced to console/UI packaging
        print(f"ERROR: {exc}", flush=True)
        log.exception("Desktop startup failed")
        return 1
    finally:
        if tray is not None:
            tray.stop()
        if server is not None:
            server.stop()
        url_file.unlink(missing_ok=True)
        lock.release()


def main() -> int:
    """CLI-compatible wrapper used by ``python -m localrag desktop``."""
    return run_desktop()
