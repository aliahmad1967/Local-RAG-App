"""Optional Windows system-tray controller (requires ``pystray``)."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..logging_conf import get_logger

log = get_logger("desktop.tray")


class TrayController:
    """Small show/quit tray icon; silently unavailable without pystray."""

    def __init__(self, on_show: Callable[[], None], on_quit: Callable[[], None]) -> None:
        self.on_show = on_show
        self.on_quit = on_quit
        self._icon = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def available() -> bool:
        try:
            import pystray  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self) -> bool:
        if not self.available():
            log.info("pystray is not installed; system tray is disabled")
            return False
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover - guarded by available()
            log.warning("Could not load tray dependencies: %s", exc)
            return False

        image = Image.new("RGB", (64, 64), "#14110e")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((8, 12, 56, 52), radius=6, fill="#e6b866")
        draw.rectangle((10, 12, 31, 52), fill="#8a6327")
        draw.line((34, 24, 50, 24), fill="#14110e", width=3)
        draw.line((34, 32, 50, 32), fill="#14110e", width=3)
        self._icon = pystray.Icon(
            "LocalRAG",
            image,
            "LocalRAG — offline books",
            menu=pystray.Menu(
                pystray.MenuItem("إظهار / Show", lambda *_: self.on_show(), default=True),
                pystray.MenuItem("خروج / Quit", lambda *_: self.on_quit()),
            ),
        )
        self._thread = threading.Thread(
            target=self._icon.run, name="localrag-tray", daemon=True
        )
        self._thread.start()
        log.info("System tray started")
        return True

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                log.debug("Could not stop tray icon: %s", exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._icon = None
