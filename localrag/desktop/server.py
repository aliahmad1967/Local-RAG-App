"""Programmatic Uvicorn host used by the desktop shell."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import httpx
import uvicorn

from ..logging_conf import get_logger

log = get_logger("desktop.server")


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an unused loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class LocalServer:
    """Run a FastAPI app in a daemon thread and stop it cleanly."""

    def __init__(
        self,
        app: Any,
        host: str = "127.0.0.1",
        port: int = 0,
        ready_path: str = "/api/health",
    ) -> None:
        if port < 0 or port > 65535:
            raise ValueError(f"Invalid port: {port}")
        self.app = app
        self.host = host
        self.port = port
        self.ready_path = ready_path
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self, timeout: float = 30.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            # Uvicorn's default formatter writes to sys.stderr, which is None
            # in a windowed PyInstaller build. Keep our UTF-8 file logging.
            log_config=None,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="localrag-uvicorn",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            servers = getattr(self._server, "servers", None)
            if servers and servers[0].sockets:
                bound = servers[0].sockets[0].getsockname()
                self.port = int(bound[1])
                break
            if not self._thread.is_alive():
                raise RuntimeError("The local web server stopped during startup.")
            time.sleep(0.05)
        else:
            raise RuntimeError("Timed out waiting for the local web server to bind.")

        if not self.wait_ready(timeout=max(1.0, timeout - 5.0)):
            self.stop()
            raise RuntimeError(f"Local web server did not become ready at {self.url}")
        log.info("Local web server ready: %s", self.url)

    def wait_ready(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        target = f"http://{self.host}:{self.port}{self.ready_path}"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(target, timeout=0.75)
                if response.status_code < 500:
                    return True
            except Exception:  # noqa: BLE001 - server may not be listening yet
                pass
            time.sleep(0.08)
        return False

    def stop(self, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():  # pragma: no cover - rare hung request
                log.warning("Uvicorn thread did not exit within %.1fs", timeout)
        self._thread = None
        self._server = None
