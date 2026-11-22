"""Single-instance lock backed by an OS-held file lock.

The file may remain after a crash; the operating system releases the lock when
the process exits, so the next launch can acquire it immediately.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

from ..logging_conf import get_logger

log = get_logger("desktop.lock")


class InstanceAlreadyRunning(RuntimeError):
    """Raised when another LocalRAG desktop process owns the lock file."""


class SingleInstanceLock:
    """Context manager around one locked byte in ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._locked = False

    def acquire(self) -> SingleInstanceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        handle.write(b"\x00")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - portable fallback for tests/dev
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise InstanceAlreadyRunning(
                "LocalRAG is already running. Check the taskbar or close the "
                f"existing process before starting another copy. ({exc})"
            ) from exc
        self._handle = handle
        self._locked = True
        log.info("Acquired desktop lock: %s", self.path)
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - portable fallback for tests/dev
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:  # pragma: no cover - best-effort cleanup
            log.debug("Could not unlock %s: %s", self.path, exc)
        finally:
            self._handle.close()
            self._handle = None
            self._locked = False
            log.info("Released desktop lock: %s", self.path)

    @property
    def locked(self) -> bool:
        return self._locked

    def __enter__(self) -> SingleInstanceLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
