"""Shared setup for isolated end-to-end verification data."""

from __future__ import annotations

import os
from pathlib import Path


def isolate_verification_data(root: Path) -> None:
    os.environ.setdefault("LOCALRAG_DATA_ROOT", str(root / "data" / "verification"))
