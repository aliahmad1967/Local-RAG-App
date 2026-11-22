"""Shared fixtures. Every test gets an isolated SQLite database under tmp_path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localrag.config import load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402


@pytest.fixture
def cfg(tmp_path: Path):
    config = load_config()
    data = tmp_path / "data"
    config.data["paths"] = {
        "data_dir": str(data),
        "db_path": str(data / "localrag.db"),
        "books_dir": str(data / "books"),
        "logs_dir": str(data / "logs"),
        "uploads_dir": str(data / "books" / "_uploads"),
    }
    config.data["ingestion"]["unload_ocr_after_job"] = False
    config.local_path = tmp_path / "config.local.yaml"
    for value in config.data["paths"].values():
        target = Path(value)
        (target.parent if target.suffix == ".db" else target).mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def db(cfg):
    return Database(cfg.path("paths", "db_path"))
