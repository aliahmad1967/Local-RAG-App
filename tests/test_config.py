from __future__ import annotations

from pathlib import Path

from localrag.config import load_config


def test_load_config_uses_data_root_environment_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCALRAG_DATA_ROOT", str(tmp_path))

    cfg = load_config()

    assert cfg.path("paths", "db_path") == tmp_path / "data" / "localrag.db"
