"""YAML configuration loading with a local override layer."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import DATA_ROOT, IS_FROZEN, PROJECT_ROOT, resolve_path

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# Bundled defaults are read-only; user settings live with the data directory.
LOCAL_CONFIG_PATH = DATA_ROOT / "config.local.yaml" if IS_FROZEN else PROJECT_ROOT / "config.local.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Small typed accessor over the nested config dictionary."""

    def __init__(
        self,
        data: dict[str, Any],
        root: Path | None = None,
        local_path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.data = data
        self.root = root or PROJECT_ROOT
        self.data_root = data_root or DATA_ROOT
        self.local_path = local_path or LOCAL_CONFIG_PATH

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, *keys: str) -> Any:
        value = self.get(*keys, default=_MISSING)
        if value is _MISSING:
            raise KeyError("Missing config key: " + ".".join(keys))
        return value

    def path(self, *keys: str, default: str = "") -> Path:
        # Runtime paths are user data (next to the .exe when frozen); every
        # other relative path is a bundled application resource.
        root = self.data_root if keys and keys[0] == "paths" else self.root
        return resolve_path(root, str(self.get(*keys, default=default)))

    def save_local(self, updates: dict[str, Any]) -> None:
        """Persist UI settings into config.local.yaml."""
        existing: dict[str, Any] = {}
        if self.local_path.exists():
            loaded = yaml.safe_load(self.local_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                existing = loaded
        self.local_path.write_text(
            yaml.safe_dump(deep_merge(existing, updates), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.data = deep_merge(self.data, updates)


_MISSING = object()


def load_config(
    path: Path | None = None,
    local_path: Path | None = None,
    root: Path | None = None,
    data_root: Path | None = None,
) -> Config:
    """Load defaults, then merge config.local.yaml on top."""
    base_path = path or DEFAULT_CONFIG_PATH
    override_path = local_path or LOCAL_CONFIG_PATH
    data_root = data_root or Path(os.environ.get("LOCALRAG_DATA_ROOT", DATA_ROOT))
    # base_path is the config *file*, so its parent is the project root.
    root = root or base_path.resolve().parent

    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Config root must be a mapping: {base_path}")
    if override_path.exists():
        override = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        if isinstance(override, dict):
            base = deep_merge(base, override)
    return Config(base, root=root, local_path=override_path, data_root=data_root)


def ensure_data_dirs(cfg: Config) -> None:
    for key in (
        ("paths", "data_dir"),
        ("paths", "db_path"),
        ("paths", "books_dir"),
        ("paths", "logs_dir"),
        ("paths", "uploads_dir"),
    ):
        raw = str(cfg.get(*key, default=""))
        # db_path is a file: create its parent instead.
        target = cfg.path(*key, default="")
        if key[-1] == "db_path":
            target = target.parent
        target.mkdir(parents=True, exist_ok=True)
        _ = raw
