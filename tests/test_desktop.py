"""Phase 5 desktop-shell tests: no GUI or real LLM required."""

from __future__ import annotations

import socket

import httpx
from fastapi import FastAPI

from localrag.desktop.lock import InstanceAlreadyRunning, SingleInstanceLock
from localrag.desktop.preflight import StartupCheck, format_log, format_startup_html
from localrag.desktop.server import LocalServer, find_free_port


def test_single_instance_lock(tmp_path):
    path = tmp_path / "data" / ".localrag.lock"
    first = SingleInstanceLock(path).acquire()
    try:
        assert first.locked
        second = SingleInstanceLock(path)
        try:
            second.acquire()
        except InstanceAlreadyRunning as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("A second instance acquired the lock")
        finally:
            second.release()
    finally:
        first.release()

    third = SingleInstanceLock(path)
    third.acquire()
    try:
        assert third.locked
    finally:
        third.release()


def test_find_free_port():
    port = find_free_port("127.0.0.1")
    assert 1024 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_local_server_starts_and_stops():
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    server = LocalServer(app, host="127.0.0.1", port=find_free_port())
    try:
        server.start(timeout=15)
        response = httpx.get(f"{server.url}api/health", timeout=3)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        server.stop()
    assert server._thread is None


def test_preflight_detects_missing_models(monkeypatch, cfg):
    monkeypatch.setattr(
        "localrag.desktop.preflight.ollama_status",
        lambda _cfg: {
            "running": True,
            "host": "http://127.0.0.1:11434",
            "models": ["qwen3.5:latest", "seriouswebby/paddleocr-vl-1.6:latest"],
            "error": None,
        },
    )
    check = StartupCheck.from_config(cfg)
    assert check.ok is False
    assert check.missing_models == ("bge-m3",)
    assert "Ollama: running" in format_log(check)

    html = format_startup_html(check, "http://127.0.0.1:9999/", language="ar")
    assert "ollama pull bge-m3" in html
    assert "http://127.0.0.1:9999/" in html
    assert "<html lang=\"ar\" dir=\"rtl\">" in html


def test_preflight_accepts_latest_suffix(monkeypatch, cfg):
    monkeypatch.setattr(
        "localrag.desktop.preflight.ollama_status",
        lambda _cfg: {
            "running": True,
            "host": "http://127.0.0.1:11434",
            "models": [
                "bge-m3:latest",
                "qwen3.5:latest",
                "seriouswebby/paddleocr-vl-1.6:latest",
            ],
            "error": None,
        },
    )
    check = StartupCheck.from_config(cfg)
    assert check.missing_models == ()
    assert check.ok is True


def test_runtime_paths_use_data_root(cfg, tmp_path):
    portable = tmp_path / "portable"
    cfg.data_root = portable
    cfg.data["paths"]["db_path"] = "data/localrag.db"
    assert cfg.path("paths", "db_path") == portable / "data" / "localrag.db"
