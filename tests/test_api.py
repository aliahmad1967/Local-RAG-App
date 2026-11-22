"""Smoke-test the FastAPI surface with an isolated database."""

from fastapi.testclient import TestClient

from localrag.api import app as app_module


def test_health_books_and_static_ui(cfg, monkeypatch):
    monkeypatch.setattr(app_module, "load_config", lambda: cfg)
    with TestClient(app_module.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ollama"]["host"].startswith("http://")

        books = client.get("/api/books")
        assert books.status_code == 200
        assert books.json() == []

        assert client.get("/").status_code == 200
        page = client.get("/ui/index.html")
        assert page.status_code == 200
        assert "charset=utf-8" in page.headers["content-type"]
        script = client.get("/ui/js/app.js")
        assert script.status_code == 200
        assert "charset=utf-8" in script.headers["content-type"]

        config = client.get("/api/config")
        assert config.status_code == 200
        assert config.json()["ocr"]["default_engine"] == "paddleocr-vl"


def test_delete_books_api(cfg, monkeypatch):
    monkeypatch.setattr(app_module, "load_config", lambda: cfg)
    with TestClient(app_module.create_app()) as client:
        db = client.app.state.db
        now = "2026-11-21T00:00:00Z"
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO books
                   (id, title, file_path, file_name, page_count, language, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("book-a", "Book A", "missing.pdf", "Book A.pdf", 10, "ar", "ingested", now, now),
            )

        assert client.get("/api/books").status_code == 200
        assert len(client.get("/api/books").json()) == 1
        result = client.delete("/api/books/book-a")
        assert result.status_code == 200
        assert result.json()["ok"] is True
        assert client.get("/api/books").json() == []
