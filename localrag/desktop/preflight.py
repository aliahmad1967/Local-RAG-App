"""Startup checks and the bilingual guidance page shown before the UI opens."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from ..config import Config
from ..ocr.vram import ollama_status


@dataclass(frozen=True)
class StartupCheck:
    ollama_running: bool
    host: str
    ollama_error: str | None
    required_models: tuple[str, ...]
    installed_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.ollama_running and not self.missing_models

    @classmethod
    def from_config(cls, cfg: Config) -> StartupCheck:
        status = ollama_status(cfg)
        installed = tuple(str(name) for name in status.get("models", []))
        required: list[str] = []

        def add(name: object) -> None:
            if name and str(name) not in required:
                required.append(str(name))

        add(cfg.get("ollama", "models", "embedding", default=None))
        add(cfg.get("generation", "model", default=None) or cfg.get("ollama", "models", "llm", default=None))
        engine = str(cfg.get("ocr", "default_engine", default="paddleocr-vl"))
        if engine and engine != "tesseract":
            add(cfg.get("ocr", "engines", engine, "model", default=None))

        warnings: list[str] = []
        if cfg.get("retrieval", "reranker", "enabled", default=True):
            warnings.append(
                "Cross-encoder weights must already be in the Hugging Face cache; "
                "LocalRAG never downloads them at runtime."
            )
        return cls(
            ollama_running=bool(status.get("running")),
            host=str(status.get("host") or cfg.get("ollama", "host", default="")),
            ollama_error=status.get("error"),
            required_models=tuple(required),
            installed_models=installed,
            missing_models=tuple(name for name in required if not _model_installed(name, installed)),
            warnings=tuple(warnings),
        )


def _model_installed(name: str, installed: tuple[str, ...]) -> bool:
    candidates = {name}
    if not name.endswith(":latest"):
        candidates.add(f"{name}:latest")
    else:
        candidates.add(name.removesuffix(":latest"))
    return bool(candidates.intersection(installed))


def format_log(check: StartupCheck) -> str:
    """One-line-per-fact console summary, safe for UTF-8 terminals."""
    lines = [
        f"Ollama: {'running' if check.ollama_running else 'DOWN'} @ {check.host}",
        f"Required models: {', '.join(check.required_models) or 'none'}",
        f"Missing models: {', '.join(check.missing_models) or 'none'}",
    ]
    if check.ollama_error:
        lines.append(f"Ollama error: {check.ollama_error}")
    lines.append(
        "Startup check: OK" if check.ok else "Startup check: ACTION REQUIRED"
    )
    return "\n".join(lines)


def format_startup_html(check: StartupCheck, app_url: str, language: str = "ar") -> str:
    """Return the guidance page; it always keeps a link to the running app."""
    lang = "ar" if str(language).lower().startswith("ar") else "en"
    text = _TEXT[lang]
    pull_items = "".join(
        f"<li><code>{escape('ollama pull ' + model)}</code></li>"
        for model in check.missing_models
    )
    if check.ollama_running:
        problem = text["missing"] if pull_items else text["warning"]
    else:
        problem = (
            f"<p>{escape(text['down'])}</p>"
            f"<pre>{escape('ollama serve')}</pre>"
        )
    pulls = (
        f"<h2>{escape(text['pull_title'])}</h2><ol>{pull_items}</ol>"
        if pull_items else ""
    )
    warnings = "".join(f"<li>{escape(item)}</li>" for item in check.warnings)
    warning_block = (
        f"<h2>{escape(text['note_title'])}</h2><ul>{warnings}</ul>" if warnings else ""
    )
    badge = "OK" if check.ok else "!"
    return f"""<!doctype html>
<html lang="{lang}" dir="{'rtl' if lang == 'ar' else 'ltr'}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalRAG — {escape(text['title'])}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center; padding:32px;
         background:#14110e; color:#efe7d8; font-family:"Segoe UI",Tahoma,sans-serif; }}
  main {{ max-width:720px; background:#211c17; border:1px solid #3d3327; border-radius:18px;
          padding:32px 36px; box-shadow:0 24px 60px #0008; }}
  h1 {{ margin:0 0 8px; color:#e6b866; font-size:26px; }}
  h2 {{ margin:24px 0 8px; font-size:17px; color:#d8c8ae; }}
  p, li {{ line-height:1.8; }}
  code, pre {{ background:#100e0b; border:1px solid #3d3327; border-radius:8px;
               padding:6px 10px; color:#f0d49a; font-family:Consolas,monospace; }}
  pre {{ padding:14px; }}
  .badge {{ display:inline-grid; place-items:center; width:34px; height:34px; border-radius:50%;
            background:#e6b866; color:#14110e; font-weight:700; margin-inline-end:10px; }}
  a {{ display:inline-block; margin-top:22px; padding:12px 18px; border-radius:10px;
       background:#e6b866; color:#14110e; text-decoration:none; font-weight:700; }}
  a:hover {{ background:#f3cc88; }}
</style>
</head>
<body>
<main>
  <h1><span class="badge">{badge}</span>LocalRAG — {escape(text['title'])}</h1>
  {problem}
  {pulls}
  {warning_block}
  <a href="{escape(app_url, quote=True)}">{escape(text['continue'])}</a>
</main>
</body>
</html>"""


_TEXT = {
    "ar": {
        "title": "فحص بدء التشغيل",
        "down": "لا يمكن الوصول إلى Ollama. شغّله ثم أعد فتح التطبيق:",
        "missing": "بعض النماذج المطلوبة غير مثبّتة. نفّذ الأوامر التالية ثم أعد فتح التطبيق:",
        "warning": "كل شيء جاهز. هذه ملاحظة فقط:",
        "pull_title": "أوامر تثبيت النماذج",
        "note_title": "ملاحظة",
        "continue": "متابعة إلى التطبيق",
    },
    "en": {
        "title": "Startup check",
        "down": "Ollama is not reachable. Start it, then reopen the app:",
        "missing": "Some required models are missing. Run these commands, then reopen the app:",
        "warning": "Everything is ready. This is only a note:",
        "pull_title": "Model install commands",
        "note_title": "Note",
        "continue": "Continue to LocalRAG",
    },
}
