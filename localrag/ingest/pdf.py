"""PDF reading, embedded-text detection and page rasterization (PyMuPDF)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from ..paths import page_image_path


@dataclass
class PDFMetadata:
    page_count: int
    title: str
    author: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"page_count": self.page_count, "title": self.title, "author": self.author}


def safe_filename(name: str) -> str:
    """Keep Unicode (incl. Arabic) filenames but strip path parts and controls."""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", str(name)).strip()
    return Path(cleaned.replace("\\", "/")).name or "book.pdf"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_document(path: Path) -> fitz.Document:
    doc = fitz.open(str(path))
    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF has no pages: {path}")
    return doc


def inspect_pdf(path: Path) -> PDFMetadata:
    with open_document(path) as doc:
        meta = doc.metadata or {}
        title = str(meta.get("title") or "").strip()
        if not title:
            title = Path(path).stem
        return PDFMetadata(
            page_count=doc.page_count,
            title=title,
            author=str(meta.get("author") or ""),
        )


def extract_embedded_text(page: fitz.Page) -> str:
    """Text layer as laid out on the page; empty for scanned images."""
    return page.get_text("text", sort=True).strip()


def has_usable_text(text: str, min_chars: int) -> bool:
    letters = sum(1 for ch in text if ch.isalnum())
    return letters >= min_chars


def rasterize_page(
    page: fitz.Page,
    out_path: Path,
    dpi: int = 250,
) -> Path:
    """Render one page to PNG at the requested DPI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    pixmap.save(str(out_path))
    return out_path


def page_image_for(
    doc: fitz.Page,
    books_root: Path,
    book_id: str,
    page_no: int,
    dpi: int = 250,
) -> Path:
    """Return the cached raster path, rendering it only if missing."""
    target = page_image_path(books_root, book_id, page_no)
    if not target.exists():
        rasterize_page(doc, target, dpi=dpi)
    return target
