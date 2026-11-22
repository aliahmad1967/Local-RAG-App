"""Generate image-only ('scanned') sample PDFs for Phase 1 verification."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def _display(text: str, language: str) -> str:
        if language != "ar":
            return text
        return get_display(arabic_reshaper.reshape(text))
except ImportError:  # pragma: no cover - fallback if shaping libs absent
    def _display(text: str, language: str) -> str:
        return text


AR_HEADING = "الفصل الأول: في شأن الكتب والمعرفة"
EN_HEADING = "Chapter One: On Books and Knowledge"

AR_SENTENCES = [
    "العلم نورٌ، والكتابُ خزنةٌ من خزائن المعرفة والتاريخ.",
    "قرأتُ في المكتبة كتابًا عن تاريخ المدن القديمة.",
    "تُحفظ الكتبُ في رفوفٍ مرتبةٍ حسبَ الموضوع واللغة.",
    "إنَّ القراءةَ اليوميةَ تُوسِّعُ المداركَ وتُنمّي الفكرَ.",
    "الـكتابُ الصديقُ الذي لا يملُّ من الحديثِ في كلِّ وقتٍ.",
    "في صفحةِ الغدِ سنقرأُ فصلاً عن الرحلاتِ والاكتشافاتِ.",
    "تحتوي المكتبةُ على مخطوطاتٍ عربيةٍ ونصوصٍ لاتينيةٍ قديمة.",
    "المعرفةُ صدى التاريخ، والصفحةُ شاهدةٌ على مرِّ الزمانِ.",
]

EN_SENTENCES = [
    "The library keeps knowledge safe for every generation of readers.",
    "Each page carries a trace of the hands that copied it long ago.",
    "A good book asks better questions than a quick answer ever could.",
    "The archivist catalogued the manuscript by title, date, and language.",
    "Reading slowly often reveals what a hurried scan misses completely.",
    "Every chapter in this volume describes a different journey.",
    "Paper, ink, and patience turned a thought into a lasting record.",
    "The final section compares Arabic and Latin sources side by side.",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\tahomabd.ttf" if bold else r"C:\Windows\Fonts\tahoma.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sample_pdf(
    path: str | Path,
    pages: int = 20,
    language: str = "ar",
    dpi: int = 150,
) -> Path:
    """Create a PDF whose pages are *only images* — forcing the OCR path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(8.27 * dpi), int(11.69 * dpi)
    margin = int(0.85 * dpi)
    heading_font = _font(int(0.26 * dpi), bold=True)
    body_font = _font(int(0.17 * dpi))

    doc = fitz.open()
    try:
        for page_no in range(1, pages + 1):
            image = Image.new("RGB", (width, height), (252, 250, 244))
            draw = ImageDraw.Draw(image)
            heading = (AR_HEADING if language == "ar" else EN_HEADING) + f"  {page_no}"
            draw.text(
                (margin, margin), _display(heading, language),
                font=heading_font, fill=(22, 20, 17),
            )
            y = margin + int(0.95 * dpi)
            pool = AR_SENTENCES if language == "ar" else EN_SENTENCES
            lines = [pool[(page_no + offset) % len(pool)] for offset in range(6)]
            for line in lines:
                draw.text(
                    (margin, y), _display(line, language),
                    font=body_font, fill=(32, 29, 25),
                )
                y += int(0.48 * dpi)
            draw.line(
                [(margin, height - margin), (width - margin, height - margin)],
                fill=(190, 178, 155), width=2,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            page = doc.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=buffer.getvalue())
        doc.save(str(path), deflate=True)
    finally:
        doc.close()
    return path


def generate_all(target_dir: str | Path) -> dict[str, Path]:
    target_dir = Path(target_dir)
    return {
        "ar": make_sample_pdf(target_dir / "sample_ar_scanned_20.pdf", 20, "ar"),
        "en": make_sample_pdf(target_dir / "sample_en_scanned_20.pdf", 20, "en"),
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for key, generated in generate_all(here).items():
        print(f"{key}: {generated.name} ({generated.stat().st_size} bytes)")
