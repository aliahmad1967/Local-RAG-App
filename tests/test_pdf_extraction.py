from pathlib import Path

import pytest

from localrag.ingest.pdf import (
    extract_embedded_text,
    has_usable_text,
    inspect_pdf,
    open_document,
    rasterize_page,
)

from .data.make_samples import make_sample_pdf


@pytest.fixture(scope="module")
def scanned_pdf(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("pdf") / "tiny_scanned.pdf"
    return make_sample_pdf(target, pages=4, language="en", dpi=100)


def test_page_count_and_empty_text_layer(scanned_pdf):
    meta = inspect_pdf(scanned_pdf)
    assert meta.page_count == 4
    with open_document(scanned_pdf) as doc:
        assert extract_embedded_text(doc[0]) == ""
        assert not has_usable_text(extract_embedded_text(doc[0]), min_chars=80)


def test_rasterization_creates_image(scanned_pdf, tmp_path):
    with open_document(scanned_pdf) as doc:
        output = rasterize_page(doc[0], tmp_path / "pages" / "p0001.png", dpi=150)
    assert output.exists()
    assert output.stat().st_size > 10_000
