from localrag.text.language import detect_language


def test_arabic():
    assert detect_language("هذا نص عربي طويل يحتوي على许多 حروف عربية فقط") == "ar"


def test_english():
    assert detect_language("This is a long English sentence with only Latin letters.") == "en"


def test_mixed():
    assert detect_language("هذا كتاب Arabic and English mixed text together here.") == "mixed"


def test_unknown_for_short_text():
    assert detect_language("abc") == "unknown"
