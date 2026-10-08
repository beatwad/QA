import pytest

from rag.lang import detect_lang


@pytest.mark.parametrize("text,lang", [
    ("What year did Tesla die?", "en"),
    ("В каком году умер Тесла?", "ru"),
    ("Когда вышел первый iPhone?", "ru"),
    ("What does «договор» mean?", "en"),
    ("Super Bowl 50 прошёл в Санта-Кларе", "ru"),
    ("1943", "en"),
    ("", "en"),
])
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang
