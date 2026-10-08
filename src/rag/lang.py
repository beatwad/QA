import re

_LETTER = re.compile(r"[^\W\d_]")
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def detect_lang(text: str) -> str:
    """'ru' if at least half of the letters are Cyrillic, else 'en'."""
    letters = _LETTER.findall(text)
    if not letters:
        return "en"
    cyrillic = sum(bool(_CYRILLIC.match(c)) for c in letters)
    return "ru" if cyrillic / len(letters) >= 0.5 else "en"
