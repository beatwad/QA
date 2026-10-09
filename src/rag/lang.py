import re

_LETTER = re.compile(r"[^\W\d_]")
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def detect_lang(text: str) -> str:
    """'ru' if at least half of the letters are Cyrillic, else 'en'."""
    # find all letters in a text
    letters = _LETTER.findall(text)
    # if no letters, e.g. string looks like "2024-01-15"
    # return default language
    if not letters:
        return "en"
    # calculate ratio of RUS/ENG, if >= 0.5 then ru, else en
    cyrillic = sum(bool(_CYRILLIC.match(c)) for c in letters)
    return "ru" if cyrillic / len(letters) >= 0.5 else "en"
