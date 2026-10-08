from datetime import date

from rag.chunking import chunk_document
from rag.documents import Document, Paragraph


def words(text: str) -> int:
    return len(text.split())


def make_doc(*paragraphs: str, acl=("research",)) -> Document:
    return Document(
        doc_id="d-en", title="T", lang="en", source="wiki", url="u", acl=list(acl),
        updated_at=date(2025, 1, 1), authority="official", status="active",
        paragraphs=[Paragraph(id=f"p{i}", text=t) for i, t in enumerate(paragraphs)],
    )


def sentence(i: int, n: int = 5) -> str:
    return " ".join([f"s{i}"] * (n - 1)) + " end."


LONG = " ".join(sentence(i) for i in range(10))  # 50 words, 10 sentences


def test_short_paragraphs_are_one_chunk_each():
    chunks = chunk_document(make_doc("one two three.", "four five."), words, max_tokens=20, overlap=0.15)
    assert [c.text for c in chunks] == ["one two three.", "four five."]
    assert [c.paragraph_ids for c in chunks] == [["p0"], ["p1"]]
    assert [c.position for c in chunks] == [0, 1]


def test_long_paragraph_respects_limit():
    chunks = chunk_document(make_doc(LONG), words, max_tokens=20, overlap=0.3)
    assert len(chunks) > 1
    assert all(words(c.text) <= 20 for c in chunks)
    assert all(c.paragraph_ids == ["p0"] for c in chunks)


def test_long_paragraph_windows_overlap():
    chunks = chunk_document(make_doc(LONG), words, max_tokens=20, overlap=0.3)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.split("end.")[-2].strip()
        assert nxt.text.startswith(last_sentence)


def test_every_sentence_is_covered():
    chunks = chunk_document(make_doc(LONG), words, max_tokens=12, overlap=0.15)
    joined = " ".join(c.text for c in chunks)
    assert all(sentence(i) in joined for i in range(10))


def test_overlong_sentence_is_split_by_words():
    run_on = " ".join(f"w{i}" for i in range(50)) + "."
    chunks = chunk_document(make_doc(run_on), words, max_tokens=20, overlap=0.15)
    assert all(words(c.text) <= 20 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == run_on.split()


def test_chunks_inherit_acl_and_detect_lang():
    chunks = chunk_document(make_doc("Hello there.", "Привет всем."), words, max_tokens=20, overlap=0.15)
    assert all(c.acl == ["research"] for c in chunks)
    assert [c.lang for c in chunks] == ["en", "ru"]


def test_empty_acl_stays_empty():
    chunks = chunk_document(make_doc("Hello there.", acl=()), words, max_tokens=20, overlap=0.15)
    assert chunks[0].acl == []
