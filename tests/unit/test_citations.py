from rag.generation.citations import extract_citations

from fakes import chunk_hit

HITS = [chunk_hit(1), chunk_hit(2), chunk_hit(3)]


def test_markers_map_to_chunk_sources():
    text, cites = extract_citations("Tesla died in 1943 [2]. He was born in Smiljan [1].", HITS)
    assert text == "Tesla died in 1943 [2]. He was born in Smiljan [1]."
    assert [(c.n, c.doc_id, c.title, c.url) for c in cites] == [
        (2, "d2", "Title 2", "https://wiki/d2"), (1, "d1", "Title 1", "https://wiki/d1")]
    assert cites[0].snippet == "Text of chunk 2."


def test_invalid_markers_are_dropped():
    text, cites = extract_citations("Fact one [7]. Fact two [0][3].", HITS)
    assert text == "Fact one. Fact two [3]."
    assert [c.n for c in cites] == [3]


def test_comma_lists_and_repeats():
    text, cites = extract_citations("A [1, 3, 9]. B [1].", HITS)
    assert text == "A [1][3]. B [1]."
    assert [c.n for c in cites] == [1, 3]


def test_no_markers():
    assert extract_citations("No citations here.", HITS) == ("No citations here.", [])
