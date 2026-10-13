from datetime import date

import pytest

from rag.config import load_config
from rag.postrank import freshness, postrank
from rag.rerank import confident
from rag.retrieval import Hit

TODAY = date(2026, 10, 1)
CFG = load_config().postrank


def hit(chunk_id, ce, doc_id=None, updated="2026-01-01", source="confluence", authority="official",
        status="active", superseded_by=None, text=None):
    return Hit(chunk_id=chunk_id, ce_score=ce, payload={
        "chunk_id": chunk_id, "doc_id": doc_id or chunk_id, "text": text or chunk_id, "updated_at": updated,
        "source": source, "authority": authority, "status": status, "superseded_by": superseded_by})


def orthogonal(hits):
    """Fake vectors: every distinct text gets its own axis (similarity 1 for equal texts, 0 otherwise)."""
    texts = [h.payload["text"] for h in hits]
    axes = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    for h, t in zip(hits, texts):
        h.vector = [1.0 if axes[t] == i else 0.0 for i in range(len(axes))]
    return hits


def ids(hits):
    return [h.chunk_id for h in hits]


def test_freshness_half_life():
    assert freshness(TODAY, TODAY, 30) == 1.0
    assert freshness(date(2026, 9, 1), TODAY, 30) == pytest.approx(0.5)
    assert freshness(date(2027, 1, 1), TODAY, 30) == 1.0  # future dates don't get a bonus above 1


def test_superseded_and_archived_are_excluded():
    hits = [hit("old", 9.0, status="archived", superseded_by="new"), hit("new", 5.0),
            hit("deprecated", 8.0, status="deprecated")]
    assert ids(postrank(orthogonal(hits), CFG, TODAY)) == ["new"]


def test_informal_copy_ranks_below_official():
    hits = [hit("slack", 5.2, source="slack", authority="informal", updated="2026-09-30"),
            hit("official", 5.0, source="confluence", authority="official", updated="2024-01-01")]
    assert ids(postrank(orthogonal(hits), CFG, TODAY)) == ["official", "slack"]


def test_freshness_reorders_near_ties():
    hits = [hit("stale", 5.0, updated="2020-01-01"), hit("fresh", 5.0, updated="2026-09-01")]
    assert ids(postrank(orthogonal(hits), CFG, TODAY)) == ["fresh", "stale"]


def test_business_signals_never_lift_off_topic_chunk():
    hits = [hit("off_topic", -3.0, updated="2026-09-30", authority="official"),
            hit("relevant", 3.0, updated="2015-01-01", source="slack", authority="informal")]
    assert ids(postrank(orthogonal(hits), CFG, TODAY)) == ["relevant", "off_topic"]


def test_per_doc_cap():
    hits = [hit(f"c{i}", 9.0 - i, doc_id="long-doc") for i in range(5)] + [hit("other", 1.0)]
    out = postrank(orthogonal(hits), CFG, TODAY)
    assert ids(out) == [f"c{i}" for i in range(CFG.max_per_doc)] + ["other"]


def test_exact_duplicate_is_dropped():
    hits = [hit("orig", 5.0, text="same text"), hit("copy", 4.9, doc_id="copy-doc", text="same text"),
            hit("other", 1.0)]
    assert ids(postrank(orthogonal(hits), CFG, TODAY)) == ["orig", "other"]


def test_near_duplicate_is_penalized():
    hits = [hit("a", 5.0), hit("a2", 5.0), hit("b", 4.0)]
    for h, v in zip(hits, ([1.0, 0.0], [0.9, 0.436], [0.0, 1.0])):  # sim(a, a2) = 0.9 < dup_threshold
        h.vector = v
    # b scores lower on its own, but a2 loses w_dup * 0.9 for repeating a
    assert ids(postrank(hits, CFG, TODAY)) == ["a", "b", "a2"]


def test_final_k():
    hits = [hit(f"c{i}", 5.0 - i * 0.01) for i in range(CFG.final_k + 5)]
    assert len(postrank(orthogonal(hits), CFG, TODAY)) == CFG.final_k


def test_gate():
    assert not confident([], threshold=-2.0)
    assert not confident([hit("weak", -2.5)], threshold=-2.0)
    assert confident([hit("weak", -2.5), hit("strong", 3.0)], threshold=-2.0)
