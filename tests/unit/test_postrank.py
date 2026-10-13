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


def orthogonal(texts):
    """Fake embedder: every distinct text gets its own axis (similarity 0 between different texts)."""
    axes = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    return [[1.0 if axes[t] == i else 0.0 for i in range(len(axes))] for t in texts]


def ids(hits):
    return [h.chunk_id for h in hits]


def test_freshness_half_life():
    assert freshness(TODAY, TODAY, 30) == 1.0
    assert freshness(date(2026, 9, 1), TODAY, 30) == pytest.approx(0.5)
    assert freshness(date(2027, 1, 1), TODAY, 30) == 1.0  # future dates don't get a bonus above 1


def test_superseded_and_archived_are_excluded():
    hits = [hit("old", 9.0, status="archived", superseded_by="new"), hit("new", 5.0),
            hit("deprecated", 8.0, status="deprecated")]
    assert ids(postrank(hits, CFG, orthogonal, TODAY)) == ["new"]


def test_informal_copy_ranks_below_official():
    hits = [hit("slack", 5.2, source="slack", authority="informal", updated="2026-09-30"),
            hit("official", 5.0, source="confluence", authority="official", updated="2024-01-01")]
    assert ids(postrank(hits, CFG, orthogonal, TODAY)) == ["official", "slack"]


def test_freshness_reorders_near_ties():
    hits = [hit("stale", 5.0, updated="2020-01-01"), hit("fresh", 5.0, updated="2026-09-01")]
    assert ids(postrank(hits, CFG, orthogonal, TODAY)) == ["fresh", "stale"]


def test_business_signals_never_lift_off_topic_chunk():
    hits = [hit("off_topic", -3.0, updated="2026-09-30", authority="official"),
            hit("relevant", 3.0, updated="2015-01-01", source="slack", authority="informal")]
    assert ids(postrank(hits, CFG, orthogonal, TODAY)) == ["relevant", "off_topic"]


def test_per_doc_cap():
    hits = [hit(f"c{i}", 9.0 - i, doc_id="long-doc") for i in range(5)] + [hit("other", 1.0)]
    out = postrank(hits, CFG, orthogonal, TODAY)
    assert ids(out) == [f"c{i}" for i in range(CFG.max_per_doc)] + ["other"]


def test_exact_duplicate_is_dropped():
    hits = [hit("orig", 5.0, text="same text"), hit("copy", 4.9, doc_id="copy-doc", text="same text"),
            hit("other", 1.0)]
    assert ids(postrank(hits, CFG, orthogonal, TODAY)) == ["orig", "other"]


def test_near_duplicate_is_penalized():
    def embed(texts):
        vecs = {"a": [1.0, 0.0], "a2": [0.9, 0.436], "b": [0.0, 1.0]}  # sim(a, a2) = 0.9 < dup_threshold
        return [vecs[t] for t in texts]

    hits = [hit("a", 5.0), hit("a2", 5.0), hit("b", 4.0)]
    # b scores lower on its own, but a2 loses w_dup * 0.9 for repeating a
    assert ids(postrank(hits, CFG, embed, TODAY)) == ["a", "b", "a2"]


def test_final_k():
    hits = [hit(f"c{i}", 5.0 - i * 0.01) for i in range(CFG.final_k + 5)]
    assert len(postrank(hits, CFG, orthogonal, TODAY)) == CFG.final_k


def test_gate():
    assert not confident([], threshold=-2.0)
    assert not confident([hit("weak", -2.5)], threshold=-2.0)
    assert confident([hit("weak", -2.5), hit("strong", 3.0)], threshold=-2.0)
