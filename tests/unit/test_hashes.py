from datetime import date

from rag.config import ChunkingConfig
from rag.indexing.documents import Document, Paragraph
from rag.indexing.ingest import content_hash, meta_hash

CHUNKING = ChunkingConfig(max_tokens=300, overlap=0.15)


def doc(**overrides) -> Document:
    fields = dict(doc_id="d-en", title="T", lang="en", source="wiki", url="u", acl=["all-employees"],
                  updated_at=date(2025, 1, 1), authority="official", status="active",
                  paragraphs=[Paragraph(id="p0", text="Hello.")])
    return Document(**{**fields, **overrides})


def test_text_change_changes_content_hash_only():
    edited = doc(paragraphs=[Paragraph(id="p0", text="Hello!")])
    assert content_hash(edited, "m", CHUNKING) != content_hash(doc(), "m", CHUNKING)
    assert meta_hash(edited) == meta_hash(doc())


def test_acl_change_changes_meta_hash_only():
    restricted = doc(acl=["hr-team"])
    assert content_hash(restricted, "m", CHUNKING) == content_hash(doc(), "m", CHUNKING)
    assert meta_hash(restricted) != meta_hash(doc())


def test_embedder_or_chunking_change_changes_content_hash():
    base = content_hash(doc(), "m", CHUNKING)
    assert content_hash(doc(), "other-model", CHUNKING) != base
    assert content_hash(doc(), "m", ChunkingConfig(max_tokens=200, overlap=0.15)) != base
