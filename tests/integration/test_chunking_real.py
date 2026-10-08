from pathlib import Path

import pytest

from rag.chunking import chunk_document, make_token_counter
from rag.documents import load_corpus

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def test_corpus_chunks_fit_embedder_limit(config):
    count = make_token_counter(config.models.embedder)
    for doc in load_corpus(CORPUS):
        chunks = chunk_document(doc, count, config.chunking.max_tokens, config.chunking.overlap)
        assert all(count(c.text) <= config.chunking.max_tokens for c in chunks), doc.doc_id
        assert {p.id for p in doc.paragraphs} == {pid for c in chunks for pid in c.paragraph_ids}
        assert all(c.lang == doc.lang for c in chunks), doc.doc_id
