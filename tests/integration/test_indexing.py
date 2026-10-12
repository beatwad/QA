import uuid
from datetime import date
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag.chunking import chunk_document, make_token_counter
from rag.documents import Document, Paragraph, load_corpus
from rag.encoders import BgeM3Encoder, Bm25Encoder
from rag.index import SPARSE, Index
from rag.ingest import DocumentUpdateService

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
SLICE = ["art00-en", "art00-ru", "art03-en", "art03-ru"]


class CountingEncoder:
    """Wraps the real dense encoder and counts how many texts were embedded."""

    def __init__(self, inner):
        self.inner, self.name, self.dim, self.embedded = inner, inner.name, inner.dim, 0

    def encode_documents(self, texts):
        self.embedded += len(texts)
        return self.inner.encode_documents(texts)

    def encode_query(self, text):
        return self.inner.encode_query(text)


@pytest.fixture(scope="session")
def bge(config):
    return BgeM3Encoder(config.models.embedder)


@pytest.fixture(scope="session")
def bm25():
    return Bm25Encoder()


@pytest.fixture(scope="session")
def count_tokens(config):
    return make_token_counter(config.models.embedder)


@pytest.fixture
def index(config, bge):
    idx = Index(QdrantClient(config.qdrant.url), f"test_{uuid.uuid4().hex[:8]}", bge.dim)
    yield idx
    idx.drop()


@pytest.fixture
def dense(bge):
    return CountingEncoder(bge)


@pytest.fixture
def service(index, dense, bm25, count_tokens, config):
    return DocumentUpdateService(index, dense, bm25, count_tokens, config.chunking)


@pytest.fixture
def docs():
    by_id = {d.doc_id: d for d in load_corpus(CORPUS)}
    return [by_id[i] for i in SLICE]


def n_chunks(doc, count_tokens, config):
    return len(chunk_document(doc, count_tokens, config.chunking.max_tokens, config.chunking.overlap))


def payloads(index, doc_id):
    points, _ = index.client.scroll(index.collection, scroll_filter=None, limit=100, with_payload=True)
    return [p.payload for p in points if p.payload["doc_id"] == doc_id]


def test_ingest_creates_one_point_per_chunk(service, index, docs, count_tokens, config):
    report = service.sync(docs)
    assert sorted(report.added) == sorted(SLICE)
    assert index.count() == sum(n_chunks(d, count_tokens, config) for d in docs)
    p = payloads(index, "art00-ru")[0]
    assert p["acl"] and p["lang"] == "ru" and p["embedder"] == config.models.embedder


def test_resync_without_changes_embeds_nothing(service, dense, docs):
    service.sync(docs)
    dense.embedded = 0
    report = service.sync(docs)
    assert sorted(report.unchanged) == sorted(SLICE)
    assert dense.embedded == 0


def test_edit_reembeds_only_that_doc(service, index, dense, docs, count_tokens, config):
    service.sync(docs)
    dense.embedded = 0
    first = docs[0].paragraphs[0]
    edited = docs[0].model_copy(update={"paragraphs": [Paragraph(id=first.id, text=first.text + " EDITED.")]})
    report = service.sync([edited, *docs[1:]])
    assert report.changed == [edited.doc_id]
    assert dense.embedded == n_chunks(edited, count_tokens, config)
    # dropped paragraphs leave no stale chunks behind
    assert index.count(edited.doc_id) == n_chunks(edited, count_tokens, config)
    assert payloads(index, edited.doc_id)[0]["text"].endswith("EDITED.")


def test_acl_only_change_updates_payload_without_reembedding(service, index, dense, docs):
    service.sync(docs)
    dense.embedded = 0
    restricted = docs[0].model_copy(update={"acl": ["hr-team"]})
    report = service.sync([restricted, *docs[1:]])
    assert report.meta_only == [restricted.doc_id]
    assert dense.embedded == 0
    assert all(p["acl"] == ["hr-team"] for p in payloads(index, restricted.doc_id))


def test_deleted_doc_loses_its_chunks(service, index, docs):
    service.sync(docs)
    report = service.sync(docs[1:])
    assert report.deleted == [docs[0].doc_id]
    assert index.count(docs[0].doc_id) == 0


def test_embedder_change_reembeds_everything(service, dense, docs):
    service.sync(docs)
    dense.name = "some-other-model"
    report = service.sync(docs)
    assert sorted(report.changed) == sorted(SLICE)


def _doc(doc_id, text):
    return Document(doc_id=doc_id, title="T", lang="ru", source="wiki", url="u", acl=["all-employees"],
                    updated_at=date(2025, 1, 1), authority="official", status="active",
                    paragraphs=[Paragraph(id=f"{doc_id}-p0", text=text)])


@pytest.mark.parametrize("query,expected", [
    ("условия договора", "contract-ru"),  # "договора" -> stem of "Договор"
    ("signed contracts", "contract-en"),  # "contracts" -> stem of "contract"
])
def test_bm25_matches_word_forms(service, index, bm25, query, expected):
    service.sync([
        _doc("contract-ru", "Договор подписан в мае 2020 года."),
        _doc("report-ru", "Отчёт о продажах за первый квартал."),
        _doc("contract-en", "The contract was signed in May 2020."),
        _doc("report-en", "Sales report for the first quarter."),
    ])
    hits = index.client.query_points(index.collection, query=bm25.encode_query(query), using=SPARSE, limit=4).points
    assert [h.payload["doc_id"] for h in hits] == [expected]
