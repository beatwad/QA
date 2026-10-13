import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag.chunking import make_token_counter
from rag.documents import load_corpus
from rag.encoders import BgeM3Encoder, Bm25Encoder
from rag.index import Index
from rag.ingest import DocumentUpdateService
from rag.rerank import CrossEncoderReranker
from rag.retrieval import Retriever

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


@pytest.fixture(scope="session")
def bge(config):
    return BgeM3Encoder(config.models.embedder)


@pytest.fixture(scope="session")
def bm25():
    return Bm25Encoder()


@pytest.fixture(scope="session")
def count_tokens(config):
    return make_token_counter(config.models.embedder)


@pytest.fixture(scope="session")
def reranker(config):
    return CrossEncoderReranker(config.models.reranker)


@pytest.fixture(scope="session")
def retriever(config, bge, bm25, count_tokens):
    """The full corpus, indexed once per test session into a throwaway collection."""
    index = Index(QdrantClient(config.qdrant.url), f"test_{uuid.uuid4().hex[:8]}", bge.dim)
    DocumentUpdateService(index, bge, bm25, count_tokens, config.chunking).sync(load_corpus(CORPUS))
    r = config.retrieval
    yield Retriever(index, bge, bm25, r.candidates, r.rrf_k, r.rrf_weights)
    index.drop()
