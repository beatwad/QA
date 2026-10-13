import uuid
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from rag.indexing.chunking import make_token_counter
from rag.config import load_config
from rag.indexing.documents import load_corpus
from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
from rag.storage.index import Index
from rag.indexing.ingest import DocumentUpdateService
from rag.search.rerank import CrossEncoderReranker
from rag.search.retrieval import Retriever


@pytest.fixture(scope="session")
def config():
    return load_config()


def _reachable(url: str) -> bool:
    try:
        httpx.get(url, timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config, items):
    llm_items = [i for i in items if "llm" in i.keywords]
    if llm_items and not _reachable(load_config().llm.base_url + "/models"):
        skip = pytest.mark.skip(reason="vLLM not reachable")
        for item in llm_items:
            item.add_marker(skip)


# Real models and a full-corpus index, shared by integration and e2e tests (created only when requested)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


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
