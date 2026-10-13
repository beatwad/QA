import pytest

from rag.chunking import make_token_counter
from rag.encoders import BgeM3Encoder, Bm25Encoder


@pytest.fixture(scope="session")
def bge(config):
    return BgeM3Encoder(config.models.embedder)


@pytest.fixture(scope="session")
def bm25():
    return Bm25Encoder()


@pytest.fixture(scope="session")
def count_tokens(config):
    return make_token_counter(config.models.embedder)
