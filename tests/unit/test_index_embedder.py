import pytest
from qdrant_client import QdrantClient

from rag.storage.index import Index


def test_collection_remembers_its_embedder_and_refuses_another():
    index = Index(QdrantClient(":memory:"), "chunks", 4)
    index.ensure_collection("BAAI/bge-m3")
    index.require_embedder("BAAI/bge-m3")
    index.ensure_collection("BAAI/bge-m3")  # existing collection, same model: fine
    with pytest.raises(RuntimeError, match="rag ingest --reset"):
        index.require_embedder("models/bge-m3-infonce")
    with pytest.raises(RuntimeError):
        index.ensure_collection("models/bge-m3-infonce")  # never mix vectors from two models
