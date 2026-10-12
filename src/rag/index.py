import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "sparse"


def point_id(chunk_id: str) -> str:
    """Qdrant needs an int or UUID id; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


class Index:
    """One collection: named vectors `dense` (cosine) + `sparse` (BM25, IDF applied by Qdrant)."""

    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist; an existing one is left as is.

        - `dense`: bi-encoder vectors, cosine distance (HNSW index).
        - `sparse`: BM25 term weights; Qdrant applies IDF at search time.
        - Keyword payload indexes on `doc_id` (sync/delete filters) and `acl` (permission filter).
        """
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("doc_id", "acl"):
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def drop(self) -> None:
        self.client.delete_collection(self.collection)

    def upsert(self, payloads: list[dict], dense: list[list[float]], sparse: list[models.SparseVector]) -> None:
        """Insert or overwrite one point per chunk; the three lists are aligned by position.

        The point id is derived from `payload["chunk_id"]`, so re-upserting a chunk replaces it.
        `wait=True` returns only once the points are searchable.
        """
        points = [
            models.PointStruct(id=point_id(p["chunk_id"]), vector={DENSE: d, SPARSE: s}, payload=p)
            for p, d, s in zip(payloads, dense, sparse)
        ]
        self.client.upsert(self.collection, points, wait=True)

    def delete_doc(self, doc_id: str, keep_chunk_ids: Iterable[str] = ()) -> None:
        """Delete a doc's points, except the given chunk ids (used after re-upserting a changed doc)."""
        keep = [point_id(c) for c in keep_chunk_ids]
        flt = _doc_filter(doc_id)
        if keep:
            flt.must_not = [models.HasIdCondition(has_id=keep)]
        self.client.delete(self.collection, models.FilterSelector(filter=flt), wait=True)

    def set_doc_payload(self, doc_id: str, payload: dict) -> None:
        """Update payload without touching the vector"""
        self.client.set_payload(self.collection, payload, points=_doc_filter(doc_id), wait=True)

    def doc_hashes(self) -> dict[str, tuple[str, str]]:
        """doc_id -> (content_hash, meta_hash), read back from the stored points."""
        out, offset = {}, None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset,
                with_payload=["doc_id", "content_hash", "meta_hash"], with_vectors=False,
            )
            for p in points:
                out[p.payload["doc_id"]] = (p.payload["content_hash"], p.payload["meta_hash"])
            if offset is None:
                return out

    def count(self, doc_id: str | None = None) -> int:
        flt = _doc_filter(doc_id) if doc_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count
