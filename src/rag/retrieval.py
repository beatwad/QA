"""Hybrid retrieval: dense + sparse search, both ACL-filtered inside Qdrant, fused with RRF."""

from dataclasses import dataclass, field

from qdrant_client import models

from rag.encoders import DenseEncoder, SparseEncoder
from rag.index import DENSE, SPARSE, Index


@dataclass
class Hit:
    chunk_id: str
    payload: dict
    rrf_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # branch -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)  # branch -> raw score (cosine / BM25)


def rrf(ranked: dict[str, list[str]], k: int, weights: dict[str, float]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists l of w_l / (k + rank_l(d)), rank from 1.

    Uses ranks, not raw scores, because cosine and BM25 aren't on comparable scales.
    Returns (id, score) best first; ties broken by id for determinism.
    """
    scores: dict[str, float] = {}
    for branch, ids in ranked.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weights[branch] / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def acl_filter(groups: list[str]) -> models.Filter:
    """Chunk visible iff its `acl` shares a group with the user's groups.
    A chunk with a missing or empty `acl` never matches (fail closed)."""
    return models.Filter(must=[models.FieldCondition(key="acl", match=models.MatchAny(any=groups))])


class Retriever:
    def __init__(self, index: Index, dense: DenseEncoder, sparse: SparseEncoder,
                 limit: int, rrf_k: int, weights: dict[str, float]):
        self.index = index
        self.dense = dense
        self.sparse = sparse
        self.limit = limit
        self.rrf_k = rrf_k
        self.weights = weights

    def search_branch(self, branch: str, query: str, groups: list[str]) -> list[Hit]:
        """One branch on its own, best first. 
        A chunk is visible to a user only if its acl shares at least one group with that list.
        No groups -> nothing is visible."""
        if not groups:
            return []
        vector = self.dense.encode_query(query) if branch == DENSE else self.sparse.encode_query(query)
        points = self.index.client.query_points(
            self.index.collection, 
            query=vector, 
            using=branch, 
            query_filter=acl_filter(groups),
            limit=self.limit, 
            with_payload=True,
        ).points
        return [Hit(chunk_id=p.payload["chunk_id"], payload=p.payload, ranks={branch: rank}, scores={branch: p.score})
                for rank, p in enumerate(points, start=1)]

    def search(self, query: str, groups: list[str]) -> list[Hit]:
        """Both branches fused with RRF -> top `limit` candidates for the reranker."""
        by_id: dict[str, Hit] = {}
        ranked = {}
        for branch in (DENSE, SPARSE):
            hits = self.search_branch(branch, query, groups)
            ranked[branch] = [h.chunk_id for h in hits]
            for h in hits:
                merged = by_id.setdefault(h.chunk_id, Hit(chunk_id=h.chunk_id, payload=h.payload))
                merged.ranks.update(h.ranks)
                merged.scores.update(h.scores)
        fused = []
        for chunk_id, score in rrf(ranked, self.rrf_k, self.weights)[: self.limit]:
            by_id[chunk_id].rrf_score = score
            fused.append(by_id[chunk_id])
        return fused
