from typing import Protocol

from rag.retrieval import Hit


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Query + chunk in one sequence -> relevance logit (raw, not squashed)."""

    def __init__(self, name: str = "BAAI/bge-reranker-v2-m3", batch_size: int = 32):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, activation_fn=torch.nn.Identity(), model_kwargs={"torch_dtype": "float16"})
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return self.model.predict([(query, t) for t in texts], batch_size=self.batch_size).tolist()


def rerank(reranker: Reranker, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Score every candidate with the cross-encoder, keep the best top_k."""
    for hit, s in zip(hits, reranker.score(query, [h.payload["text"] for h in hits])):
        hit.ce_score = s
    return sorted(hits, key=lambda h: -h.ce_score)[:top_k]


def confident(hits: list[Hit], threshold: float) -> bool:
    """Confidence gate on the RAW cross-encoder score of the best remaining chunk.
    Below threshold -> answer 'not found' without calling the LLM."""
    return bool(hits) and max(h.ce_score for h in hits) >= threshold
