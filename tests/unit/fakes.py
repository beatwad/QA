import copy

from rag.search.retrieval import Hit


class FakeLLM:
    """Returns canned replies (streamed in small pieces) and records the messages it got.
    `reply` is one string for every call, or a list consumed one per call."""

    def __init__(self, reply: str | list[str] = "", error: Exception | None = None):
        self.replies = list(reply) if isinstance(reply, list) else None
        self.reply, self.error, self.calls = reply, error, []

    def stream(self, messages, max_tokens, temperature):
        self.calls.append(messages)
        if self.error:
            raise self.error
        reply = self.replies.pop(0) if self.replies is not None else self.reply
        for i in range(0, len(reply), 4):
            yield reply[i : i + 4]


def words(text: str) -> int:
    return len(text.split())


def chunk_hit(n: int, text: str | None = None) -> Hit:
    return Hit(chunk_id=f"d{n}-c000", payload={
        "chunk_id": f"d{n}-c000", "doc_id": f"d{n}", "title": f"Title {n}", "url": f"https://wiki/d{n}",
        "source": "wiki", "updated_at": "2025-01-01", "text": text or f"Text of chunk {n}."})


def live_hit(n: int, text: str | None = None) -> Hit:
    """A retrieved chunk with everything post-ranking needs (active, official, own axis vector)."""
    h = chunk_hit(n, text)
    h.payload.update({"status": "active", "superseded_by": None, "authority": "official", "source": "wiki"})
    h.vector = [1.0 if i == n else 0.0 for i in range(10)]
    return h


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, groups):
        self.queries.append(query)
        return copy.deepcopy(self.hits)  # the pipeline sets scores on hits; keep the originals clean


class FakeReranker:
    """Same score for every chunk."""

    def __init__(self, score: float):
        self.value = score

    def score(self, query, texts):
        return [self.value] * len(texts)
