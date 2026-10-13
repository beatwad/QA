from rag.retrieval import Hit


class FakeLLM:
    """Returns a canned reply (streamed in small pieces) and records the messages it got."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def stream(self, messages, max_tokens, temperature):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for i in range(0, len(self.reply), 4):
            yield self.reply[i : i + 4]


def words(text: str) -> int:
    return len(text.split())


def chunk_hit(n: int, text: str | None = None) -> Hit:
    return Hit(chunk_id=f"d{n}-c000", payload={
        "chunk_id": f"d{n}-c000", "doc_id": f"d{n}", "title": f"Title {n}", "url": f"https://wiki/d{n}",
        "source": "wiki", "updated_at": "2025-01-01", "text": text or f"Text of chunk {n}."})
