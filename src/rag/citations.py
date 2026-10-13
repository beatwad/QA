"""Turns the LLM's [n] markers into real sources the user can click."""

import re
from dataclasses import dataclass

from rag.retrieval import Hit

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1]  or  [1, 2]
SNIPPET_CHARS = 200


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    url: str
    snippet: str


def extract_citations(answer: str, hits: list[Hit]) -> tuple[str, list[Citation]]:
    """Map [n] markers to the prompt's chunks (1-based).

    Markers pointing outside 1..len(hits) are removed from the text. Citations are listed
    once each, in order of first appearance.
    """
    cited: dict[int, Citation] = {}

    def keep_valid(m: re.Match) -> str:
        valid = [n for n in (int(x) for x in m.group(1).split(",")) if 1 <= n <= len(hits)]
        for n in valid:
            if n not in cited:
                p = hits[n - 1].payload
                cited[n] = Citation(n=n, chunk_id=p["chunk_id"], doc_id=p["doc_id"], title=p["title"],
                                    url=p["url"], snippet=p["text"][:SNIPPET_CHARS])
        return "".join(f"[{n}]" for n in valid)

    text = _MARKER.sub(keep_valid, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # " ." left where a marker was removed
    return text, list(cited.values())
