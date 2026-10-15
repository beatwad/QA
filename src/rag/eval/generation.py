"""Generation subset for an LLM judge.

Runs the serving pipeline on the golden questions tagged `generation` and saves, per question, what a
judge needs: the reference answers, the exact context the LLM saw, and the answer with its citations.
The judge then scores each answered row:
  faithful   every claim in the answer is supported by the passages it cites
  correct    the answer agrees with the reference answers
  citations  each [n] points to a passage that contains the claim it's attached to
"""

from collections.abc import Iterator

from rag.generation.llm import LLM
from rag.serving.pipeline import Pipeline


class RecordingLLM:
    """Passes calls through and keeps the last prompt, so the dump shows exactly what the LLM saw."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.messages: list[dict] | None = None

    def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        self.messages = messages
        return self.llm.stream(messages, max_tokens, temperature)


def run_generation(pipeline: Pipeline, llm: RecordingLLM, golden: list[dict],
                   users: dict[str, list[str]]) -> list[dict]:
    """`llm` must be the pipeline's LLM. Each question gets a new conversation, so there is no condensation."""
    rows = []
    for g in (g for g in golden if "generation" in g["tags"]):
        llm.messages = None
        result = dict(pipeline.ask(g["user"], users[g["user"]], g["question"]))["result"]
        rows.append({
            "id": g["id"], "lang": g["lang"], "question": g["question"], "answers": g["answers"],
            "gold_paragraph": g["gold_paragraph"], "status": result["status"], "answer": result["answer"],
            "citations": [{"n": c["n"], "chunk_id": c["chunk_id"]} for c in result["citations"]],
            "prompt": llm.messages[-1]["content"] if llm.messages else None,  # None: the gate blocked it
        })
    return rows
