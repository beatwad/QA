"""Prompt assembly: system rules + numbered chunks (strongest first) + recent turns + question,
kept under the LLM's context window."""

from rag.indexing.chunking import TokenCounter
from rag.search.retrieval import Hit

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer employees' questions using ONLY the numbered context passages.
Rules:
- If any passage contains the answer, give it, even if the passage is in another language than the question or repeats another passage.
- Write the answer as plain text and put the passage number after each claim as [n], e.g. [1] or [2][3]. Never reply with only [n] markers.
- Reply with exactly {NOT_FOUND} and nothing else only if NO passage contains the answer.
- Write the whole answer in the language of the question, even if the passages are in another language: translate names and terms from the passages instead of copying them.
- Be concise."""

# appended to the user message when the first answer came out in the wrong language
LANGUAGE_REMINDER = {"en": "Write the whole answer in English only.", "ru": "Напиши весь ответ только на русском языке."}

# chat-template tokens around each message (user doesn't see them)
# used to avoid token count under-estimation which can lead to rejection from vLLM
PER_MESSAGE_OVERHEAD = 8  


def format_chunk(n: int, hit: Hit) -> str:
    p = hit.payload
    return f"[{n}] {p['title']} ({p['source']}, updated {p['updated_at']})\n{p['text']}"


def build_messages(question: str, hits: list[Hit], history: list[dict], count: TokenCounter,
                   budget: int) -> tuple[list[dict], list[Hit]]:
    """Returns (messages, hits actually included). Hits are numbered [1..n] in the given order.

    Budget priority: system rules and question always; then chunks in rank order; then the most
    recent history turns. Whatever doesn't fit is left out.

    We also return list of hits to extract citation from them on the later stages.
    """
    used = count(SYSTEM) + count(question) + 3 * PER_MESSAGE_OVERHEAD

    # Add chunks to message until we hit the budget
    # Chunks have highest priority, so we add them first
    blocks, included = [], []
    for hit in hits:
        block = format_chunk(len(blocks) + 1, hit)
        cost = count(block) + 2  # blank line between passages
        if used + cost > budget:
            break
        blocks.append(block)
        included.append(hit)
        used += cost

    # Add history turns until we hit the budget
    turns: list[dict] = []
    for turn in reversed(history):
        cost = count(turn["content"]) + PER_MESSAGE_OVERHEAD
        if used + cost > budget:
            break
        turns.insert(0, turn)
        used += cost

    user = "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, *turns, {"role": "user", "content": user}], included


def is_not_found(answer: str) -> bool:
    return answer.strip().startswith(NOT_FOUND)
