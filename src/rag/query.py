"""Condensates multi-turn conversation: (recent turns + follow-up) to one standalone 
question for retrieval. Most of the follow-up questions demand context for retrieval's correct work,
this function provides the necessary context."""

import logging

from rag.llm import LLM, complete

log = logging.getLogger(__name__)

CONDENSE_SYSTEM = """Rewrite the user's last question as a standalone question that can be understood
without the conversation. Resolve pronouns and references using the conversation.
Keep the language of the last question. Output only the rewritten question."""


def condense(llm: LLM, history: list[dict], question: str) -> str:
    """No history -> the question as is. LLM error or empty output -> the raw question (degraded recall)."""
    if not history:
        return question
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}"},
    ]
    try:
        rewritten = complete(llm, messages, max_tokens=128).strip()
    except Exception:
        log.warning("condensation failed, using the raw question", exc_info=True)
        return question
    return rewritten or question
