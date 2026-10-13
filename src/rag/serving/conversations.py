"""In-memory conversation history. Lost on restart and not shared between workers
(see .claude/doc/todo.md)."""

from collections import defaultdict


class ConversationStore:
    def __init__(self):
        # keyed by (user, conversation_id): one user can never read another user's history,
        # which may quote documents the other user isn't allowed to see
        self._messages: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def history(self, user: str, conversation_id: str) -> list[dict]:
        return list(self._messages.get((user, conversation_id), []))

    def append(self, user: str, conversation_id: str, question: str, answer: str) -> None:
        self._messages[(user, conversation_id)] += [
            {"role": "user", "content": question}, {"role": "assistant", "content": answer}]
