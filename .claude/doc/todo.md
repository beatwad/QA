# Future work

Known limitations accepted for now, to revisit later.

## Conversation memory is in-process only

`src/rag/serving/conversations.py` keeps conversation history in a Python dict inside the API process.

- A restart (deploy, crash) loses every conversation: the next follow-up is treated as a first question.
- It doesn't work with more than one API worker or replica: each process has its own memory, so a follow-up routed to a different worker has no history.
- Nothing expires: memory grows with every conversation until the process restarts.

Options: Redis (TTL per conversation, shared across workers) or a database table if history should be kept for analytics and the training flywheel (design.md → "Query logs").
