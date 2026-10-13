# Future work

Known limitations accepted for now, to revisit later.

## Conversation memory is in-process only

`src/rag/serving/conversations.py` keeps conversation history in a Python dict inside the API process.

- A restart (deploy, crash) loses every conversation: the next follow-up is treated as a first question.
- It doesn't work with more than one API worker or replica: each process has its own memory, so a follow-up routed to a different worker has no history.
- Nothing expires: memory grows with every conversation until the process restarts.

Options: Redis (TTL per conversation, shared across workers) or a database table if history should be kept for analytics and the training flywheel (design.md → "Query logs").

## Replace `logging` with loguru

Logging uses the standard library: `logging.getLogger(__name__)` in `search/retrieval.py`, `generation/query.py` and `serving/pipeline.py`, configured once with `logging.basicConfig` in `cli.py` (`rag serve`).

Switch to [loguru](https://github.com/Delgan/loguru):

- `from loguru import logger` replaces the per-module `log = logging.getLogger(__name__)`.
- `log.warning(..., exc_info=True)` becomes `logger.opt(exception=True).warning(...)` (or `logger.exception(...)` where the level can be ERROR).
- The per-request line in `serving/pipeline.py` is built with `json.dumps`; loguru's `logger.bind(...)` plus `serialize=True` on the sink can produce structured JSON logs instead.
- uvicorn and other libraries still log through `logging`: route them into loguru with an `InterceptHandler` so all output has one format.
- Tests that check log output (none yet) would use loguru's `caplog` integration recipe, since pytest's `caplog` only sees `logging`.
