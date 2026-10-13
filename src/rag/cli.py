from pathlib import Path

import typer

from rag.config import load_config

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """Ops commands for the RAG bot."""


@app.command()
def ingest(corpus: Path = Path("data/corpus"), reset: bool = False):
    """Sync the corpus into Qdrant (only changed docs are re-embedded)."""
    from qdrant_client import QdrantClient

    from rag.indexing.chunking import make_token_counter
    from rag.indexing.documents import load_corpus
    from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
    from rag.storage.index import Index
    from rag.indexing.ingest import DocumentUpdateService

    cfg = load_config()
    dense = BgeM3Encoder(cfg.models.embedder)
    index = Index(QdrantClient(cfg.qdrant.url), cfg.qdrant.collection, dense.dim)
    if reset:
        index.drop()
    service = DocumentUpdateService(index, dense, Bm25Encoder(), make_token_counter(cfg.models.embedder), cfg.chunking)
    report = service.sync(load_corpus(corpus))
    for name, ids in vars(report).items():
        typer.echo(f"{name:10} {len(ids)}")
    typer.echo(f"points     {index.count()}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080, corpus: Path = Path("data/corpus"),
          users: Path = Path("data/users.yaml")):
    """Run the API (models load at startup)."""
    import uvicorn
    from qdrant_client import QdrantClient

    from rag.serving.api import create_app
    from rag.serving.auth import Users
    from rag.indexing.chunking import make_token_counter
    from rag.serving.conversations import ConversationStore
    from rag.indexing.documents import load_corpus
    from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
    from rag.storage.index import Index
    from rag.indexing.ingest import DocumentUpdateService
    from rag.generation.llm import OpenAICompatLLM
    from rag.serving.pipeline import Pipeline
    from rag.search.rerank import CrossEncoderReranker
    from rag.search.retrieval import Retriever

    _setup_logging()
    cfg = load_config()
    dense, sparse = BgeM3Encoder(cfg.models.embedder), Bm25Encoder()
    index = Index(QdrantClient(cfg.qdrant.url), cfg.qdrant.collection, dense.dim)
    service = DocumentUpdateService(index, dense, sparse, make_token_counter(cfg.models.embedder), cfg.chunking)
    r = cfg.retrieval
    pipeline = Pipeline(cfg, Retriever(index, dense, sparse, r.candidates, r.rrf_k, r.rrf_weights),
                        CrossEncoderReranker(cfg.models.reranker), OpenAICompatLLM(cfg.llm.base_url, cfg.llm.model),
                        make_token_counter(cfg.llm.tokenizer), ConversationStore())
    app_ = create_app(pipeline, Users.load(users), lambda: service.sync(load_corpus(corpus)))
    uvicorn.run(app_, host=host, port=port, log_config=None)


def _setup_logging(level: str = "INFO") -> None:
    """Log through loguru at `level`, and route libraries that use stdlib logging (uvicorn, httpx, ...)
    into it, so everything comes out in one format."""
    import inspect
    import logging
    import sys

    from loguru import logger

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                lvl: str | int = logger.level(record.levelname).name
            except ValueError:
                lvl = record.levelno
            # walk out of the logging module so loguru reports the caller's file and line
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame, depth = frame.f_back, depth + 1
            logger.opt(depth=depth, exception=record.exc_info).log(lvl, record.getMessage())

    logger.remove()
    # diagnose=False: tracebacks don't print local variable values (queries, user groups) into logs
    logger.add(sys.stderr, level=level, diagnose=False)
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)
    # one INFO line per HTTP call (Hugging Face, Qdrant, vLLM) drowns everything else
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
