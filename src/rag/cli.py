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


@app.command("eval")
def eval_(limit: int = typer.Option(None, help="answerable questions per language (default: all)"),
          out: Path = typer.Option(None, help="also write the numbers to this JSON file"),
          golden: Path = Path("data/golden.jsonl"), users: Path = Path("data/users.yaml")):
    """Offline metrics: Recall@5/10, MRR, NDCG@10 per stage and language pair, plus gate calibration."""
    import json
    from datetime import date

    import yaml
    from qdrant_client import QdrantClient

    from rag.eval.runner import Evaluator, format_report, sample
    from rag.search.rerank import CrossEncoderReranker
    from rag.search.retrieval import Retriever
    from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
    from rag.storage.index import Index

    _setup_logging()
    cfg = load_config()
    dense = BgeM3Encoder(cfg.models.embedder)
    r = cfg.retrieval
    retriever = Retriever(Index(QdrantClient(cfg.qdrant.url), cfg.qdrant.collection, dense.dim), dense, Bm25Encoder(),
                          r.candidates, r.rrf_k, r.rrf_weights)
    retriever.index.require_embedder(dense.name)
    evaluator = Evaluator(cfg, retriever, CrossEncoderReranker(cfg.models.reranker),
                          yaml.safe_load(users.read_text()), date.today())
    questions = sample([json.loads(line) for line in golden.read_text().splitlines()], limit)
    retrieval, gate_scores = evaluator.retrieval(questions)
    gate = evaluator.gate(questions, gate_scores)
    typer.echo(format_report(retrieval, gate, cfg.rerank.gate_threshold))
    if out:
        out.write_text(json.dumps({"embedder": cfg.models.embedder, "retrieval": retrieval, "gate": gate}, indent=1))


@app.command("eval-generation")
def eval_generation(out: Path = Path("data/eval/generation.jsonl"), golden: Path = Path("data/golden.jsonl"),
                    users: Path = Path("data/users.yaml")):
    """Answer the golden `generation` subset with the real pipeline and dump it for an LLM judge."""
    import json

    import yaml
    from qdrant_client import QdrantClient

    from rag.eval.generation import RecordingLLM, run_generation
    from rag.generation.llm import OpenAICompatLLM
    from rag.indexing.chunking import make_token_counter
    from rag.search.rerank import CrossEncoderReranker
    from rag.search.retrieval import Retriever
    from rag.serving.conversations import ConversationStore
    from rag.serving.pipeline import Pipeline
    from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
    from rag.storage.index import Index

    _setup_logging()
    cfg = load_config()
    dense = BgeM3Encoder(cfg.models.embedder)
    r = cfg.retrieval
    retriever = Retriever(Index(QdrantClient(cfg.qdrant.url), cfg.qdrant.collection, dense.dim), dense, Bm25Encoder(),
                          r.candidates, r.rrf_k, r.rrf_weights)
    retriever.index.require_embedder(dense.name)
    llm = RecordingLLM(OpenAICompatLLM(cfg.llm.base_url, cfg.llm.model))
    pipeline = Pipeline(cfg, retriever, CrossEncoderReranker(cfg.models.reranker), llm,
                        make_token_counter(cfg.llm.tokenizer), ConversationStore())
    rows = run_generation(pipeline, llm, [json.loads(line) for line in golden.read_text().splitlines()],
                          yaml.safe_load(users.read_text()))
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    statuses = [row["status"] for row in rows]
    typer.echo(f"{len(rows)} questions -> {out}: "
               + ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses))))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080, corpus: Path = Path("data/corpus"),
          users: Path = Path("data/users.yaml")):
    """Run the API (models load at startup)."""
    import uvicorn
    import yaml
    from qdrant_client import QdrantClient

    from rag.serving.api import create_app
    from rag.serving.auth import Users
    from rag.indexing.chunking import make_token_counter
    from rag.serving.conversations import ConversationStore
    from rag.indexing.documents import load_corpus
    from rag.storage.encoders import BgeM3Encoder, Bm25Encoder
    from rag.storage.index import Index
    from rag.indexing.ingest import DocumentUpdateService
    from rag.indexing.importer import Importer, fetch_page
    from rag.generation.llm import OpenAICompatLLM
    from rag.serving.pipeline import Pipeline
    from rag.search.rerank import CrossEncoderReranker
    from rag.search.retrieval import Retriever

    _setup_logging()
    cfg = load_config()
    dense, sparse = BgeM3Encoder(cfg.models.embedder), Bm25Encoder()
    index = Index(QdrantClient(cfg.qdrant.url), cfg.qdrant.collection, dense.dim)
    index.require_embedder(dense.name)  # deploy block: refuse to serve queries against another model's vectors
    service = DocumentUpdateService(index, dense, sparse, make_token_counter(cfg.models.embedder), cfg.chunking)
    r = cfg.retrieval
    pipeline = Pipeline(cfg, Retriever(index, dense, sparse, r.candidates, r.rrf_k, r.rrf_weights),
                        CrossEncoderReranker(cfg.models.reranker), OpenAICompatLLM(cfg.llm.base_url, cfg.llm.model),
                        make_token_counter(cfg.llm.tokenizer), ConversationStore())
    def sync():
        return service.sync(load_corpus(corpus))

    def near_duplicates(paragraphs: list[str], doc_id: str) -> list[bool]:
        vectors = dense.encode_documents(paragraphs)
        return [index.nearest_score(v, doc_id) >= cfg.postrank.dup_threshold for v in vectors]

    groups = sorted({g for gs in yaml.safe_load(users.read_text()).values() for g in gs})
    importer = Importer(corpus, fetch_page, near_duplicates, sync, groups,
                        sorted(cfg.postrank.half_life_days), sorted(cfg.postrank.authority_prior))
    app_ = create_app(pipeline, Users.load(users), sync, importer)
    uvicorn.run(app_, host=host, port=port, log_config=None)


@app.command()
def train():
    """Fine-tune the dense bi-encoder with InfoNCE (config `train`). To compare with the base model:
    set models.embedder to the printed directory, `rag ingest --reset`, then `rag eval` vs data/eval/baseline.json."""
    from rag.train.train import train as run

    _setup_logging()
    typer.echo(run(load_config().train))


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
