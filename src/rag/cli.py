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
