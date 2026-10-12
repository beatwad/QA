"""Document Update Service: diff the corpus against the index by hash and sync."""

import hashlib
import json
from dataclasses import dataclass, field

from rag.chunking import TokenCounter, chunk_document
from rag.config import ChunkingConfig
from rag.documents import Document
from rag.encoders import DenseEncoder, SparseEncoder
from rag.index import Index


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_hash(doc: Document, embedder: str, chunking: ChunkingConfig) -> str:
    """Changes when anything that affects the vectors changes: text, embedder or chunking."""
    paragraphs = [p.model_dump() for p in doc.paragraphs]
    return _sha([paragraphs, embedder, chunking.model_dump()])


def meta_hash(doc: Document) -> str:
    return _sha(doc.model_dump(mode="json", exclude={"paragraphs"}))


def doc_payload(doc: Document) -> dict:
    """Doc-level fields copied onto every chunk point (filtering + post-ranking)."""
    meta = doc.model_dump(mode="json", exclude={"paragraphs", "lang"})
    return {**meta, "meta_hash": meta_hash(doc)}


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)  # re-embedded
    meta_only: list[str] = field(default_factory=list)  # payload update, no re-embedding
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


class DocumentUpdateService:
    def __init__(self, index: Index, dense: DenseEncoder, sparse: SparseEncoder,
                 count_tokens: TokenCounter, chunking: ChunkingConfig):
        self.index = index
        self.dense = dense
        self.sparse = sparse
        self.count_tokens = count_tokens
        self.chunking = chunking

    def sync(self, docs: list[Document]) -> SyncReport:
        """Bring the index in line with `docs` (the full current corpus), doing the minimum work.

        Compares each doc's hashes with the ones stored in the index:
        - not indexed           -> chunk + embed (added)
        - content hash differs  -> re-embed this doc only (changed)
        - only meta hash differs -> payload update, no re-embedding (meta_only)
        - both equal            -> nothing (unchanged)
        Docs in the index but not in `docs` have their chunks removed (deleted).
        """
        self.index.ensure_collection()
        stored = self.index.doc_hashes()
        report = SyncReport()
        for doc in docs:
            new_content = content_hash(doc, self.dense.name, self.chunking)
            old = stored.get(doc.doc_id)
            # if document is new - add it's content
            if old is None:
                self._embed(doc, new_content)
                report.added.append(doc.doc_id)
            # content hash differs - re-embed
            elif old[0] != new_content:
                self._embed(doc, new_content)
                report.changed.append(doc.doc_id)
            # meta hash differs - update payload
            elif old[1] != meta_hash(doc):
                self.index.set_doc_payload(doc.doc_id, doc_payload(doc))
                report.meta_only.append(doc.doc_id)
            # else do nothing
            else:
                report.unchanged.append(doc.doc_id)
        # remove from index doc ids that were not found in storage
        for doc_id in sorted(stored.keys() - {d.doc_id for d in docs}):
            self.index.delete_doc(doc_id)
            report.deleted.append(doc_id)
        return report

    def _embed(self, doc: Document, content: str) -> None:
        """Chunk the doc, embed every chunk (dense + sparse) and write it to the index.

        Each point's payload = chunk fields + doc metadata + `content` (the doc's content hash)
        + embedder name. Chunks left over from the doc's previous version are then deleted.
        """
        chunks = chunk_document(doc, self.count_tokens, self.chunking.max_tokens, self.chunking.overlap)
        texts = [c.text for c in chunks]
        payloads = [
            {**c.model_dump(), **doc_payload(doc), "content_hash": content, "embedder": self.dense.name}
            for c in chunks
        ]
        self.index.upsert(payloads, self.dense.encode_documents(texts),
                          self.sparse.encode_documents(texts, [c.lang for c in chunks]))
        # upsert first, then drop leftover chunks, so the doc is never missing from the index
        self.index.delete_doc(doc.doc_id, keep_chunk_ids=[c.chunk_id for c in chunks])
