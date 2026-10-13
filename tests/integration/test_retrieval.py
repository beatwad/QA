import json
import uuid
from pathlib import Path

import pytest
import yaml
from qdrant_client import QdrantClient

from rag.indexing.documents import load_corpus
from rag.storage.index import DENSE, SPARSE, Index
from rag.indexing.ingest import DocumentUpdateService
from rag.search.retrieval import Retriever

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parents[2] / "data"
USERS = yaml.safe_load((DATA / "users.yaml").read_text())
GOLDEN = [json.loads(line) for line in (DATA / "golden.jsonl").read_text().splitlines()]
DOCS = {d.doc_id: d for d in load_corpus(DATA / "corpus")}


def make_retriever(config, bge, bm25, count_tokens, docs):
    index = Index(QdrantClient(config.qdrant.url), f"test_{uuid.uuid4().hex[:8]}", bge.dim)
    DocumentUpdateService(index, bge, bm25, count_tokens, config.chunking).sync(docs)
    r = config.retrieval
    return Retriever(index, bge, bm25, r.candidates, r.rrf_k, r.rrf_weights)


def restricted_questions():
    """Every question whose gold doc is not open to all employees (~400 queries)."""
    return [g for g in GOLDEN if g["gold_doc"] and DOCS[g["gold_doc"]].acl != ["all-employees"]]


@pytest.mark.parametrize("user", ["alice", "bob", "hana"])
def test_permission_leak_canary(retriever, user):
    groups = USERS[user]
    for g in restricted_questions():
        for branch in (DENSE, SPARSE):
            for hit in retriever.search_branch(branch, g["question"], groups):
                assert set(hit.payload["acl"]) & set(groups), f"{user} saw {hit.chunk_id} via {branch}"


def test_restricted_docs_are_found_when_allowed(retriever):
    """Positive control: the canary means something only if these docs are retrievable at all."""
    for g in (g for g in GOLDEN if g["kind"] == "acl"):
        doc_ids = {h.payload["doc_id"] for h in retriever.search(g["question"], USERS["root"])}
        assert g["gold_doc"] in doc_ids, g["id"]


def test_no_groups_sees_nothing(retriever):
    assert retriever.search("What year did Tesla die?", []) == []


def test_fused_hits_carry_both_branches(retriever):
    hits = retriever.search("What year did Tesla die?", USERS["alice"])
    assert len(hits) == retriever.limit
    assert all(a.rrf_score >= b.rrf_score for a, b in zip(hits, hits[1:]))
    both = [h for h in hits if set(h.ranks) == {DENSE, SPARSE}]
    assert both and all(set(h.scores) == {DENSE, SPARSE} for h in both)
    # every candidate carries its stored dense vector, including sparse-only hits
    assert all(h.vector and len(h.vector) == retriever.dense.dim for h in hits)


def test_missing_acl_is_never_returned(config, bge, bm25):
    index = Index(QdrantClient(config.qdrant.url), f"test_{uuid.uuid4().hex[:8]}", bge.dim)
    index.ensure_collection()
    try:
        text = "Nikola Tesla died on 7 January 1943."
        base = {"chunk_id": "x", "doc_id": "x", "text": text, "lang": "en"}
        payloads = [{**base, "chunk_id": "empty-acl", "acl": []}, {**base, "chunk_id": "no-acl"}]
        index.upsert(payloads, bge.encode_documents([text] * 2), bm25.encode_documents([text] * 2, ["en"] * 2))
        assert index.count() == 2
        retriever = Retriever(index, bge, bm25, 10, 60, {DENSE: 1.0, SPARSE: 1.0})
        assert retriever.search("When did Tesla die?", USERS["root"]) == []
    finally:
        index.drop()


def test_cross_lingual_query_found_by_dense_only(config, bge, bm25, count_tokens):
    """RU question against an EN-only index: BM25 can't match across scripts, dense can."""
    docs = [DOCS[i] for i in ("art00-en", "art01-en", "art03-en")]
    retriever = make_retriever(config, bge, bm25, count_tokens, docs)
    try:
        gold = "art03-p00"
        question = "В каком году умер Тесла?"
        dense = retriever.search_branch(DENSE, question, USERS["root"])
        sparse = retriever.search_branch(SPARSE, question, USERS["root"])
        assert gold in dense[0].payload["paragraph_ids"]
        assert all(gold not in h.payload["paragraph_ids"] for h in sparse)
        fused = retriever.search(question, USERS["root"])
        assert gold in fused[0].payload["paragraph_ids"]
    finally:
        retriever.index.drop()
