"""Full stack through the HTTP API: real index, encoders, reranker and vLLM."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.serving.api import create_app
from rag.serving.auth import Users
from rag.indexing.chunking import make_token_counter
from rag.serving.conversations import ConversationStore
from rag.indexing.documents import load_corpus
from rag.generation.llm import OpenAICompatLLM
from rag.serving.pipeline import Pipeline
from rag.search.retrieval import Retriever

pytestmark = pytest.mark.llm

DATA = Path(__file__).resolve().parents[2] / "data"
USERS = Users.load(DATA / "users.yaml")
DOCS = {d.doc_id: d for d in load_corpus(DATA / "corpus")}
GOLDEN = [json.loads(line) for line in (DATA / "golden.jsonl").read_text().splitlines()]


class BrokenEncoder:
    """A dense encoder that is down."""

    def __init__(self, dim):
        self.name, self.dim = "broken", dim

    def encode_query(self, text):
        raise RuntimeError("GPU fell off the bus")


@pytest.fixture(scope="module")
def make_client(config, retriever, reranker):
    count = make_token_counter(config.llm.tokenizer)

    def _make(llm_url=config.llm.base_url, dense=None):
        ret = retriever if dense is None else Retriever(retriever.index, dense, retriever.sparse, retriever.limit,
                                                        retriever.rrf_k, retriever.weights)
        pipeline = Pipeline(config, ret, reranker, OpenAICompatLLM(llm_url, config.llm.model, timeout=5), count,
                            ConversationStore())
        return TestClient(create_app(pipeline, USERS, sync=lambda: None))
    return _make


def ask(client, question, user="alice", conversation_id=None):
    resp = client.post("/v1/ask", json={"question": question, "conversation_id": conversation_id},
                       headers={"X-User": user})
    assert resp.status_code == 200
    events = []
    for block in resp.text.strip().split("\n\n"):
        kind, data = block.split("\n")
        events.append((kind.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    meta = events[0][1]
    final = next(d for k, d in events if k == "result")
    tokens = "".join(d["text"] for k, d in events if k == "token")
    return meta, final, tokens


@pytest.mark.parametrize("question", ["What year did Tesla die?", "В каком году умер Тесла?"])
def test_answer_with_valid_citation(make_client, question):
    _, final, tokens = ask(make_client(), question)
    assert final["status"] == "answered", final
    assert "1943" in final["answer"] and tokens
    assert any(c["doc_id"].startswith("art03-") for c in final["citations"])


def test_follow_up_in_same_conversation(make_client):
    client = make_client()
    meta, _, _ = ask(client, "Who was Nikola Tesla?")
    meta2, final, _ = ask(client, "When did he die?", conversation_id=meta["conversation_id"])
    assert "Tesla" in meta2["query"]
    assert final["status"] == "answered" and "1943" in final["answer"]


def test_unanswerable_question_is_not_found(make_client):
    _, final, tokens = ask(make_client(), "How many moons does Mars have?")
    assert final["status"] == "not_found" and final["citations"] == [] and tokens == ""


def test_restricted_doc_never_leaks_through_the_api(make_client):
    client = make_client()
    for g in (g for g in GOLDEN if g["kind"] == "acl"):
        _, final, _ = ask(client, g["question"], user="alice")
        assert final["status"] == "not_found", g["id"]
        visible = set(USERS.groups("alice"))
        assert all(set(DOCS[c["doc_id"]].acl) & visible for c in final["citations"])


def test_llm_down_returns_links(make_client):
    _, final, tokens = ask(make_client(llm_url="http://localhost:9/v1"), "What year did Tesla die?")
    assert final["status"] == "llm_unavailable" and tokens == ""
    assert final["citations"] and all(c["url"] for c in final["citations"])


def test_dense_branch_down_still_answers_from_sparse(make_client, retriever):
    client = make_client(dense=BrokenEncoder(retriever.dense.dim))
    _, final, _ = ask(client, "What year did Tesla die?")
    assert final["status"] == "answered" and "1943" in final["answer"]
