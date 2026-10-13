"""Retrieval -> cross-encoder -> post-ranking -> gate, with real models on the full corpus."""
import json
import random
from datetime import date
from pathlib import Path

import pytest
import yaml

from rag.postrank import postrank
from rag.rerank import confident, rerank

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parents[2] / "data"
USERS = yaml.safe_load((DATA / "users.yaml").read_text())
GOLDEN = [json.loads(line) for line in (DATA / "golden.jsonl").read_text().splitlines()]
TODAY = date(2026, 10, 1)


@pytest.fixture(scope="module")
def run(config, retriever, reranker):
    """question, user -> (final chunks, gate passed?)"""
    def _run(g):
        hits = rerank(reranker, g["question"], retriever.search(g["question"], USERS[g["user"]]), config.rerank.top_k)
        final = postrank(hits, config.postrank, TODAY)
        return final, confident(final, config.rerank.gate_threshold)
    return _run


def tagged(tag):
    return [g for g in GOLDEN if tag in g["tags"]]


def test_gate_fires_on_unanswerable_and_hidden_questions(run):
    for g in (g for g in GOLDEN if g["kind"] in ("unanswerable", "acl")):
        _, passed = run(g)
        assert not passed, g["id"]


def test_gate_lets_answerable_questions_through(run):
    sample = random.Random(0).sample([g for g in GOLDEN if g["kind"] == "answerable"], 100)
    passed = sum(run(g)[1] for g in sample)
    assert passed >= 90


def test_superseded_copy_never_reaches_the_llm(run):
    assert tagged("superseded")
    for g in tagged("superseded"):
        final, _ = run(g)
        assert not any(h.payload["doc_id"].endswith("-old") for h in final)
        assert g["gold_doc"] in [h.payload["doc_id"] for h in final]


def test_contradicting_informal_copy_is_dropped(run):
    """The Slack copy differs from the official paragraph by one fact: the official one wins on authority,
    then the Slack copy is removed as its near-duplicate, so the LLM never sees the contradiction."""
    assert tagged("informal")
    for g in tagged("informal"):
        order = [h.payload["doc_id"] for h in run(g)[0]]
        assert order[0] == g["gold_doc"], g["id"]
        assert g["gold_doc"] + "-slack" not in order, g["id"]


def test_duplicate_copies_are_deduplicated(run):
    assert tagged("duplicated")
    for g in tagged("duplicated"):
        texts = [h.payload["text"] for h in run(g)[0]]
        assert len(texts) == len(set(texts)), g["id"]
