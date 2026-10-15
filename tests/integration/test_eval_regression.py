"""Retrieval quality floor: fails if a change makes the system noticeably worse at finding gold paragraphs.

Full run (rag eval, 1190 questions per language) measured Recall@5 0.98-0.99 at every stage on the
whole index; the floor leaves room for sampling noise on 100 questions per language.
"""
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from rag.eval.runner import Evaluator, sample

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parents[2] / "data"
RECALL_AT_5_FLOOR = 0.95


@pytest.fixture(scope="module")
def results(config, retriever, reranker):
    golden = [json.loads(line) for line in (DATA / "golden.jsonl").read_text().splitlines()]
    evaluator = Evaluator(config, retriever, reranker, yaml.safe_load((DATA / "users.yaml").read_text()),
                          date(2026, 10, 1))
    whole_index = [("EN→all", "en", None), ("RU→all", "ru", None)]
    return evaluator.retrieval(sample(golden, limit=100), settings=whole_index)[0]


@pytest.mark.parametrize("setting", ["EN→all", "RU→all"])
@pytest.mark.parametrize("stage", ["dense", "rrf", "rerank", "postrank"])
def test_recall_at_5_floor(results, setting, stage):
    assert results[setting][stage]["recall@5"] >= RECALL_AT_5_FLOOR, results[setting][stage]
