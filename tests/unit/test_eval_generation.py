from datetime import date

from rag.config import load_config
from rag.eval.generation import RecordingLLM, run_generation
from rag.serving.conversations import ConversationStore
from rag.serving.pipeline import Pipeline

from fakes import FakeLLM, FakeReranker, FakeRetriever, live_hit, words

CFG = load_config()
GOLDEN = [
    {"id": "q1-en", "lang": "en", "user": "root", "question": "What year did Tesla die?", "answers": ["1943"],
     "gold_paragraph": "d1-p00", "tags": ["generation"]},
    {"id": "q2-en", "lang": "en", "user": "root", "question": "Capital of Poland?", "answers": ["Warsaw"],
     "gold_paragraph": "d2-p00", "tags": []},
]


def make(ce):
    llm = RecordingLLM(FakeLLM("Tesla died in 1943 [1]."))
    pipeline = Pipeline(CFG, FakeRetriever([live_hit(1, "Tesla died in 1943.")]), FakeReranker(ce), llm, words,
                        ConversationStore(), today=lambda: date(2026, 10, 1))
    return pipeline, llm


def test_dumps_answer_citations_and_the_prompt_the_llm_saw():
    rows = run_generation(*make(ce=5.0), GOLDEN, {"root": ["all-employees"]})
    assert [r["id"] for r in rows] == ["q1-en"]  # only the generation subset
    row = rows[0]
    assert row["status"] == "answered" and row["answer"] == "Tesla died in 1943 [1]."
    assert row["citations"] == [{"n": 1, "chunk_id": "d1-c000"}]
    assert "[1] Title 1" in row["prompt"] and "Tesla died in 1943." in row["prompt"]


def test_gate_blocked_question_has_no_prompt():
    row = run_generation(*make(ce=-5.0), GOLDEN, {"root": ["all-employees"]})[0]
    assert row["status"] == "not_found" and row["prompt"] is None
