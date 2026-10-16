from datetime import date

import pytest

from rag.config import load_config
from rag.serving.conversations import ConversationStore
from rag.storage.index import DENSE, SPARSE
from rag.serving.pipeline import MESSAGES, Pipeline
from rag.search.retrieval import Hit, Retriever

from fakes import FakeLLM, FakeReranker, FakeRetriever, live_hit, words

CFG = load_config()
HITS = [live_hit(1, "Tesla died in 1943."), live_hit(2, "Warsaw is the capital of Poland.")]


def make(llm, ce=5.0, conversations=None):
    return Pipeline(CFG, FakeRetriever(HITS), FakeReranker(ce), llm, words, conversations or ConversationStore(),
                    today=lambda: date(2026, 10, 1))


def run(pipeline, question="What year did Tesla die?", user="alice", conversation_id="c1"):
    return list(pipeline.ask(user, ["all-employees"], question, conversation_id))


def final(events):
    return next(data for kind, data in events if kind == "result")


def test_answer_streams_tokens_then_result_then_done():
    events = run(make(FakeLLM("Tesla died in 1943 [1].")))
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-2:] == ["result", "done"]
    assert set(kinds[1:-2]) == {"token"}
    assert "".join(d["text"] for k, d in events if k == "token") == "Tesla died in 1943 [1]."
    out = final(events)
    assert out["status"] == "answered"
    assert [c["n"] for c in out["citations"]] == [1]


def test_low_confidence_skips_the_llm():
    llm = FakeLLM("should not be called")
    events = run(make(llm, ce=-5.0), question="Сколько спутников у Марса?")
    assert llm.calls == []
    assert final(events) == {"status": "not_found", "answer": MESSAGES["not_found"]["ru"], "citations": []}
    assert "token" not in [k for k, _ in events]


@pytest.mark.parametrize("reply", ["NOT_FOUND", "  NOT_FOUND", "NOT_FOUND."])
def test_sentinel_never_reaches_the_user(reply):
    events = run(make(FakeLLM(reply)))
    assert "token" not in [k for k, _ in events]
    assert final(events)["status"] == "not_found"


def test_answer_starting_like_the_sentinel_is_streamed_intact():
    events = run(make(FakeLLM("NO. Tesla died in 1943 [1].")))
    assert "".join(d["text"] for k, d in events if k == "token") == "NO. Tesla died in 1943 [1]."


@pytest.mark.parametrize("reply", ["[1] [2] NOT_FOUND", "Two-thirds of the population. Therefore: NOT_FOUND"])
def test_sentinel_after_other_text_is_not_found(reply):
    out = final(run(make(FakeLLM(reply))))
    assert out == {"status": "not_found", "answer": MESSAGES["not_found"]["en"], "citations": []}


@pytest.mark.parametrize("reply", ["[1] [2]", "[1, 2]", " [1]\n"])
def test_answer_without_text_returns_ranked_links(reply):
    out = final(run(make(FakeLLM(reply))))
    assert out["status"] == "llm_unavailable"
    assert [c["doc_id"] for c in out["citations"]] == ["d1", "d2"]


def test_wrong_language_answer_is_retried_with_a_reminder():
    llm = FakeLLM(["Тесла умер в 1943 году [1].", "Tesla died in 1943 [1]."])
    out = final(run(make(llm)))
    assert out["status"] == "answered" and out["answer"] == "Tesla died in 1943 [1]."
    assert len(llm.calls) == 2
    assert llm.calls[1][-1]["content"].endswith("Write the whole answer in English only.")


def test_still_wrong_language_after_the_retry_returns_ranked_links():
    out = final(run(make(FakeLLM(["Тесла умер в 1943 году [1].", "Тесла умер в 1943 [1]."]))))
    assert out["status"] == "llm_unavailable"


def test_russian_question_with_a_latin_only_name_is_not_retried():
    llm = FakeLLM("Dolby Digital [1]")
    out = final(run(make(llm), question="Какой звук поддерживает Sky Box?"))
    assert out["status"] == "answered" and len(llm.calls) == 1


def test_russian_question_answered_in_an_english_sentence_is_retried():
    llm = FakeLLM(["Tesla died in 1943 [1].", "Тесла умер в 1943 году [1]."])
    out = final(run(make(llm), question="В каком году умер Тесла?"))
    assert out["answer"] == "Тесла умер в 1943 году [1]." and len(llm.calls) == 2


def test_llm_down_returns_ranked_links():
    out = final(run(make(FakeLLM(error=ConnectionError("vLLM down")))))
    assert out["status"] == "llm_unavailable"
    assert out["answer"] == MESSAGES["llm_unavailable"]["en"]
    assert [c["doc_id"] for c in out["citations"]] == ["d1", "d2"]


def test_follow_up_is_condensed_with_the_conversation():
    llm = FakeLLM(["Tesla was an inventor [1].", "When did Nikola Tesla die?", "In 1943 [1]."])
    pipeline = make(llm)
    run(pipeline, question="Who was Tesla?")
    events = run(pipeline, question="When did he die?")
    assert events[0] == ("meta", {"conversation_id": "c1", "query": "When did Nikola Tesla die?"})
    assert pipeline.retriever.queries[-1] == "When did Nikola Tesla die?"
    condense_prompt = llm.calls[1][-1]["content"]
    assert "Who was Tesla?" in condense_prompt and "Tesla was an inventor [1]." in condense_prompt


def test_users_cannot_read_each_others_conversations():
    llm = FakeLLM("Tesla died in 1943 [1].")
    pipeline = make(llm)
    run(pipeline, user="alice", conversation_id="shared-id")
    calls_before = len(llm.calls)
    run(pipeline, user="bob", question="When did he die?", conversation_id="shared-id")
    assert len(llm.calls) == calls_before + 1  # answer only: bob has no history, so no condensation


def test_new_conversation_gets_an_id():
    events = run(make(FakeLLM("Tesla died in 1943 [1].")), conversation_id=None)
    assert len(events[0][1]["conversation_id"]) == 32


class StubRetriever(Retriever):
    """Real fusion logic; branches and vector lookup are stubbed."""

    def __init__(self, broken):
        super().__init__(index=self, dense=None, sparse=None, limit=10, rrf_k=60,
                         weights={DENSE: 1.0, SPARSE: 1.0})
        self.broken = broken

    def search_branch(self, branch, query, groups, langs=None):
        if branch in self.broken:
            raise RuntimeError(f"{branch} down")
        return [Hit(chunk_id=f"{branch}-1", payload={}, ranks={branch: 1},
                    vector=[1.0] if branch == DENSE else None)]

    def dense_vectors(self, chunk_ids):
        return {c: [0.5] for c in chunk_ids}


def test_one_failed_branch_serves_the_other():
    hits = StubRetriever(broken={DENSE}).search("q", ["g"])
    assert [h.chunk_id for h in hits] == ["sparse-1"]
    assert hits[0].vector == [0.5]


def test_both_branches_failing_raises():
    with pytest.raises(RuntimeError):
        StubRetriever(broken={DENSE, SPARSE}).search("q", ["g"])
