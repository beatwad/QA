from rag.query import condense

from fakes import FakeLLM

HISTORY = [{"role": "user", "content": "Who was Nikola Tesla?"},
           {"role": "assistant", "content": "An inventor [1]."}]


def test_first_turn_is_not_rewritten():
    llm = FakeLLM("should not be used")
    assert condense(llm, [], "Who was Tesla?") == "Who was Tesla?"
    assert llm.calls == []


def test_follow_up_is_rewritten_with_history():
    llm = FakeLLM("When did Nikola Tesla die?")
    assert condense(llm, HISTORY, "When did he die?") == "When did Nikola Tesla die?"
    assert "Who was Nikola Tesla?" in llm.calls[0][-1]["content"]


def test_llm_failure_falls_back_to_raw_question():
    assert condense(FakeLLM(error=ConnectionError("vLLM down")), HISTORY, "When did he die?") == "When did he die?"


def test_empty_rewrite_falls_back_to_raw_question():
    assert condense(FakeLLM("  "), HISTORY, "When did he die?") == "When did he die?"
