"""Real vLLM: grounded answers with citations, answer language, NOT_FOUND, condensation."""
from pathlib import Path

import pytest

from rag.chunking import make_token_counter
from rag.citations import extract_citations
from rag.documents import load_corpus
from rag.lang import detect_lang
from rag.llm import OpenAICompatLLM
from rag.prompt import build_messages, is_not_found
from rag.query import condense
from rag.retrieval import Hit

pytestmark = pytest.mark.llm

DOCS = {d.doc_id: d for d in load_corpus(Path(__file__).resolve().parents[2] / "data" / "corpus")}


@pytest.fixture(scope="module")
def llm(config):
    return OpenAICompatLLM(config.llm.base_url, config.llm.model)


@pytest.fixture(scope="module")
def answer(config, llm):
    count = make_token_counter(config.llm.tokenizer)

    def _answer(question, hits, history=()):
        messages, used = build_messages(question, hits, list(history), count,
                                        config.llm.context_window - config.llm.max_tokens)
        raw = "".join(llm.stream(messages, config.llm.max_tokens, temperature=0.0))
        text, cites = extract_citations(raw, used)
        return raw, text, cites
    return _answer


def paragraph(doc_id: str, idx: int) -> Hit:
    doc = DOCS[doc_id]
    p = doc.paragraphs[idx]
    payload = {**doc.model_dump(mode="json", exclude={"paragraphs"}), "chunk_id": f"{p.id}-{doc.lang}", "text": p.text}
    return Hit(chunk_id=payload["chunk_id"], payload=payload)


def context(lang: str) -> list[Hit]:
    """Gold paragraph (Tesla's death, 1943) in slot [2] between two distractors."""
    return [paragraph(f"art00-{lang}", 0), paragraph(f"art03-{lang}", 0), paragraph(f"art01-{lang}", 0)]


@pytest.mark.parametrize("question,ctx_lang,answer_lang", [
    ("What year did Tesla die?", "en", "en"),
    ("В каком году умер Тесла?", "ru", "ru"),
    ("В каком году умер Тесла?", "en", "ru"),  # cross-lingual: answer in the question's language
])
def test_grounded_answer_cites_the_gold_passage(answer, question, ctx_lang, answer_lang):
    raw, text, cites = answer(question, context(ctx_lang))
    assert not is_not_found(raw), raw
    assert "1943" in text
    assert 2 in [c.n for c in cites], raw
    assert detect_lang(text) == answer_lang, text


@pytest.mark.parametrize("question", ["How many moons does Mars have?", "Сколько спутников у Марса?"])
def test_answer_not_in_context_gives_not_found(answer, question):
    raw, _, cites = answer(question, context("en"))
    assert is_not_found(raw), raw
    assert cites == []


@pytest.mark.parametrize("history,follow_up,expected", [
    ([{"role": "user", "content": "Who was Nikola Tesla?"},
      {"role": "assistant", "content": "A Serbian-American inventor and engineer [1]."}],
     "When did he die?", "Tesla"),
    ([{"role": "user", "content": "Кто такой Никола Тесла?"},
      {"role": "assistant", "content": "Сербско-американский изобретатель и инженер [1]."}],
     "Когда он умер?", "Тесл"),
])
def test_condensation_resolves_the_reference(llm, history, follow_up, expected):
    standalone = condense(llm, history, follow_up)
    assert expected in standalone, standalone
    assert detect_lang(standalone) == detect_lang(follow_up)
