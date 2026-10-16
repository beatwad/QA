"""Serving pipeline: one question in, a stream of events out.

Events (type, data), in order:
  meta       {conversation_id, query}            query = the standalone (condensed) question
  token      {text}                              zero or more, only when an answer is generated
  result     {status, answer, citations}         status: answered | not_found | llm_unavailable
  done       {}
"""

import json
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import date

from loguru import logger

from rag.indexing.chunking import TokenCounter
from rag.generation.citations import SNIPPET_CHARS, extract_citations
from rag.config import Config
from rag.serving.conversations import ConversationStore
from rag.lang import detect_lang
from rag.generation.llm import LLM, complete
from rag.search.postrank import postrank
from rag.generation.prompt import LANGUAGE_REMINDER, NOT_FOUND, build_messages
from rag.generation.query import condense
from rag.search.rerank import Reranker, confident, rerank
from rag.search.retrieval import Hit, Retriever

Event = tuple[str, dict]

MESSAGES = {
    "not_found": {
        "en": "I couldn't find this in the documents you have access to.",
        "ru": "Не удалось найти ответ в доступных вам документах.",
    },
    "llm_unavailable": {
        "en": "The answer generator is unavailable right now. These documents look most relevant:",
        "ru": "Генератор ответов сейчас недоступен. Наиболее подходящие документы:",
    },
}


class Pipeline:
    def __init__(self, cfg: Config, retriever: Retriever, reranker: Reranker, llm: LLM,
                 count_tokens: TokenCounter, conversations: ConversationStore,
                 today: Callable[[], date] = date.today):
        self.cfg = cfg
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.count_tokens = count_tokens
        self.conversations = conversations
        self.today = today

    def ask(self, user: str, groups: list[str], question: str, conversation_id: str | None = None) -> Iterator[Event]:
        conversation_id = conversation_id or uuid.uuid4().hex
        lang = detect_lang(question)
        history = self.conversations.history(user, conversation_id)[-self.cfg.llm.history_turns:]

        query = condense(self.llm, history, question)
        yield "meta", {"conversation_id": conversation_id, "query": query}

        hits = rerank(self.reranker, query, self.retriever.search(query, groups), self.cfg.rerank.top_k)
        final = postrank(hits, self.cfg.postrank, self.today())

        if not confident(final, self.cfg.rerank.gate_threshold):
            status, answer, citations = "not_found", MESSAGES["not_found"][lang], []
        else:
            status, answer, citations = yield from self._generate(query, final, history, lang)

        # save before the last events: the client may disconnect as soon as it sees them
        self.conversations.append(user, conversation_id, question, answer)
        logger.info(json.dumps({"user": user, "conversation_id": conversation_id, "query": query, "status": status,
                             "chunks": [h.chunk_id for h in final], "cited": [c["chunk_id"] for c in citations]},
                            ensure_ascii=False))
        yield "result", {"status": status, "answer": answer, "citations": citations}
        yield "done", {}

    def _generate(self, query: str, final: list[Hit], history: list[dict], lang: str):
        """Streams token events; returns (status, answer, citations)."""
        budget = self.cfg.llm.context_window - self.cfg.llm.max_tokens
        messages, used = build_messages(query, final, history, self.count_tokens, budget)
        raw = ""
        try:
            for text in _hold_back_sentinel(self.llm.stream(messages, self.cfg.llm.max_tokens,
                                                             self.cfg.llm.temperature)):
                if text is None:  # the model said NOT_FOUND
                    return "not_found", MESSAGES["not_found"][lang], []
                raw += text
                yield "token", {"text": text}
        except Exception:
            logger.opt(exception=True).warning("LLM failed, returning ranked links")
            return _links_fallback(final, lang)
        problem = _problem(raw, lang)
        if problem == "language":
            # the model copied text from a passage in the other language: ask once more, with a reminder
            logger.warning("LLM answered in the wrong language, retrying")
            reminded = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + "\n\n" + LANGUAGE_REMINDER[lang]}]
            try:
                raw = complete(self.llm, reminded, self.cfg.llm.max_tokens, self.cfg.llm.temperature)
            except Exception:
                logger.opt(exception=True).warning("LLM retry failed, returning ranked links")
                return _links_fallback(final, lang)
            problem = _problem(raw, lang)
        if problem == "not_found":  # the start of the answer may have been streamed already; `result` supersedes it
            return "not_found", MESSAGES["not_found"][lang], []
        if problem:
            logger.warning("LLM answer unusable ({}), returning ranked links", problem)
            return _links_fallback(final, lang)
        answer, cites = extract_citations(raw, used)
        return "answered", answer, [asdict(c) for c in cites]


def _hold_back_sentinel(pieces: Iterator[str]) -> Iterator[str | None]:
    """Pass text through, except: hold back the start of the answer until it's clear whether it's
    the NOT_FOUND sentinel. Yields None (and stops) if it is, so the raw sentinel never reaches the user."""
    buffer = ""
    for piece in pieces:
        # pass-through mode
        if buffer is None:
            yield piece
            continue
        # Use buffer to check if message starts with NOT_FOUND
        buffer += piece
        head = buffer.lstrip()
        # If message starts with NOT_FOUND - yield None and exit
        if head.startswith(NOT_FOUND):
            yield None
            return
        # If it seems like message is going to begin with NOT_FOUND - 
        # wait until has enough characters to decide
        if len(head) < len(NOT_FOUND) and NOT_FOUND.startswith(head):
            continue  # could still turn into NOT_FOUND
        # If message is non NOT_FOUND - yield the whole buffer and set it to None
        # to turn the function to pass-through mode
        yield buffer
        buffer = None
    # In case when the answer is shorter than NOT_FOUND length
    if buffer:
        yield buffer


def _problem(raw: str, lang: str) -> str | None:
    """What is wrong with the model's raw answer, if anything: "not_found" (the sentinel anywhere in it),
    "empty" (only [n] markers, no words), "language" (not in the question's language)."""
    if NOT_FOUND in raw:
        return "not_found"
    text = re.sub(r"\[[\d,\s]+\]", "", raw)
    if not re.search(r"\w", text):
        return "empty"
    # English answers contain no Cyrillic. A Russian answer may be Latin-only (a name, a term), so only
    # a whole English sentence counts as the wrong language.
    if lang == "en":
        wrong = bool(re.search("[а-яё]", text, re.IGNORECASE))
    else:
        wrong = detect_lang(text) == "en" and len(text.split()) >= 4
    if wrong:
        return "language"
    return None


def _links_fallback(final: list[Hit], lang: str) -> tuple[str, str, list[dict]]:
    return "llm_unavailable", MESSAGES["llm_unavailable"][lang], [_link(i, h) for i, h in enumerate(final, start=1)]


def _link(n: int, hit: Hit) -> dict:
    """Turn a retrieved chunk (Hit) into a plain dict describing a source link. 
    Used for the fallback when the LLM is unavailable."""
    p = hit.payload
    return {"n": n, "chunk_id": p["chunk_id"], "doc_id": p["doc_id"], "title": p["title"], "url": p["url"],
            "snippet": p["text"][:SNIPPET_CHARS]}
