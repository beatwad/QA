import re
from collections.abc import Callable

from pydantic import BaseModel

from rag.indexing.documents import Document
from rag.lang import detect_lang

TokenCounter = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    position: int
    text: str
    paragraph_ids: list[str]
    acl: list[str]
    lang: str


def make_token_counter(model_name: str) -> TokenCounter:
    """Counts tokens with the embedding model's own tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(model_name)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def chunk_document(doc: Document, count: TokenCounter, max_tokens: int, overlap: float) -> list[Chunk]:
    """
    One paragraph = one chunk; a paragraph over max_tokens is split into
    sentence windows that share ~overlap * max_tokens of trailing sentences.
    """
    texts = []
    for p in doc.paragraphs:
        pieces = [p.text] if count(p.text) <= max_tokens else _windows(p.text, count, max_tokens, overlap)
        texts += [(p.id, t) for t in pieces]
    return [
        Chunk(chunk_id=f"{doc.doc_id}-c{i:03d}", doc_id=doc.doc_id, position=i, text=text,
              paragraph_ids=[pid], acl=doc.acl, lang=detect_lang(text))
        for i, (pid, text) in enumerate(texts)
    ]


def _units(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Sentences; a sentence longer than max_tokens is broken into word runs."""
    units = []
    for sent in _SENTENCE_END.split(text):
        # if sentence length <= max_tokens - add it
        if count(sent) <= max_tokens:
            units.append(sent)
            continue
        # else split sentence by runs, each has lenght <= max_tokens
        # and add one-by-one
        run: list[str] = []
        for word in sent.split():
            if run and count(" ".join(run + [word])) > max_tokens:
                units.append(" ".join(run))
                run = []
            run.append(word)
        units.append(" ".join(run))
    return units


def _windows(text: str, count: TokenCounter, max_tokens: int, overlap: float) -> list[str]:
    """
    Group sentences into windows of at most max_tokens tokens.
    Consecutive windows share up to overlap * max_tokens tokens of trailing sentences.
    """
    units = _units(text, count, max_tokens)
    windows, start = [], 0
    while True:
        end = start + 1
        # keep adding sentences to windows list while the joined text still fits in max_tokens
        while end < len(units) and count(" ".join(units[start : end + 1])) <= max_tokens:
            end += 1
        windows.append(" ".join(units[start:end]))
        if end == len(units):
            return windows
        # step back over trailing sentences that fit in the overlap budget, keeping progress
        next_start = end
        while next_start - 1 > start and count(" ".join(units[next_start - 1 : end])) <= overlap * max_tokens:
            next_start -= 1
        start = next_start
