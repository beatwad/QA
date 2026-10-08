import json
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel


class Paragraph(BaseModel):
    id: str
    text: str


class Document(BaseModel):
    doc_id: str
    title: str
    lang: str
    source: str
    url: str
    acl: list[str] = []  # empty -> nobody can see it (fail closed)
    updated_at: date
    authority: str
    status: str
    superseded_by: str | None = None
    paragraphs: list[Paragraph]


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Join data/corpus/<doc_id>.json (content) with meta.yaml (metadata)."""
    meta = {m["doc_id"]: m for m in yaml.safe_load((corpus_dir / "meta.yaml").read_text())}
    docs = []
    for path in sorted(corpus_dir.glob("*.json")):
        content = json.loads(path.read_text())
        if content["doc_id"] not in meta:
            raise ValueError(f"{path.name}: no entry in meta.yaml")
        docs.append(Document(**{**meta[content["doc_id"]], "paragraphs": content["paragraphs"]}))
    return docs
