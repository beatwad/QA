"""Import a web page into the corpus: download, clean, dedupe, write data/corpus files, sync the index."""

import json
import re
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import yaml

from rag.lang import detect_lang

MIN_PARAGRAPH_CHARS = 60  # drops headings, captions and other fragments

Fetch = Callable[[str], tuple[str, str]]  # url -> (title, text)
NearDuplicates = Callable[[list[str], str], list[bool]]  # (paragraphs, doc_id) -> is it already indexed elsewhere?


class ImportRejected(ValueError):
    """The request can't be served (bad settings, unreachable or empty page)."""


def fetch_page(url: str) -> tuple[str, str]:
    """Wikipedia pages come from the API (clean text); anything else goes through trafilatura."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ImportRejected("url must start with http:// or https://")
    wiki = re.fullmatch(r"([a-z-]+)\.wikipedia\.org", parsed.hostname)
    if wiki and parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        resp = requests.get(f"https://{wiki[1]}.wikipedia.org/w/api.php", timeout=30,
                            headers={"User-Agent": "rag-import/0.1"},
                            params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                    "titles": title, "format": "json"})
        resp.raise_for_status()
        page = next(iter(resp.json()["query"]["pages"].values()))
        if "extract" not in page:
            raise ImportRejected(f"Wikipedia has no page {title!r}")
        return page["title"], page["extract"]

    import trafilatura

    html = trafilatura.fetch_url(url)
    text = html and trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        raise ImportRejected("could not download or extract text from the page")
    meta = trafilatura.extract_metadata(html)
    return (meta.title if meta and meta.title else parsed.hostname), text


def split_paragraphs(text: str) -> list[str]:
    """One paragraph per line; drops wiki headings ("== History =="), short fragments and exact repeats."""
    seen, out = set(), []
    for line in text.splitlines():
        line = " ".join(line.split())
        key = line.casefold()
        if len(line) < MIN_PARAGRAPH_CHARS or re.fullmatch(r"=+ .* =+", line) or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def doc_id_for(url: str) -> str:
    """Stable id from the URL, so importing the same page again updates its document."""
    parsed = urlparse(url)
    return re.sub(r"[\W_]+", "-", unquote(f"{parsed.hostname}{parsed.path}").lower()).strip("-")[:100]


class Importer:
    def __init__(self, corpus_dir: Path, fetch: Fetch, near_duplicates: NearDuplicates, sync: Callable[[], object],
                 groups: list[str], sources: list[str], authorities: list[str]):
        self.corpus_dir, self.fetch, self.near_duplicates, self.sync = corpus_dir, fetch, near_duplicates, sync
        self.options = {"groups": groups, "sources": sources, "authorities": authorities}
        self._lock = threading.Lock()  # meta.yaml is rewritten as a whole

    def run(self, url: str, acl: list[str], source: str, authority: str,
            title: str | None = None, lang: str | None = None) -> dict:
        # an empty ACL would make the document invisible to everyone (fail closed), so it is refused here
        if not acl:
            raise ImportRejected("pick at least one ACL group")
        for value, allowed, name in ((acl, self.options["groups"], "ACL group"),
                                     ([source], self.options["sources"], "source"),
                                     ([authority], self.options["authorities"], "authority")):
            unknown = [v for v in value if v not in allowed]
            if unknown:
                raise ImportRejected(f"unknown {name}: {', '.join(unknown)}")

        page_title, text = self.fetch(url)
        doc_id = doc_id_for(url)
        paragraphs = split_paragraphs(text)
        fresh = [p for p, dup in zip(paragraphs, self.near_duplicates(paragraphs, doc_id)) if not dup]
        if not fresh:
            raise ImportRejected("no usable paragraphs (empty page, or everything is already indexed)")

        doc = {"doc_id": doc_id, "title": title or page_title, "lang": lang or detect_lang(" ".join(fresh)),
               "paragraphs": [{"id": f"{doc_id}-p{i:03d}", "text": p} for i, p in enumerate(fresh)]}
        meta = {"doc_id": doc_id, "title": doc["title"], "lang": doc["lang"], "source": source, "url": url,
                "acl": acl, "updated_at": date.today(), "authority": authority, "status": "active",
                "superseded_by": None}
        with self._lock:
            (self.corpus_dir / f"{doc_id}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            self._upsert_meta(meta)
            report = self.sync()
        return {"doc_id": doc_id, "title": doc["title"], "lang": doc["lang"], "paragraphs": len(fresh),
                "skipped_duplicates": len(paragraphs) - len(fresh), "sync": report}

    def _upsert_meta(self, entry: dict) -> None:
        path = self.corpus_dir / "meta.yaml"
        rows = [m for m in yaml.safe_load(path.read_text()) if m["doc_id"] != entry["doc_id"]]
        path.write_text(yaml.safe_dump(rows + [entry], allow_unicode=True, sort_keys=False))
