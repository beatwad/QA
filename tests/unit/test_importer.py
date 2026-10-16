import json

import pytest
import yaml
from fastapi.testclient import TestClient

from rag.indexing.importer import Importer, ImportRejected, doc_id_for, split_paragraphs
from rag.indexing.ingest import SyncReport
from rag.serving.api import create_app
from rag.serving.auth import Users

LONG = "This is a sufficiently long paragraph about Tesla's early life and where he was born. "
TEXT = f"""== Early life ==
{LONG}One.
short
{LONG}One.
{LONG.upper()}Two is different.
"""
SETTINGS = dict(url="https://en.wikipedia.org/wiki/Nikola_Tesla", acl=["all-employees"], source="wiki",
                authority="official")


def make(tmp_path, dup=lambda paras, doc_id: [False] * len(paras), text=TEXT):
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump([{"doc_id": "other", "title": "Other"}]))
    synced = []

    def sync():
        synced.append(True)
        return SyncReport(added=["x"])

    imp = Importer(tmp_path, lambda url: ("Nikola Tesla", text), dup, sync,
                   ["all-employees", "hr-team"], ["wiki", "slack"], ["official", "informal"])
    return imp, synced


def test_split_drops_headings_short_lines_and_repeats():
    assert split_paragraphs(TEXT) == [LONG + "One.", LONG.upper() + "Two is different."]


def test_doc_id_is_stable_filesystem_safe_and_handles_cyrillic():
    assert doc_id_for("https://en.wikipedia.org/wiki/Nikola_Tesla") == "en-wikipedia-org-wiki-nikola-tesla"
    assert doc_id_for("https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D1%81%D0%BB%D0%B0") == "ru-wikipedia-org-wiki-тесла"
    assert "/" not in doc_id_for("https://x.org/../../etc/passwd")


def test_import_writes_document_and_metadata_then_syncs(tmp_path):
    imp, synced = make(tmp_path)
    result = imp.run(**SETTINGS)
    doc = json.loads((tmp_path / "en-wikipedia-org-wiki-nikola-tesla.json").read_text())
    assert [p["id"] for p in doc["paragraphs"]] == ["en-wikipedia-org-wiki-nikola-tesla-p000",
                                                    "en-wikipedia-org-wiki-nikola-tesla-p001"]
    meta = {m["doc_id"]: m for m in yaml.safe_load((tmp_path / "meta.yaml").read_text())}
    assert meta["en-wikipedia-org-wiki-nikola-tesla"]["acl"] == ["all-employees"] and "other" in meta
    assert synced and result["paragraphs"] == 2 and result["lang"] == "en"


def test_reimport_replaces_the_metadata_entry(tmp_path):
    imp, _ = make(tmp_path)
    imp.run(**SETTINGS)
    imp.run(**{**SETTINGS, "acl": ["hr-team"]})
    rows = yaml.safe_load((tmp_path / "meta.yaml").read_text())
    assert [r["acl"] for r in rows if r["doc_id"].startswith("en-wiki")] == [["hr-team"]]


def test_paragraphs_already_indexed_elsewhere_are_skipped(tmp_path):
    imp, _ = make(tmp_path, dup=lambda paras, doc_id: [True, False])
    result = imp.run(**SETTINGS)
    assert result["paragraphs"] == 1 and result["skipped_duplicates"] == 1


@pytest.mark.parametrize("override", [{"acl": []}, {"acl": ["nobody"]}, {"source": "fax"}, {"authority": "meh"}])
def test_bad_settings_are_refused_before_anything_is_written(tmp_path, override):
    imp, synced = make(tmp_path)
    with pytest.raises(ImportRejected):
        imp.run(**{**SETTINGS, **override})
    assert synced == [] and not list(tmp_path.glob("*.json"))


def test_nothing_usable_is_refused(tmp_path):
    imp, synced = make(tmp_path, text="== Only a heading ==")
    with pytest.raises(ImportRejected):
        imp.run(**SETTINGS)
    assert synced == []


def test_import_endpoint_is_admins_only_and_reports_rejections(tmp_path):
    imp, _ = make(tmp_path)
    c = TestClient(create_app(None, Users({"alice": ["all-employees"], "ops": ["all-employees", "admin"]}),
                              lambda: None, imp))  # no pipeline needed: the import routes don't ask questions
    assert c.post("/v1/admin/import", json=SETTINGS).status_code == 401
    assert c.post("/v1/admin/import", json=SETTINGS, headers={"X-User": "alice"}).status_code == 403
    assert c.get("/v1/admin/import/options", headers={"X-User": "alice"}).status_code == 403
    assert c.get("/v1/admin/import/options", headers={"X-User": "ops"}).json()["sources"] == ["wiki", "slack"]
    assert c.post("/v1/admin/import", json={**SETTINGS, "acl": []}, headers={"X-User": "ops"}).status_code == 422
    ok = c.post("/v1/admin/import", json=SETTINGS, headers={"X-User": "ops"})
    assert ok.status_code == 200 and ok.json()["sync"] == {"added": 1, "changed": 0, "meta_only": 0, "deleted": 0,
                                                            "unchanged": 0}
