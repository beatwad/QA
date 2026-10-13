import json
from datetime import date

from fastapi.testclient import TestClient

from rag.serving.api import create_app
from rag.serving.auth import Users
from rag.config import load_config
from rag.serving.conversations import ConversationStore
from rag.indexing.ingest import SyncReport
from rag.serving.pipeline import Pipeline

from fakes import FakeLLM, FakeReranker, FakeRetriever, live_hit, words

USERS = Users({"alice": ["all-employees"], "ops": ["all-employees", "admin"]})


def client(synced=None):
    pipeline = Pipeline(load_config(), FakeRetriever([live_hit(1, "Tesla died in 1943.")]), FakeReranker(5.0),
                        FakeLLM("Tesla died in 1943 [1]."), words, ConversationStore(),
                        today=lambda: date(2026, 10, 1))

    def sync():
        synced.append(True)
        return SyncReport(added=["a", "b"], unchanged=["c"])

    return TestClient(create_app(pipeline, USERS, sync))


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        kind, data = block.split("\n")
        events.append((kind.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_missing_or_unknown_user_is_refused():
    c = client()
    assert c.post("/v1/ask", json={"question": "q"}).status_code == 401
    assert c.post("/v1/ask", json={"question": "q"}, headers={"X-User": "mallory"}).status_code == 401


def test_ask_streams_sse_events_in_order():
    resp = client().post("/v1/ask", json={"question": "What year did Tesla die?"}, headers={"X-User": "alice"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and "token" in kinds and kinds[-2:] == ["result", "done"]
    assert events[-2][1]["citations"][0]["url"] == "https://wiki/d1"


def test_admin_sync_is_admins_only():
    synced = []
    c = client(synced)
    assert c.post("/v1/admin/sync").status_code == 401
    assert c.post("/v1/admin/sync", headers={"X-User": "alice"}).status_code == 403
    assert synced == []
    resp = c.post("/v1/admin/sync", headers={"X-User": "ops"})
    assert resp.status_code == 200 and synced == [True]
    assert resp.json() == {"added": 2, "changed": 0, "meta_only": 0, "deleted": 0, "unchanged": 1}


def test_health():
    assert client().get("/health").json() == {"status": "ok"}
