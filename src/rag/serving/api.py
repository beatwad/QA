"""FastAPI app: POST /v1/ask (SSE stream), POST /v1/admin/sync (admins only), GET /health."""

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.serving.auth import ADMIN_GROUP, Users
from rag.serving.pipeline import Event, Pipeline


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


def sse(events: Iterator[Event]) -> Iterator[str]:
    """Server-Sent Events - push a stream of messages from server to a client over one HTTP response"""
    for kind, data in events:
        yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: Pipeline, users: Users, sync: Callable[[], object]) -> FastAPI:
    app = FastAPI(title="RAG Q&A bot")

    def current_user(x_user: str | None = Header(default=None)) -> tuple[str, list[str]]:
        """Fail closed: missing or unknown user -> 401, before any search runs."""
        groups = users.groups(x_user)
        if groups is None:
            raise HTTPException(401, "unknown user")
        return x_user, groups

    @app.post("/v1/ask")
    def ask(req: AskRequest, who: tuple[str, list[str]] = Depends(current_user)):
        user, groups = who
        return StreamingResponse(sse(pipeline.ask(user, groups, req.question, req.conversation_id)),
                                 media_type="text/event-stream")

    @app.post("/v1/admin/sync")
    def admin_sync(who: tuple[str, list[str]] = Depends(current_user)):
        if ADMIN_GROUP not in who[1]:
            raise HTTPException(403, "admins only")
        report = sync()
        return {name: len(ids) for name, ids in asdict(report).items()}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
