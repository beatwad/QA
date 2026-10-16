"""FastAPI app: POST /v1/ask (SSE stream), POST /v1/admin/sync and /v1/admin/import (admins only), GET /v1/me, GET /health."""

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict

import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.indexing.importer import Importer, ImportRejected
from rag.serving.auth import ADMIN_GROUP, Users
from rag.serving.pipeline import Event, Pipeline


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class ImportRequest(BaseModel):
    url: str
    acl: list[str]
    source: str
    authority: str
    title: str | None = None
    lang: str | None = None


def sse(events: Iterator[Event]) -> Iterator[str]:
    """Server-Sent Events - push a stream of messages from server to a client over one HTTP response"""
    for kind, data in events:
        yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: Pipeline, users: Users, sync: Callable[[], object],
               importer: Importer | None = None) -> FastAPI:
    app = FastAPI(title="RAG Q&A bot")

    def current_user(x_user: str | None = Header(default=None)) -> tuple[str, list[str]]:
        """Fail closed: missing or unknown user -> 401, before any search runs."""
        groups = users.groups(x_user)
        if groups is None:
            raise HTTPException(401, "unknown user")
        return x_user, groups

    @app.get("/v1/me")
    def me(who: tuple[str, list[str]] = Depends(current_user)):
        return {"user": who[0], "groups": who[1]}

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

    if importer:
        def admin_user(who: tuple[str, list[str]] = Depends(current_user)) -> None:
            if ADMIN_GROUP not in who[1]:
                raise HTTPException(403, "admins only")

        @app.get("/v1/admin/import/options", dependencies=[Depends(admin_user)])
        def import_options():
            return importer.options

        @app.post("/v1/admin/import", dependencies=[Depends(admin_user)])
        def admin_import(req: ImportRequest):
            try:
                result = importer.run(**req.model_dump())
            except ImportRejected as e:
                raise HTTPException(422, str(e))
            except requests.RequestException as e:
                raise HTTPException(502, f"download failed: {e}")
            return {**result, "sync": {name: len(ids) for name, ids in asdict(result["sync"]).items()}}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
