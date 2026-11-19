"""Streaming chat API for the cPanel PHP gateway, independent of batch polling.

Run a single process: uvicorn chat_service.main:app --host 127.0.0.1 --port 8010
Only the cPanel server holds this service's bearer key. Browsers use api.php.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from chat_service.config import ChatConfig
from chat_service.inference import ChatRuntime
from chat_service.usage import QuotaExceeded, UsageStore

UNAVAILABLE = "Chat is coming online. Please check back soon."
UPSTREAM_RATE_LIMITED = "Chat has reached its shared usage limit. Please try again later."


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=64)
    stream: bool = True

    @model_validator(mode="after")
    def conversation(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must be from the user")
        if sum(len(message.content) for message in self.messages) > 32000:
            raise ValueError("Conversation is too long")
        return self


def event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(config: ChatConfig | None = None, upstream: httpx.AsyncClient | None = None) -> FastAPI:
    settings = config or ChatConfig.load()
    runtime = ChatRuntime(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.upstream = upstream or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0),
        )
        app.state.usage = UsageStore(settings.db_path)
        app.state.active = 0
        try:
            yield
        finally:
            app.state.usage.close()
            if upstream is None:
                await app.state.upstream.aclose()

    app = FastAPI(title="OserAI Chat", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Validation diagnostics can include submitted messages; don't echo them.
        return JSONResponse({"message": "Please shorten your message or start a new conversation."}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"message": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    def authorize(authorization: str | None = Header(None)):
        if not settings.ready:
            raise HTTPException(503, UNAVAILABLE)
        supplied = (authorization or "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied.encode(), settings.gateway_key.encode()):
            raise HTTPException(401, "This connection is not authorized.")

    async def model_available() -> bool:
        return await runtime.model_available(app.state.upstream)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/api/status")
    async def status(authorization: str | None = Header(None)):
        authorize(authorization)
        ready = await model_available()
        return JSONResponse({
            "ready": ready,
            "model_label": settings.model_label,
            "max_input_chars": min(settings.max_input_chars, 16000),
            "message": "Ready to chat" if ready else UNAVAILABLE,
        }, headers={"Cache-Control": "no-store"})

    @app.post("/api/chat")
    async def chat(
        body: ChatRequest, request: Request,
        authorization: str | None = Header(None),
        x_visitor_id: str = Header(default="", pattern=r"^[a-f0-9]{64}$"),
        x_visitor_network: str = Header(default="", pattern=r"^[a-f0-9]{64}$"),
    ):
        authorize(authorization)
        if not x_visitor_id or not x_visitor_network:
            raise HTTPException(400, "Please refresh the page and try again.")
        if sum(len(message.content) for message in body.messages) > settings.max_input_chars:
            raise HTTPException(422, "This conversation is full. Please start a new chat.")
        if app.state.active >= settings.max_concurrent:
            raise HTTPException(503, "All chat slots are busy. Please try again shortly.", headers={"Retry-After": "10"})
        if not await model_available():
            raise HTTPException(503, UNAVAILABLE)
        # Recheck after awaiting the upstream model inventory.
        if app.state.active >= settings.max_concurrent:
            raise HTTPException(503, "All chat slots are busy. Please try again shortly.")
        request_id = str(uuid.uuid4())
        try:
            app.state.usage.reserve(request_id, x_visitor_id, x_visitor_network, settings)
        except QuotaExceeded:
            raise HTTPException(429, "You've reached the chat limit for now. Please try again later.", headers={"Retry-After": "60"})
        app.state.active += 1

        async def generate():
            started = time.monotonic()
            outcome, tokens, answer = "interrupted", None, ""
            finished = False
            try:
                # Enforce a total deadline as well as HTTP socket timeouts.
                async with asyncio.timeout(settings.timeout_s):
                    yield "start", {"request_id": request_id, "model_label": settings.model_label}
                    async with aclosing(runtime.stream(app.state.upstream, [message.model_dump() for message in body.messages])) as chunks:
                        async for chunk in chunks:
                            if await request.is_disconnected():
                                return
                            if chunk.completion_tokens is not None:
                                tokens = chunk.completion_tokens
                            if chunk.text:
                                answer += chunk.text
                                if len(answer) > 100000:
                                    raise ValueError("Completion is too large")
                                yield "token", {"text": chunk.text}
                            finished = finished or chunk.finished
                    if not finished or not answer.strip():
                        raise ValueError("Incomplete completion")
                outcome = "completed"
                yield "done", {"completion_tokens": tokens, "elapsed_s": round(time.monotonic() - started, 2)}
            except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
                outcome = "error"
                if (settings.inference_backend == "groq" and isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 429):
                    yield "error", {"message": UPSTREAM_RATE_LIMITED, "status": 429}
                else:
                    yield "error", {"message": "The reply was interrupted. Please try again in a moment."}
            finally:
                app.state.active -= 1
                app.state.usage.finish(request_id, outcome, time.monotonic() - started, tokens)

        if not body.stream:
            result = {"text": ""}
            failure = None
            async for name, payload in generate():
                if name == "error":
                    failure = payload
                elif name == "token":
                    result["text"] += payload["text"]
                else:
                    result.update(payload)
            if failure:
                return JSONResponse({"message": failure["message"]}, status_code=failure.get("status", 503))
            if not result["text"] or "elapsed_s" not in result:
                return JSONResponse({"message": "The reply was interrupted. Please try again."}, status_code=503)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})

        async def frames():
            async for name, payload in generate():
                yield event(name, payload)

        return StreamingResponse(frames(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
        })

    return app


app = create_app()
