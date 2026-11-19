"""Streaming chat API for the cPanel PHP gateway, independent of batch polling.

Run a single process: uvicorn chat_service.main:app --host 127.0.0.1 --port 8010
Only the cPanel server holds this service's bearer key. Browsers use api.php.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from chat_service.config import ChatConfig
from chat_service.inference import ChatRuntime
from chat_service.knowledge import guide_context, load_guides, match_guide
from chat_service.pilot import PilotAccounts, PilotUnavailable, PilotUsage, prompt_token_bound
from chat_service.usage import QuotaExceeded, UsageStore

UNAVAILABLE = "Chat is coming online. Please check back soon."
UPSTREAM_RATE_LIMITED = "Chat has reached its shared usage limit. Please try again later."
SIGN_IN_REQUIRED = "Sign in with your OserAI access code to chat."
LOG = logging.getLogger(__name__)


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


class SignInRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)


def event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(config: ChatConfig | None = None, upstream: httpx.AsyncClient | None = None) -> FastAPI:
    settings = config or ChatConfig.load()
    runtime = ChatRuntime(settings)
    accounts = PilotAccounts(settings)
    guides = load_guides()
    render_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        app.state.upstream = upstream or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0),
        )
        app.state.usage = None if settings.accounts_required else UsageStore(settings.db_path)
        app.state.pilot = PilotUsage(settings) if settings.accounts_required and settings.ready else None
        app.state.active = 0
        try:
            if app.state.pilot is not None:
                await app.state.pilot.start()
            yield
        finally:
            if app.state.pilot is not None:
                await app.state.pilot.close()
            if app.state.usage is not None:
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

    @app.exception_handler(PilotUnavailable)
    async def quota_unavailable(request, exc):
        LOG.warning("OserAI pilot usage storage is unavailable")
        return JSONResponse({"message": UNAVAILABLE}, status_code=503)

    @app.exception_handler(QuotaExceeded)
    async def rate_limited(request, exc):
        message = str(exc) or "You've reached the chat limit for now. Please try again later."
        return JSONResponse({"message": message}, status_code=429, headers={"Retry-After": "60"})

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

    @app.get("/guides/{slug}", response_class=PlainTextResponse, name="read_guide")
    async def read_guide(slug: str):
        guide = guides.get(slug)
        if guide is None:
            raise HTTPException(404, "Guide not found.")
        return f"{guide.title}\n\n{guide.text}\n"

    @app.get("/api/status")
    async def status(authorization: str | None = Header(None), x_chat_account: str = Header(default="", max_length=105)):
        authorize(authorization)
        account = accounts.verify(x_chat_account)
        if not settings.accounts_required or account:
            ready = await model_available()
        else:
            ready = True
        quota = await app.state.pilot.allowance(account) if account else None
        return JSONResponse({
            "ready": ready,
            "model_label": settings.model_label,
            "max_input_chars": min(settings.max_input_chars, 16000),
            "message": "Ready to chat" if ready else UNAVAILABLE,
            "accounts_required": bool(settings.accounts_required),
            "signed_in": account is not None,
            "quota": quota,
        }, headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    async def login(body: SignInRequest, authorization: str | None = Header(None)):
        authorize(authorization)
        proof = accounts.sign_in(body.code)
        if proof is None:
            raise HTTPException(401, "That access code is invalid.")
        return JSONResponse({"account_credential": proof}, headers={"Cache-Control": "no-store"})

    @app.post("/api/chat")
    async def chat(
        body: ChatRequest, request: Request,
        authorization: str | None = Header(None),
        x_visitor_id: str = Header(default="", pattern=r"^[a-f0-9]{64}$"),
        x_visitor_network: str = Header(default="", pattern=r"^[a-f0-9]{64}$"),
        x_chat_account: str = Header(default="", max_length=105),
    ):
        authorize(authorization)
        account = accounts.verify(x_chat_account)
        if settings.accounts_required and account is None:
            raise HTTPException(401, SIGN_IN_REQUIRED)
        if not x_visitor_id or not x_visitor_network:
            raise HTTPException(400, "Please refresh the page and try again.")
        if sum(len(message.content) for message in body.messages) > settings.max_input_chars:
            raise HTTPException(422, "This conversation is full. Please start a new chat.")
        if app.state.active >= settings.max_concurrent:
            raise HTTPException(503, "All chat slots are busy. Please try again shortly.", headers={"Retry-After": "10"})
        if not account and not await model_available():
            raise HTTPException(503, UNAVAILABLE)
        # Recheck after awaiting the upstream model inventory.
        if app.state.active >= settings.max_concurrent:
            raise HTTPException(503, "All chat slots are busy. Please try again shortly.")
        source = match_guide(guides, [message.content for message in body.messages if message.role == "user"])
        messages = [{
            "role": "system",
            "content": (
                "You are OserAI, an AI assistant. Give clear, practical answers. "
                "Say when you are unsure. Ask a short clarifying question "
                "when a missing detail would change the answer. "
                "When asked who you are, introduce yourself as OserAI. "
                f"When asked what powers this chat, say it uses "
                f"{settings.model_name} served by {settings.inference_backend}."
                + guide_context(source)
            ),
        }]
        messages.extend(message.model_dump() for message in body.messages)
        try:
            prompt_bound = prompt_token_bound(messages) if account else 0
        except UnicodeError:
            raise HTTPException(422, "Please check the text in your message and try again.") from None

        request_id = str(uuid.uuid4())
        reply_limit = None
        app.state.active += 1
        try:
            if account:
                reply_limit = await app.state.pilot.reserve(request_id, account, x_visitor_network, prompt_bound)
            else:
                app.state.usage.reserve(request_id, x_visitor_id, x_visitor_network, settings)
        except BaseException:
            app.state.active -= 1
            raise

        async def generate():
            started = time.monotonic()
            outcome, tokens, answer = "interrupted", None, ""
            total_tokens, quota = None, None
            finished = False
            try:
                # Enforce a total deadline as well as HTTP socket timeouts.
                async with asyncio.timeout(settings.timeout_s):
                    yield "start", {"request_id": request_id, "model_label": settings.model_label}
                    async with aclosing(runtime.stream(app.state.upstream, messages, reply_limit)) as chunks:
                        async for chunk in chunks:
                            if chunk.completion_tokens is not None:
                                tokens = chunk.completion_tokens
                            if chunk.total_tokens is not None:
                                total_tokens = chunk.total_tokens
                            if await request.is_disconnected():
                                return
                            if chunk.text:
                                answer += chunk.text
                                if len(answer) > 100000:
                                    raise ValueError("Completion is too large")
                                yield "token", {"text": chunk.text}
                            finished = finished or chunk.finished
                    if not finished or not answer.strip():
                        raise ValueError("Incomplete completion")
                    if source:
                        guide_url = (
                            f"{render_url}/guides/{source.slug}"
                            if render_url else request.url_for("read_guide", slug=source.slug)
                        )
                        yield "token", {"text": f"\n\n**Guide consulted**\n- [{source.title}]({guide_url})"}
                outcome = "completed"
            except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
                outcome = "error"
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404, 422, 429):
                    total_tokens = 0
                if (settings.inference_backend == "groq" and isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 429):
                    yield "error", {"message": UPSTREAM_RATE_LIMITED, "status": 429}
                else:
                    yield "error", {"message": "The reply was interrupted. Please try again in a moment."}
                return
            finally:
                app.state.active -= 1
                if account:
                    try:
                        if total_tokens is not None:
                            if total_tokens > prompt_bound + reply_limit:
                                LOG.warning("Groq reported more tokens than the pilot reserved")
                            await app.state.pilot.finish(request_id, total_tokens)
                        if outcome == "completed":
                            quota = await app.state.pilot.allowance(account)
                    except PilotUnavailable:
                        LOG.warning("OserAI pilot could not reconcile usage. The remaining allowance may be delayed.")
                else:
                    app.state.usage.finish(request_id, outcome, time.monotonic() - started, tokens)
            yield "done", {"completion_tokens": tokens, "elapsed_s": round(time.monotonic() - started, 2), "quota": quota}

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
            async with aclosing(generate()) as frames:
                async for name, payload in frames:
                    yield event(name, payload)

        return StreamingResponse(frames(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
        })

    return app


app = create_app()
