from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from chat_service.config import ChatConfig
from chat_service.main import create_app

KEY = "local-test-key-only-never-a-real-deployment-secret"
HEADERS = {"Authorization": "Bearer " + KEY, "X-Visitor-ID": "a" * 64, "X-Visitor-Network": "b" * 64}


def client_for(content=None, model="qwen3:4b", sent=None):
    config = ChatConfig(gateway_key=KEY, inference_backend="ollama", model_name="qwen3:4b", db_path=":memory:", max_concurrent=1, max_output_tokens=512)
    if content is None:
        content = b'\n'.join(json.dumps(chunk, ensure_ascii=False).encode() for chunk in [
            {"message": {"thinking": "Do not expose this internal text.", "content": "Hello "}, "done": False},
            {"message": {"content": "world 🌱"}, "done": False},
            {"message": {"content": "!"}, "done": True, "eval_count": 8},
        ])

    def handler(request):
        if sent is not None:
            sent.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": model}]})
        assert request.url.path == "/api/chat"
        return httpx.Response(200, content=content, headers={"Content-Type": "application/x-ndjson"})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(create_app(config, upstream))


@pytest.mark.parametrize("streaming", [True, False])
def test_ollama_conversation_thinking_context_and_final_usage(streaming):
    sent = []
    messages = [{"role": "user", "content": "I am Ada."}, {"role": "assistant", "content": "Hi Ada."}, {"role": "user", "content": "My name?"}]
    with client_for(sent=sent) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": messages, "stream": streaming})
        assert result.status_code == 200
        assert "internal text" not in result.text
        payload = json.loads(sent[-1].content)
        system_message, *history = payload["messages"]
        assert system_message["role"] == "system"
        assert history == messages
        assert payload["think"] is False
        assert payload["options"] == {"num_ctx": 4096, "num_predict": 512, "temperature": 0.7}
        assert payload["keep_alive"] == "10m"
        if streaming:
            assert "event: done" in result.text and '"completion_tokens": 8' in result.text
        else:
            assert result.json()["text"] == "Hello world 🌱!"
            assert result.json()["completion_tokens"] == 8
            assert result.json()["elapsed_s"] >= 0
            assert "request_id" in result.json()
        assert client.app.state.active == 0
        assert client.app.state.usage.connection.execute("SELECT status, completion_tokens FROM chat_requests").fetchone() == ("completed", 8)


@pytest.mark.parametrize("content", [
    b'{"message":{"content":"partial"},"done":false}\n',
    b'{"error":"PRIVATE CUDA FAILURE"}\n',
    b'not-json\n',
    b'{"message":{"content":""},"done":true,"eval_count":0}\n',
])
@pytest.mark.parametrize("streaming", [True, False])
def test_ollama_failed_replies_are_not_successful_or_leaked(content, streaming):
    with client_for(content) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "Hi"}], "stream": streaming})
        assert "PRIVATE" not in result.text
        assert "event: done" not in result.text
        if streaming:
            assert "event: error" in result.text
        else:
            assert result.status_code == 503
            assert "text" not in result.json()
        assert client.app.state.active == 0
        assert client.app.state.usage.connection.execute("SELECT status FROM chat_requests").fetchone() == ("error",)


def test_ollama_inventory_requires_the_exact_model_tag():
    sent = []
    with client_for(model="qwen3:8b", sent=sent) as client:
        assert client.get("/api/status", headers=HEADERS).json()["ready"] is False
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status_code == 503
        assert all(request.method == "GET" for request in sent)


def test_ollama_missing_counts_remain_unknown():
    with client_for(b'{"message":{"content":"Hello"},"done":true}\n') as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "Hi"}], "stream": False})
        assert result.json()["completion_tokens"] is None


def test_aborted_generation_closes_the_upstream_before_releasing_capacity():
    closed = []

    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps({"message": {"content": "x" * 100001}, "done": False}).encode() + b"\n"

        async def aclose(self):
            closed.append(True)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
        return httpx.Response(200, stream=OversizedStream())

    config = ChatConfig(gateway_key=KEY, inference_backend="ollama", model_name="qwen3:4b", db_path=":memory:")
    with TestClient(create_app(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "Hi"}], "stream": False})
        assert result.status_code == 503
        assert closed == [True]
        assert client.app.state.active == 0


def test_config_loads_ollama_and_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("OSERAI_CHAT_INFERENCE_BACKEND", "ollama")
    assert ChatConfig.load().inference_backend == "ollama"
    monkeypatch.setenv("OSERAI_CHAT_INFERENCE_BACKEND", "unrecognized")
    with pytest.raises(ValueError, match="parallax, ollama, groq"):
        ChatConfig.load()
