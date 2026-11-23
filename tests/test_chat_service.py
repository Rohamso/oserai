from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from chat_service.config import GUIDE_LABEL, GUIDE_MODEL, ChatConfig
from chat_service.main import create_app
from scripts.package_chat_site import FILES, FRONTEND_FILES, package

KEY = "test-gateway-key-for-the-test-suite-only-12345"
HEADERS = {"Authorization": f"Bearer {KEY}", "X-Visitor-ID": "a" * 64, "X-Visitor-Network": "b" * 64}
MODEL = "test/chat-model"
GROQ_KEY = "test-groq-provider-key"
GROQ_MODEL = "qwen/qwen3.8-27b"


def stream(*texts, usage=True, finish=True):
    frames = ["data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) for text in texts]
    if finish:
        frames.append('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}')
        if usage:
            frames.append('data: {"choices":[],"usage":{"completion_tokens":7}}')
        frames.append("data: [DONE]")
    return ("\n\n".join(frames) + "\n\n").encode()


@pytest.fixture
def config():
    return ChatConfig(gateway_key=KEY, model_name=MODEL, db_path=":memory:")


def make_client(config, content=None, status=200, model=MODEL, requests=None):
    def handler(request):
        if requests is not None:
            requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model}]})
        return httpx.Response(status, content=content if content is not None else stream("Hello ", "world 🌱"), headers={"Content-Type": "text/event-stream"})
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(create_app(config, upstream))


def test_chat_stream_preserves_conversation_and_records_real_usage(config):
    sent = []
    messages = [{"role": "user", "content": "My name is Ada."}, {"role": "assistant", "content": "Hi Ada."}, {"role": "user", "content": "What's my name?"}]
    with make_client(config, requests=sent) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": messages})
        assert result.status_code == 200
        assert "text/event-stream" in result.headers["content-type"]
        assert 'event: token' in result.text and "world 🌱" in result.text
        assert 'event: done' in result.text
        assert '"completion_tokens": 7' in result.text
        assert "Guide consulted" not in result.text
        payload = json.loads(sent[-1].content)
        system_message, *history = payload["messages"]
        assert system_message["role"] == "system"
        assert "OserAI guide excerpt" not in system_message["content"]
        assert history == messages
        assert payload["stream"] is True
        assert payload["model"] == MODEL
        assert payload["stream_options"]["include_usage"] is True
        record = client.app.state.usage.connection.execute("SELECT status, completion_tokens FROM chat_requests").fetchone()
        assert record == ("completed", 7)
        assert client.app.state.active == 0


@pytest.mark.parametrize(("questions", "slug", "excerpt", "streaming"), [
    (["How do I add my Mac as a worker?"], "parallax-cluster", "parallax join", False),
    (["How do I join your network?"], "parallax-cluster", "parallax join", False),
    (["My worker can't find the scheduler."], "parallax-discovery", "Privacy & Security → Local Network", True),
    (["How do I add my Mac as a worker?", "What about on Linux?"], "parallax-cluster", "Linux GPUs", False),
    (["Where do I get the OserAI Trainer run ID?"], "trainer-worker", "whoever is", False),
])
def test_chat_consults_and_cites_a_relevant_guide(config, monkeypatch, questions, slug, excerpt, streaming):
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://pilot.example")
    sent = []
    history = [{"role": "user", "content": questions[0]}]
    if len(questions) > 1:
        history.extend([{"role": "assistant", "content": "Try it on the local network."},
                        {"role": "user", "content": questions[1]}])

    with make_client(config, requests=sent) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": history, "stream": streaming})
        assert result.status_code == 200
        source = client.get(f"/guides/{slug}")
        assert source.status_code == 200
        assert source.headers["content-type"].startswith("text/plain")
        assert excerpt in source.text
        assert client.get("/guides/chat-cpanel").status_code == 404

    system_message, *forwarded = json.loads(sent[-1].content)["messages"]
    assert system_message["role"] == "system"
    assert excerpt in system_message["content"]
    assert forwarded == history
    reply = result.text if streaming else result.json()["text"]
    assert "Guide consulted" in reply
    assert f"https://pilot.example/guides/{slug}" in reply
    if streaming:
        assert reply.index("Guide consulted") < reply.index("event: done")


def test_unrelated_worker_question_does_not_use_project_docs(config):
    sent = []
    with make_client(config, requests=sent) as client:
        result = client.post("/api/chat", headers=HEADERS, json={
            "messages": [{"role": "user", "content": "How do I install a worker for Celery?"}],
            "stream": False,
        })
    assert result.json()["text"] == "Hello world 🌱"
    assert "OserAI guide excerpt" not in json.loads(sent[-1].content)["messages"][0]["content"]


def test_guide_preview_uses_its_own_endpoint_and_key(config):
    guide_key = "g" * 42
    config = replace(config, inference_backend="groq", groq_api_key=GROQ_KEY, model_name=GROQ_MODEL,
                     guide_url="https://guide.example/v1/", guide_api_key=guide_key)
    sent = []

    def handler(request):
        sent.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": GROQ_MODEL}]})
        is_guide = request.url.host == "guide.example"
        return httpx.Response(200, json={
            "model": GUIDE_MODEL if is_guide else GROQ_MODEL,
            "choices": [{"message": {"content": "A real reply"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 44, "completion_tokens": 3, "total_tokens": 47},
        })

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(config, upstream)) as client:
        preview = client.post("/api/chat", headers=HEADERS, json={"stream": False, "messages": [
            {"role": "user", "content": "/guide How do I add my Mac as a worker?"},
        ]})
        regular = client.post("/api/chat", headers=HEADERS, json={"stream": False, "messages": [
            {"role": "user", "content": "Hello"},
        ]})
    assert preview.status_code == regular.status_code == 200
    assert preview.json()["model_label"] == GUIDE_LABEL
    assert preview.json()["text"].startswith(f"**{GUIDE_LABEL} (preview)**\n\nA real reply")
    assert preview.json()["completion_tokens"] == 3
    assert "Guide consulted" in preview.json()["text"]
    assert regular.json()["text"] == "A real reply"
    guide_request, inventory, regular_request = sent
    assert str(guide_request.url) == "https://guide.example/v1/chat/completions"
    assert guide_request.headers["authorization"] == f"Bearer {guide_key}"
    payload = json.loads(guide_request.content)
    assert payload["model"] == GUIDE_MODEL and payload["max_completion_tokens"] == 160
    assert payload["messages"][-1]["content"] == "How do I add my Mac as a worker?"
    assert "parallax join" in payload["messages"][0]["content"]
    assert inventory.url.host == regular_request.url.host == "api.groq.com"
    assert regular_request.headers["authorization"] == f"Bearer {GROQ_KEY}"


def test_unconfigured_guide_preview_returns_unavailable_without_calling_a_provider(config):
    sent = []
    with make_client(config, requests=sent) as client:
        response = client.post("/api/chat", headers=HEADERS, json={"messages": [
            {"role": "user", "content": "/guide What is OserAI?"},
        ]})
    assert response.status_code == 503
    assert "preview is unavailable" in response.json()["message"]
    assert sent == []


@pytest.mark.parametrize("content,status", [
    (stream("Partial reply", finish=False), 200),
    (b"data: invalid-json\n\n", 200),
    (b'data: {"error":{"message":"PRIVATE UPSTREAM DETAILS"}}\n\n', 200),
    (b"private engine traceback", 503),
    (stream("", usage=False), 200),
])
def test_failed_streams_release_capacity_and_do_not_claim_completion(config, content, status):
    with make_client(config, content=content, status=status) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}]})
        assert "event: error" in result.text
        assert "event: done" not in result.text
        assert "PRIVATE" not in result.text and "traceback" not in result.text
        assert client.app.state.active == 0
        assert client.app.state.usage.connection.execute("SELECT status FROM chat_requests").fetchone() == ("error",)


def test_missing_token_counts_remain_unknown_instead_of_estimated(config):
    with make_client(config, content=stream("A real reply", usage=False)) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}]})
        assert '"completion_tokens": null' in result.text
        assert client.app.state.usage.connection.execute("SELECT completion_tokens FROM chat_requests").fetchone() == (None,)


def test_parallax_supports_completed_json_transport_without_changing_runtime_protocol(config):
    sent = []
    with make_client(config, requests=sent) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert result.status_code == 200
        assert result.json()["text"] == "Hello world 🌱"
        assert result.json()["completion_tokens"] == 7
        assert json.loads(sent[-1].content)["stream"] is True
        assert client.app.state.active == 0


def test_gateway_requires_private_bearer_key(config):
    with make_client(config) as client:
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        status = client.get("/api/status", headers=HEADERS)
        assert status.json()["ready"] is True
        assert KEY not in status.text and MODEL not in status.text


def test_unconfigured_gateway_is_explicitly_unavailable(config):
    with make_client(replace(config, gateway_key="")) as client:
        assert client.get("/health").json() == {"ok": True}
        result = client.get("/api/status", headers=HEADERS)
        assert result.status_code == 503
        assert "coming online" in result.json()["message"]


def test_wrong_model_does_not_consume_chat_request(config):
    sent = []
    with make_client(config, model="other/model", requests=sent) as client:
        assert client.get("/api/status", headers=HEADERS).json()["ready"] is False
        assert client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503
        assert all(request.method == "GET" for request in sent)
        assert client.app.state.usage.connection.execute("SELECT COUNT(*) FROM chat_requests").fetchone() == (0,)


def test_rate_limit_persists_across_separate_requests(config):
    with make_client(replace(config, requests_per_minute=1)) as client:
        first = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}]})
        second = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "again"}]})
        assert "event: done" in first.text
        assert second.status_code == 429
        assert second.headers["retry-after"] == "60"


def test_busy_gateway_returns_useful_capacity_error(config):
    with make_client(config) as client:
        client.app.state.active = config.max_concurrent
        result = client.post("/api/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "hi"}]})
        assert result.status_code == 503
        assert "busy" in result.json()["message"]


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "SECRET INVALID INPUT"}], [{"role": "assistant", "content": "SECRET INVALID INPUT"}], [{"role": "user", "content": "x" * 16001}]])
def test_invalid_history_never_echoes_private_message_content(config, messages):
    with make_client(config) as client:
        result = client.post("/api/chat", headers=HEADERS, json={"messages": messages})
        assert result.status_code == 422
        assert "SECRET INVALID INPUT" not in result.text


def test_cpanel_archive_contains_only_public_assets(tmp_path):
    import zipfile
    output = tmp_path / "chat.zip"
    package(output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(FILES)
        assert all(entry.create_system == 3 and (entry.external_attr >> 16) & 0o777 == 0o644
                   for entry in archive.infolist())
        assert "index.html" in archive.namelist()
        assert not any(name.endswith(".env") or "config.php" in name for name in archive.namelist())
        assert "csrf_token" in archive.read("api.php").decode()

    package(output, frontend_only=True)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(FRONTEND_FILES)
        assert {"index.html", ".htaccess", "assets/app.js"} <= set(archive.namelist())
        assert "api.php" not in archive.namelist() and ".user.ini" not in archive.namelist()
        assert all(entry.create_system == 3 and (entry.external_attr >> 16) & 0o777 == 0o644
                   for entry in archive.infolist())


@pytest.fixture
def groq_config(config):
    return replace(config, inference_backend="groq", groq_api_key=GROQ_KEY, model_name=GROQ_MODEL)


def groq_client(config, handler):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(create_app(config, upstream))


@pytest.mark.parametrize("streaming", [True, False])
def test_groq_authenticates_and_preserves_history_and_usage(groq_config, streaming):
    messages = [
        {"role": "user", "content": "Remember the word cedar."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Which word?"},
    ]
    paths = []

    def handler(request):
        assert request.url.host == "api.groq.com"
        assert request.headers["Authorization"] == "Bearer " + GROQ_KEY
        assert "X-Visitor-ID" not in request.headers
        paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": GROQ_MODEL}]})
        payload = json.loads(request.content)
        system_message = payload["messages"][0]
        assert system_message["role"] == "system"
        assert "You are OserAI" in system_message["content"]
        assert f"{GROQ_MODEL} served by groq" in system_message["content"]
        assert payload == {
            "model": GROQ_MODEL, "messages": [system_message, *messages], "stream": False,
            "max_completion_tokens": groq_config.max_output_tokens, "temperature": 0.7,
        }
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Cedar.", "reasoning": "Provider scratchpad"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        })

    with groq_client(groq_config, handler) as client:
        status = client.get("/api/status", headers=HEADERS)
        assert status.json()["ready"] is True
        result = client.post("/api/chat", headers=HEADERS, json={"messages": messages, "stream": streaming})
        assert result.status_code == 200
        assert GROQ_KEY not in status.text + result.text
        assert "Provider scratchpad" not in result.text
        if streaming:
            assert 'event: done' in result.text
            assert '"text": "Cedar."' in result.text
        else:
            assert result.json()["text"] == "Cedar."
            assert result.json()["completion_tokens"] == 3
        assert paths == ["/openai/v1/models", "/openai/v1/models", "/openai/v1/chat/completions"]
        usage = client.app.state.usage.connection.execute("SELECT status, completion_tokens FROM chat_requests").fetchone()
        assert usage == ("completed", 3)
        assert client.app.state.active == 0


def test_groq_key_is_required_and_loaded_from_the_environment(monkeypatch):
    monkeypatch.setenv("OSERAI_CHAT_INFERENCE_BACKEND", "groq")
    monkeypatch.setenv("OSERAI_CHAT_GATEWAY_KEY", KEY)
    monkeypatch.setenv("OSERAI_CHAT_MODEL_NAME", GROQ_MODEL)
    monkeypatch.setenv("OSERAI_CHAT_DB_PATH", ":memory:")
    monkeypatch.delenv("OSERAI_CHAT_GROQ_API_KEY", raising=False)

    def handler(request):
        pytest.fail("An unconfigured gateway must not contact Groq")

    with groq_client(ChatConfig.load(), handler) as client:
        assert client.get("/api/status", headers=HEADERS).status_code == 503
    monkeypatch.setenv("OSERAI_CHAT_GROQ_API_KEY", GROQ_KEY)
    config = ChatConfig.load()
    assert config.ready
    assert GROQ_KEY not in repr(config)


@pytest.mark.parametrize(("code", "body", "expected"), [
    (429, {"error": {"message": "Private Groq quota details"}}, 429),
    (200, {"choices": []}, 503),
    (200, {"choices": [{"message": {"content": None, "reasoning": "Private scratchpad"}, "finish_reason": "length"}]}, 503),
])
def test_groq_failure_is_sanitized_and_releases_capacity(groq_config, code, body, expected):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": GROQ_MODEL}]})
        return httpx.Response(code, json=body)

    with groq_client(groq_config, handler) as client:
        result = client.post("/api/chat", headers=HEADERS, json={
            "messages": [{"role": "user", "content": "Hello"}], "stream": False,
        })
        assert result.status_code == expected
        assert "Private" not in result.text
        if expected == 429:
            assert "shared usage limit" in result.json()["message"]
        assert client.app.state.active == 0
        usage = client.app.state.usage.connection.execute("SELECT status, completion_tokens FROM chat_requests").fetchone()
        assert usage == ("error", None)
