from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from chat_service.config import ChatConfig
from chat_service.main import create_app
from chat_service.pilot import PilotAccounts, PilotUsage, prompt_token_bound
from chat_service.usage import QuotaExceeded
from tests.test_chat_service import GROQ_KEY, GROQ_MODEL, HEADERS, KEY


CODE = "A" * 43
OTHER_CODE = "B" * 43


@pytest.fixture
def pilot_config():
    return ChatConfig(
        gateway_key=KEY, inference_backend="groq", groq_api_key=GROQ_KEY, model_name=GROQ_MODEL,
        db_path=":memory:", accounts_required=1, max_output_tokens=512,
        pilot_accounts=json.dumps({
            "alice": hashlib.sha256(CODE.encode()).hexdigest(),
            "bob": hashlib.sha256(OTHER_CODE.encode()).hexdigest(),
        }),
    )


@pytest.fixture
def postgres_url():
    url = os.getenv("OSERAI_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set OSERAI_TEST_POSTGRES_URL to run the PostgreSQL integration tests")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import make_conninfo
    from psycopg.sql import SQL, Identifier

    schema = "pilot_test_" + uuid.uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
        try:
            yield make_conninfo(url, options=f"-c search_path={schema}")
        finally:
            db.execute(SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema)))


def test_codes_are_private_and_rotating_one_revokes_its_sessions(pilot_config):
    accounts = PilotAccounts(pilot_config)
    assert accounts.sign_in("wrong") is None
    assert accounts.verify("alice." + "a" * 64) is None
    alice = accounts.sign_in(CODE)
    bob = accounts.sign_in(OTHER_CODE)
    assert CODE not in alice
    assert accounts.verify(alice) == "alice"
    assert accounts.verify(bob) == "bob"
    digests = json.loads(pilot_config.pilot_accounts)
    rotated_code = "C" * 43
    digests["alice"] = hashlib.sha256(rotated_code.encode()).hexdigest()
    rotated = PilotAccounts(replace(pilot_config, pilot_accounts=json.dumps(digests)))
    assert rotated.verify(alice) is None
    assert rotated.verify(bob) == "bob"
    assert rotated.verify(rotated.sign_in(rotated_code)) == "alice"


def test_pilot_fails_closed_without_durable_storage(pilot_config):
    with TestClient(create_app(pilot_config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/status", headers=HEADERS).status_code == 503
        assert client.post("/api/login", headers=HEADERS, json={"code": CODE}).status_code == 503


def client_for(settings, calls, modes=None):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": GROQ_MODEL}]})
        calls.append(json.loads(request.content))
        mode = next(modes, "normal") if modes else "normal"
        if mode == "rate-limited":
            return httpx.Response(429, json={"error": {"message": "private provider detail"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Answer 🌱"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 201, "completion_tokens": 23, "total_tokens": 224}
            if mode != "unknown-usage" else {},
        })

    return TestClient(create_app(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))))


def login(client, code):
    result = client.post("/api/login", headers=HEADERS, json={"code": code})
    assert result.status_code == 200, result.text
    return {**HEADERS, "X-Chat-Account": result.json()["account_credential"]}


def talk(client, headers, text="Hello", stream=False):
    return client.post("/api/chat", headers=headers, json={
        "stream": stream, "messages": [{"role": "user", "content": text}],
    })


def test_api_admission_debits_total_tokens_across_devices_and_restarts(pilot_config, postgres_url):
    settings = replace(pilot_config, database_url=postgres_url, user_tokens_per_day=1000)
    calls = []
    with client_for(settings, calls) as client:
        guest = client.get("/api/status", headers=HEADERS).json()
        assert guest["accounts_required"] is True and guest["signed_in"] is False and guest["quota"] is None
        assert talk(client, HEADERS).status_code == 401
        assert client.post("/api/login", json={"code": CODE}).status_code == 401
        assert client.post("/api/login", headers=HEADERS, json={"code": "wrong"}).status_code == 401
        assert not calls

        alice = login(client, CODE)
        first = talk(client, alice)
        assert first.status_code == 200, first.text
        assert first.json()["quota"]["remaining"] == 776
        assert first.json()["quota"]["resets_at"].endswith("T00:00:00+00:00")
        assert talk(client, alice, text="x" * 1000).status_code == 429
        assert len(calls) == 1
        assert 64 <= calls[0]["max_completion_tokens"] <= 512

    with client_for(settings, calls) as restarted:
        phone = {**login(restarted, CODE), "X-Visitor-ID": "3" * 64, "X-Visitor-Network": "4" * 64}
        bob = login(restarted, OTHER_CODE)
        assert restarted.get("/api/status", headers=phone).json()["quota"]["remaining"] == 776
        assert restarted.get("/api/status", headers=bob).json()["quota"]["remaining"] == 1000
        streamed = talk(restarted, phone, stream=True)
        assert streamed.status_code == 200 and "event: done" in streamed.text
        assert '"remaining": 552' in streamed.text
        assert calls[-1]["max_completion_tokens"] < calls[0]["max_completion_tokens"]


def test_unknown_groq_usage_holds_the_reservation(pilot_config, postgres_url):
    settings = replace(pilot_config, database_url=postgres_url)
    calls = []
    modes = iter(["rate-limited", "unknown-usage"])
    with client_for(settings, calls, modes) as client:
        headers = login(client, CODE)
        assert talk(client, headers).status_code == 429
        assert client.get("/api/status", headers=headers).json()["quota"]["remaining"] == 5000
        answer = talk(client, headers, text="How do I add my Mac as an OserAI worker?")
        assert answer.status_code == 200
        sent = calls[-1]
        assert "parallax join" in sent["messages"][0]["content"]
        reserved = prompt_token_bound(sent["messages"]) + sent["max_completion_tokens"]
        assert answer.json()["quota"]["remaining"] == 5000 - reserved


@pytest.mark.parametrize("limit", ["shared_tokens_per_minute", "shared_tokens_per_day", "user_tokens_per_day"])
def test_reservations_are_atomic_between_processes(pilot_config, postgres_url, limit):
    config = replace(pilot_config, database_url=postgres_url, max_output_tokens=300, **{limit: 550})

    async def check():
        first, second = PilotUsage(config), PilotUsage(config)
        try:
            await first.start()
            await second.start()
            account = "alice" if limit == "user_tokens_per_day" else "bob"
            results = await asyncio.gather(
                first.reserve(str(uuid.uuid4()), "alice", "x", 200),
                second.reserve(str(uuid.uuid4()), account, "y", 200),
                return_exceptions=True,
            )
            assert results.count(300) == 1
            assert sum(isinstance(result, QuotaExceeded) for result in results) == 1
        finally:
            await first.close()
            await second.close()

    asyncio.run(check())
