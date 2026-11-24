from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from agent import parallax_client
from agent.parallax_client import ParallaxError, run_inference

MODEL = "Qwen/Qwen3-0.6B"
ANSWER = {"choices": [{"message": {"content": "Paris."}}], "usage": {"completion_tokens": 3}}


def response(payload, status=200):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "http://parallax/v1/chat/completions"))


@pytest.fixture
def serving_model(monkeypatch):
    def get(url, timeout):
        assert url == "http://parallax/v1/models"
        assert 0 < timeout <= 10
        return response({"data": [{"id": MODEL}]})
    monkeypatch.setattr(httpx, "get", get)


@pytest.mark.parametrize("url", ["http://parallax", "http://parallax/", "http://parallax/v1/"])
def test_completion_contract_and_end_to_end_metrics(monkeypatch, serving_model, url):
    clock = iter([1.0, 1.25, 3.0])
    monkeypatch.setattr(parallax_client, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    def post(url, json, timeout):
        assert url == "http://parallax/v1/chat/completions"
        assert json == {
            "model": MODEL, "messages": [{"role": "user", "content": "Capital of France?"}],
            "max_tokens": 50, "temperature": 0, "seed": 42, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        assert timeout == 29.75
        return response(ANSWER)
    monkeypatch.setattr(httpx, "post", post)

    result = run_inference(url, MODEL, "Capital of France?", 50, 30)
    assert result.output_text == "Paris."
    assert result.tokens_generated == 3
    assert result.duration_s == 2.0
    assert result.tokens_per_sec == 1.5


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("timed out"), 503, 429])
def test_upstream_failures_are_retryable_agent_errors(monkeypatch, serving_model, failure):
    def post(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return response({"error": {"message": "not ready"}}, failure)
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ParallaxError, match="Parallax request failed"):
        run_inference("http://parallax", MODEL, "hi", 50, 30)


@pytest.mark.parametrize("payload", [
    {}, {"choices": []}, {"choices": None}, {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 1}},
    {"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": -1}},
    {"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": True}},
    {"choices": [{"message": {"content": "hi"}}], "usage": {}},
    {"error": {"message": "model unavailable"}},
])
def test_bad_completions_do_not_become_successful_billable_jobs(monkeypatch, serving_model, payload):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response(payload))
    with pytest.raises(ParallaxError, match="Invalid Parallax completion response"):
        run_inference("http://parallax", MODEL, "hi", 50, 30)


def test_invalid_json_is_an_inference_error(monkeypatch, serving_model):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(
        200, text="not JSON", request=httpx.Request("POST", "http://parallax/v1/chat/completions"),
    ))
    with pytest.raises(ParallaxError, match="Invalid Parallax completion response"):
        run_inference("http://parallax", MODEL, "hi", 50, 30)


def test_wrong_cluster_model_never_submits_completion(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: response({"data": [{"id": "different-model"}]}))
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("wrong model was submitted"))
    with pytest.raises(ParallaxError, match="not serving the configured model"):
        run_inference("http://parallax", MODEL, "hi", 50, 30)


@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": [None]}])
def test_bad_model_inventory_is_an_inference_error(monkeypatch, payload):
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: response(payload))
    with pytest.raises(ParallaxError, match="Invalid Parallax /v1/models response"):
        run_inference("http://parallax", MODEL, "hi", 50, 30)
