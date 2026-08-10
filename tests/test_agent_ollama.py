from __future__ import annotations

import httpx
import pytest

from agent.ollama_client import OllamaError, run_inference


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self) -> dict:
        return self._payload


def test_run_inference_parses_ollama_response(monkeypatch):
    def fake_post(url, json, timeout):
        assert json["options"]["temperature"] == 0
        assert json["options"]["seed"] == 42
        return _FakeResponse({
            "response": "The water cycle moves water through evaporation and condensation.",
            "eval_count": 12,
            "eval_duration": 2_000_000_000,  # 2s in ns
        })

    monkeypatch.setattr(httpx, "post", fake_post)

    result = run_inference("http://localhost:11434", "llama3.1:8b", "explain the water cycle", 100, 30)

    assert result.tokens_generated == 12
    assert result.duration_s == 2.0
    assert result.tokens_per_sec == 6.0
    assert "water cycle" in result.output_text


def test_run_inference_raises_ollama_error_on_transport_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaError):
        run_inference("http://localhost:11434", "llama3.1:8b", "hi", 100, 30)


def test_run_inference_raises_ollama_error_on_timeout(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OllamaError):
        run_inference("http://localhost:11434", "llama3.1:8b", "hi", 100, 30)
