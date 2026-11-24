"""Submit a whole request to a Gradient Parallax scheduler or standalone server.

Parallax owns its workers, model shards and request routing. OserAI sees the
cluster as one inference node and measures end-to-end request latency.
"""

from __future__ import annotations

import time

import httpx

from common.inference import InferenceError, InferenceResult


class ParallaxError(InferenceError):
    pass


def run_inference(
    parallax_url: str, model_name: str, prompt: str, max_tokens: int, timeout_s: float,
) -> InferenceResult:
    base_url = parallax_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    start = time.monotonic()
    try:
        # The scheduler may override the request's model with its configured
        # model. Check the advertised model before charging usage to a batch.
        models = httpx.get(f"{base_url}/models", timeout=min(timeout_s, 10.0))
        models.raise_for_status()
        try:
            model_ids = {model["id"] for model in models.json()["data"]}
        except (ValueError, KeyError, TypeError) as exc:
            raise ParallaxError("Invalid Parallax /v1/models response") from exc
        if model_name not in model_ids:
            raise ParallaxError(f"Parallax is not serving the configured model {model_name}")
        remaining_s = timeout_s - (time.monotonic() - start)
        if remaining_s <= 0:
            raise ParallaxError("Parallax model check exhausted the job timeout")
        response = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 42,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=remaining_s,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ParallaxError(f"Parallax request failed: {exc}") from exc

    duration_s = time.monotonic() - start
    try:
        data = response.json()
        output_text = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("expected a non-empty text answer")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("usage.completion_tokens must be a non-negative integer")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ParallaxError(f"Invalid Parallax completion response: {exc}") from exc

    return InferenceResult(
        output_text=output_text,
        tokens_generated=tokens,
        duration_s=duration_s,
        tokens_per_sec=tokens / duration_s if duration_s > 0 else 0.0,
    )
