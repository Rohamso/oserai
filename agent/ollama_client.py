"""Thin client for a local Ollama server."""

from __future__ import annotations

import time
import httpx

from common.inference import InferenceError, InferenceResult


class OllamaError(InferenceError):
    pass


OllamaResult = InferenceResult


def run_inference(
    ollama_url: str, model_name: str, prompt: str, max_tokens: int, timeout_s: float,
) -> OllamaResult:
    """Run one deterministic inference call against Ollama's /api/generate.

    temperature=0 and seed=42 reduce sampling variation for hash comparison.
    Identical output still depends on model weights, quantization and runtime;
    a mismatch alone is not evidence that a contributor acted dishonestly.
    """
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": max_tokens},
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(str(exc)) from exc

    data = resp.json()
    wall_duration = time.monotonic() - start
    output_text = data.get("response", "")
    tokens_generated = data.get("eval_count", 0) or 0
    eval_duration_ns = data.get("eval_duration", 0) or 0
    duration_s = (eval_duration_ns / 1e9) if eval_duration_ns else wall_duration
    tokens_per_sec = tokens_generated / duration_s if duration_s > 0 else 0.0

    return OllamaResult(
        output_text=output_text,
        tokens_generated=tokens_generated,
        duration_s=duration_s,
        tokens_per_sec=tokens_per_sec,
    )
