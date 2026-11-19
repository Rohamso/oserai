"""Normalize replies from Parallax, Ollama, and Groq for the chat gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from chat_service.config import ChatConfig


@dataclass(frozen=True)
class Chunk:
    text: str = ""
    completion_tokens: int | None = None
    finished: bool = False


class ChatRuntime:
    def __init__(self, config: ChatConfig):
        self.config = config

    @property
    def upstream_headers(self) -> dict[str, str]:
        key = {
            "groq": self.config.groq_api_key.strip(),
            "parallax": self.config.parallax_api_key,
        }.get(self.config.inference_backend)
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def model_available(self, client: httpx.AsyncClient) -> bool:
        settings = self.config
        try:
            if settings.inference_backend == "ollama":
                response = await client.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=5.0)
                response.raise_for_status()
                names = {model.get("name", model.get("model")) for model in response.json()["models"]}
            else:
                response = await client.get(settings.api_url + "/models", headers=self.upstream_headers, timeout=5.0)
                response.raise_for_status()
                names = {model["id"] for model in response.json()["data"]}
            return settings.model_name in names
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            return False

    async def stream(self, client: httpx.AsyncClient, messages: list[dict]):
        settings = self.config
        if settings.inference_backend == "groq":
            # The completed JSON response includes usage and works with the cPanel proxy's buffered mode.
            response = await client.post(
                settings.api_url + "/chat/completions",
                headers=self.upstream_headers,
                json={
                    "model": settings.model_name, "messages": messages, "stream": False,
                    "max_completion_tokens": settings.max_output_tokens, "temperature": 0.7,
                },
            )
            response.raise_for_status()
            value = response.json()
            choices = value.get("choices")
            if value.get("error") or not isinstance(choices, list) or not choices:
                raise ValueError("Invalid Groq completion")
            choice = choices[0]
            text = (choice.get("message") or {}).get("content")
            if not isinstance(text, str) or not text.strip() or choice.get("finish_reason") not in ("stop", "length"):
                raise ValueError("Incomplete Groq completion")
            count = (value.get("usage") or {}).get("completion_tokens")
            yield Chunk(text=text, completion_tokens=count if type(count) is int and count >= 0 else None, finished=True)
            return

        ollama = settings.inference_backend == "ollama"
        if ollama:
            url = settings.ollama_url.rstrip("/") + "/api/chat"
            payload = {
                "model": settings.model_name, "messages": messages, "stream": True,
                "think": False, "keep_alive": "10m",
                "options": {"num_ctx": settings.ollama_context_length,
                            "num_predict": settings.max_output_tokens, "temperature": 0.7},
            }
        else:
            url = settings.api_url + "/chat/completions"
            payload = {
                "model": settings.model_name, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": settings.max_output_tokens, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        async with client.stream("POST", url, headers=self.upstream_headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if not ollama:
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        yield Chunk(finished=True)
                        return
                value = json.loads(line)
                if value.get("error"):
                    raise ValueError("Upstream completion error")
                if ollama:
                    text = (value.get("message") or {}).get("content")
                    count = value.get("eval_count")
                    done = value.get("done") is True
                    yield Chunk(text=text if isinstance(text, str) else "",
                                completion_tokens=count if type(count) is int and count >= 0 else None,
                                finished=done)
                    if done:
                        return
                else:
                    count = (value.get("usage") or {}).get("completion_tokens")
                    if type(count) is int and count >= 0:
                        yield Chunk(completion_tokens=count)
                    for choice in value.get("choices", []):
                        text = (choice.get("delta") or {}).get("content")
                        yield Chunk(text=text if isinstance(text, str) else "",
                                    finished=choice.get("finish_reason") is not None)
