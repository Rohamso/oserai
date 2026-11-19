from __future__ import annotations

import os
from dataclasses import dataclass, field


GROQ_API_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class ChatConfig:
    gateway_key: str = ""
    inference_backend: str = "parallax"
    parallax_url: str = "http://127.0.0.1:3001"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_context_length: int = 4096
    model_name: str = ""
    model_label: str = "OserAI Chat"
    parallax_api_key: str = ""
    db_path: str = "out/chat-usage.db"
    max_output_tokens: int = 768
    max_input_chars: int = 32000
    timeout_s: float = 150.0
    max_concurrent: int = 4
    requests_per_minute: int = 10
    requests_per_day: int = 100
    groq_api_key: str = field(default="", repr=False)
    accounts_required: int = 0
    pilot_accounts: str = field(default="", repr=False)
    database_url: str = field(default="", repr=False)
    user_tokens_per_day: int = 5000
    shared_tokens_per_day: int = 100000
    shared_tokens_per_minute: int = 6000
    shared_requests_per_day: int = 800
    shared_requests_per_minute: int = 20

    @classmethod
    def load(cls) -> "ChatConfig":
        values = {}
        for name, setting in cls.__dataclass_fields__.items():
            raw = os.getenv(f"OSERAI_CHAT_{name.upper()}")
            if raw is not None:
                values[name] = type(setting.default)(raw)
        config = cls(**values)
        if any(getattr(config, name) <= 0 for name in (
            "max_output_tokens", "max_input_chars", "timeout_s", "max_concurrent",
            "requests_per_minute", "requests_per_day", "ollama_context_length",
            "user_tokens_per_day", "shared_tokens_per_day", "shared_tokens_per_minute",
            "shared_requests_per_day", "shared_requests_per_minute",
        )):
            raise ValueError("Chat limits must be positive")
        if not config.parallax_url.startswith(("http://", "https://")):
            raise ValueError("OSERAI_CHAT_PARALLAX_URL must be an HTTP(S) URL")
        if not config.ollama_url.startswith(("http://", "https://")):
            raise ValueError("OSERAI_CHAT_OLLAMA_URL must be an HTTP(S) URL")
        if config.inference_backend not in ("parallax", "ollama", "groq"):
            raise ValueError("OSERAI_CHAT_INFERENCE_BACKEND must be one of: parallax, ollama, groq")
        return config

    @property
    def api_url(self) -> str:
        if self.inference_backend == "groq":
            return GROQ_API_URL
        base = self.parallax_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    @property
    def ready(self) -> bool:
        return (
            len(self.gateway_key) >= 32
            and bool(self.model_name)
            and (self.inference_backend != "groq" or bool(self.groq_api_key.strip()))
            and (
                not self.accounts_required
                or (self.inference_backend == "groq" and bool(self.database_url and self.pilot_accounts))
            )
        )
