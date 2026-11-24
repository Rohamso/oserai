"""Agent configuration: node_config.yaml with AGENT_* environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from common.inference import InferenceBackend

_ENV_PREFIX = "AGENT_"


@dataclass
class AgentConfig:
    node_name: str
    coordinator_url: str
    model_name: str
    max_concurrent_jobs: int = 1
    city: Optional[str] = None
    country: Optional[str] = None
    hardware: Optional[str] = None
    measured_watts: Optional[float] = None  # Phase 2 placeholder
    grid_zone: Optional[str] = None  # Phase 2 placeholder
    ollama_url: str = "http://localhost:11434"
    heartbeat_interval_s: float = 30.0
    poll_interval_s: float = 3.0
    poll_backoff_cap_s: float = 300.0
    state_path: str = ".node_state.json"
    inference_backend: InferenceBackend = "ollama"
    parallax_url: str = "http://localhost:3001"

    def __post_init__(self) -> None:
        if self.inference_backend not in ("ollama", "parallax"):
            raise ValueError("inference_backend must be ollama or parallax")
        url = self.parallax_url if self.inference_backend == "parallax" else self.ollama_url
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"{self.inference_backend} URL must start with http:// or https://")

    @classmethod
    def load(cls, path: str = "node_config.yaml") -> "AgentConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"node config not found: {path}")
        data: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        location = data.get("location") or {}

        values: dict[str, Any] = {
            "node_name": data.get("node_name"),
            "coordinator_url": data.get("coordinator_url"),
            "model_name": data.get("model_name"),
            "max_concurrent_jobs": data.get("max_concurrent_jobs", 1),
            "city": location.get("city"),
            "country": location.get("country"),
            "hardware": data.get("hardware"),
            "measured_watts": data.get("measured_watts"),
            "grid_zone": data.get("grid_zone"),
            "ollama_url": data.get("ollama_url", "http://localhost:11434"),
            "heartbeat_interval_s": data.get("heartbeat_interval_s", 30.0),
            "poll_interval_s": data.get("poll_interval_s", 3.0),
            "poll_backoff_cap_s": data.get("poll_backoff_cap_s", 300.0),
            "state_path": data.get("state_path", ".node_state.json"),
            "inference_backend": data.get("inference_backend", "ollama"),
            "parallax_url": data.get("parallax_url", "http://localhost:3001"),
        }

        for key, current in list(values.items()):
            env_key = _ENV_PREFIX + key.upper()
            if env_key not in os.environ:
                continue
            raw = os.environ[env_key]
            if isinstance(current, bool):
                values[key] = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                values[key] = int(raw)
            elif isinstance(current, float):
                values[key] = float(raw)
            else:
                values[key] = raw

        if not values["node_name"] or not values["coordinator_url"] or not values["model_name"]:
            raise ValueError("node_config.yaml must set node_name, coordinator_url, and model_name")
        return cls(**values)
