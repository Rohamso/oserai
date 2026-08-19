"""Training agent configuration: train_node_config.yaml with TRAINAGENT_*
environment overrides. Mirrors agent/config.py's pattern -- a separate
dataclass since a training node's config has nothing in common with an
inference node's (dataset shard, run binding, device, no model_name/Ollama).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_ENV_PREFIX = "TRAINAGENT_"


@dataclass
class TrainAgentConfig:
    node_name: str
    coordinator_url: str
    run_id: str
    shard_index: int
    num_shards: int = 2
    device: str = "cpu"
    city: Optional[str] = None
    country: Optional[str] = None
    hardware: Optional[str] = None
    heartbeat_interval_s: float = 30.0
    poll_interval_s: float = 5.0
    poll_backoff_cap_s: float = 300.0
    state_path: str = ".train_node_state.json"

    @classmethod
    def load(cls, path: str = "train_node_config.yaml") -> "TrainAgentConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"train node config not found: {path}")
        data: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        location = data.get("location") or {}

        values: dict[str, Any] = {
            "node_name": data.get("node_name"),
            "coordinator_url": data.get("coordinator_url"),
            "run_id": data.get("run_id"),
            "shard_index": data.get("shard_index"),
            "num_shards": data.get("num_shards", 2),
            "device": data.get("device", "cpu"),
            "city": location.get("city"),
            "country": location.get("country"),
            "hardware": data.get("hardware"),
            "heartbeat_interval_s": data.get("heartbeat_interval_s", 30.0),
            "poll_interval_s": data.get("poll_interval_s", 5.0),
            "poll_backoff_cap_s": data.get("poll_backoff_cap_s", 300.0),
            "state_path": data.get("state_path", ".train_node_state.json"),
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

        if not values["node_name"] or not values["coordinator_url"] or not values["run_id"]:
            raise ValueError("train_node_config.yaml must set node_name, coordinator_url, and run_id")
        if values["shard_index"] is None:
            raise ValueError("train_node_config.yaml must set shard_index (0, 1, ... one per known machine)")
        return cls(**values)
