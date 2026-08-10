"""Coordinator configuration: YAML file with environment variable overrides.

Env vars take the form COORDINATOR_<FIELD_NAME> (e.g. COORDINATOR_PORT=9000).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "COORDINATOR_"

_DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "db_path": "coordinator.db",
    "heartbeat_timeout_s": 90,
    "default_verification_pct": 0.05,
    "default_payout_rate": 0.00001,
    "default_job_timeout_s": 300,
    "dispatch_interval_s": 2.0,
    "max_job_attempts": 3,
    "log_level": "INFO",
}


@dataclass
class CoordinatorConfig:
    host: str = _DEFAULTS["host"]
    port: int = _DEFAULTS["port"]
    db_path: str = _DEFAULTS["db_path"]
    heartbeat_timeout_s: int = _DEFAULTS["heartbeat_timeout_s"]
    default_verification_pct: float = _DEFAULTS["default_verification_pct"]
    default_payout_rate: float = _DEFAULTS["default_payout_rate"]
    default_job_timeout_s: int = _DEFAULTS["default_job_timeout_s"]
    dispatch_interval_s: float = _DEFAULTS["dispatch_interval_s"]
    max_job_attempts: int = _DEFAULTS["max_job_attempts"]
    log_level: str = _DEFAULTS["log_level"]

    @classmethod
    def load(cls, path: str | None = None) -> "CoordinatorConfig":
        cfg_path = path or os.environ.get("COORDINATOR_CONFIG_PATH", "coordinator_config.yaml")
        data: dict[str, Any] = {}
        p = Path(cfg_path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}

        values = dict(_DEFAULTS)
        values.update(data)
        for key, default in _DEFAULTS.items():
            env_key = _ENV_PREFIX + key.upper()
            if env_key in os.environ:
                values[key] = type(default)(os.environ[env_key])
        return cls(**values)
