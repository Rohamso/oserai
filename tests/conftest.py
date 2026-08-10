from __future__ import annotations

import pytest

from coordinator.config import CoordinatorConfig
from coordinator.db import Database

MODEL = "llama3.1:8b-instruct-q4_K_M"


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def config() -> CoordinatorConfig:
    return CoordinatorConfig(
        heartbeat_timeout_s=90,
        max_job_attempts=3,
        default_verification_pct=0.05,
        default_payout_rate=0.00001,
        default_job_timeout_s=300,
        dispatch_interval_s=2.0,
    )


def make_node(db: Database, name: str = "node-a", model_name: str = MODEL, **history) -> str:
    node_id = db.register_node(
        name=name, token=f"tok-{name}", city="Toronto", country="CA",
        hardware="test rig", model_name=model_name, max_concurrent_jobs=1,
        measured_watts=None, grid_zone=None,
    )
    tokens = history.get("tokens_total")
    duration = history.get("duration_total")
    if tokens and duration:
        db.record_node_job_success(node_id, tokens, duration)
    db.update_heartbeat(node_id, "idle", cpu_pct=10.0, ram_free_mb=4096.0)
    return node_id


def make_batch_and_job(db: Database, model_name: str = MODEL, timeout_s: int = 300) -> tuple[str, str]:
    batch_id = db.create_batch(
        model_name=model_name, max_tokens=100, timeout_s=timeout_s,
        verification_pct=0.0, payout_rate=0.00001, total_jobs=1,
    )
    job_id = db.create_job(batch_id, "test prompt", max_tokens=100, timeout_s=timeout_s)
    return batch_id, job_id
