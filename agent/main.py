"""Register, heartbeat, and run jobs through Ollama or a Parallax cluster."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil

from agent.config import AgentConfig
from agent.coordinator_client import CoordinatorClient
from agent.inference import run_inference
from common.hashing import sha256_hex
from common.inference import InferenceError
from common.logging_setup import log_event, setup_logging

logger = setup_logging("agent")


class AgentState:
    def __init__(self) -> None:
        self.status = "idle"
        self.shutdown_requested = threading.Event()


def _load_saved_credentials(state_path: str) -> Optional[dict[str, Any]]:
    p = Path(state_path)
    if not p.exists():
        return None
    try:
        saved = json.loads(p.read_text())
        if isinstance(saved, dict) and all(isinstance(saved.get(key), str) for key in ("node_id", "token")):
            return saved
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _registration_identity(config: AgentConfig) -> dict[str, str]:
    return {
        "coordinator_url": config.coordinator_url.rstrip("/"),
        "model_name": config.model_name,
        "inference_backend": config.inference_backend,
        "inference_url": (config.parallax_url if config.inference_backend == "parallax" else config.ollama_url).rstrip("/"),
    }


def _save_credentials(config: AgentConfig, node_id: str, token: str) -> None:
    Path(config.state_path).write_text(json.dumps({
        "node_id": node_id, "token": token, "identity": _registration_identity(config),
    }))


def _register_payload(config: AgentConfig) -> dict[str, Any]:
    return {
        "name": config.node_name,
        "model_name": config.model_name,
        "inference_backend": config.inference_backend,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "city": config.city,
        "country": config.country,
        "hardware": config.hardware,
        "measured_watts": config.measured_watts,
        "grid_zone": config.grid_zone,
    }


def ensure_registered(client: CoordinatorClient, config: AgentConfig) -> None:
    """Reuse a saved node_id/token if the coordinator still recognizes it
    (same DB, agent just restarted); otherwise register fresh."""
    saved = _load_saved_credentials(config.state_path)
    if saved and saved.get("identity") == _registration_identity(config):
        client.node_id, client.token = saved["node_id"], saved["token"]
        status = client.heartbeat(status="idle", cpu_pct=0.0, ram_free_mb=0.0)
        if status == 200:
            log_event(logger, "resumed_existing_registration", node_id=client.node_id)
            return
        log_event(logger, "saved_registration_invalid_reregistering", node_id=client.node_id, status=status)

    if saved:
        # Retire the old registration when a state file is reused for another
        # backend/model/endpoint. The old credentials cannot authenticate to
        # an unrelated coordinator.
        client.node_id, client.token = saved["node_id"], saved["token"]
        client.deregister()

    node_id, token = client.register(_register_payload(config))
    _save_credentials(config, node_id, token)
    log_event(logger, "node_registered", node_id=node_id, name=config.node_name)


def _heartbeat_loop(client: CoordinatorClient, state: AgentState, interval_s: float) -> None:
    while not state.shutdown_requested.is_set():
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_free_mb = psutil.virtual_memory().available / (1024 * 1024)
        client.heartbeat(status=state.status, cpu_pct=cpu_pct, ram_free_mb=ram_free_mb)
        state.shutdown_requested.wait(interval_s)


def run_job(client: CoordinatorClient, config: AgentConfig, job: dict[str, Any], state: AgentState) -> None:
    job_id = job["job_id"]
    log_event(logger, "job_running", job_id=job_id)
    state.status = "busy"
    try:
        if job.get("inference_backend", "ollama") != config.inference_backend:
            raise InferenceError("Job inference backend does not match this agent's configuration")
        result = run_inference(
            config, job["model_name"], job["prompt"], job["max_tokens"], job["timeout_s"],
        )
        client.post_result(job_id, {
            "node_id": client.node_id,
            "success": True,
            "output_text": result.output_text,
            "output_sha256": sha256_hex(result.output_text),
            "tokens_generated": result.tokens_generated,
            "duration_s": result.duration_s,
            "tokens_per_sec": result.tokens_per_sec,
        })
        log_event(logger, "job_completed", job_id=job_id, tokens=result.tokens_generated, tps=round(result.tokens_per_sec, 2))
    except InferenceError as exc:
        client.post_result(job_id, {"node_id": client.node_id, "success": False, "error": str(exc)})
        log_event(logger, "job_failed", job_id=job_id, error=str(exc))
    finally:
        state.status = "idle"


def run(config_path: str = "node_config.yaml") -> None:
    config = AgentConfig.load(config_path)
    client = CoordinatorClient(config.coordinator_url, backoff_cap_s=config.poll_backoff_cap_s)
    state = AgentState()

    def _handle_signal(signum: int, frame: Any) -> None:
        log_event(logger, "shutdown_signal_received", signal=signum)
        state.shutdown_requested.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ensure_registered(client, config)

    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(client, state, config.heartbeat_interval_s), daemon=True,
    )
    hb_thread.start()

    poll_delay = config.poll_interval_s
    # Phase 1 runs one job at a time regardless of max_concurrent_jobs; that
    # field is recorded and sent to the coordinator for future use.
    while not state.shutdown_requested.is_set():
        try:
            job = client.next_job()
        except httpx.HTTPError as exc:
            log_event(logger, "coordinator_unreachable", error=str(exc), backoff_s=poll_delay)
            state.shutdown_requested.wait(poll_delay)
            poll_delay = min(poll_delay * 2, config.poll_backoff_cap_s)
            continue

        poll_delay = config.poll_interval_s
        if job is None:
            state.shutdown_requested.wait(config.poll_interval_s)
            continue

        run_job(client, config, job, state)

    log_event(logger, "shutting_down", node_id=client.node_id)
    client.deregister()
    client.close()
    log_event(logger, "shutdown_complete", node_id=client.node_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed inference node agent")
    parser.add_argument("--config", default="node_config.yaml")
    args = parser.parse_args()
    run(args.config)
