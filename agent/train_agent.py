"""Training node agent: registers with the coordinator, heartbeats, pulls
the current DiLoCo round's global weights, runs H local AdamW steps on its
own data shard, and uploads the resulting pseudo-gradient (delta).

Separate entrypoint from agent/main.py -- a training round holds a loaded
model + optimizer in memory for potentially many GPU-minutes, a completely
different lifecycle from the inference agent's seconds-scale poll/run/report
loop, and keeps torch off machines that only ever run agent/main.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil
import torch

from agent.coordinator_client import CoordinatorClient
from agent.train_config import TrainAgentConfig
from common.logging_setup import log_event, setup_logging
from training.checkpoint_io import bytes_to_state_dict, state_dict_to_bytes
from training.config import GPTConfig
from training.dataset import get_batch, load_dataset, shard_for_node
from training.model import GPT

logger = setup_logging("train_agent")


class AgentState:
    def __init__(self) -> None:
        self.status = "idle"
        self.shutdown_requested = threading.Event()


def _load_saved_credentials(state_path: str) -> Optional[dict[str, str]]:
    p = Path(state_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_credentials(state_path: str, node_id: str, token: str) -> None:
    Path(state_path).write_text(json.dumps({"node_id": node_id, "token": token}))


def _register_payload(config: TrainAgentConfig) -> dict[str, Any]:
    return {
        "name": config.node_name,
        "city": config.city,
        "country": config.country,
        "hardware": config.hardware,
    }


def ensure_registered(client: CoordinatorClient, config: TrainAgentConfig) -> None:
    """Reuse a saved node_id/token if the coordinator still recognizes it
    (same DB, agent just restarted); otherwise register fresh."""
    saved = _load_saved_credentials(config.state_path)
    if saved:
        client.node_id, client.token = saved["node_id"], saved["token"]
        status = client.heartbeat(status="idle", cpu_pct=0.0, ram_free_mb=0.0)
        if status == 200:
            log_event(logger, "resumed_existing_registration", node_id=client.node_id)
            return
        log_event(logger, "saved_registration_invalid_reregistering", node_id=client.node_id, status=status)

    node_id, token = client.register_training(_register_payload(config))
    _save_credentials(config.state_path, node_id, token)
    log_event(logger, "training_node_registered", node_id=node_id, name=config.node_name)


def _heartbeat_loop(client: CoordinatorClient, state: AgentState, interval_s: float) -> None:
    while not state.shutdown_requested.is_set():
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_free_mb = psutil.virtual_memory().available / (1024 * 1024)
        client.heartbeat(status=state.status, cpu_pct=cpu_pct, ram_free_mb=ram_free_mb)
        state.shutdown_requested.wait(interval_s)


def run_round(
    client: CoordinatorClient, config: TrainAgentConfig,
    round_info: dict[str, Any], model: GPT, shard: torch.Tensor, state: AgentState,
) -> None:
    round_id = round_info["round_id"]
    log_event(logger, "round_started", round_id=round_id, round_num=round_info["round_num"])
    state.status = "busy"
    t0 = time.time()
    try:
        weights = client.download_checkpoint(round_info["base_checkpoint_id"])
        model.load_state_dict(bytes_to_state_dict(weights, map_location=config.device))
        # Snapshot pre-training weights -- optimizer.step() mutates the live
        # parameter tensors in place, so this must be a real copy, not a view.
        start_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

        optimizer = torch.optim.AdamW(model.parameters(), lr=round_info["inner_lr"])
        model.train()
        last_loss = None
        for _ in range(round_info["inner_steps_h"]):
            x, y = get_batch(shard, round_info["block_size"], round_info["inner_batch_size"], config.device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        end_sd = model.state_dict()
        # DiLoCo pseudo-gradient: theta_start_of_round - theta_after_H_local_steps.
        delta = {k: (start_sd[k] - end_sd[k]).cpu() for k in start_sd}
        wall_time_s = time.time() - t0

        result = client.upload_delta(
            round_id, state_dict_to_bytes(delta),
            local_loss=last_loss or 0.0,
            local_steps_completed=round_info["inner_steps_h"],
            wall_time_s=wall_time_s,
        )
        if result is None:
            log_event(logger, "round_discarded_already_closed", round_id=round_id)
        else:
            log_event(
                logger, "round_completed", round_id=round_id,
                loss=round(last_loss, 4) if last_loss is not None else None,
                wall_time_s=round(wall_time_s, 1),
            )
    finally:
        state.status = "idle"


def run(config_path: str = "train_node_config.yaml") -> None:
    config = TrainAgentConfig.load(config_path)
    client = CoordinatorClient(config.coordinator_url, backoff_cap_s=config.poll_backoff_cap_s)
    state = AgentState()

    def _handle_signal(signum: int, frame: Any) -> None:
        log_event(logger, "shutdown_signal_received", signal=signum)
        state.shutdown_requested.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ensure_registered(client, config)

    run_info = client.get_run(config.run_id)
    gpt_config = GPTConfig(
        vocab_size=run_info["vocab_size"], block_size=run_info["block_size"],
        n_layer=run_info["n_layer"], n_head=run_info["n_head"], n_embd=run_info["n_embd"],
    )
    model = GPT(gpt_config).to(config.device)

    data = load_dataset(run_info["dataset_name"])
    shard = shard_for_node(data, config.shard_index, config.num_shards).to(config.device)
    log_event(
        logger, "shard_loaded", shard_index=config.shard_index,
        num_shards=config.num_shards, shard_len=len(shard),
    )

    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(client, state, config.heartbeat_interval_s), daemon=True,
    )
    hb_thread.start()

    poll_delay = config.poll_interval_s
    while not state.shutdown_requested.is_set():
        try:
            round_info = client.get_current_round(config.run_id)
        except httpx.HTTPError as exc:
            log_event(logger, "coordinator_unreachable", error=str(exc), backoff_s=poll_delay)
            state.shutdown_requested.wait(poll_delay)
            poll_delay = min(poll_delay * 2, config.poll_backoff_cap_s)
            continue

        poll_delay = config.poll_interval_s
        if round_info is None or round_info["status"] != "open" or round_info["already_submitted"]:
            state.shutdown_requested.wait(config.poll_interval_s)
            continue

        run_round(client, config, round_info, model, shard, state)

    log_event(logger, "shutting_down", node_id=client.node_id)
    client.deregister()
    client.close()
    log_event(logger, "shutdown_complete", node_id=client.node_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training node agent (DiLoCo)")
    parser.add_argument("--config", default="train_node_config.yaml")
    args = parser.parse_args()
    run(args.config)
