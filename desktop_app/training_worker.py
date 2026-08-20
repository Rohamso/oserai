"""QThread wrapper around agent/train_agent.py's DiLoCo loop.

Deliberately thin: this module doesn't reimplement registration, heartbeat,
polling, or the local-step math -- it imports and drives the exact same
functions the CLI agent (`python -m agent.train_agent`) uses, so the
desktop app and the CLI agent are two front-ends for one tested training
loop, not two independently-maintained implementations. The only new thing
here is turning that loop's blocking, signal-handler-driven CLI shape into
something a GUI can start/stop and get live status out of.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import httpx
import torch
from PySide6.QtCore import QThread, Signal

from agent.coordinator_client import CoordinatorClient
from agent.train_agent import (
    LORA_BACKEND_PREFIX,
    AgentState,
    _heartbeat_loop,
    ensure_registered,
    run_round,
    run_round_lora,
)
from agent.train_config import TrainAgentConfig
from training.chat_dataset import load_dolly_chat_tokens
from training.config import GPTConfig
from training.dataset import load_dataset, shard_for_node
from training.lora import LoraRunConfig, build_lora_adapter, load_frozen_base, load_tokenizer
from training.model import GPT


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices


class TrainingWorker(QThread):
    status_changed = Signal(str)
    log_message = Signal(str)
    round_completed = Signal(int, float)
    error = Signal(str)

    def __init__(self, config: TrainAgentConfig) -> None:
        super().__init__()
        self.config = config
        self.state = AgentState()

    def stop(self) -> None:
        self.state.shutdown_requested.set()

    def run(self) -> None:  # noqa: C901 -- mirrors agent/train_agent.py's run(), same shape
        client = CoordinatorClient(self.config.coordinator_url, backoff_cap_s=self.config.poll_backoff_cap_s)
        try:
            self.status_changed.emit("Registering with coordinator...")
            ensure_registered(client, self.config)

            run_info = client.get_run(self.config.run_id)
            is_lora = run_info["model_backend"].startswith(LORA_BACKEND_PREFIX)

            self.status_changed.emit("Loading model (first time may take a while)...")
            if is_lora:
                if run_info["dataset_name"] != "dolly_chat":
                    raise ValueError(
                        f"lora backend only supports dataset_name=dolly_chat currently, "
                        f"got {run_info['dataset_name']!r}"
                    )
                lora_config = LoraRunConfig(
                    base_model_id=run_info["base_model_id"], lora_r=run_info["lora_r"],
                    lora_alpha=run_info["lora_alpha"], lora_dropout=run_info["lora_dropout"],
                    lora_target_modules=tuple(run_info["lora_target_modules"].split(",")),
                )
                base = load_frozen_base(lora_config.base_model_id, self.config.device)
                tokenizer = load_tokenizer(lora_config.base_model_id)
                model: Any = build_lora_adapter(base, lora_config)
                data = load_dolly_chat_tokens(tokenizer)
            else:
                gpt_config = GPTConfig(
                    vocab_size=run_info["vocab_size"], block_size=run_info["block_size"],
                    n_layer=run_info["n_layer"], n_head=run_info["n_head"], n_embd=run_info["n_embd"],
                )
                model = GPT(gpt_config).to(self.config.device)
                data = load_dataset(run_info["dataset_name"])

            shard = shard_for_node(data, self.config.shard_index, self.config.num_shards).to(self.config.device)
            self.log_message.emit(f"Data shard loaded ({len(shard):,} tokens).")

            hb_thread = threading.Thread(
                target=_heartbeat_loop, args=(client, self.state, self.config.heartbeat_interval_s), daemon=True,
            )
            hb_thread.start()

            self.status_changed.emit("Idle -- waiting for a round to open")
            poll_delay = self.config.poll_interval_s
            while not self.state.shutdown_requested.is_set():
                try:
                    round_info = client.get_current_round(self.config.run_id)
                except httpx.HTTPError as exc:
                    self.log_message.emit(f"Coordinator unreachable: {exc}")
                    self.state.shutdown_requested.wait(poll_delay)
                    poll_delay = min(poll_delay * 2, self.config.poll_backoff_cap_s)
                    continue

                poll_delay = self.config.poll_interval_s
                if round_info is None or round_info["status"] != "open" or round_info["already_submitted"]:
                    self.state.shutdown_requested.wait(self.config.poll_interval_s)
                    continue

                self.status_changed.emit(f"Training round {round_info['round_num']}...")
                if is_lora:
                    loss = run_round_lora(client, self.config, round_info, model, shard, self.state)
                else:
                    loss = run_round(client, self.config, round_info, model, shard, self.state)
                self.round_completed.emit(round_info["round_num"], loss if loss is not None else float("nan"))
                self.status_changed.emit("Idle -- waiting for a round to open")

            self.status_changed.emit("Deregistering...")
            client.deregister()
        except Exception as exc:  # noqa: BLE001 -- surfaced to the GUI, not swallowed
            self.error.emit(str(exc))
        finally:
            client.close()
            self.status_changed.emit("Stopped")
