"""Model architecture and DiLoCo hyperparameter defaults.

~51M params at these defaults (nanoGPT-shaped decoder-only transformer,
GPT-2 BPE vocab) -- small enough to train on a single consumer GPU in a
demo-able time window, sized to not be dwarfed by the 50257-entry tied
embedding/head. Pass larger n_layer/n_head/n_embd/block_size in a
POST /training/runs body for a closer-to-GPT-2-small run if you have real
GPUs and the patience to let DiLoCo rounds run longer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.2
    bias: bool = False


# DiLoCo round defaults, used both by scripts/train_local.py (as plain
# single-machine hyperparameters) and by coordinator training_runs rows.
DEFAULT_INNER_LR = 1e-3
DEFAULT_INNER_STEPS_H = 50
DEFAULT_INNER_BATCH_SIZE = 64
DEFAULT_OUTER_LR = 0.7
DEFAULT_OUTER_MOMENTUM = 0.9
DEFAULT_MAX_ROUNDS = 200
DEFAULT_ROUND_TIMEOUT_S = 600
DEFAULT_QUORUM = 1
