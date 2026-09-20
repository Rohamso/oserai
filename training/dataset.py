"""GPT-2 BPE tokenization (via tiktoken) + batching for the bundled training
corpora. One shared encoder everywhere -- pretraining, SFT, and generation
all use the same vocab, so checkpoints stay wire-compatible across datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken
import torch

# Path(__file__).parent works when running from source, but this module
# ships inside a PyInstaller-frozen app too (desktop_app), where pure-Python
# modules like this one live inside an in-memory PYZ archive -- __file__
# doesn't point to a real directory there, so a bundled data file has to be
# resolved against sys._MEIPASS (where PyInstaller unpacks `datas`) instead.
if hasattr(sys, "_MEIPASS"):
    DATA_DIR = Path(sys._MEIPASS) / "training" / "data"
else:
    DATA_DIR = Path(__file__).parent / "data"

VOCAB_SIZE = 50257

# Alpaca-style instruction/response template, shared by the SFT data-prep
# script and the coordinator's /generate endpoint so training and inference
# see identical formatting.
CHAT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("gpt2")
    return _encoder


def eot_token_id() -> int:
    return get_encoder().eot_token


# dataset_name -> bundled corpus file. Every file is plain text; for
# dolly_chat each example is already formatted with CHAT_TEMPLATE and
# separated by "<|endoftext|>" by scripts/prepare_training_data.py, so
# tokenization/windowing below is identical across all three datasets.
_DATASET_FILES = {
    "tiny_shakespeare": DATA_DIR / "tiny_shakespeare.txt",
    "tinystories": DATA_DIR / "tinystories.txt",
    "dolly_chat": DATA_DIR / "dolly15k_chat.txt",
}

DATASET_NAMES = frozenset(_DATASET_FILES)

# Datasets formatted as chat turns rather than raw prose -- used by the
# coordinator to decide whether to wrap /generate prompts in CHAT_TEMPLATE.
CHAT_DATASETS = {"dolly_chat"}


def load_dataset(name: str) -> torch.Tensor:
    """Tokenize a bundled corpus into a single 1D tensor of token ids."""
    path = _DATASET_FILES.get(name)
    if path is None:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(_DATASET_FILES)}")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/prepare_training_data.py` first"
        )
    text = path.read_text(encoding="utf-8")
    ids = get_encoder().encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long)


def shard_for_node(data: torch.Tensor, shard_index: int, num_shards: int) -> torch.Tensor:
    """Contiguous, non-overlapping slice of the corpus for one training node.

    Simple fixed data-parallel split -- with exactly num_shards known machines
    there's no need for anything fancier than dividing the corpus evenly.
    """
    n = len(data)
    shard_size = n // num_shards
    start = shard_index * shard_size
    end = n if shard_index == num_shards - 1 else start + shard_size
    return data[start:end]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random minibatch of (input, target) sequences from data."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
