"""Chat-formatted training data for the LoRA path (training/lora.py) --
loads the already-fetched Dolly instruction/response pairs
(training/data/dolly15k_chat.txt, produced by scripts/prepare_training_data.py)
and re-tokenizes each example through the base model's own chat template
(tokenizer.apply_chat_template) instead of the tiktoken/CHAT_TEMPLATE path
used by the from-scratch model -- better transfer since it matches the
format the base model was already instruct-tuned on.

Packs all examples into one token stream so training/dataset.py's
shard_for_node/get_batch keep working completely unchanged for sharding --
SFT data looks like any other corpus to the rest of the DiLoCo pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from training.dataset import DATA_DIR
from training.dataset import get_batch as _get_batch

DOLLY_CHAT_PATH = DATA_DIR / "dolly15k_chat.txt"

_INSTRUCTION_MARK = "### Instruction:\n"
_RESPONSE_MARK = "\n\n### Response:\n"


def _parse_dolly_examples(text: str) -> list[tuple[str, str]]:
    examples = []
    for block in text.split("<|endoftext|>"):
        block = block.strip()
        if not block.startswith(_INSTRUCTION_MARK):
            continue
        rest = block[len(_INSTRUCTION_MARK):]
        instruction, _, response = rest.partition(_RESPONSE_MARK)
        instruction, response = instruction.strip(), response.strip()
        if instruction and response:
            examples.append((instruction, response))
    return examples


def load_dolly_chat_tokens(
    tokenizer: PreTrainedTokenizerBase, path: Path = DOLLY_CHAT_PATH, max_examples: int | None = None,
) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python scripts/prepare_training_data.py` first")
    examples = _parse_dolly_examples(path.read_text())
    if max_examples is not None:
        examples = examples[:max_examples]

    all_ids: list[int] = []
    for instruction, response in examples:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        all_ids.extend(tokenizer.apply_chat_template(messages, tokenize=True)["input_ids"])
    return torch.tensor(all_ids, dtype=torch.long)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, labels) for a HF causal LM's forward(labels=...), which
    does its own internal shift -- unlike training/dataset.py's get_batch
    (used by the scratch-GPT path, already pre-shifted for a loss computed
    manually), labels here is the *same* sequence as input_ids."""
    x, _ = _get_batch(data, block_size, batch_size, device)
    return x, x.clone()
