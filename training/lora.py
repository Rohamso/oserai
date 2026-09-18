"""LoRA fine-tuning on a real pretrained instruct model, as the DiLoCo
payload -- instead of training a model's knowledge from zero (see
training/model.py, the from-scratch path), freeze a real pretrained model
and train only small low-rank adapter matrices on top of it.

Two things this buys over the from-scratch path:
  - Quality: adapting knowledge that's already there instead of building it
    from a demo-sized corpus.
  - Bandwidth: a LoRA delta is ~10-30MB (this module's default config)
    vs. gigabytes for a full-model delta -- the difference between "movable
    over a contributor's home upload connection every round" and not.

The base model is always frozen (requires_grad=False, no optimizer state)
-- only adapter parameters ever get gradients or get exchanged over the
network. peft's get_peft_model_state_dict()/set_peft_model_state_dict()
return/accept plain dict[str, Tensor], so training/checkpoint_io.py and
training/outer_optim.py need no changes at all to carry LoRA deltas instead
of full-model deltas.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

DEFAULT_INNER_LR = 1e-4
DEFAULT_INNER_STEPS_H = 30
DEFAULT_INNER_BATCH_SIZE = 4
DEFAULT_BLOCK_SIZE = 512


@dataclass
class LoraRunConfig:
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGET_MODULES)


StateDict = dict[str, torch.Tensor]

# One process, one copy of each base model -- it's frozen and read-only
# across every run/round, so every caller in this process (coordinator's
# /generate, a training agent's round loop) shares the same weights instead
# of re-loading a multi-GB model repeatedly.
_base_model_cache: dict[tuple[str, str, torch.dtype], torch.nn.Module] = {}
_tokenizer_cache: dict[str, PreTrainedTokenizerBase] = {}


def _dtype_for_device(device: str) -> torch.dtype:
    # fp16 on CUDA/MPS keeps a 1.5B-param frozen base to ~3GB instead of
    # ~6GB in fp32 -- the difference between fitting and OOMing on an
    # 8-9GB-RAM consumer machine. CPU-only falls back to fp32 since several
    # ops used here (e.g. some attention kernels) don't have fast fp16 CPU
    # paths.
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def _from_pretrained_prefer_cache(loader, model_id: str, **kwargs):
    # Try the local HF cache first, with no network call at all -- on a
    # contributor's machine that already has this model, a from_pretrained()
    # call with no local_files_only still reaches out to the Hub to check
    # for updates, and on a network with broken/filtered IPv6 that HEAD/etag
    # request can hang for tens of minutes with no error (observed: a
    # SYN_SENT connection that never resolved). Only the very first
    # download on a machine needs the network at all.
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return loader.from_pretrained(model_id, **kwargs)


def load_tokenizer(base_model_id: str = DEFAULT_BASE_MODEL_ID) -> PreTrainedTokenizerBase:
    if base_model_id not in _tokenizer_cache:
        tok = _from_pretrained_prefer_cache(AutoTokenizer, base_model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _tokenizer_cache[base_model_id] = tok
    return _tokenizer_cache[base_model_id]


def load_frozen_base(
    base_model_id: str = DEFAULT_BASE_MODEL_ID, device: str = "cpu", dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Download (once, cached under ~/.cache/huggingface) and load the base
    model, frozen. Cached per (model_id, device, dtype) for the life of the
    process.

    dtype defaults to _dtype_for_device(device) (fp32 on CPU, for training
    stability) but callers that only ever run inference on CPU -- the
    coordinator, which never trains, only aggregates deltas and generates
    -- should pass dtype=torch.float16 explicitly: halves the ~6GB fp32
    footprint of a 1.5B model to ~3GB, which is the difference between
    fitting and thrashing on an 8-9GB-RAM machine, and fp16 CPU matmul
    works fine for inference (just slower per-op than fp32, an easy trade
    given memory is the binding constraint here)."""
    dtype = dtype if dtype is not None else _dtype_for_device(device)
    key = (base_model_id, device, dtype)
    if key not in _base_model_cache:
        model = _from_pretrained_prefer_cache(AutoModelForCausalLM, base_model_id, dtype=dtype)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        _base_model_cache[key] = model
    return _base_model_cache[key]


def build_lora_adapter(base_model: torch.nn.Module, config: LoraRunConfig) -> PeftModel:
    """Wrap a frozen base model with a fresh LoRA adapter. Trainable params
    are cast to float32 regardless of the (possibly fp16) base -- a standard
    mixed-precision trick: the adapter is tiny, so the memory cost of fp32
    is negligible, but AdamW over it is far more numerically stable than
    fp16."""
    lora_config = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules), bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return peft_model


@contextmanager
def temporary_lora_adapter(base_model: torch.nn.Module, config: LoraRunConfig):
    """Attach a fresh LoRA adapter to a (possibly cached/shared) frozen base
    model for the duration of the `with` block, then detach it.

    get_peft_model mutates the base module in place to attach adapter
    hooks -- fine for a training agent, which builds exactly one adapter
    per process lifetime, but the coordinator is a long-lived server that
    calls build_lora_adapter repeatedly (once per run creation, once per
    /generate call) against the *same* cached base model
    (load_frozen_base's module-level cache). Without unloading between
    calls, peft warns "modifying a model with PEFT for a second time" and
    adapters stack on the base's modules instead of each call getting a
    clean, isolated adapter. Use this wherever a cached base might be
    reused; training/train_agent.py's one-shot build_lora_adapter call
    doesn't need it.
    """
    peft_model = build_lora_adapter(base_model, config)
    try:
        yield peft_model
    finally:
        peft_model.unload()


def get_adapter_state_dict(peft_model: PeftModel) -> StateDict:
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(peft_model).items()}


def load_adapter_state_dict(peft_model: PeftModel, state_dict: StateDict) -> None:
    set_peft_model_state_dict(peft_model, state_dict)


def local_lora_steps(
    peft_model: PeftModel, get_batch_fn, inner_lr: float, inner_steps_h: int, inner_batch_size: int,
    block_size: int, device: str,
) -> tuple[StateDict, float]:
    """Run H local AdamW steps over adapter params only (base stays
    frozen), mirroring the DiLoCo local-training math in
    agent/train_agent.py's run_round for the scratch-GPT path. Returns the
    pseudo-gradient delta (start - end) and the last step's loss.

    get_batch_fn(block_size, batch_size, device) -> (input_ids, labels),
    e.g. training/chat_dataset.get_batch.
    """
    start_sd = get_adapter_state_dict(peft_model)

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=inner_lr)
    peft_model.train()

    last_loss = None
    for _ in range(inner_steps_h):
        input_ids, labels = get_batch_fn(block_size, inner_batch_size, device)
        out = peft_model(input_ids=input_ids, labels=labels)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()
        last_loss = out.loss.item()

    end_sd = get_adapter_state_dict(peft_model)
    delta = {k: (start_sd[k].float() - end_sd[k].float()).cpu() for k in start_sd}
    return delta, (last_loss or 0.0)


@torch.no_grad()
def generate_chat_reply(
    peft_model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, prompt: str, device: str,
    max_new_tokens: int = 300, temperature: float = 0.8,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    peft_model.eval()
    out = peft_model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=max(temperature, 1e-3),
        top_k=200, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
