"""Standalone single-machine LoRA fine-tuning + chat sampling, no
coordinator involved -- proves the base model + LoRA adapter + chat dataset
loop actually works before spinning up real training nodes. Same
de-risk-before-networking role as scripts/train_local.py (the from-scratch
path); unlike that script, only the (tiny) LoRA adapter is saved/loaded --
the frozen base model is re-downloaded/cached by transformers, never
touched by torch.save.

    python scripts/train_local_lora.py --steps 200 --out out/lora_adapter.pt
    python scripts/train_local_lora.py --sample-only --checkpoint out/lora_adapter.pt \\
        --prompt "What is the capital of France?"

Run `python scripts/prepare_training_data.py` first to fetch
training/data/dolly15k_chat.txt.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.chat_dataset import get_batch, load_dolly_chat_tokens
from training.lora import (
    DEFAULT_BASE_MODEL_ID,
    LoraRunConfig,
    build_lora_adapter,
    generate_chat_reply,
    get_adapter_state_dict,
    load_adapter_state_dict,
    load_frozen_base,
    load_tokenizer,
)


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def save_adapter(path: Path, peft_model, config: LoraRunConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": get_adapter_state_dict(peft_model),
            "base_model_id": config.base_model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": list(config.lora_target_modules),
        },
        path,
    )


def load_adapter_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = LoraRunConfig(
        base_model_id=ckpt["base_model_id"], lora_r=ckpt["lora_r"], lora_alpha=ckpt["lora_alpha"],
        lora_dropout=ckpt["lora_dropout"], lora_target_modules=tuple(ckpt["lora_target_modules"]),
    )
    base = load_frozen_base(config.base_model_id, device)
    tokenizer = load_tokenizer(config.base_model_id)
    peft_model = build_lora_adapter(base, config)
    load_adapter_state_dict(peft_model, ckpt["adapter_state_dict"])
    return peft_model, tokenizer


@torch.no_grad()
def estimate_loss(peft_model, data: torch.Tensor, block_size: int, batch_size: int, device: str, iters: int = 10) -> float:
    peft_model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        losses[i] = peft_model(input_ids=x, labels=y).loss.item()
    peft_model.train()
    return losses.mean().item()


def train(args: argparse.Namespace) -> None:
    device = args.device
    config = LoraRunConfig(
        base_model_id=args.base_model_id, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    base = load_frozen_base(config.base_model_id, device)
    tokenizer = load_tokenizer(config.base_model_id)
    peft_model = build_lora_adapter(base, config)
    peft_model.print_trainable_parameters()

    data = load_dolly_chat_tokens(tokenizer, max_examples=args.max_examples)
    n = int(0.95 * len(data))
    train_data, val_data = data[:n], data[n:]

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    min_lr = args.lr / 10
    warmup_steps = max(1, args.steps // 20)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = get_lr(step, warmup_steps, args.steps, args.lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        loss = peft_model(input_ids=x, labels=y).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_loss(peft_model, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.steps}  train_loss {loss.item():.4f}  val_loss {val_loss:.4f}  lr {lr:.2e}  ({elapsed:.0f}s)")

    out_path = Path(args.out)
    save_adapter(out_path, peft_model, config)
    print(f"saved adapter to {out_path}")

    print("\n--- sample ---")
    print(generate_chat_reply(peft_model, tokenizer, args.prompt, device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="only tokenize the first N Dolly examples -- much faster startup for a quick smoke test",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--out", default="out/lora_adapter.pt")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--sample-only", action="store_true", help="skip training, just load --checkpoint and sample")
    parser.add_argument("--checkpoint", default="out/lora_adapter.pt")
    args = parser.parse_args()

    if args.sample_only:
        peft_model, tokenizer = load_adapter_checkpoint(Path(args.checkpoint), args.device)
        print(generate_chat_reply(peft_model, tokenizer, args.prompt, args.device))
        return

    train(args)


if __name__ == "__main__":
    main()
