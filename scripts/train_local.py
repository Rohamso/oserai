"""Standalone single-machine training + sampling, no coordinator involved.

De-risks the ML before any networking: proves the model/dataset/training
loop actually works and produces reasonable output, for either stage of the
pretrain -> SFT chat pipeline.

    # 1. pretrain on TinyStories
    python scripts/train_local.py --dataset tinystories --steps 3000 --out out/pretrain.pt

    # 2. instruction-tune from the pretrain checkpoint on Dolly
    python scripts/train_local.py --dataset dolly_chat --init-checkpoint out/pretrain.pt \\
        --steps 3000 --out out/sft.pt

    # 3. chat with the result
    python scripts/train_local.py --sample-only --checkpoint out/sft.pt --prompt "What is the capital of France?"

Run `python scripts/prepare_training_data.py` first to fetch the
tinystories/dolly_chat corpora -- tiny_shakespeare ships bundled already.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import GPTConfig
from training.dataset import CHAT_DATASETS, CHAT_TEMPLATE, DATASET_NAMES, VOCAB_SIZE, eot_token_id, get_batch, get_encoder, load_dataset
from training.model import GPT


def save_checkpoint(path: Path, model: GPT, config: GPTConfig, dataset_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.__dict__, "dataset_name": dataset_name},
        path,
    )


def load_checkpoint(path: Path, device: str) -> tuple[GPT, str]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("dataset_name", "tiny_shakespeare")


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, block_size: int, batch_size: int, device: str, iters: int = 20) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def sample(model: GPT, prompt: str, device: str, dataset_name: str, max_new_tokens: int = 300) -> str:
    encoder = get_encoder()
    is_chat = dataset_name in CHAT_DATASETS
    prompt_text = CHAT_TEMPLATE.format(instruction=prompt) if is_chat else prompt
    encoded = encoder.encode(prompt_text, allowed_special={"<|endoftext|>"})

    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    stop_id = eot_token_id() if is_chat else None
    out = model.generate(idx, max_new_tokens=max_new_tokens, stop_token_id=stop_id)

    if is_chat:
        new_tokens = out[0, len(encoded):].tolist()
        if new_tokens and new_tokens[-1] == stop_id:
            new_tokens = new_tokens[:-1]
        return encoder.decode(new_tokens)
    return encoder.decode(out[0].tolist())


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr -- the flat LR previously
    used here let loss destabilize/climb in the back half of long runs."""
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def train(args: argparse.Namespace) -> None:
    device = args.device
    data = load_dataset(args.dataset)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config).to(device)
    if args.init_checkpoint:
        init_model, _ = load_checkpoint(Path(args.init_checkpoint), device)
        if init_model.config.__dict__ != config.__dict__:
            raise ValueError(
                f"--init-checkpoint config {init_model.config.__dict__} does not match "
                f"the requested config {config.__dict__} -- pass matching --n-layer/--n-head/--n-embd/--block-size"
            )
        model.load_state_dict(init_model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 20)

    print(f"dataset: {args.dataset}  params: {sum(p.numel() for p in model.parameters()):,}  device: {device}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = get_lr(step, warmup_steps, args.steps, args.lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_loss(model, val_data, config.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.steps}  train_loss {loss.item():.4f}  val_loss {val_loss:.4f}  lr {lr:.2e}  ({elapsed:.0f}s)")

    out_path = Path(args.out)
    save_checkpoint(out_path, model, config, args.dataset)
    print(f"saved checkpoint to {out_path}")

    print("\n--- sample ---")
    print(sample(model, args.prompt, device, args.dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_NAMES), default="tinystories")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=None, help="cosine decay floor (default: lr/10)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="linear warmup steps (default: steps//20)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--out", default="out/train_local.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--init-checkpoint", default=None, help="continue from a prior checkpoint's weights (e.g. SFT from a pretrain run) instead of random init")
    parser.add_argument("--sample-only", action="store_true", help="skip training, just load --checkpoint and sample")
    parser.add_argument("--checkpoint", default="out/train_local.pt")
    args = parser.parse_args()

    if args.sample_only:
        model, dataset_name = load_checkpoint(Path(args.checkpoint), args.device)
        print(sample(model, args.prompt, args.device, dataset_name))
        return

    train(args)


if __name__ == "__main__":
    main()
