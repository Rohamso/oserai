"""Run a bounded local LoRA CPT/SFT rehearsal using verified Parquet shards."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from training.adaptation_data import sha256
from training.lora import LoraRunConfig, build_lora_adapter


SEED = 42
RECIPE_PATH = Path(__file__).resolve().parents[1] / "training" / "adaptation_recipe.json"


def read_rows(path: Path, tokenizer_hash: str, phase: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["phase"] != phase or manifest["tokenizer"]["sha256"] != tokenizer_hash:
        raise ValueError(f"Wrong phase or tokenizer for {path}")
    rows = []
    for shard in manifest["shards"]:
        filename = path / shard["file"]
        if filename.parent != path or sha256(filename) != shard["sha256"]:
            raise ValueError(f"Shard integrity check failed for {filename}")
        rows.extend(pq.read_table(filename).to_pylist())
    return manifest, rows


def loss(model, pair: tuple[list[int], list[int]], device: str) -> torch.Tensor:
    tokens, labels = pair
    targets = torch.tensor(labels[1:])
    positions = torch.where(targets != -100)[0]
    if not positions.numel():
        raise ValueError("Training example has no unmasked next-token labels")
    logits = model(input_ids=torch.tensor([tokens], device=device),
                   logits_to_keep=positions.to(device), use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits[0].float(), targets[positions].to(device))


@torch.inference_mode()
def evaluate(model, samples: list[tuple[list[int], list[int]]], device: str) -> float:
    model.eval()
    return sum(loss(model, pair, device).item() for pair in samples) / len(samples)


def train(model, samples, device: str, phase: str, recipe: dict) -> dict:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=recipe["peak_lr"], betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=recipe.get("weight_decay", 0.01))
    warmup = max(1, min(8, len(samples) // 10))
    losses = []
    started = time.monotonic()
    model.train()
    for step, pair in enumerate(samples, 1):
        lr = recipe["peak_lr"] * min(1, step / warmup)
        if phase == "cpt" and step > warmup:
            progress = (step - warmup) / max(1, len(samples) - warmup)
            lr = recipe["min_lr"] + (recipe["peak_lr"] - recipe["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        value = loss(model, pair, device)
        item = value.item()
        if not math.isfinite(item):
            raise RuntimeError(f"Non-finite {phase} loss at step {step}")
        value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(item)
        if step == 1 or step % 16 == 0 or step == len(samples):
            recent = losses[-16:]
            print(f"{phase} {step}/{len(samples)} mean loss {sum(recent) / len(recent):.4f}", flush=True)
    return {"steps": len(samples), "warmup_steps": warmup, "seconds": time.monotonic() - started,
            "tokens_seen": sum(len(pair[0]) for pair in samples), "mean_train_loss": sum(losses) / len(losses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--cpt-window", type=int, default=256)
    parser.add_argument("--cpt-steps", type=int, default=48)
    parser.add_argument("--sft-epochs", type=int, choices=(2, 3), default=2)
    parser.add_argument("--sft-window", type=int, default=192,
                        help="Keep complete dialogues no longer than this local training window")
    parser.add_argument("--resume-cpt", type=Path, help="Start SFT from a saved CPT adapter")
    args = parser.parse_args()
    if min(args.cpt_window, args.sft_window) < 2 or args.cpt_steps < 1:
        parser.error("Training windows must be at least two tokens with positive CPT steps")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable")

    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    tokenizer_hash = sha256(args.base / "tokenizer.json")
    identity, cpt = read_rows(args.data / "cpt", tokenizer_hash, "cpt")
    _, sft = read_rows(args.data / "sft-train", tokenizer_hash, "sft")
    _, sft_eval = read_rows(args.data / "sft-eval", tokenizer_hash, "sft")
    sft = [row for row in sft if len(row["input_ids"]) <= args.sft_window]
    sft_eval = [row for row in sft_eval if len(row["input_ids"]) <= args.sft_window]
    if not sft or not sft_eval:
        raise ValueError("The SFT window excludes all training or evaluation dialogues")
    if len(cpt) < 2 or set(ref["record_id"] for row in sft for ref in row["references"]) & set(
        ref["record_id"] for row in sft_eval for ref in row["references"]
    ):
        raise ValueError("Evaluation requires a separate CPT row and disjoint SFT conversations")
    eval_start = len(cpt) - 1
    # A document can run across packed rows. Hold every row that contains the
    # final row's documents out of training to prevent verse or prose overlap.
    held_documents = {ref["record_id"] for ref in cpt[-1]["references"]}
    cpt_train = [row for row in cpt[:eval_start] if not held_documents.intersection(
        ref["record_id"] for ref in row["references"])]
    if not cpt_train or args.cpt_window > identity["context"]:
        raise ValueError("No disjoint CPT data, or requested window exceeds the packed context")
    rng = random.Random(SEED)
    cpt_samples = []
    for _ in range(args.cpt_steps):
        row = rng.choice(cpt_train)["input_ids"]
        start = rng.randint(0, len(row) - args.cpt_window)
        tokens = row[start:start + args.cpt_window]
        cpt_samples.append((tokens, tokens))
    held_tokens = cpt[-1]["input_ids"]
    cpt_eval = [(tokens, tokens) for offset in range(0, len(held_tokens), args.cpt_window * 4)
                if len(tokens := held_tokens[offset:offset + args.cpt_window]) >= 2][:8]
    sft_samples = []
    for _ in range(args.sft_epochs):
        shuffled = list(sft)
        rng.shuffle(shuffled)
        sft_samples.extend((row["input_ids"], row["labels"]) for row in shuffled)
    sft_eval_samples = [(row["input_ids"], row["labels"]) for row in sft_eval]
    torch.manual_seed(SEED)
    if args.device == "cpu":
        torch.set_num_threads(4)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16, local_files_only=True)
    base.to(args.device)
    base.requires_grad_(False)
    source = identity["tokenizer"]
    if args.resume_cpt:
        adapter_config = json.loads((args.resume_cpt / "adapter_config.json").read_text(encoding="utf-8"))
        if adapter_config["base_model_name_or_path"] != source["model"] or adapter_config["revision"] != source["revision"]:
            raise ValueError("CPT adapter base and revision must match the data manifests")
        model = PeftModel.from_pretrained(base, args.resume_cpt, is_trainable=True)
    else:
        model = build_lora_adapter(base, LoraRunConfig(base_model_id=source["model"]))
    model.peft_config["default"].base_model_name_or_path = source["model"]
    model.peft_config["default"].revision = source["revision"]
    model.print_trainable_parameters()

    def scores():
        values = {"cpt_heldout_loss": evaluate(model, cpt_eval, args.device),
                  "sft_heldout_loss": evaluate(model, sft_eval_samples, args.device)}
        print(values, flush=True)
        return values

    args.output.mkdir(parents=True)
    if args.resume_cpt:
        with model.disable_adapter():
            before = scores()
        cpt_report = {"resumed_adapter_sha256": sha256(args.resume_cpt / "adapter_model.safetensors")}
    else:
        before = scores()
        cpt_report = train(model, cpt_samples, args.device, "cpt", recipe["cpt"])
        model.save_pretrained(args.output / "cpt-adapter", safe_serialization=True)
    cpt_scores = scores()
    sft_report = train(model, sft_samples, args.device, "sft", recipe["sft"])
    model.save_pretrained(args.output / "sft-adapter", safe_serialization=True)
    base = model.unload()
    model = PeftModel.from_pretrained(base, args.output / "sft-adapter", is_trainable=False)
    final = scores()
    report = {
        "kind": "local-lora-rehearsal", "base": source, "base_weights_sha256": sha256(args.base / "model.safetensors"),
        "device": args.device, "seed": SEED, "cpt_window": args.cpt_window,
        "sft_window": args.sft_window, "sft_records": len(sft), "sft_epochs": args.sft_epochs,
        "data_manifests": {split: sha256(args.data / split / "manifest.json") for split in ("cpt", "sft-train", "sft-eval")},
        "eval_samples": {"cpt": len(cpt_eval), "sft": len(sft_eval_samples)},
        "before": before, "cpt": {**cpt_report, **cpt_scores}, "sft": {**sft_report, **final},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
