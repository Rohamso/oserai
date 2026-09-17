"""Import a scripts/train_local.py or scripts/train_local_lora.py checkpoint
into the coordinator's DB as a completed training run, so the dashboard /
POST /training/runs/{id}/generate can serve it immediately -- no need to
re-run training through the coordinator+train_agent distributed path just
to chat with a model you already trained locally.

    python scripts/import_local_checkpoint.py out/sft.pt --name "local-sft-demo"
    python scripts/import_local_checkpoint.py out/lora_adapter.pt --name "local-lora-demo"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordinator.config import CoordinatorConfig  # noqa: E402
from coordinator.db import Database  # noqa: E402
from training.checkpoint_io import state_dict_to_bytes  # noqa: E402


def import_lora_checkpoint(db: Database, ckpt: dict, name: str) -> str:
    from training.lora import load_tokenizer  # noqa: E402 (heavy deps, only needed on this path)

    base_model_id = ckpt["base_model_id"]
    tokenizer = load_tokenizer(base_model_id)
    model_backend = "lora_" + base_model_id.rsplit("/", 1)[-1].lower()

    # inner_*/outer_*/quorum/round_timeout_s/max_rounds are DiLoCo round
    # params -- meaningless for an already-trained, externally-imported
    # checkpoint, so they're set to inert placeholders rather than left
    # unset (the schema requires them).
    run_id = db.create_training_run(
        name=name, vocab_size=tokenizer.vocab_size, dataset_name="dolly_chat",
        inner_lr=1e-3, inner_steps_h=1, inner_batch_size=1,
        outer_lr=1.0, outer_momentum=0.0, max_rounds=1, round_timeout_s=1, quorum=1,
        model_backend=model_backend, base_model_id=base_model_id,
        lora_r=ckpt["lora_r"], lora_alpha=ckpt["lora_alpha"], lora_dropout=ckpt["lora_dropout"],
        lora_target_modules=",".join(ckpt["lora_target_modules"]),
    )
    checkpoint_id = db.save_checkpoint(
        run_id, "global_weights", state_dict_to_bytes(ckpt["adapter_state_dict"]), round_num=0,
    )
    db.advance_run_round(run_id, 0, checkpoint_id, None)
    return run_id


def import_scratch_gpt_checkpoint(db: Database, ckpt: dict, name: str) -> str:
    config = ckpt["config"]
    dataset_name = ckpt.get("dataset_name", "tiny_shakespeare")

    run_id = db.create_training_run(
        name=name, n_layer=config["n_layer"], n_head=config["n_head"], n_embd=config["n_embd"],
        block_size=config["block_size"], vocab_size=config["vocab_size"], dataset_name=dataset_name,
        inner_lr=1e-3, inner_steps_h=1, inner_batch_size=1,
        outer_lr=1.0, outer_momentum=0.0, max_rounds=1, round_timeout_s=1, quorum=1,
    )
    checkpoint_id = db.save_checkpoint(
        run_id, "global_weights", state_dict_to_bytes(ckpt["model_state_dict"]), round_num=0,
    )
    db.advance_run_round(run_id, 0, checkpoint_id, None)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="path to a train_local.py or train_local_lora.py .pt checkpoint")
    parser.add_argument("--name", default=None, help="training run name (default: checkpoint filename)")
    parser.add_argument(
        "--db-path", default=None,
        help="coordinator DB path (default: coordinator_config.yaml / CoordinatorConfig default)",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    name = args.name or Path(args.checkpoint).stem

    db_path = args.db_path or CoordinatorConfig.load().db_path
    db = Database(db_path)

    if "adapter_state_dict" in ckpt:
        run_id = import_lora_checkpoint(db, ckpt, name)
    else:
        run_id = import_scratch_gpt_checkpoint(db, ckpt, name)

    db.set_run_status(run_id, "completed", completed_at=time.time())
    db.close()

    print(f"imported {args.checkpoint} as training run {run_id} ({name!r}) into {db_path}")
    print("start (or restart) the coordinator, then chat via the dashboard 'Talk to a trained run' box, or:")
    print(
        f'  curl -X POST http://localhost:8000/training/runs/{run_id}/generate '
        f'-H "Content-Type: application/json" -d \'{{"prompt": "Hello!"}}\''
    )


if __name__ == "__main__":
    main()
