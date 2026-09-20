"""One-time fetch + format of the two chat-pipeline corpora into
training/data/*.txt, the same bundled-file pattern tiny_shakespeare.txt
already uses. Run once before starting a pretrain or SFT run:

    python scripts/prepare_training_data.py

Both datasets are plain text with stories/examples already separated by the
literal string "<|endoftext|>" -- training/dataset.py tokenizes them with
that exact special token, so pretraining and SFT data look identical to the
rest of the DiLoCo pipeline (one token stream, random contiguous windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import CHAT_TEMPLATE, DATA_DIR  # noqa: E402

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStoriesV2-GPT4-train.txt"
)
DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)
EOT = "<|endoftext|>"


def fetch_tinystories(max_bytes: int, force: bool) -> None:
    out_path = DATA_DIR / "tinystories.txt"
    if out_path.exists() and not force:
        print(f"skip {out_path} (already exists, pass --force to refetch)")
        return

    print(f"fetching first {max_bytes / 1e6:.0f}MB of TinyStories...")
    with httpx.stream(
        "GET", TINYSTORIES_URL, follow_redirects=True,
        headers={"Range": f"bytes=0-{max_bytes - 1}"}, timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        chunks = []
        total = 0
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
    text = b"".join(chunks).decode("utf-8", errors="ignore")

    # Trim the trailing partial story back to the last complete boundary.
    last_eot = text.rfind(EOT)
    if last_eot == -1:
        raise RuntimeError("downloaded chunk contains no <|endoftext|> boundary -- try a larger --tinystories-mb")
    text = text[: last_eot + len(EOT)]

    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({len(text) / 1e6:.1f}MB, {text.count(EOT)} stories)")


def fetch_dolly(force: bool) -> None:
    out_path = DATA_DIR / "dolly15k_chat.txt"
    if out_path.exists() and not force:
        print(f"skip {out_path} (already exists, pass --force to refetch)")
        return

    print("fetching databricks-dolly-15k...")
    resp = httpx.get(DOLLY_URL, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()

    examples = []
    for line in resp.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = row["instruction"].strip()
        if row.get("context", "").strip():
            instruction = f"{instruction}\n\n{row['context'].strip()}"
        response = row["response"].strip()
        examples.append(CHAT_TEMPLATE.format(instruction=instruction) + response + "\n" + EOT)

    text = "\n".join(examples) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({len(text) / 1e6:.1f}MB, {len(examples)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tinystories-mb", type=int, default=50,
        help="how much of the (multi-GB) TinyStories corpus to keep, in MB (default: 50)",
    )
    parser.add_argument("--force", action="store_true", help="refetch even if the output file already exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetch_tinystories(args.tinystories_mb * 1_000_000, args.force)
    fetch_dolly(args.force)


if __name__ == "__main__":
    main()
