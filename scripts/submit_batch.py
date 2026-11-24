"""Submit a JSONL prompt file to the coordinator and tail batch progress."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help='JSONL file of {"prompt": ...} lines')
    parser.add_argument("--coordinator-url", default="http://localhost:8000")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--inference-backend", choices=("ollama", "parallax"), default="ollama")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--verification-pct", type=float, default=None)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.inference_backend == "parallax" and args.verification_pct not in (None, 0):
        parser.error("Parallax batches require --verification-pct 0 (or omit the flag)")

    prompts = load_prompts(args.file)
    print(f"submitting {len(prompts)} prompts (backend={args.inference_backend}, model={args.model_name})...")

    payload: dict = {
        "prompts": prompts,
        "model_name": args.model_name,
        "inference_backend": args.inference_backend,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
    }
    if args.verification_pct is not None:
        payload["verification_pct"] = args.verification_pct

    resp = httpx.post(f"{args.coordinator_url}/batches", json=payload, timeout=30)
    resp.raise_for_status()
    batch = resp.json()
    batch_id = batch["batch_id"]
    print(f"batch {batch_id} submitted ({batch['total_jobs']} jobs). tailing progress...")

    while True:
        resp = httpx.get(f"{args.coordinator_url}/batches/{batch_id}", timeout=30)
        resp.raise_for_status()
        p = resp.json()
        done = p["completed"] + p["verified"] + p["mismatched"] + p["failed"]
        pct = int(100 * done / p["total_jobs"]) if p["total_jobs"] else 100
        eta = f"{p['eta_seconds']:.0f}s" if p["eta_seconds"] is not None else "?"
        sys.stdout.write(
            f"\r[{pct:3d}%] {done}/{p['total_jobs']} done "
            f"(queued={p['queued']} dispatched={p['dispatched']} running={p['running']} "
            f"failed={p['failed']} retries={p['retry_count']}) "
            f"{p['throughput_tokens_per_sec']:.1f} tok/s, eta {eta}   "
        )
        sys.stdout.flush()
        if done >= p["total_jobs"]:
            break
        time.sleep(args.poll_interval)

    print(f"\nbatch complete. Results: {args.coordinator_url}/batches/{batch_id}/jobs")


if __name__ == "__main__":
    main()
