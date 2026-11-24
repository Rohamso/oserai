"""Measure sequential inference latency and usage against an agent's backend.

Run with `python -m scripts.benchmark_inference prompts.jsonl --config ...`.
This calls the runtime directly, without submitting jobs or ledger entries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent.config import AgentConfig
from agent.inference import run_inference
from common.inference import InferenceError
from scripts.submit_batch import load_prompts


def benchmark(config: AgentConfig, prompts: list[str], max_tokens: int, timeout_s: float) -> dict:
    results = []
    started = time.monotonic()
    for index, prompt in enumerate(prompts):
        request_start = time.monotonic()
        try:
            result = run_inference(config, config.model_name, prompt, max_tokens, timeout_s)
            results.append({
                "index": index,
                "success": True,
                "latency_s": time.monotonic() - request_start,
                "tokens_generated": result.tokens_generated,
                "output_text": result.output_text,
            })
        except InferenceError as exc:
            results.append({
                "index": index, "success": False,
                "latency_s": time.monotonic() - request_start, "error": str(exc),
            })
    wall_time_s = time.monotonic() - started
    successful = [r for r in results if r["success"]]
    tokens = sum(r["tokens_generated"] for r in successful)
    return {
        "inference_backend": config.inference_backend,
        "model_name": config.model_name,
        "requests": len(results),
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "wall_time_s": wall_time_s,
        "tokens_generated": tokens,
        "end_to_end_tokens_per_sec": tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "median_latency_s": statistics.median(r["latency_s"] for r in successful) if successful else None,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help='JSONL file of {"prompt": ...} lines')
    parser.add_argument("--config", required=True, help="Agent YAML config (Ollama or Parallax)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", help="Write the full JSON report, including answers")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout_s <= 0:
        parser.error("--max-tokens and --timeout-s must be positive")
    config = AgentConfig.load(args.config)
    prompts = load_prompts(args.file)
    if not prompts:
        parser.error("prompt file is empty")
    report = benchmark(config, prompts, args.max_tokens, args.timeout_s)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    for result in report["results"]:
        if not result["success"]:
            print(f"request {result['index']} failed: {result['error']}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
