"""Generate a JSONL file of short summarization/QA test prompts for demos."""

from __future__ import annotations

import argparse
import json
import random

_TOPICS = [
    "the water cycle", "the French Revolution", "how neural networks learn",
    "the history of the printing press", "photosynthesis", "the Roman aqueduct system",
    "how vaccines work", "the causes of the 2008 financial crisis", "black holes",
    "the invention of the internet", "how bees communicate", "the Silk Road",
    "the Manhattan Project", "coral reef ecosystems", "the fall of the Berlin Wall",
]

_TEMPLATES = [
    "In two sentences, explain {topic}.",
    "Summarize {topic} for a curious 10-year-old in 3 sentences.",
    "What is the single most important fact about {topic}? Answer in one sentence.",
    "Give a concise, factual overview of {topic} in under 50 words.",
]


def generate_prompts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        rng.choice(_TEMPLATES).format(topic=rng.choice(_TOPICS))
        for _ in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="test_batch.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = generate_prompts(args.count, args.seed)
    with open(args.output, "w") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
