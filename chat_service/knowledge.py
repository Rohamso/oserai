"""Retrieve sections from the public OserAI contributor guides."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARALLAX_GUIDE = "docs/parallax-mvp.md"
TRAINER_GUIDE = "desktop_app/README.md"
# These specific sections can be retrieved and read via the public guides route.
TOPICS = {
    "parallax-cluster": (
        PARALLAX_GUIDE, "1. Start Parallax separately",
        "start install setup join add connect contribute compute scheduler ready mac macos linux windows phone mobile android ios model",
    ),
    "parallax-discovery": (
        PARALLAX_GUIDE, "Troubleshoot worker discovery",
        "find discover discovery scheduler network wifi firewall permission permissions denied blocked port error route troubleshoot troubleshooting",
    ),
    "parallax-coordinator": (
        PARALLAX_GUIDE, "2. Start an isolated OserAI coordinator",
        "coordinator database sqlite isolated python uvicorn",
    ),
    "parallax-bridge": (
        PARALLAX_GUIDE, "3. Start the bridge",
        "bridge config configuration registration register reregister endpoint agent state",
    ),
    "parallax-batch": (
        PARALLAX_GUIDE, "4. Submit a batch and read answers",
        "batch batches submit job jobs prompts answers results dashboard verification tokens",
    ),
    "parallax-benchmark": (
        PARALLAX_GUIDE, "5. Benchmark before expanding",
        "benchmark benchmarks latency throughput compare measurement dropout recover recovery acceptance offline",
    ),
    "trainer-worker": (
        TRAINER_GUIDE, "What a contributor needs to fill in",
        "training trainer train diloco lora shard shards cuda mps cpu run",
    ),
}
PROJECT_TERMS = {"oserai", "parallax", "diloco"}
WORKER_TERMS = {"worker", "workers", "contribute", "contributing", "contributor", "contributors", "node", "nodes"}
WORKER_TASK_TERMS = {
    "join", "connect", "discover", "scheduler", "mac", "macos", "linux", "windows",
    "phone", "mobile", "android", "ios", "training", "compute", "need",
}
FOLLOW_UP = re.compile(r"^(?:and\b|what about\b|what if\b|how about\b|can it\b|does it\b)", re.I)


@dataclass(frozen=True)
class GuideSection:
    slug: str
    title: str
    text: str
    terms: frozenset[str]


def load_guides(root: Path = ROOT) -> dict[str, GuideSection]:
    guides = {}
    for filename in (PARALLAX_GUIDE, TRAINER_GUIDE):
        try:
            markdown = (root / filename).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        parts = re.split(r"^## (.+)$", markdown, flags=re.MULTILINE)
        sections = dict(zip(parts[1::2], parts[2::2]))
        for slug, (source, heading, keywords) in TOPICS.items():
            if source == filename and heading in sections:
                title = "OserAI Parallax guide" if source == PARALLAX_GUIDE else "OserAI Trainer guide"
                guides[slug] = GuideSection(slug, f"{title} · {heading}", sections[heading].strip(), frozenset(keywords.split()))
    return guides


def match_guide(guides: dict[str, GuideSection], questions: list[str]) -> GuideSection | None:
    latest = questions[-1]
    current = set(re.findall(r"[a-z0-9]+", latest.lower()))
    prior = set()
    if len(questions) > 1 and FOLLOW_UP.match(latest.lstrip()):
        prior = set(re.findall(r"[a-z0-9]+", questions[-2].lower()))
    context = current | prior
    worker_question = context & WORKER_TERMS and context & WORKER_TASK_TERMS
    if not (context & PROJECT_TERMS or worker_question or {"join", "network"} <= context):
        return None

    best = None
    best_score = 0
    for guide in guides.values():
        matches = len(current & guide.terms)
        score = matches * 3 + len(prior & guide.terms)
        if matches and score > best_score:
            best, best_score = guide, score
    return best


def guide_context(guide: GuideSection | None) -> str:
    if guide is None:
        return ""
    return (
        "\n\nUse this OserAI guide excerpt for relevant project questions. Copy command flags "
        "and paths accurately. Say when the guide does not cover a detail. The gateway adds "
        f"the guide link after the reply.\n\n{guide.title}\n{guide.text[:2400]}"
    )
