"""Node scoring for job dispatch."""

from __future__ import annotations

from typing import Any


def score_node(node: dict[str, Any], job: dict[str, Any], median_tokens_per_sec: float) -> float:
    """Score how good a fit `node` is to run `job`. Higher wins.

    Phase 1: prefer the node with the highest historical (lifetime) tokens/sec.
    A node with no completed jobs yet defaults to the network median so new
    nodes get a fair shot at work instead of starving behind established ones.

    Phase 2 will extend this to fold in `measured_watts` and the grid carbon
    intensity for `grid_zone`, so a fast-but-dirty node can be scored below a
    slower-but-cleaner one. Keep this function's signature stable so that
    change is additive (new optional kwargs / fields on `node`), not a rewrite
    of the dispatch loop that calls it.
    """
    jobs_done = node.get("jobs_done", 0) or 0
    duration_total = node.get("duration_total", 0.0) or 0.0
    if jobs_done > 0 and duration_total > 0:
        tokens_total = node.get("tokens_total", 0) or 0
        return tokens_total / duration_total
    return median_tokens_per_sec


def pick_best_node(candidates: list[dict[str, Any]], job: dict[str, Any], median_tokens_per_sec: float) -> dict[str, Any] | None:
    """Pick the highest-scoring node for `job` out of `candidates`. None if empty."""
    if not candidates:
        return None
    return max(candidates, key=lambda n: score_node(n, job, median_tokens_per_sec))
