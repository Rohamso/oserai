"""Aggregation logic behind GET /stats and GET /batches/{id}."""

from __future__ import annotations

import time
from typing import Any

from coordinator.db import Database

COMPLETED_STATUSES = ("completed", "verified", "mismatched")
PRIMARY_TERMINAL_STATUSES = ("completed", "verified", "mismatched", "failed")


def compute_batch_progress(db: Database, batch: dict[str, Any]) -> dict[str, Any]:
    jobs = db.list_jobs_for_batch(batch["id"])
    primary = [j for j in jobs if not j["is_duplicate"]]

    counts = {s: 0 for s in ("queued", "dispatched", "running", "completed", "verified", "mismatched", "failed")}
    for j in primary:
        counts[j["status"]] = counts.get(j["status"], 0) + 1

    retry_count = sum(j["requeue_count"] for j in jobs)

    done_jobs = [j for j in jobs if j["status"] in COMPLETED_STATUSES and j["duration_s"]]
    total_tokens = sum(j["tokens_generated"] or 0 for j in done_jobs)
    total_duration = sum(j["duration_s"] or 0 for j in done_jobs)
    throughput = total_tokens / total_duration if total_duration > 0 else 0.0

    done_primary = sum(counts[s] for s in PRIMARY_TERMINAL_STATUSES)
    remaining = max(batch["total_jobs"] - done_primary, 0)
    avg_duration = (total_duration / len(done_jobs)) if done_jobs else None
    if remaining == 0:
        eta = 0.0
    elif avg_duration:
        eta = remaining * avg_duration
    else:
        eta = None

    return {
        "batch_id": batch["id"],
        "model_name": batch["model_name"],
        "total_jobs": batch["total_jobs"],
        "queued": counts["queued"],
        "dispatched": counts["dispatched"],
        "running": counts["running"],
        "completed": counts["completed"],
        "verified": counts["verified"],
        "mismatched": counts["mismatched"],
        "failed": counts["failed"],
        "retry_count": retry_count,
        "throughput_tokens_per_sec": throughput,
        "eta_seconds": eta,
    }


def compute_node_stats(node: dict[str, Any], heartbeat_timeout_s: int) -> dict[str, Any]:
    jobs_done = node["jobs_done"]
    jobs_failed = node["jobs_failed"]
    total_attempts = jobs_done + jobs_failed
    failure_rate = jobs_failed / total_attempts if total_attempts else 0.0
    tps = node["tokens_total"] / node["duration_total"] if node["duration_total"] else 0.0

    # Phase 1 uptime is a point-in-time snapshot (online now vs. not), not a
    # time-weighted history -- good enough for a PoC dashboard.
    now = time.time()
    if node["status"] == "offline":
        uptime_pct = 0.0
    elif node["last_heartbeat"] and (now - node["last_heartbeat"]) < heartbeat_timeout_s:
        uptime_pct = 100.0
    else:
        uptime_pct = 0.0

    return {
        "node_id": node["id"],
        "name": node["name"],
        "status": node["status"],
        "jobs_done": jobs_done,
        "jobs_failed": jobs_failed,
        "tokens_per_sec_avg": tps,
        "failure_rate": failure_rate,
        "mismatch_count": node["mismatch_count"],
        "uptime_pct": uptime_pct,
    }
