"""Background dispatch cycle: heartbeat sweep, timeout/requeue handling,
job assignment, and verification-pair comparison.

Everything here operates on a `Database` instance and is plain synchronous
Python so it can be unit tested without spinning up FastAPI or asyncio.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from common.hashing import sha256_hex
from common.logging_setup import log_event
from coordinator.config import CoordinatorConfig
from coordinator.db import Database
from coordinator.scoring import pick_best_node

ACTIVE_JOB_STATUSES = ("dispatched", "running")

logger = logging.getLogger("coordinator.dispatch")


def _requeue_or_fail(
    db: Database, job: dict[str, Any], max_attempts: int, reason: str, free_node: bool = True,
) -> None:
    """free_node=False when the node itself was just marked offline -- it must
    stay offline, not bounce back to idle and get handed the next job too."""
    node_id = job.get("assigned_node_id")
    if node_id:
        db.record_node_requeue(node_id)
        if free_node:
            db.set_node_status(node_id, "idle")
    if job["attempt_count"] >= max_attempts:
        db.terminal_fail_job(job["id"], reason)
        log_event(logger, "job_failed_terminal", job_id=job["id"], reason=reason, attempts=job["attempt_count"])
    else:
        db.requeue_job(job["id"])
        log_event(logger, "job_requeued", job_id=job["id"], reason=reason, attempts=job["attempt_count"])


def sweep_offline_nodes(db: Database, config: CoordinatorConfig, now: Optional[float] = None) -> list[str]:
    """Mark nodes with a stale heartbeat offline and requeue/fail any jobs they held."""
    now = now if now is not None else time.time()
    cutoff = now - config.heartbeat_timeout_s
    newly_offline = db.sweep_offline_nodes(cutoff)
    for node_id in newly_offline:
        log_event(logger, "node_offline", node_id=node_id)
        for job in db.list_stuck_jobs(ACTIVE_JOB_STATUSES):
            if job["assigned_node_id"] == node_id:
                _requeue_or_fail(db, job, config.max_job_attempts, reason="node_offline", free_node=False)
    return newly_offline


def sweep_job_timeouts(db: Database, config: CoordinatorConfig, now: Optional[float] = None) -> list[str]:
    """Requeue/fail jobs that have exceeded their per-job timeout, even if the node
    is still heartbeating (covers a stuck/hung inference call)."""
    now = now if now is not None else time.time()
    timed_out = []
    for job in db.list_stuck_jobs(ACTIVE_JOB_STATUSES):
        if not job["dispatched_at"]:
            continue
        if now - job["dispatched_at"] > job["timeout_s"]:
            timed_out.append(job["id"])
            _requeue_or_fail(db, job, config.max_job_attempts, reason="timeout")
    return timed_out


def dispatch_queued_jobs(db: Database, config: CoordinatorConfig) -> list[tuple[str, str]]:
    """Assign queued jobs to idle, online nodes running the matching model.

    Returns list of (job_id, node_id) assignments made this cycle.
    """
    assignments: list[tuple[str, str]] = []
    reserved_node_ids: set[str] = set()
    median = db.median_tokens_per_sec()

    queued_jobs = _queued_jobs_with_model(db)
    targets = {(j["model_name"], j["inference_backend"]) for j in queued_jobs}
    for model_name, inference_backend in targets:
        queued = [j for j in queued_jobs if (j["model_name"], j["inference_backend"]) == (model_name, inference_backend)]
        idle_nodes = [
            n for n in db.list_nodes_by_status_and_model("idle", model_name, inference_backend)
            if n["id"] not in reserved_node_ids
        ]
        for job in queued:
            excluded = _excluded_node_for_verification(db, job)
            candidates = [n for n in idle_nodes if n["id"] != excluded and n["id"] not in reserved_node_ids]
            best = pick_best_node(candidates, job, median)
            if best is None:
                continue
            db.assign_job(job["id"], best["id"])
            db.set_node_status(best["id"], "busy")
            reserved_node_ids.add(best["id"])
            assignments.append((job["id"], best["id"]))
            log_event(logger, "job_dispatched", job_id=job["id"], node_id=best["id"], model_name=model_name)
    return assignments


def _queued_jobs_with_model(db: Database) -> list[dict[str, Any]]:
    """All queued jobs, each annotated with its batch's model_name."""
    out = []
    batches_cache: dict[str, dict[str, Any]] = {}
    for batch in db.list_batches():
        batches_cache[batch["id"]] = batch
        for job in db.list_jobs_for_batch(batch["id"]):
            if job["status"] == "queued":
                job = dict(job)
                job["model_name"] = batch["model_name"]
                job["inference_backend"] = batch["inference_backend"]
                out.append(job)
    return out


def _excluded_node_for_verification(db: Database, job: dict[str, Any]) -> Optional[str]:
    """If this job is one half of a verification pair, don't send it to the same
    node that already holds (or ran) the sibling job."""
    group_id = job.get("verification_group_id")
    if not group_id:
        return None
    sibling = db.get_verification_sibling(job["id"], group_id)
    if sibling and sibling.get("assigned_node_id"):
        return sibling["assigned_node_id"]
    return None


def check_verification_pair(db: Database, job_id: str) -> None:
    """After a job completes, if it belongs to a verification pair and both
    halves are now complete, compare output hashes and record the result."""
    job = db.get_job(job_id)
    if job is None or not job.get("verification_group_id"):
        return
    sibling = db.get_verification_sibling(job_id, job["verification_group_id"])
    if sibling is None or sibling["status"] != "completed":
        return
    if job["status"] != "completed":
        return

    match = job["output_sha256"] == sibling["output_sha256"] and bool(job["output_sha256"])
    db.set_verification_result(job["id"], match)
    db.set_verification_result(sibling["id"], match)
    if not match:
        db.record_node_mismatch(job["assigned_node_id"])
        db.record_node_mismatch(sibling["assigned_node_id"])
        log_event(
            logger, "verification_mismatch",
            job_id=job["id"], sibling_job_id=sibling["id"],
            node_id=job["assigned_node_id"], sibling_node_id=sibling["assigned_node_id"],
            output_a=job["output_text"], output_b=sibling["output_text"],
            sha_a=job["output_sha256"], sha_b=sibling["output_sha256"],
        )
    else:
        log_event(
            logger, "verification_match",
            job_id=job["id"], sibling_job_id=sibling["id"],
            node_id=job["assigned_node_id"], sibling_node_id=sibling["assigned_node_id"],
        )


def handle_job_result(
    db: Database, config: CoordinatorConfig, job_id: str, node_id: str,
    success: bool, output_text: str, output_sha256: str,
    tokens_generated: int, duration_s: float, tokens_per_sec: float,
    error: Optional[str],
) -> dict[str, Any]:
    """Apply a POST /jobs/{id}/result payload. Returns the updated job row."""
    job = db.get_job(job_id)
    if job is None:
        raise ValueError(f"unknown job {job_id}")
    if job["assigned_node_id"] != node_id:
        raise ValueError(f"job {job_id} is not assigned to node {node_id}")
    if job["status"] in ("completed", "verified", "mismatched"):
        # Result submission retries after a lost HTTP response must not pay
        # twice or free a node that has already started another job.
        return job
    if job["status"] not in ACTIVE_JOB_STATUSES:
        raise ValueError(f"job {job_id} is not active")

    db.set_node_status(node_id, "idle")

    if not success:
        db.record_node_job_failure(node_id)
        db.fail_job_attempt(job_id, error or "unknown error")
        _requeue_or_fail(db, job, config.max_job_attempts, reason=error or "reported_failure")
        return db.get_job(job_id)

    computed_sha = sha256_hex(output_text)
    db.complete_job(
        job_id, output_text, output_sha256 or computed_sha,
        tokens_generated, duration_s, tokens_per_sec,
    )
    db.record_node_job_success(node_id, tokens_generated, duration_s)

    batch = db.get_batch(job["batch_id"])
    payout_rate = batch["payout_rate"] if batch else config.default_payout_rate
    db.append_ledger(node_id, job_id, tokens_generated, duration_s, tokens_generated * payout_rate)

    check_verification_pair(db, job_id)
    return db.get_job(job_id)


def run_dispatch_cycle(db: Database, config: CoordinatorConfig) -> None:
    """One full tick of the background dispatch loop."""
    sweep_offline_nodes(db, config)
    sweep_job_timeouts(db, config)
    dispatch_queued_jobs(db, config)
