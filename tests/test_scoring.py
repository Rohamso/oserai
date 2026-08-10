from __future__ import annotations

from coordinator.dispatch import dispatch_queued_jobs
from coordinator.scoring import pick_best_node, score_node
from tests.conftest import MODEL, make_batch_and_job, make_node


def test_score_node_uses_historical_tokens_per_sec():
    node = {"jobs_done": 4, "duration_total": 20.0, "tokens_total": 200}
    assert score_node(node, job={}, median_tokens_per_sec=5.0) == 10.0


def test_score_node_defaults_new_node_to_median():
    node = {"jobs_done": 0, "duration_total": 0.0, "tokens_total": 0}
    assert score_node(node, job={}, median_tokens_per_sec=7.5) == 7.5


def test_pick_best_node_prefers_higher_score():
    fast = {"id": "fast", "jobs_done": 10, "tokens_total": 1000, "duration_total": 10.0}
    slow = {"id": "slow", "jobs_done": 10, "tokens_total": 100, "duration_total": 10.0}
    best = pick_best_node([slow, fast], job={}, median_tokens_per_sec=1.0)
    assert best["id"] == "fast"


def test_pick_best_node_empty_candidates_returns_none():
    assert pick_best_node([], job={}, median_tokens_per_sec=1.0) is None


def test_dispatch_assigns_queued_job_to_fastest_idle_matching_node(db, config):
    slow_id = make_node(db, name="slow-node", tokens_total=10, duration_total=10.0)   # 1 tok/s
    fast_id = make_node(db, name="fast-node", tokens_total=100, duration_total=10.0)  # 10 tok/s
    make_node(db, name="wrong-model", model_name="other-model", tokens_total=1000, duration_total=1.0)
    _, job_id = make_batch_and_job(db)

    assignments = dispatch_queued_jobs(db, config)

    assert (job_id, fast_id) in assignments
    job = db.get_job(job_id)
    assert job["status"] == "dispatched"
    assert job["assigned_node_id"] == fast_id
    assert db.get_node(fast_id)["status"] == "busy"
    assert db.get_node(slow_id)["status"] == "idle"


def test_dispatch_skips_offline_nodes(db, config):
    node_id = make_node(db, name="offline-node")
    db.set_node_status(node_id, "offline")
    _, job_id = make_batch_and_job(db)

    assignments = dispatch_queued_jobs(db, config)

    assert assignments == []
    assert db.get_job(job_id)["status"] == "queued"


def test_dispatch_sends_verification_duplicate_to_a_different_node(db, config):
    node_a = make_node(db, name="node-a", tokens_total=100, duration_total=10.0)
    node_b = make_node(db, name="node-b", tokens_total=100, duration_total=10.0)

    batch_id = db.create_batch(
        model_name=MODEL, max_tokens=100, timeout_s=300,
        verification_pct=1.0, payout_rate=0.00001, total_jobs=1,
    )
    group_id = "grp-1"
    primary_id = db.create_job(batch_id, "prompt", 100, 300, verification_group_id=group_id, is_duplicate=False)
    dup_id = db.create_job(batch_id, "prompt", 100, 300, verification_group_id=group_id, is_duplicate=True)

    dispatch_queued_jobs(db, config)

    primary = db.get_job(primary_id)
    dup = db.get_job(dup_id)
    assert primary["assigned_node_id"] is not None
    assert dup["assigned_node_id"] is not None
    assert primary["assigned_node_id"] != dup["assigned_node_id"]
    assert {primary["assigned_node_id"], dup["assigned_node_id"]} == {node_a, node_b}
