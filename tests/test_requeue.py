from __future__ import annotations

from coordinator.dispatch import handle_job_result, sweep_job_timeouts
from tests.conftest import make_batch_and_job, make_node


def test_timed_out_job_is_requeued_and_node_freed(db, config):
    node_id = make_node(db)
    _, job_id = make_batch_and_job(db, timeout_s=5)
    db.assign_job(job_id, node_id)
    db.set_node_status(node_id, "busy")
    db.mark_running(job_id)

    now = db.get_job(job_id)["dispatched_at"] + 10  # well past the 5s timeout

    timed_out = sweep_job_timeouts(db, config, now=now)

    assert timed_out == [job_id]
    job = db.get_job(job_id)
    assert job["status"] == "queued"
    assert job["assigned_node_id"] is None
    assert job["requeue_count"] == 1
    assert db.get_node(node_id)["status"] == "idle"
    assert db.get_node(node_id)["requeue_count"] == 1


def test_job_fails_terminally_after_max_attempts(db, config):
    node_id = make_node(db)
    _, job_id = make_batch_and_job(db, timeout_s=5)

    for attempt in range(config.max_job_attempts):
        db.assign_job(job_id, node_id)
        db.mark_running(job_id)
        now = db.get_job(job_id)["dispatched_at"] + 10
        sweep_job_timeouts(db, config, now=now)

    job = db.get_job(job_id)
    assert job["status"] == "failed"
    assert job["requeue_count"] == config.max_job_attempts - 1


def test_failed_result_below_max_attempts_gets_requeued(db, config):
    node_id = make_node(db)
    _, job_id = make_batch_and_job(db)
    db.assign_job(job_id, node_id)
    db.mark_running(job_id)

    job = handle_job_result(db, config, job_id, node_id, False, "", "", 0, 0.0, 0.0, "ollama timeout")

    assert job["status"] == "queued"
    assert job["assigned_node_id"] is None
    assert db.get_node(node_id)["jobs_failed"] == 1


def test_non_timed_out_running_job_is_left_alone(db, config):
    node_id = make_node(db)
    _, job_id = make_batch_and_job(db, timeout_s=300)
    db.assign_job(job_id, node_id)
    db.mark_running(job_id)

    now = db.get_job(job_id)["dispatched_at"] + 1  # well within timeout
    timed_out = sweep_job_timeouts(db, config, now=now)

    assert timed_out == []
    assert db.get_job(job_id)["status"] == "running"
