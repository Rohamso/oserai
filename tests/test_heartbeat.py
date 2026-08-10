from __future__ import annotations

from coordinator.dispatch import sweep_offline_nodes
from tests.conftest import make_batch_and_job, make_node


def test_node_with_stale_heartbeat_marked_offline(db, config):
    node_id = make_node(db)
    now = db.get_node(node_id)["last_heartbeat"] + config.heartbeat_timeout_s + 1

    offline = sweep_offline_nodes(db, config, now=now)

    assert offline == [node_id]
    assert db.get_node(node_id)["status"] == "offline"


def test_node_with_recent_heartbeat_stays_online(db, config):
    node_id = make_node(db)
    now = db.get_node(node_id)["last_heartbeat"] + 1  # well inside the 90s window

    offline = sweep_offline_nodes(db, config, now=now)

    assert offline == []
    assert db.get_node(node_id)["status"] == "idle"


def test_offline_node_requeues_its_in_flight_job(db, config):
    node_id = make_node(db)
    _, job_id = make_batch_and_job(db)
    db.assign_job(job_id, node_id)
    db.mark_running(job_id)

    now = db.get_node(node_id)["last_heartbeat"] + config.heartbeat_timeout_s + 1
    sweep_offline_nodes(db, config, now=now)

    job = db.get_job(job_id)
    assert job["status"] == "queued"
    assert job["assigned_node_id"] is None
    assert job["requeue_count"] == 1
    assert db.get_node(node_id)["requeue_count"] == 1
    # regression: requeuing the job must not bounce the node back to idle --
    # it's offline, and shouldn't be handed the very next queued job.
    assert db.get_node(node_id)["status"] == "offline"


def test_offline_node_is_not_immediately_reassigned_the_requeued_job(db, config):
    from coordinator.dispatch import dispatch_queued_jobs

    node_id = make_node(db)  # only node in the network
    _, job_id = make_batch_and_job(db)
    db.assign_job(job_id, node_id)
    db.mark_running(job_id)

    now = db.get_node(node_id)["last_heartbeat"] + config.heartbeat_timeout_s + 1
    sweep_offline_nodes(db, config, now=now)
    dispatch_queued_jobs(db, config)  # background loop would run this right after

    job = db.get_job(job_id)
    assert job["status"] == "queued"
    assert job["assigned_node_id"] is None
    assert db.get_node(node_id)["status"] == "offline"


def test_new_node_never_heartbeated_is_not_immediately_offline_before_timeout():
    # A node created "now" with a fresh heartbeat should not be swept this instant.
    from coordinator.db import Database
    from coordinator.config import CoordinatorConfig

    database = Database(":memory:")
    cfg = CoordinatorConfig(heartbeat_timeout_s=90)
    node_id = make_node(database)

    offline = sweep_offline_nodes(database, cfg)  # uses real time.time()

    assert offline == []
    assert database.get_node(node_id)["status"] == "idle"
    database.close()
