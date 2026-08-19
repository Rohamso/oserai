from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from coordinator.db import Database
from coordinator.dispatch import sweep_offline_nodes
from coordinator.training_dispatch import (
    aggregate_closed_rounds,
    close_ready_rounds,
    start_run,
    submit_delta,
)
from training.checkpoint_io import state_dict_to_bytes


def make_training_node(db: Database, name: str = "gpu-a") -> str:
    node_id = db.register_node(
        name=name, token=f"tok-{name}", city=None, country=None,
        hardware="rtx-4090", model_name="nanogpt-shakespeare-char",
        max_concurrent_jobs=1, measured_watts=None, grid_zone=None, role="training",
    )
    db.update_heartbeat(node_id, "idle", cpu_pct=5.0, ram_free_mb=8192.0)
    return node_id


def make_run(db: Database, round_timeout_s: int = 600, quorum: int = 1, max_rounds: int = 5) -> str:
    run_id = db.create_training_run(
        name="test-run", n_layer=1, n_head=1, n_embd=4, block_size=8, vocab_size=10,
        dataset_name="tiny_shakespeare", inner_lr=1e-3, inner_steps_h=5, inner_batch_size=4,
        outer_lr=0.5, outer_momentum=0.9, max_rounds=max_rounds,
        round_timeout_s=round_timeout_s, quorum=quorum,
    )
    initial_weights = {"w": torch.tensor([1.0, 2.0])}
    checkpoint_id = db.save_checkpoint(run_id, "global_weights", state_dict_to_bytes(initial_weights), round_num=0)
    db.advance_run_round(run_id, 0, checkpoint_id, None)
    return run_id


def fake_delta_bytes(values: list[float]) -> bytes:
    return state_dict_to_bytes({"w": torch.tensor(values)})


def test_fast_close_when_both_online_nodes_submit(db):
    node_a = make_training_node(db, "gpu-a")
    node_b = make_training_node(db, "gpu-b")
    run_id = make_run(db)
    start_run(db, run_id)
    round_row = db.get_current_round(run_id)
    assert round_row["status"] == "open"

    submit_delta(db, round_row["id"], node_a, fake_delta_bytes([0.1, 0.2]), 1.0, 5, 1.0)
    submit_delta(db, round_row["id"], node_b, fake_delta_bytes([0.3, 0.4]), 1.1, 5, 1.2)

    # well before deadline -- fast-close should still fire since both online nodes reported
    closed = close_ready_rounds(db, now=round_row["opened_at"] + 1)
    assert closed == [round_row["id"]]
    assert aggregate_closed_rounds(db) == [round_row["id"]]

    updated = db.get_round(round_row["id"])
    assert updated["status"] == "applied"
    assert updated["participant_count"] == 2

    run = db.get_training_run(run_id)
    assert run["current_round"] == 1
    next_round = db.get_current_round(run_id)
    assert next_round["round_num"] == 1
    assert next_round["status"] == "open"


def test_slow_close_with_single_straggler(db):
    node_a = make_training_node(db, "gpu-a")
    make_training_node(db, "gpu-b")  # stays online, never submits
    run_id = make_run(db, round_timeout_s=5)
    start_run(db, run_id)
    round_row = db.get_current_round(run_id)

    submit_delta(db, round_row["id"], node_a, fake_delta_bytes([0.1, 0.1]), 1.0, 5, 1.0)

    # before deadline: only 1 of 2 online nodes reported, not ready
    assert close_ready_rounds(db, now=round_row["opened_at"] + 1) == []

    # after deadline: closes anyway with quorum=1 satisfied by node_a alone
    closed = close_ready_rounds(db, now=round_row["deadline_at"] + 1)
    assert closed == [round_row["id"]]
    aggregate_closed_rounds(db)

    updated = db.get_round(round_row["id"])
    assert updated["status"] == "applied"
    assert updated["participant_count"] == 1


def test_round_skipped_when_nobody_submits(db):
    make_training_node(db, "gpu-a")
    run_id = make_run(db, round_timeout_s=5)
    start_run(db, run_id)
    round_row = db.get_current_round(run_id)
    initial_checkpoint = round_row["base_checkpoint_id"]

    close_ready_rounds(db, now=round_row["deadline_at"] + 1)
    aggregate_closed_rounds(db)

    updated = db.get_round(round_row["id"])
    assert updated["status"] == "skipped"
    assert updated["participant_count"] == 0

    run = db.get_training_run(run_id)
    assert run["latest_checkpoint_id"] == initial_checkpoint  # weights carried forward unchanged

    next_round = db.get_current_round(run_id)
    assert next_round["round_num"] == 1
    assert next_round["base_checkpoint_id"] == initial_checkpoint


def test_duplicate_delta_from_same_node_is_ignored(db):
    node_a = make_training_node(db, "gpu-a")
    run_id = make_run(db)
    start_run(db, run_id)
    round_row = db.get_current_round(run_id)

    first = submit_delta(db, round_row["id"], node_a, fake_delta_bytes([0.1, 0.1]), 1.0, 5, 1.0)
    second = submit_delta(db, round_row["id"], node_a, fake_delta_bytes([0.2, 0.2]), 1.0, 5, 1.0)

    assert first["inserted"] is True
    assert second["inserted"] is False
    assert len(db.list_deltas_for_round(round_row["id"])) == 1


def test_offline_node_drops_out_of_fast_close_wait_set(db, config):
    node_a = make_training_node(db, "gpu-a")
    node_b = make_training_node(db, "gpu-b")
    run_id = make_run(db, round_timeout_s=600)
    start_run(db, run_id)
    round_row = db.get_current_round(run_id)

    submit_delta(db, round_row["id"], node_a, fake_delta_bytes([0.1, 0.1]), 1.0, 5, 1.0)

    # node_b's heartbeat goes stale past config.heartbeat_timeout_s; node_a
    # keeps heartbeating (refreshed here, after pinning node_b's cutoff, so
    # its timestamp lands strictly after cutoff). Sweep offline the same way
    # the inference path does (reused, role-agnostic -- see test_heartbeat.py
    # for direct coverage of the sweep itself).
    cutoff_now = db.get_node(node_b)["last_heartbeat"] + config.heartbeat_timeout_s + 1e-6
    db.update_heartbeat(node_a, "idle", cpu_pct=5.0, ram_free_mb=8192.0)
    sweep_offline_nodes(db, config, now=cutoff_now)
    assert db.get_node(node_b)["status"] == "offline"
    assert db.get_node(node_a)["status"] != "offline"

    # fast-close now fires off node_a alone, since node_b no longer counts as online
    closed = close_ready_rounds(db, now=round_row["opened_at"] + 1)
    assert closed == [round_row["id"]]


def test_round_closable_after_coordinator_restart(tmp_path):
    db_path = str(tmp_path / "test.db")
    db1 = Database(db_path)
    node_a = make_training_node(db1, "gpu-a")
    run_id = make_run(db1, round_timeout_s=5)
    start_run(db1, run_id)
    round_row = db1.get_current_round(run_id)
    submit_delta(db1, round_row["id"], node_a, fake_delta_bytes([0.1, 0.1]), 1.0, 5, 1.0)
    db1.close()

    db2 = Database(db_path)
    try:
        closed = close_ready_rounds(db2, now=round_row["deadline_at"] + 1)
        assert closed == [round_row["id"]]
        aggregate_closed_rounds(db2)
        updated = db2.get_round(round_row["id"])
        assert updated["status"] == "applied"
        assert updated["participant_count"] == 1
    finally:
        db2.close()
