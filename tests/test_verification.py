from __future__ import annotations

from coordinator.dispatch import check_verification_pair, handle_job_result
from tests.conftest import MODEL, make_node


def _setup_pair(db):
    node_a = make_node(db, name="node-a")
    node_b = make_node(db, name="node-b")
    batch_id = db.create_batch(
        model_name=MODEL, max_tokens=100, timeout_s=300,
        verification_pct=1.0, payout_rate=0.00001, total_jobs=1,
    )
    group_id = "grp-verify"
    primary_id = db.create_job(batch_id, "prompt", 100, 300, verification_group_id=group_id, is_duplicate=False)
    dup_id = db.create_job(batch_id, "prompt", 100, 300, verification_group_id=group_id, is_duplicate=True)
    db.assign_job(primary_id, node_a)
    db.assign_job(dup_id, node_b)
    db.mark_running(primary_id)
    db.mark_running(dup_id)
    return node_a, node_b, primary_id, dup_id


def test_matching_hashes_mark_both_jobs_verified(db, config):
    node_a, node_b, primary_id, dup_id = _setup_pair(db)

    handle_job_result(db, config, primary_id, node_a, True, "same output", "abc123", 10, 1.0, 10.0, None)
    # only one half done so far -- nothing to compare yet
    assert db.get_job(primary_id)["status"] == "completed"

    handle_job_result(db, config, dup_id, node_b, True, "same output", "abc123", 10, 1.0, 10.0, None)

    assert db.get_job(primary_id)["status"] == "verified"
    assert db.get_job(dup_id)["status"] == "verified"
    assert db.get_node(node_a)["mismatch_count"] == 0
    assert db.get_node(node_b)["mismatch_count"] == 0


def test_mismatched_hashes_flag_both_and_increment_mismatch_count(db, config):
    node_a, node_b, primary_id, dup_id = _setup_pair(db)

    handle_job_result(db, config, primary_id, node_a, True, "output one", "aaa111", 10, 1.0, 10.0, None)
    handle_job_result(db, config, dup_id, node_b, True, "output two (corrupted)", "bbb222", 10, 1.0, 10.0, None)

    assert db.get_job(primary_id)["status"] == "mismatched"
    assert db.get_job(dup_id)["status"] == "mismatched"
    assert db.get_node(node_a)["mismatch_count"] == 1
    assert db.get_node(node_b)["mismatch_count"] == 1


def test_check_verification_pair_is_noop_until_both_complete(db):
    node_a, node_b, primary_id, dup_id = _setup_pair(db)
    db.complete_job(primary_id, "text", "sha", 10, 1.0, 10.0)

    check_verification_pair(db, primary_id)

    # sibling still running -- primary should remain 'completed', not verified/mismatched
    assert db.get_job(primary_id)["status"] == "completed"


def test_ledger_gets_one_row_per_completed_job(db, config):
    node_a, node_b, primary_id, dup_id = _setup_pair(db)
    handle_job_result(db, config, primary_id, node_a, True, "out", "sha1", 20, 2.0, 10.0, None)
    handle_job_result(db, config, dup_id, node_b, True, "out", "sha1", 20, 2.0, 10.0, None)

    ledger = db.list_ledger()
    assert len(ledger) == 2
    assert {row["job_id"] for row in ledger} == {primary_id, dup_id}
    for row in ledger:
        assert row["tokens"] == 20
        assert row["payout"] == 20 * config.default_payout_rate
