"""Background training-round cycle: round close (fast/timeout), aggregation
via the DiLoCo outer optimizer, and round/run advancement.

Mirrors dispatch.py's role for the inference job queue: pure functions over
a `Database` instance, unit-testable without FastAPI or a real GPU (see
tests/test_training_dispatch.py, which uses tiny hand-built tensors).

Scoping note: this PoC treats "all currently online role='training' nodes"
as the participant pool for whichever run is active -- reasonable for a
demo with a small, fixed set of known training machines. A deployment
running multiple concurrent runs on partially-overlapping node sets would
need an explicit run<->node membership table; out of scope here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from common.logging_setup import log_event
from coordinator.db import Database
from training.checkpoint_io import bytes_to_state_dict, state_dict_to_bytes
from training.outer_optim import average_state_dicts, nesterov_outer_step

logger = logging.getLogger("coordinator.training_dispatch")


def _online_training_node_count(db: Database) -> int:
    return sum(1 for n in db.list_nodes_by_role("training") if n["status"] != "offline")


def _round_ready_to_close(db: Database, round_row: dict[str, Any], now: float) -> bool:
    if now >= round_row["deadline_at"]:
        return True
    online = _online_training_node_count(db)
    if online == 0:
        return False
    submitted = len(db.list_deltas_for_round(round_row["id"]))
    return submitted >= online


def close_ready_rounds(db: Database, now: Optional[float] = None) -> list[str]:
    """Move any 'open' round past its close condition into 'aggregating'."""
    now = now if now is not None else time.time()
    closed = []
    for round_row in db.list_rounds_by_status("open"):
        if _round_ready_to_close(db, round_row, now):
            db.set_round_status(round_row["id"], "aggregating")
            log_event(logger, "round_closed", round_id=round_row["id"], run_id=round_row["run_id"])
            closed.append(round_row["id"])
    return closed


def aggregate_closed_rounds(db: Database) -> list[str]:
    """Average submitted deltas for every 'aggregating' round, apply the outer
    optimizer step (or skip if nobody submitted), open the next round or
    complete the run."""
    processed = []
    for round_row in db.list_rounds_by_status("aggregating"):
        _aggregate_one_round(db, round_row)
        processed.append(round_row["id"])
    return processed


def _aggregate_one_round(db: Database, round_row: dict[str, Any]) -> None:
    run = db.get_training_run(round_row["run_id"])
    deltas = db.list_deltas_for_round(round_row["id"])

    if not deltas:
        db.set_round_status(round_row["id"], "skipped", participant_count=0)
        log_event(logger, "round_skipped", round_id=round_row["id"], run_id=run["id"])
        _advance_or_complete(
            db, run, round_row,
            new_checkpoint_id=run["latest_checkpoint_id"],
            new_outer_state_id=run["outer_state_checkpoint_id"],
        )
        return

    delta_state_dicts = [
        bytes_to_state_dict(db.get_checkpoint_blob(d["checkpoint_id"])) for d in deltas
    ]
    avg_delta = average_state_dicts(delta_state_dicts)

    global_state = bytes_to_state_dict(db.get_checkpoint_blob(round_row["base_checkpoint_id"]))
    momentum_buffer = None
    if run["outer_state_checkpoint_id"]:
        momentum_buffer = bytes_to_state_dict(db.get_checkpoint_blob(run["outer_state_checkpoint_id"]))

    new_global, new_buffer = nesterov_outer_step(
        global_state, avg_delta, momentum_buffer, run["outer_lr"], run["outer_momentum"],
    )

    new_checkpoint_id = db.save_checkpoint(
        run["id"], "global_weights", state_dict_to_bytes(new_global), round_num=round_row["round_num"] + 1,
    )
    new_outer_state_id = db.save_checkpoint(
        run["id"], "outer_state", state_dict_to_bytes(new_buffer), round_num=round_row["round_num"] + 1,
    )

    db.set_round_status(
        round_row["id"], "applied", result_checkpoint_id=new_checkpoint_id, participant_count=len(deltas),
    )
    log_event(logger, "round_applied", round_id=round_row["id"], run_id=run["id"], participants=len(deltas))

    _advance_or_complete(
        db, run, round_row, new_checkpoint_id=new_checkpoint_id, new_outer_state_id=new_outer_state_id,
    )


def _advance_or_complete(
    db: Database, run: dict[str, Any], round_row: dict[str, Any],
    new_checkpoint_id: str, new_outer_state_id: Optional[str],
) -> None:
    next_round_num = round_row["round_num"] + 1
    db.advance_run_round(run["id"], next_round_num, new_checkpoint_id, new_outer_state_id)

    if next_round_num >= run["max_rounds"]:
        db.set_run_status(run["id"], "completed", completed_at=time.time())
        log_event(logger, "run_completed", run_id=run["id"], rounds=next_round_num)
        return

    deadline = time.time() + run["round_timeout_s"]
    db.create_training_round(run["id"], next_round_num, new_checkpoint_id, deadline)
    log_event(logger, "round_opened", run_id=run["id"], round_num=next_round_num)


def start_run(db: Database, run_id: str) -> None:
    """pending -> running: open round 0 from the initial checkpoint already
    saved at run-creation time."""
    run = db.get_training_run(run_id)
    if run is None or run["status"] != "pending":
        raise ValueError(f"run {run_id} is not pending")
    db.set_run_status(run_id, "running")
    deadline = time.time() + run["round_timeout_s"]
    db.create_training_round(run_id, 0, run["latest_checkpoint_id"], deadline)
    log_event(logger, "run_started", run_id=run_id)


def submit_delta(
    db: Database, round_id: str, node_id: str, blob: bytes,
    local_loss: Optional[float], local_steps_completed: Optional[int], wall_time_s: Optional[float],
) -> dict[str, Any]:
    """Apply a POST /training/rounds/{id}/delta payload. Raises ValueError if
    the round is no longer open -- callers map this to HTTP 409; the agent
    just discards and waits for the next round, not a fatal error."""
    round_row = db.get_round(round_id)
    if round_row is None:
        raise ValueError(f"unknown round {round_id}")
    if round_row["status"] != "open":
        raise ValueError(f"round {round_id} is no longer open (status={round_row['status']})")

    run = db.get_training_run(round_row["run_id"])
    checkpoint_id = db.save_checkpoint(
        run["id"], "delta", blob, round_num=round_row["round_num"], node_id=node_id,
    )
    inserted = db.record_delta(round_id, node_id, checkpoint_id, local_loss, local_steps_completed, wall_time_s)
    log_event(logger, "delta_submitted", round_id=round_id, node_id=node_id, inserted=inserted)
    return {"inserted": inserted, "checkpoint_id": checkpoint_id}


def run_training_cycle(db: Database) -> None:
    """One full tick of the background training-round loop."""
    close_ready_rounds(db)
    aggregate_closed_rounds(db)
