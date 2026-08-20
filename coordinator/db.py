"""SQLite data layer for the coordinator.

Every query lives in exactly one method on `Database`. Nothing outside this
module writes raw SQL. That keeps a future swap to Postgres to: (1) replace
the connection/cursor plumbing in `__init__`/`_execute`, (2) adjust the
handful of SQLite-specific bits (AUTOINCREMENT, `?` placeholders,
`INSERT ... RETURNING` support). Call sites elsewhere never change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from common.hashing import sha256_bytes

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    token               TEXT NOT NULL,
    city                TEXT,
    country              TEXT,
    hardware            TEXT,
    model_name          TEXT NOT NULL,
    role                TEXT NOT NULL DEFAULT 'inference',
    max_concurrent_jobs INTEGER NOT NULL DEFAULT 1,
    measured_watts      REAL,
    grid_zone           TEXT,
    status              TEXT NOT NULL DEFAULT 'idle',
    last_heartbeat      REAL,
    cpu_pct             REAL,
    ram_free_mb         REAL,
    created_at          REAL NOT NULL,
    jobs_done           INTEGER NOT NULL DEFAULT 0,
    jobs_failed         INTEGER NOT NULL DEFAULT 0,
    tokens_total        INTEGER NOT NULL DEFAULT 0,
    duration_total       REAL NOT NULL DEFAULT 0,
    mismatch_count       INTEGER NOT NULL DEFAULT 0,
    requeue_count        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS batches (
    id                TEXT PRIMARY KEY,
    model_name        TEXT NOT NULL,
    max_tokens        INTEGER NOT NULL,
    timeout_s         INTEGER NOT NULL,
    verification_pct  REAL NOT NULL,
    payout_rate       REAL NOT NULL,
    created_at        REAL NOT NULL,
    total_jobs        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id                     TEXT PRIMARY KEY,
    batch_id               TEXT NOT NULL REFERENCES batches(id),
    prompt                 TEXT NOT NULL,
    max_tokens             INTEGER NOT NULL,
    timeout_s              INTEGER NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'queued',
    assigned_node_id       TEXT,
    attempt_count          INTEGER NOT NULL DEFAULT 0,
    requeue_count          INTEGER NOT NULL DEFAULT 0,
    verification_group_id TEXT,
    is_duplicate           INTEGER NOT NULL DEFAULT 0,
    verified               INTEGER,
    created_at             REAL NOT NULL,
    dispatched_at           REAL,
    completed_at             REAL,
    output_text            TEXT,
    output_sha256           TEXT,
    tokens_generated         INTEGER,
    duration_s               REAL,
    tokens_per_sec           REAL,
    error                    TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    node_id     TEXT,
    ts          REAL NOT NULL,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id    TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    tokens     INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    ts         REAL NOT NULL,
    payout     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_assigned ON jobs(assigned_node_id);
CREATE INDEX IF NOT EXISTS idx_jobs_verification_group ON jobs(verification_group_id);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);
CREATE INDEX IF NOT EXISTS idx_ledger_node ON ledger(node_id);

CREATE TABLE IF NOT EXISTS training_runs (
    id                        TEXT PRIMARY KEY,
    name                      TEXT NOT NULL,
    status                    TEXT NOT NULL DEFAULT 'pending',
    model_backend             TEXT NOT NULL DEFAULT 'scratch_gpt',
    -- n_layer/n_head/n_embd/block_size: scratch_gpt only (architecture of
    -- the from-scratch model). base_model_id/lora_*: lora backends only
    -- (which pretrained model + adapter shape). Nullable because a run
    -- only ever populates one set, depending on model_backend.
    n_layer                   INTEGER,
    n_head                    INTEGER,
    n_embd                    INTEGER,
    block_size                INTEGER,
    vocab_size                INTEGER NOT NULL,
    base_model_id             TEXT,
    lora_r                    INTEGER,
    lora_alpha                INTEGER,
    lora_dropout              REAL,
    lora_target_modules       TEXT,
    dataset_name              TEXT NOT NULL,
    inner_lr                  REAL NOT NULL,
    inner_steps_h             INTEGER NOT NULL,
    inner_batch_size          INTEGER NOT NULL,
    outer_lr                  REAL NOT NULL,
    outer_momentum            REAL NOT NULL,
    max_rounds                INTEGER NOT NULL,
    round_timeout_s           INTEGER NOT NULL,
    quorum                    INTEGER NOT NULL DEFAULT 1,
    current_round             INTEGER NOT NULL DEFAULT 0,
    created_at                REAL NOT NULL,
    completed_at              REAL,
    latest_checkpoint_id      TEXT,
    outer_state_checkpoint_id TEXT
);

CREATE TABLE IF NOT EXISTS training_rounds (
    id                   TEXT PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES training_runs(id),
    round_num            INTEGER NOT NULL,
    status               TEXT NOT NULL DEFAULT 'open',
    base_checkpoint_id   TEXT NOT NULL,
    result_checkpoint_id TEXT,
    opened_at            REAL NOT NULL,
    deadline_at          REAL NOT NULL,
    closed_at            REAL,
    participant_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_deltas (
    id                    TEXT PRIMARY KEY,
    round_id              TEXT NOT NULL REFERENCES training_rounds(id),
    node_id               TEXT NOT NULL REFERENCES nodes(id),
    checkpoint_id         TEXT NOT NULL,
    local_loss            REAL,
    local_steps_completed INTEGER,
    wall_time_s           REAL,
    submitted_at          REAL NOT NULL,
    UNIQUE(round_id, node_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES training_runs(id),
    kind       TEXT NOT NULL,
    round_num  INTEGER,
    node_id    TEXT,
    sha256     TEXT NOT NULL,
    byte_size  INTEGER NOT NULL,
    created_at REAL NOT NULL,
    blob       BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_rounds_run ON training_rounds(run_id);
CREATE INDEX IF NOT EXISTS idx_training_deltas_round ON training_deltas(round_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
"""


def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


class Database:
    """Thread-safe wrapper around a single SQLite connection.

    SQLite serializes writes anyway; a single connection guarded by a lock
    keeps the PoC simple and avoids "database is locked" errors under the
    FastAPI threadpool.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # nodes
    # ------------------------------------------------------------------ #

    def register_node(
        self,
        name: str,
        token: str,
        city: Optional[str],
        country: Optional[str],
        hardware: Optional[str],
        model_name: str,
        max_concurrent_jobs: int,
        measured_watts: Optional[float],
        grid_zone: Optional[str],
        role: str = "inference",
    ) -> str:
        node_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO nodes
                   (id, name, token, city, country, hardware, model_name, role,
                    max_concurrent_jobs, measured_watts, grid_zone, status,
                    last_heartbeat, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'idle', ?, ?)""",
                (
                    node_id, name, token, city, country, hardware, model_name, role,
                    max_concurrent_jobs, measured_watts, grid_zone, now, now,
                ),
            )
            self._conn.commit()
        return node_id

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            return _row(cur.fetchone())

    def list_nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM nodes ORDER BY created_at")
            return _rows(cur.fetchall())

    def list_nodes_by_status_and_model(self, status: str, model_name: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM nodes WHERE status = ? AND model_name = ?",
                (status, model_name),
            )
            return _rows(cur.fetchall())

    def list_nodes_by_role(self, role: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM nodes WHERE role = ? ORDER BY created_at", (role,))
            return _rows(cur.fetchall())

    def update_heartbeat(self, node_id: str, status: str, cpu_pct: float, ram_free_mb: float) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE nodes SET last_heartbeat = ?, status = ?, cpu_pct = ?, ram_free_mb = ?
                   WHERE id = ?""",
                (time.time(), status, cpu_pct, ram_free_mb, node_id),
            )
            self._conn.commit()

    def set_node_status(self, node_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE nodes SET status = ? WHERE id = ?", (status, node_id))
            self._conn.commit()

    def sweep_offline_nodes(self, cutoff_ts: float) -> list[str]:
        """Mark nodes with no heartbeat since cutoff_ts as offline. Returns their ids."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT id FROM nodes
                   WHERE status != 'offline' AND (last_heartbeat IS NULL OR last_heartbeat < ?)""",
                (cutoff_ts,),
            )
            ids = [r["id"] for r in cur.fetchall()]
            if ids:
                self._conn.executemany(
                    "UPDATE nodes SET status = 'offline' WHERE id = ?",
                    [(i,) for i in ids],
                )
                self._conn.commit()
            return ids

    def deregister_node(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE nodes SET status = 'offline' WHERE id = ?", (node_id,))
            self._conn.commit()

    def record_node_job_success(self, node_id: str, tokens: int, duration_s: float) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE nodes SET jobs_done = jobs_done + 1,
                       tokens_total = tokens_total + ?, duration_total = duration_total + ?
                   WHERE id = ?""",
                (tokens, duration_s, node_id),
            )
            self._conn.commit()

    def record_node_job_failure(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET jobs_failed = jobs_failed + 1 WHERE id = ?", (node_id,)
            )
            self._conn.commit()

    def record_node_mismatch(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET mismatch_count = mismatch_count + 1 WHERE id = ?", (node_id,)
            )
            self._conn.commit()

    def record_node_requeue(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET requeue_count = requeue_count + 1 WHERE id = ?", (node_id,)
            )
            self._conn.commit()

    def median_tokens_per_sec(self) -> float:
        """Median lifetime tokens/sec across nodes that have completed >=1 job."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT tokens_total, duration_total FROM nodes WHERE jobs_done > 0 AND duration_total > 0"
            )
            rates = sorted(r["tokens_total"] / r["duration_total"] for r in cur.fetchall())
        if not rates:
            return 10.0  # PoC fallback default before any node has history
        mid = len(rates) // 2
        if len(rates) % 2 == 0:
            return (rates[mid - 1] + rates[mid]) / 2
        return rates[mid]

    # ------------------------------------------------------------------ #
    # batches
    # ------------------------------------------------------------------ #

    def create_batch(
        self, model_name: str, max_tokens: int, timeout_s: int,
        verification_pct: float, payout_rate: float, total_jobs: int,
    ) -> str:
        batch_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO batches
                   (id, model_name, max_tokens, timeout_s, verification_pct,
                    payout_rate, created_at, total_jobs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (batch_id, model_name, max_tokens, timeout_s, verification_pct,
                 payout_rate, time.time(), total_jobs),
            )
            self._conn.commit()
        return batch_id

    def get_batch(self, batch_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,))
            return _row(cur.fetchone())

    def list_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM batches ORDER BY created_at DESC")
            return _rows(cur.fetchall())

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def create_job(
        self, batch_id: str, prompt: str, max_tokens: int, timeout_s: int,
        verification_group_id: Optional[str] = None, is_duplicate: bool = False,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO jobs
                   (id, batch_id, prompt, max_tokens, timeout_s, status,
                    verification_group_id, is_duplicate, created_at)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (job_id, batch_id, prompt, max_tokens, timeout_s,
                 verification_group_id, int(is_duplicate), time.time()),
            )
            self._conn.commit()
        self.log_job_event(job_id, "queued", node_id=None)
        return job_id

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            return _row(cur.fetchone())

    def list_queued_jobs(self, model_name: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """SELECT j.* FROM jobs j JOIN batches b ON j.batch_id = b.id
                   WHERE j.status = 'queued' AND b.model_name = ?
                   ORDER BY j.created_at""",
                (model_name,),
            )
            return _rows(cur.fetchall())

    def list_jobs_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE batch_id = ? ORDER BY created_at", (batch_id,)
            )
            return _rows(cur.fetchall())

    def get_dispatched_job_for_node(self, node_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE status = 'dispatched' AND assigned_node_id = ? LIMIT 1",
                (node_id,),
            )
            return _row(cur.fetchone())

    def get_verification_sibling(self, job_id: str, group_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE verification_group_id = ? AND id != ?",
                (group_id, job_id),
            )
            return _row(cur.fetchone())

    def assign_job(self, job_id: str, node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status = 'dispatched', assigned_node_id = ?,
                       dispatched_at = ?, attempt_count = attempt_count + 1
                   WHERE id = ?""",
                (node_id, time.time(), job_id),
            )
            self._conn.commit()
        self.log_job_event(job_id, "dispatched", node_id=node_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
            self._conn.commit()
        job = self.get_job(job_id)
        self.log_job_event(job_id, "running", node_id=job["assigned_node_id"] if job else None)

    def complete_job(
        self, job_id: str, output_text: str, output_sha256: str,
        tokens_generated: int, duration_s: float, tokens_per_sec: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status = 'completed', output_text = ?, output_sha256 = ?,
                       tokens_generated = ?, duration_s = ?, tokens_per_sec = ?, completed_at = ?
                   WHERE id = ?""",
                (output_text, output_sha256, tokens_generated, duration_s,
                 tokens_per_sec, time.time(), job_id),
            )
            self._conn.commit()
        job = self.get_job(job_id)
        self.log_job_event(job_id, "completed", node_id=job["assigned_node_id"] if job else None)

    def fail_job_attempt(self, job_id: str, error: str) -> None:
        """Record a failed attempt without deciding retry vs. terminal failure."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET error = ? WHERE id = ?", (error, job_id)
            )
            self._conn.commit()

    def requeue_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status = 'queued', assigned_node_id = NULL,
                       dispatched_at = NULL, requeue_count = requeue_count + 1
                   WHERE id = ?""",
                (job_id,),
            )
            self._conn.commit()
        self.log_job_event(job_id, "requeued", node_id=None)

    def terminal_fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?",
                (error, time.time(), job_id),
            )
            self._conn.commit()
        job = self.get_job(job_id)
        self.log_job_event(job_id, "failed", node_id=job["assigned_node_id"] if job else None, error=error)

    def set_verification_result(self, job_id: str, verified: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, verified = ? WHERE id = ?",
                ("verified" if verified else "mismatched", int(verified), job_id),
            )
            self._conn.commit()
        job = self.get_job(job_id)
        self.log_job_event(
            job_id, "verified" if verified else "mismatched",
            node_id=job["assigned_node_id"] if job else None,
        )

    def list_stuck_jobs(self, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders})", statuses
            )
            return _rows(cur.fetchall())

    # ------------------------------------------------------------------ #
    # job events (structured transition log, queryable for debugging)
    # ------------------------------------------------------------------ #

    def log_job_event(self, job_id: str, event_type: str, node_id: Optional[str], **detail: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO job_events (job_id, event_type, node_id, ts, detail_json) VALUES (?, ?, ?, ?, ?)",
                (job_id, event_type, node_id, time.time(), json.dumps(detail) if detail else None),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # ledger
    # ------------------------------------------------------------------ #

    def append_ledger(self, node_id: str, job_id: str, tokens: int, duration_s: float, payout: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger (node_id, job_id, tokens, duration_s, ts, payout) VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, job_id, tokens, duration_s, time.time(), payout),
            )
            self._conn.commit()

    def list_ledger(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM ledger ORDER BY ts DESC LIMIT ?", (limit,))
            return _rows(cur.fetchall())

    def total_payouts(self) -> float:
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(SUM(payout), 0) AS total FROM ledger")
            return cur.fetchone()["total"]

    # ------------------------------------------------------------------ #
    # training runs / rounds / deltas / checkpoints (DiLoCo)
    #
    # This module never imports torch -- checkpoints.blob is opaque bytes
    # to everything here, encoded/decoded only by training/checkpoint_io.py.
    # ------------------------------------------------------------------ #

    def create_training_run(
        self, name: str, vocab_size: int, dataset_name: str, inner_lr: float, inner_steps_h: int,
        inner_batch_size: int, outer_lr: float, outer_momentum: float,
        max_rounds: int, round_timeout_s: int, quorum: int,
        model_backend: str = "scratch_gpt",
        n_layer: Optional[int] = None, n_head: Optional[int] = None,
        n_embd: Optional[int] = None, block_size: Optional[int] = None,
        base_model_id: Optional[str] = None, lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None, lora_dropout: Optional[float] = None,
        lora_target_modules: Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO training_runs
                   (id, name, status, model_backend, n_layer, n_head, n_embd, block_size, vocab_size,
                    base_model_id, lora_r, lora_alpha, lora_dropout, lora_target_modules,
                    dataset_name, inner_lr, inner_steps_h, inner_batch_size, outer_lr,
                    outer_momentum, max_rounds, round_timeout_s, quorum, created_at)
                   VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, name, model_backend, n_layer, n_head, n_embd, block_size, vocab_size,
                    base_model_id, lora_r, lora_alpha, lora_dropout, lora_target_modules,
                    dataset_name, inner_lr, inner_steps_h, inner_batch_size, outer_lr,
                    outer_momentum, max_rounds, round_timeout_s, quorum, time.time(),
                ),
            )
            self._conn.commit()
        return run_id

    def get_training_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,))
            return _row(cur.fetchone())

    def list_training_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM training_runs ORDER BY created_at DESC")
            return _rows(cur.fetchall())

    def set_run_status(self, run_id: str, status: str, completed_at: Optional[float] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE training_runs SET status = ?, completed_at = ? WHERE id = ?",
                (status, completed_at, run_id),
            )
            self._conn.commit()

    def advance_run_round(
        self, run_id: str, current_round: int,
        latest_checkpoint_id: str, outer_state_checkpoint_id: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE training_runs
                   SET current_round = ?, latest_checkpoint_id = ?, outer_state_checkpoint_id = ?
                   WHERE id = ?""",
                (current_round, latest_checkpoint_id, outer_state_checkpoint_id, run_id),
            )
            self._conn.commit()

    def create_training_round(
        self, run_id: str, round_num: int, base_checkpoint_id: str, deadline_at: float,
    ) -> str:
        round_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO training_rounds
                   (id, run_id, round_num, status, base_checkpoint_id, opened_at, deadline_at)
                   VALUES (?, ?, ?, 'open', ?, ?, ?)""",
                (round_id, run_id, round_num, base_checkpoint_id, time.time(), deadline_at),
            )
            self._conn.commit()
        return round_id

    def get_round(self, round_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM training_rounds WHERE id = ?", (round_id,))
            return _row(cur.fetchone())

    def get_current_round(self, run_id: str) -> Optional[dict[str, Any]]:
        """Most recently opened round for a run, regardless of status -- callers
        check `status` themselves (an agent polling mid-aggregation just waits)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM training_rounds WHERE run_id = ? ORDER BY round_num DESC LIMIT 1",
                (run_id,),
            )
            return _row(cur.fetchone())

    def list_rounds_by_status(self, status: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM training_rounds WHERE status = ?", (status,))
            return _rows(cur.fetchall())

    def set_round_status(
        self, round_id: str, status: str,
        result_checkpoint_id: Optional[str] = None, participant_count: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE training_rounds SET status = ?,
                       result_checkpoint_id = COALESCE(?, result_checkpoint_id),
                       participant_count = COALESCE(?, participant_count),
                       closed_at = ?
                   WHERE id = ?""",
                (status, result_checkpoint_id, participant_count, time.time(), round_id),
            )
            self._conn.commit()

    def record_delta(
        self, round_id: str, node_id: str, checkpoint_id: str,
        local_loss: Optional[float], local_steps_completed: Optional[int], wall_time_s: Optional[float],
    ) -> bool:
        """Idempotent: a retried upload from a node that lost the HTTP response
        but whose delta already landed is silently ignored. Returns True if this
        call actually inserted a new row."""
        delta_id = str(uuid.uuid4())
        with self._lock:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO training_deltas
                   (id, round_id, node_id, checkpoint_id, local_loss,
                    local_steps_completed, wall_time_s, submitted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (delta_id, round_id, node_id, checkpoint_id, local_loss,
                 local_steps_completed, wall_time_s, time.time()),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def list_deltas_for_round(self, round_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM training_deltas WHERE round_id = ?", (round_id,))
            return _rows(cur.fetchall())

    def save_checkpoint(
        self, run_id: str, kind: str, blob: bytes,
        round_num: Optional[int] = None, node_id: Optional[str] = None,
    ) -> str:
        checkpoint_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO checkpoints
                   (id, run_id, kind, round_num, node_id, sha256, byte_size, created_at, blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (checkpoint_id, run_id, kind, round_num, node_id,
                 sha256_bytes(blob), len(blob), time.time(), blob),
            )
            self._conn.commit()
        return checkpoint_id

    def get_checkpoint_blob(self, checkpoint_id: str) -> Optional[bytes]:
        with self._lock:
            cur = self._conn.execute("SELECT blob FROM checkpoints WHERE id = ?", (checkpoint_id,))
            row = cur.fetchone()
            return row["blob"] if row is not None else None

    def get_checkpoint_meta(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, run_id, kind, round_num, node_id, sha256, byte_size, created_at "
                "FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            return _row(cur.fetchone())
