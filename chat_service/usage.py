"""Anonymous request quotas and aggregate usage; message content is not stored."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from chat_service.config import ChatConfig


class QuotaExceeded(Exception):
    pass


class UsageStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS chat_requests (
                id TEXT PRIMARY KEY,
                visitor TEXT NOT NULL,
                network TEXT NOT NULL,
                started_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'started',
                duration_s REAL,
                completion_tokens INTEGER
            );
            CREATE INDEX IF NOT EXISTS chat_visitor_time ON chat_requests(visitor, started_at);
            CREATE INDEX IF NOT EXISTS chat_network_time ON chat_requests(network, started_at);
        """)

    def reserve(self, request_id: str, visitor: str, network: str, config: ChatConfig) -> None:
        now = time.time()
        with self.lock, self.connection:
            recent = self.connection.execute(
                "SELECT COUNT(*) FROM chat_requests WHERE network = ? AND started_at > ?",
                (network, now - 60),
            ).fetchone()[0]
            daily = self.connection.execute(
                "SELECT COUNT(*) FROM chat_requests WHERE visitor = ? AND started_at > ?",
                (visitor, now - 86400),
            ).fetchone()[0]
            if recent >= config.requests_per_minute or daily >= config.requests_per_day:
                raise QuotaExceeded
            self.connection.execute(
                "INSERT INTO chat_requests (id, visitor, network, started_at) VALUES (?, ?, ?, ?)",
                (request_id, visitor, network, now),
            )
            self.connection.execute("DELETE FROM chat_requests WHERE started_at < ?", (now - 7 * 86400,))

    def finish(self, request_id: str, status: str, duration_s: float, tokens: int | None) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE chat_requests SET status = ?, duration_s = ?, completion_tokens = ? WHERE id = ?",
                (status, duration_s, tokens, request_id),
            )

    def close(self) -> None:
        self.connection.close()
