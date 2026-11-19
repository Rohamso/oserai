"""Invite credentials and durable admission control for the Groq pilot."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from chat_service.config import ChatConfig
from chat_service.usage import QuotaExceeded

ACCOUNT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
ACCESS_CODE = re.compile(r"[A-Za-z0-9_-]{43}")
DIGEST = re.compile(r"[a-f0-9]{64}")
MIN_REPLY_TOKENS = 64
DAY = timedelta(days=1)
MINUTE = timedelta(minutes=1)


class PilotAccounts:
    def __init__(self, settings: ChatConfig):
        self.gateway_key = settings.gateway_key.encode()
        self.digests: dict[str, str] = {}
        if not settings.accounts_required or not settings.pilot_accounts:
            return
        try:
            entries = json.loads(settings.pilot_accounts)
            if not isinstance(entries, dict) or not entries:
                raise ValueError
            if any(not ACCOUNT_ID.fullmatch(account) or not isinstance(digest, str) or not DIGEST.fullmatch(digest)
                   for account, digest in entries.items()):
                raise ValueError
            if len(set(entries.values())) != len(entries):
                raise ValueError
        except ValueError:
            raise ValueError("OSERAI_CHAT_PILOT_ACCOUNTS must map unique account IDs to unique SHA-256 digests") from None
        self.digests = entries

    def _proof(self, account: str, digest: str) -> str:
        return hmac.new(self.gateway_key, f"oserai-pilot-session:{account}:{digest}".encode(), "sha256").hexdigest()

    def sign_in(self, code: str) -> str | None:
        if not ACCESS_CODE.fullmatch(code):
            return None
        supplied = hashlib.sha256(code.encode()).hexdigest()
        for account, digest in self.digests.items():
            if hmac.compare_digest(supplied, digest):
                return f"{account}.{self._proof(account, digest)}"
        return None

    def verify(self, credential: str) -> str | None:
        account, _, proof = credential.partition(".")
        digest = self.digests.get(account)
        if digest and DIGEST.fullmatch(proof) and hmac.compare_digest(proof, self._proof(account, digest)):
            return account
        return None


def prompt_token_bound(messages: list[dict]) -> int:
    # Qwen's text tokenizer merges bytes. Reserve one token per UTF-8 byte
    # plus room for roles, control tokens and the provider's chat template.
    return 64 + sum(len(message["content"].encode("utf-8")) + 24 for message in messages)


def utc_day(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


class PilotUnavailable(Exception):
    pass


class PilotUsage:
    def __init__(self, settings: ChatConfig):
        from psycopg import Error
        from psycopg_pool import AsyncConnectionPool

        self.settings = settings
        self.db_error = Error
        self.pool = AsyncConnectionPool(
            settings.database_url, min_size=0, max_size=4, max_idle=60, timeout=5,
            kwargs={"connect_timeout": 5}, open=False,
        )

    @asynccontextmanager
    async def connection(self):
        try:
            async with self.pool.connection() as db:
                yield db
        except self.db_error as exc:
            raise PilotUnavailable from exc

    async def start(self):
        await self.pool.open()
        async with self.connection() as db:
            # Also serialize first-time schema creation across gateway processes.
            await db.execute("SELECT pg_advisory_xact_lock(hashtext('oserai-chat-pilot-schema'))")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_pilot_usage (
                    request_id UUID PRIMARY KEY,
                    account TEXT NOT NULL,
                    network TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    tokens INTEGER NOT NULL CHECK (tokens >= 0)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS chat_pilot_usage_started_at
                ON chat_pilot_usage(started_at)
            """)

    def _allowance(self, spent: int, day: datetime) -> dict:
        limit = self.settings.user_tokens_per_day
        return {"limit": limit, "remaining": max(0, limit - spent), "resets_at": (day + DAY).isoformat()}

    async def allowance(self, account: str) -> dict:
        day = utc_day(datetime.now(UTC))
        async with self.connection() as db:
            row = await (await db.execute("""
                SELECT COALESCE(SUM(tokens), 0) FROM chat_pilot_usage
                WHERE account = %s AND started_at >= %s
            """, (account, day))).fetchone()
        return self._allowance(row[0], day)

    async def reserve(self, request_id: str, account: str, network: str, prompt_bound: int) -> int:
        settings = self.settings
        async with self.connection() as db:
            # Check and debit under one transaction even when two services share the database.
            await db.execute("LOCK TABLE chat_pilot_usage IN SHARE ROW EXCLUSIVE MODE")
            now = datetime.now(UTC)
            day = utc_day(now)
            minute_ago = now - MINUTE
            row = await (await db.execute("""
                SELECT COALESCE(SUM(tokens) FILTER (WHERE account = %s AND started_at >= %s), 0),
                       COALESCE(SUM(tokens), 0),
                       COALESCE(SUM(tokens) FILTER (WHERE started_at > %s), 0),
                       COUNT(*) FILTER (WHERE started_at > %s AND network = %s),
                       COUNT(*) FILTER (WHERE account = %s),
                       COUNT(*) FILTER (WHERE started_at > %s),
                       COUNT(*)
                FROM chat_pilot_usage WHERE started_at > %s
            """, (account, day, minute_ago, minute_ago, network, account, minute_ago, now - DAY))).fetchone()
            user_tokens, daily_tokens, minute_tokens, network_requests, user_requests, minute_requests, daily_requests = row
            if network_requests >= settings.requests_per_minute or user_requests >= settings.requests_per_day:
                raise QuotaExceeded("You've reached the chat request limit. Please try again later.")
            daily_left = settings.shared_tokens_per_day - daily_tokens
            if daily_requests >= settings.shared_requests_per_day or daily_left < prompt_bound + MIN_REPLY_TOKENS:
                raise QuotaExceeded("OserAI has reached its shared daily limit for this request. Try again later.")
            user_left = settings.user_tokens_per_day - user_tokens
            if user_left < prompt_bound + MIN_REPLY_TOKENS:
                raise QuotaExceeded("This request exceeds your remaining daily allowance. Try a shorter message or a new chat, or wait for the reset.")
            minute_left = settings.shared_tokens_per_minute - minute_tokens
            if minute_requests >= settings.shared_requests_per_minute or minute_left < prompt_bound + MIN_REPLY_TOKENS:
                raise QuotaExceeded("OserAI's shared minute limit is busy. Try a shorter message or try again in a minute.")
            reply_limit = min(settings.max_output_tokens, min(user_left, daily_left, minute_left) - prompt_bound)
            await db.execute("""
                INSERT INTO chat_pilot_usage(request_id, account, network, started_at, tokens)
                VALUES (%s, %s, %s, %s, %s)
            """, (request_id, account, network, now, prompt_bound + reply_limit))
            await db.execute("DELETE FROM chat_pilot_usage WHERE started_at < %s", (now - 7 * DAY,))
        return reply_limit

    async def finish(self, request_id: str, total_tokens: int):
        async with self.connection() as db:
            await db.execute(
                "UPDATE chat_pilot_usage SET tokens = %s WHERE request_id = %s", (total_tokens, request_id),
            )

    async def close(self):
        await self.pool.close()
