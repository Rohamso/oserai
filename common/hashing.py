"""Shared hashing helpers used by both the coordinator and the agent."""

import hashlib


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
