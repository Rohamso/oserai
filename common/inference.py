"""Contract shared by inference backends and coordinator payloads."""

from dataclasses import dataclass
from typing import Literal

InferenceBackend = Literal["ollama", "parallax"]


class InferenceError(Exception):
    """An inference request failed or returned an unusable response."""


@dataclass
class InferenceResult:
    output_text: str
    tokens_generated: int
    duration_s: float
    tokens_per_sec: float
