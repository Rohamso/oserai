"""Wire format for state_dicts moving over HTTP as opaque bytes.

Used by both the coordinator (aggregation, generation) and every training
agent (local steps) -- one definition keeps them in sync.
"""

from __future__ import annotations

import io

import torch


def state_dict_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location=map_location, weights_only=True)
