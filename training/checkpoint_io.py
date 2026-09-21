"""Wire format for state_dicts moving over HTTP as opaque bytes.

Used by both the coordinator (aggregation, generation) and every training
agent (local steps) -- one definition keeps them in sync.
"""

from __future__ import annotations

import io

import torch


def state_dict_to_bytes(state_dict: dict[str, torch.Tensor], dtype: torch.dtype | None = None) -> bytes:
    """dtype: downcast every tensor before serializing (e.g. torch.float16
    for a delta upload, halving the bytes that cross a contributor's home
    connection) -- None keeps each tensor's own dtype, the right choice for
    checkpoint storage where full precision should be preserved. The
    aggregation math in training/outer_optim.py upcasts via .float() before
    averaging, so a fp16 delta round-trips through bytes_to_state_dict and
    into that math with no separate decompression step needed."""
    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location=map_location, weights_only=True)
