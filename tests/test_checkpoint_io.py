from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.checkpoint_io import bytes_to_state_dict, state_dict_to_bytes
from training.outer_optim import average_state_dicts


def test_state_dict_to_bytes_default_preserves_dtype_and_values():
    sd = {"w": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    round_tripped = bytes_to_state_dict(state_dict_to_bytes(sd))
    assert round_tripped["w"].dtype == torch.float32
    assert torch.equal(round_tripped["w"], sd["w"])


def test_state_dict_to_bytes_fp16_shrinks_payload():
    # A few thousand elements so fixed torch.save framing overhead doesn't
    # dominate and mask the ~2x element-size difference fp16 should give.
    sd = {"w": torch.randn(4096, dtype=torch.float32)}
    fp32_bytes = state_dict_to_bytes(sd)
    fp16_bytes = state_dict_to_bytes(sd, dtype=torch.float16)
    assert len(fp16_bytes) < len(fp32_bytes) * 0.6


def test_state_dict_to_bytes_fp16_round_trips_through_averaging():
    # A LoRA/DiLoCo delta: serialize fp16 (agent upload path), deserialize,
    # then feed straight into average_state_dicts exactly like
    # coordinator/training_dispatch.py does -- its own .float() upcast
    # should absorb the fp16 tensor with no separate decompression step.
    original = {"w": torch.tensor([1.0, -2.5, 0.125], dtype=torch.float32)}
    compressed = bytes_to_state_dict(state_dict_to_bytes(original, dtype=torch.float16))
    assert compressed["w"].dtype == torch.float16

    avg = average_state_dicts([compressed])
    assert avg["w"].dtype == torch.float32
    # fp16 has ~3 decimal digits of precision -- values here are exactly
    # representable in fp16, so this should be an exact match, not just close.
    assert torch.allclose(avg["w"], original["w"], atol=1e-3)
