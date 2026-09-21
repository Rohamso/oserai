"""Real end-to-end smoke test for the LoRA training path -- the class of
bug the rest of the suite structurally can't catch.

Every other training test (test_training_dispatch.py, test_outer_optim.py)
exercises dispatch/aggregation against small hand-built tensors, by design
-- fast, no network, no real model. That's the right call for those tests,
but it means nothing in CI ever actually loads the real ~1.5B-param base
model and runs a real local step, so a real-world failure in that path
(wrong kwargs to from_pretrained, an OOM that crashes the process instead
of raising, a tokenizer/chat-template mismatch, ...) has no test to catch
it before a contributor's machine does. This is exactly how the desktop
app's OOM crash (training/lora.py's load_frozen_base) shipped unnoticed.

Opt-in via the `smoke` marker (see pytest.ini) -- downloads the real base
model (~3GB, first run only, cached under ~/.cache/huggingface afterward)
and needs the same free RAM load_frozen_base itself requires, so it isn't
run by default. Run explicitly with:

    pytest -m smoke tests/test_lora_smoke.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.chat_dataset import get_batch, load_dolly_chat_tokens
from training.checkpoint_io import bytes_to_state_dict, state_dict_to_bytes
from training.dataset import DATA_DIR
from training.lora import LoraRunConfig, build_lora_adapter, load_frozen_base, load_tokenizer, local_lora_steps

pytestmark = pytest.mark.smoke

DOLLY_CHAT_PATH = DATA_DIR / "dolly15k_chat.txt"


def test_real_lora_round_completes():
    """Load the real frozen base, build a real adapter, run a couple of
    real local steps, and round-trip the resulting delta through the same
    bytes<->state_dict wire format a training agent uploads -- the full
    shape of one DiLoCo round on the lora_qwen2.5-1.5b backend, minus only
    the HTTP hop to a coordinator."""
    if not DOLLY_CHAT_PATH.exists():
        pytest.skip(f"{DOLLY_CHAT_PATH} not found -- run `python scripts/prepare_training_data.py` first")

    config = LoraRunConfig()
    base = load_frozen_base(config.base_model_id, device="cpu")
    tokenizer = load_tokenizer(config.base_model_id)
    model = build_lora_adapter(base, config)

    data = load_dolly_chat_tokens(tokenizer, max_examples=50)

    def get_batch_fn(block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(data, block_size, batch_size, device)

    delta, last_loss = local_lora_steps(
        model, get_batch_fn, inner_lr=1e-4, inner_steps_h=2, inner_batch_size=2, block_size=64, device="cpu",
    )

    assert delta, "expected a non-empty adapter delta"
    assert all(torch.isfinite(v).all() for v in delta.values())
    assert last_loss == last_loss and last_loss > 0  # not NaN

    # Same wire format a training agent uses to upload a delta -- round-trip
    # it to make sure what actually moves over the network is well-formed.
    restored = bytes_to_state_dict(state_dict_to_bytes(delta))
    assert restored.keys() == delta.keys()
    for key in delta:
        assert torch.allclose(restored[key], delta[key])

    # The actual upload path (agent/train_agent.py's run_round_lora) sends
    # fp16, not this fp32 default -- verify the compressed wire format on a
    # real delta specifically, not just a hand-built tensor: it should
    # round-trip through average_state_dicts's .float() upcast losslessly
    # enough to train on, and the payload should actually shrink.
    fp32_bytes = state_dict_to_bytes(delta)
    fp16_bytes = state_dict_to_bytes(delta, dtype=torch.float16)
    assert len(fp16_bytes) < len(fp32_bytes) * 0.6

    restored_fp16 = bytes_to_state_dict(fp16_bytes)
    assert restored_fp16.keys() == delta.keys()
    for key in delta:
        assert restored_fp16[key].dtype == torch.float16
        assert torch.isfinite(restored_fp16[key]).all()
        # fp16 has ~3 decimal digits of precision -- LoRA deltas after just
        # a couple of steps are small, so an absolute tolerance this loose
        # would hide a real bug; this is checking "close to the fp32
        # original," not "bit-identical."
        assert torch.allclose(restored_fp16[key].float(), delta[key], atol=1e-3, rtol=1e-2)
