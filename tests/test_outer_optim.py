from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.outer_optim import average_state_dicts, nesterov_outer_step


def test_average_state_dicts_simple_mean():
    sds = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, 4.0])}]
    avg = average_state_dicts(sds)
    assert torch.allclose(avg["w"], torch.tensor([2.0, 3.0]))


def test_average_state_dicts_single_input_is_passthrough():
    sds = [{"w": torch.tensor([5.0])}]
    avg = average_state_dicts(sds)
    assert torch.allclose(avg["w"], torch.tensor([5.0]))


def test_average_state_dicts_requires_at_least_one():
    with pytest.raises(ValueError):
        average_state_dicts([])


def test_nesterov_outer_step_first_call_hand_computed():
    global_state = {"w": torch.tensor([10.0])}
    avg_delta = {"w": torch.tensor([1.0])}

    new_global, new_buffer = nesterov_outer_step(
        global_state, avg_delta, momentum_buffer=None, outer_lr=0.5, outer_momentum=0.9,
    )

    # buf = 0.9*0 + 1.0 = 1.0; update = 1.0 + 0.9*1.0 = 1.9; new = 10.0 - 0.5*1.9 = 9.05
    assert torch.allclose(new_buffer["w"], torch.tensor([1.0]))
    assert torch.allclose(new_global["w"], torch.tensor([9.05]))


def test_nesterov_outer_step_momentum_buffer_persists_across_calls():
    global_state = {"w": torch.tensor([10.0])}
    _, buf1 = nesterov_outer_step(
        global_state, {"w": torch.tensor([1.0])}, momentum_buffer=None, outer_lr=0.5, outer_momentum=0.9,
    )
    new_global, buf2 = nesterov_outer_step(
        {"w": torch.tensor([9.05])}, {"w": torch.tensor([2.0])},
        momentum_buffer=buf1, outer_lr=0.5, outer_momentum=0.9,
    )

    # buf = 0.9*1.0 + 2.0 = 2.9; update = 2.0 + 0.9*2.9 = 4.61; new = 9.05 - 0.5*4.61 = 6.745
    assert torch.allclose(buf2["w"], torch.tensor([2.9]))
    assert torch.allclose(new_global["w"], torch.tensor([6.745]))
