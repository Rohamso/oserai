"""DiLoCo outer optimizer: average worker pseudo-gradients, apply an
SGD-with-Nesterov-momentum step to the global weights.

Pseudo-gradient convention (per the DiLoCo paper): a worker's delta is
`theta_start_of_round - theta_after_H_local_steps`. Local training moves
theta toward lower loss, so this delta behaves exactly like a standard
gradient (points toward higher loss) and plugs directly into ordinary SGD.
"""

from __future__ import annotations

import torch

StateDict = dict[str, torch.Tensor]


def average_state_dicts(state_dicts: list[StateDict]) -> StateDict:
    if not state_dicts:
        raise ValueError("average_state_dicts requires at least one state_dict")
    keys = state_dicts[0].keys()
    return {
        key: torch.mean(torch.stack([sd[key].float() for sd in state_dicts]), dim=0)
        for key in keys
    }


def nesterov_outer_step(
    global_state: StateDict,
    avg_delta: StateDict,
    momentum_buffer: StateDict | None,
    outer_lr: float,
    outer_momentum: float,
) -> tuple[StateDict, StateDict]:
    """One outer SGD+Nesterov step. Returns (new_global_state, new_momentum_buffer)."""
    new_global: StateDict = {}
    new_buffer: StateDict = {}
    for key, grad in avg_delta.items():
        buf = momentum_buffer[key] if momentum_buffer is not None else torch.zeros_like(grad)
        buf = outer_momentum * buf + grad
        update = grad + outer_momentum * buf
        new_buffer[key] = buf
        new_global[key] = global_state[key] - outer_lr * update
    return new_global, new_buffer
