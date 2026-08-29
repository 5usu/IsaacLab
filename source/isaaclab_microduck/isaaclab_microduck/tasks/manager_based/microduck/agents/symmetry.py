# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bilateral symmetry transform for the unified MicroDuck policy interface."""

from __future__ import annotations

import torch
from tensordict import TensorDict

_JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_JOINT_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

# Observation order is defined by ObservationsCfg.PolicyCfg in base_env_cfg.py.
_OBS_PERM = (
    (0, 1, 2)
    + (3, 4, 5)
    + (6, 7, 8)
    + tuple(9 + index for index in _JOINT_PERM)
    + tuple(23 + index for index in _JOINT_PERM)
    + tuple(37 + index for index in _JOINT_PERM)
    + (51, 52, 53, 54)
    + (55, 56, 57, 58, 59, 60)
)
_OBS_SIGN = (
    (-1.0, 1.0, -1.0)
    + (1.0, -1.0, 1.0)
    + (1.0, -1.0, -1.0)
    + _JOINT_SIGN
    + _JOINT_SIGN
    + _JOINT_SIGN
    + (1.0, 1.0, -1.0, -1.0)
    + (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
)

_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _transform_tensors(device: torch.device):
    tensors = _CACHE.get(device)
    if tensors is None:
        tensors = (
            torch.tensor(_OBS_PERM, dtype=torch.long, device=device),
            torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device),
            torch.tensor(_JOINT_PERM, dtype=torch.long, device=device),
            torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device),
        )
        _CACHE[device] = tensors
    return tensors


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append left-right mirrored policy observations and actions to a batch."""
    del env
    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        policy = obs["policy"]
        obs_perm, obs_sign, _, _ = _transform_tensors(policy.device)
        if policy.shape[-1] != len(obs_perm):
            raise ValueError(f"Expected 61 policy observations, received {policy.shape[-1]}.")
        obs_aug["policy"][batch_size:] = policy[:, obs_perm] * obs_sign

    actions_aug = None
    if actions is not None:
        _, _, action_perm, action_sign = _transform_tensors(actions.device)
        if actions.shape[-1] != len(action_perm):
            raise ValueError(f"Expected 14 actions, received {actions.shape[-1]}.")
        mirrored = actions[:, action_perm] * action_sign
        actions_aug = torch.cat((actions, mirrored), dim=0)

    return obs_aug, actions_aug
