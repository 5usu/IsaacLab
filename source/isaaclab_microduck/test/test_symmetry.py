# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MicroDuck bilateral symmetry augmentation."""

import torch
from isaaclab_microduck.tasks.manager_based.microduck.agents.symmetry import compute_symmetric_states
from tensordict import TensorDict


def test_symmetry_shapes_and_involution():
    policy = torch.randn(3, 61)
    critic = torch.randn(3, 64)
    actions = torch.randn(3, 14)
    obs = TensorDict({"policy": policy, "critic": critic}, batch_size=[3])

    mirrored_obs, mirrored_actions = compute_symmetric_states(None, obs, actions)
    assert mirrored_obs.batch_size == torch.Size([6])
    assert mirrored_actions.shape == (6, 14)

    mirrored_only = TensorDict({"policy": mirrored_obs["policy"][3:], "critic": critic}, batch_size=[3])
    restored_obs, restored_actions = compute_symmetric_states(None, mirrored_only, mirrored_actions[3:])
    torch.testing.assert_close(restored_obs["policy"][3:], policy)
    torch.testing.assert_close(restored_actions[3:], actions)
