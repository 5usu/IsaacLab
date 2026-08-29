# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_microduck.actuators import BamActuator, BamActuatorCfg

from isaaclab.utils.types import ArticulationActions


def _make_actuator(*, backlash: bool = False) -> BamActuator:
    joint_names = ["left_hip", "right_hip", "passive_jaw"]
    if backlash:
        joint_names += ["passive_left_hip_backlash", "passive_right_hip_backlash"]
    cfg = BamActuatorCfg(
        joint_names_expr=[".*"],
        minimum_delay=0,
        maximum_delay=0,
        supply_voltage_range=(7.5, 7.5),
        voltage_drop_gain_range=(0.0, 0.0),
        encoder_through_backlash=backlash,
    )
    return BamActuator(
        cfg,
        joint_names=joint_names,
        joint_ids=torch.arange(len(joint_names)),
        num_envs=2,
        device="cpu",
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
        effort_limit=10.0,
        velocity_limit=20.0,
    )


def test_passive_joints_receive_no_effort() -> None:
    actuator = _make_actuator()
    action = ArticulationActions(
        joint_positions=torch.tensor([[0.2, -0.2, 0.0], [0.2, -0.2, 0.0]]),
        joint_velocities=torch.zeros(2, 3),
        joint_efforts=torch.zeros(2, 3),
    )
    output = actuator.compute(action, torch.zeros(2, 3), torch.zeros(2, 3))

    assert output.joint_positions is None
    assert output.joint_velocities is None
    assert torch.all(output.joint_efforts[:, :2].abs() > 0.0)
    assert torch.equal(output.joint_efforts[:, 2], torch.zeros(2))


def test_backlash_position_is_used_as_encoder_feedback() -> None:
    actuator = _make_actuator(backlash=True)
    target = torch.zeros(2, 5)
    target[:, :2] = 0.1
    joint_pos = torch.zeros(2, 5)
    joint_pos[:, 3:] = 0.1
    action = ArticulationActions(
        joint_positions=target,
        joint_velocities=torch.zeros(2, 5),
        joint_efforts=torch.zeros(2, 5),
    )
    output = actuator.compute(action, joint_pos, torch.zeros(2, 5))

    assert torch.allclose(output.joint_efforts, torch.zeros(2, 5), atol=1.0e-7)
