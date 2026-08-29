# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration-level tests for MicroDuck command generators."""

from isaaclab_microduck.tasks.mdp.commands import (
    CyclicPhaseCommandCfg,
    MicroDuckVelocityCommandCfg,
    RelativeHeadingVelocityCommandCfg,
    SitStandCommandCfg,
    UniformVectorCommandCfg,
)


def test_command_configs_resolve_their_terms():
    velocity_ranges = MicroDuckVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
    )

    assert MicroDuckVelocityCommandCfg(asset_name="robot", ranges=velocity_ranges).class_type.endswith(
        ":MicroDuckVelocityCommand"
    )
    assert RelativeHeadingVelocityCommandCfg(asset_name="robot", ranges=velocity_ranges).class_type.endswith(
        ":RelativeHeadingVelocityCommand"
    )
    assert CyclicPhaseCommandCfg().class_type.endswith(":CyclicPhaseCommand")
    assert UniformVectorCommandCfg(ranges=((-1.0, 1.0),)).class_type.endswith(":UniformVectorCommand")
    assert SitStandCommandCfg(asset_name="robot", ranges=velocity_ranges).class_type.endswith(":SitStandCommand")


def test_vector_command_requires_explicit_ranges():
    cfg = UniformVectorCommandCfg(ranges=((-0.1, 0.2),) * 4, zero_command_prob=0.1)

    assert len(cfg.ranges) == 4
    assert cfg.zero_command_prob == 0.1
