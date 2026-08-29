# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-data configurations for MicroDuck command generators."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .runtime_commands import (
        CyclicPhaseCommand,
        MicroDuckVelocityCommand,
        RelativeHeadingVelocityCommand,
        SitStandCommand,
        UniformVectorCommand,
    )


@configclass
class MicroDuckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with explicit turn-in-place samples."""

    class_type: type[MicroDuckVelocityCommand] | str = "{DIR}.runtime_commands:MicroDuckVelocityCommand"
    rel_turn_in_place_envs: float = 0.0
    """Fraction of resamples reserved for turning in place."""


@configclass
class RelativeHeadingVelocityCommandCfg(MicroDuckVelocityCommandCfg):
    """Velocity command whose yaw-rate slot carries relative heading error."""

    class_type: type[RelativeHeadingVelocityCommand] | str = "{DIR}.runtime_commands:RelativeHeadingVelocityCommand"


@configclass
class CyclicPhaseCommandCfg(CommandTermCfg):
    """Configuration for a continuous cosine/sine phase command."""

    class_type: type[CyclicPhaseCommand] | str = "{DIR}.runtime_commands:CyclicPhaseCommand"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    period: float = 4.0
    randomize_phase: bool = True


@configclass
class UniformVectorCommandCfg(CommandTermCfg):
    """Configuration for a sampled vector command."""

    class_type: type[UniformVectorCommand] | str = "{DIR}.runtime_commands:UniformVectorCommand"
    ranges: tuple[tuple[float, float], ...] = MISSING
    zero_command_prob: float = 0.0


@configclass
class SitStandCommandCfg(UniformVelocityCommandCfg):
    """Configuration for a binary posture command with a slewed target."""

    class_type: type[SitStandCommand] | str = "{DIR}.runtime_commands:SitStandCommand"
    sit_probability: float = 0.5
    ramp_time: float = 2.0
    sit_height: float = 0.060
    stand_height: float = 0.115


# Configuration aliases retained for the original task vocabulary.
VelocityCommandCommandOnlyCfg = MicroDuckVelocityCommandCfg
GroundPickPhaseCommandCfg = CyclicPhaseCommandCfg
UniformPoseCommandCfg = UniformVectorCommandCfg
