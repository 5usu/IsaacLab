# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backlash variants of the MicroDuck environments."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_microduck.assets import (
    MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG,
    MICRODUCK_ROLLERS_BACKLASH_CFG,
    MICRODUCK_WALK_BACKLASH_CFG,
)
from isaaclab_microduck.tasks import mdp

from .base_env_cfg import (
    MicroDuckVelocityFlatEnvCfg,
    MicroDuckVelocityFlatEnvCfg_PLAY,
    MicroDuckVelocityRoughEnvCfg,
    MicroDuckVelocityRoughEnvCfg_PLAY,
)
from .recovery_env_cfg import (
    MicroDuckSitStandFlatEnvCfg,
    MicroDuckSitStandFlatEnvCfg_PLAY,
    MicroDuckSitStandRoughEnvCfg,
    MicroDuckSitStandRoughEnvCfg_PLAY,
    MicroDuckStandUpFlatEnvCfg,
    MicroDuckStandUpFlatEnvCfg_PLAY,
    MicroDuckStandUpRoughEnvCfg,
    MicroDuckStandUpRoughEnvCfg_PLAY,
    MicroDuckVelStandFlatEnvCfg,
    MicroDuckVelStandFlatEnvCfg_PLAY,
    MicroDuckVelStandRoughEnvCfg,
    MicroDuckVelStandRoughEnvCfg_PLAY,
)
from .roller_env_cfg import (
    MicroDuckRollerCrouchFlatEnvCfg,
    MicroDuckRollerCrouchFlatEnvCfg_PLAY,
    MicroDuckRollerSlopeEnvCfg,
    MicroDuckRollerSlopeEnvCfg_PLAY,
    MicroDuckSwizzleFlatEnvCfg,
    MicroDuckSwizzleFlatEnvCfg_PLAY,
    MicroDuckVelocityRollersFlatEnvCfg,
    MicroDuckVelocityRollersFlatEnvCfg_PLAY,
)
from .skill_env_cfg import (
    MicroDuckBackflipFlatEnvCfg,
    MicroDuckBackflipFlatEnvCfg_PLAY,
    MicroDuckBallKickFlatEnvCfg,
    MicroDuckBallKickFlatEnvCfg_PLAY,
    MicroDuckGroundPickFlatEnvCfg,
    MicroDuckGroundPickFlatEnvCfg_PLAY,
    MicroDuckGroundPickRoughEnvCfg,
    MicroDuckGroundPickRoughEnvCfg_PLAY,
)

_BACKLASH_LEFT_FOOT_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll_left_hip_yaw_motor/yaw2roll/"
    "hip_l_left_hip_roll_motor/hip_l/upper_leg_left_left_hip_pitch_motor/upper_leg_left/"
    "leg_left_knee_motor/leg/{ankle}_left_ankle_motor/{ankle}"
)
_BACKLASH_RIGHT_FOOT_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll_right_hip_yaw_motor/bearing_roll/"
    "hip_l_2_right_hip_roll_motor/hip_l_2/upper_leg_right_right_hip_pitch_motor/upper_leg_right/"
    "leg_2_right_knee_motor/leg_2/{ankle}_right_ankle_motor/{ankle}"
)
_BACKLASH_HEAD_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/neck_neck_pitch_motor/neck/"
    "neck_pitch_head_pitch_motor/neck_pitch/yaw_roll_motion_head_yaw_motor/yaw_roll_motion/"
    "jaw_soft_head_roll_motor/jaw_soft"
)


def _apply_backlash(cfg, robot_cfg, *, roller: bool = False) -> None:
    """Swap the robot and preserve the 14-element servo encoder interface."""
    cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name)
        group.joint_pos.func = mdp.joint_pos_rel_backlash
        group.joint_vel.func = mdp.joint_vel_rel_backlash

    left_ankle = "ankle_l_v1" if roller else "ankle_left"
    right_ankle = "ankle_r_v1" if roller else "ankle_right"
    cfg.scene.left_foot_contact.prim_path = _BACKLASH_LEFT_FOOT_PATH.replace("{ankle}", left_ankle)
    cfg.scene.right_foot_contact.prim_path = _BACKLASH_RIGHT_FOOT_PATH.replace("{ankle}", right_ankle)
    cfg.scene.head_ground_contact.prim_path = _BACKLASH_HEAD_PATH


class _WalkBacklashMixin:
    def __post_init__(self):
        super().__post_init__()
        _apply_backlash(self, MICRODUCK_WALK_BACKLASH_CFG)


class _FullCollisionBacklashMixin:
    def __post_init__(self):
        super().__post_init__()
        _apply_backlash(self, MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG)


class _RollerBacklashMixin:
    def __post_init__(self):
        super().__post_init__()
        _apply_backlash(self, MICRODUCK_ROLLERS_BACKLASH_CFG, roller=True)


@configclass
class MicroDuckVelocityFlatBacklashEnvCfg(_WalkBacklashMixin, MicroDuckVelocityFlatEnvCfg):
    pass


@configclass
class MicroDuckVelocityFlatBacklashEnvCfg_PLAY(_WalkBacklashMixin, MicroDuckVelocityFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckVelocityRoughBacklashEnvCfg(_WalkBacklashMixin, MicroDuckVelocityRoughEnvCfg):
    pass


@configclass
class MicroDuckVelocityRoughBacklashEnvCfg_PLAY(_WalkBacklashMixin, MicroDuckVelocityRoughEnvCfg_PLAY):
    pass


@configclass
class MicroDuckVelStandFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckVelStandFlatEnvCfg):
    pass


@configclass
class MicroDuckVelStandFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckVelStandFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckVelStandRoughBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckVelStandRoughEnvCfg):
    pass


@configclass
class MicroDuckVelStandRoughBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckVelStandRoughEnvCfg_PLAY):
    pass


@configclass
class MicroDuckStandUpFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckStandUpFlatEnvCfg):
    pass


@configclass
class MicroDuckStandUpFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckStandUpFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckStandUpRoughBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckStandUpRoughEnvCfg):
    pass


@configclass
class MicroDuckStandUpRoughBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckStandUpRoughEnvCfg_PLAY):
    pass


@configclass
class MicroDuckSitStandFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckSitStandFlatEnvCfg):
    pass


@configclass
class MicroDuckSitStandFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckSitStandFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckSitStandRoughBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckSitStandRoughEnvCfg):
    pass


@configclass
class MicroDuckSitStandRoughBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckSitStandRoughEnvCfg_PLAY):
    pass


@configclass
class MicroDuckGroundPickFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckGroundPickFlatEnvCfg):
    pass


@configclass
class MicroDuckGroundPickFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckGroundPickFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckGroundPickRoughBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckGroundPickRoughEnvCfg):
    pass


@configclass
class MicroDuckGroundPickRoughBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckGroundPickRoughEnvCfg_PLAY):
    pass


@configclass
class MicroDuckBallKickFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckBallKickFlatEnvCfg):
    pass


@configclass
class MicroDuckBallKickFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckBallKickFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckBackflipFlatBacklashEnvCfg(_FullCollisionBacklashMixin, MicroDuckBackflipFlatEnvCfg):
    pass


@configclass
class MicroDuckBackflipFlatBacklashEnvCfg_PLAY(_FullCollisionBacklashMixin, MicroDuckBackflipFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckVelocityRollersFlatBacklashEnvCfg(_RollerBacklashMixin, MicroDuckVelocityRollersFlatEnvCfg):
    pass


@configclass
class MicroDuckVelocityRollersFlatBacklashEnvCfg_PLAY(_RollerBacklashMixin, MicroDuckVelocityRollersFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckSwizzleFlatBacklashEnvCfg(_RollerBacklashMixin, MicroDuckSwizzleFlatEnvCfg):
    pass


@configclass
class MicroDuckSwizzleFlatBacklashEnvCfg_PLAY(_RollerBacklashMixin, MicroDuckSwizzleFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckRollerCrouchFlatBacklashEnvCfg(_RollerBacklashMixin, MicroDuckRollerCrouchFlatEnvCfg):
    pass


@configclass
class MicroDuckRollerCrouchFlatBacklashEnvCfg_PLAY(_RollerBacklashMixin, MicroDuckRollerCrouchFlatEnvCfg_PLAY):
    pass


@configclass
class MicroDuckRollerSlopeBacklashEnvCfg(_RollerBacklashMixin, MicroDuckRollerSlopeEnvCfg):
    pass


@configclass
class MicroDuckRollerSlopeBacklashEnvCfg_PLAY(_RollerBacklashMixin, MicroDuckRollerSlopeEnvCfg_PLAY):
    pass
